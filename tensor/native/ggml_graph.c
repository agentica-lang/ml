/* The compiled graphs of the ggml backend (`src/lower.agc`): ONE ggml graph
   per traced evaluation, built once per structure and cached for the process.

   A graph is built node by node in the order `lower.agc` sends (a topological
   order); `compile` makes the cgraph, gives the fills, the finiteness checks
   and the index checks of `gather` one small buffer, and checks every node
   against the backend. The leaves (inputs, constants) and the outputs own no
   memory: every call points them at the caller's tensors (`bind`) and at
   fresh ones (`output`), so nothing is copied in or out; a leaf or output of
   another type (an I32 tensor against the F32 graph) goes through a buffer of
   its own and a converting copy instead. Intermediates are not the graph's
   either: one graph allocator per device holds them, shared by every graph of
   that device. So a graph waiting in the cache costs its tensors' metadata
   only. Every run, under the device lock, re-plans the shared allocator when
   another graph ran last, computes, and points the leaves and outputs back at
   nothing. The allocator keeps the largest buffer any graph needed for the
   rest of the process.

   Three locks, in this order: the device lock (1), `cache_lock` (2) and the
   deferred list's lock (3). `cache_lock` (the slots, who holds a graph, the
   pins, the counters) is held briefly and is all a lease's or a pin's free
   hook takes; the device lock is held through builds and runs. A graph the
   hook must drop is destroyed under the device lock later (`tg_defer`).

   Layout: a row-major shape [d0, ..., dk] is a ggml tensor with ne[0] = dk,
   ne[1] = dk-1, ..., so trailing-axis broadcasting is ggml's own repeat.

   A node may be "lazy": its tensor holds fewer elements than its recorded
   shape and repeats to it (a fill, a `broadcast_to`). Binary ops take a lazy
   right operand as ggml's broadcast; anything else materializes it. A fill is
   "zero" when the trace's value is 0: the zero branch of a `where` drops out.
   Zero-ness is part of the cache key, the value itself is set per run.

   Every value is F32 here: a Bool is 0/1 and an integer exact below 2^24. */
#include "ggml_shared.h"

#include "ggml-alloc.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The op codes `lower.agc` sends; keep both lists in one order. */
enum {
  OP_LEAF = 0, /* an input or a constant, bound per run */
  OP_FULL,     /* a fill: one element, lazy; ints: [zero] */
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_DIV,
  OP_NEG,
  OP_EXP,
  OP_LOG,
  OP_RELU,
  OP_SQRT,
  OP_SIGMOID,
  OP_TANH,
  OP_MATMUL,
  OP_TRANSPOSE,
  OP_RESHAPE,
  OP_BROADCAST,
  OP_SUM, /* ints: the input's rank, then the reduced axes */
  OP_MEAN,
  OP_MAX,
  OP_IDENTITY, /* stop_grad, copy, contiguous, a Bool or int cast to float */
  OP_ARANGE,   /* ints: start, stop, step */
  OP_EQ,
  OP_LT,
  OP_WHERE,
  OP_SLICE,  /* ints: the starts, row-major */
  OP_GATHER, /* ints: the source's rank */
  OP_GELU_ERF,
  OP_CONCAT, /* ints: the axis, then 1 for a stack (each arg gains that axis) */
  OP_CONV1D, /* ints: stride, padding */
  OP_SCATTER_ADD, /* ints: the rows */
  OP_PERMUTE,     /* ints: the axes, row-major */
  OP_ATTENTION,   /* args: q, k, v, then an optional mask; ints: the bits of the scale, a double, and 1 when flash attention may take it */
  OP_ROPE,        /* args: x, positions; ints: dims, neox, the bits of the base, inverse */
  OP_BATCH_MATMUL,
  OP_NORM,        /* a layer norm over the last axis; ints: eps as the bits of an F32 */
  OP_RMS_NORM,    /* an rms norm over the last axis; ints: eps as the bits of an F32 */
  OP_SET_ROWS,    /* args: a cache leaf, the rows, their positions; written in place, its result is an output that is the leaf */
  OP_TOP_K,       /* the indices of the k largest of a vector, as F32; ints: k */
  OP_CAST,        /* ints: 1 for F16, 0 for F32; a node that already has the type is passed through */
  OP_COUNT
};

static const char *op_names[OP_COUNT] = {
  "leaf", "full", "add", "sub", "mul", "div", "neg", "exp", "log", "relu", "sqrt", "sigmoid", "tanh",
  "matmul", "transpose", "reshape", "broadcast_to", "sum", "mean", "max", "identity", "arange", "eq", "lt", "where",
  "slice", "gather", "gelu_erf", "concat", "conv1d", "scatter_add", "permute", "attention", "rope", "batch_matmul",
  "norm", "rms_norm", "set_rows", "top_k", "cast",
};

/* A leaf or an output: the tensor of the graph, its own memory (made on the
   first call that needs it) and, for this call, the tensor it converts to or
   from. */
struct port {
  struct ggml_tensor *t;
  ggml_backend_buffer_t own;
  struct ggml_tensor *conv; /* the bound tensor, converted to or from `own`; else NULL. Its memory is `held` */
  bool set;                               /* bound or prepared for the next run */
  bool donated;                           /* a leaf a `set_rows` writes in place, or the output that is that leaf: no copy in or out */
  struct tg_storage *held;                /* the memory of the bound tensor, kept until the run is over */
};

struct gnode {
  struct ggml_tensor *t;
  int64_t ne[4];
  int op;
  int lazy; /* t has fewer elements than ne and repeats to it */
  int zero; /* a fill of 0.0, possibly broadcast or reshaped */
  int arg;  /* a transpose's argument, for matmul's layout */
  int in_place; /* a set_rows that writes its leaf: its result is that leaf, an output with no copy and no check */
  int rank; /* of the recorded shape */
  struct ggml_tensor *guard; /* a gather's or scatter_add's count of indices out of range */
  struct ggml_tensor *dense; /* the contiguous copy of a view, made once for every op that needs one */
  struct ggml_tensor *at; /* the node as the I32 positions of a rope, made once */
  struct ggml_tensor *half; /* the node as the F16 mask of a flash attention, made once */
};

#define MAX_OUTPUTS 256
#define MAX_ARGS 256
#define MAX_ENTRIES 32

/* One graph of the cache. */
struct entry {
  struct deferred link; /* to destroy it from a free hook */
  char *key;
  int64_t device;
  ggml_backend_t backend;  /* the device's, open for the process */
  struct ggml_context *ctx; /* op tensors and the graph; no data */
  struct ggml_context *fill_ctx; /* fills and checks, in fill_buffer */
  ggml_backend_buffer_t fill_buffer;
  struct ggml_context *port_ctx; /* leaves and outputs; no memory */
  struct ggml_cgraph *graph;
  struct gnode *nodes;
  int64_t count, capacity;
  struct port *leaves;
  int64_t leaf_count;
  struct ggml_tensor *fill_store; /* one F32 per fill; each fill is a view */
  int64_t fill_count;
  struct port outs[MAX_OUTPUTS];
  struct ggml_tensor *checks; /* the sum of 0 * output per output (NaN unless finite), then the guards */
  int64_t output_count, guard_count;
  struct ggml_tensor **inner; /* what the shared allocator places, every run */
  int64_t inner_count;
  bool compiled, in_use;
  int pins;     /* pins on it: never evicted while above 0 (under cache_lock) */
  int64_t used; /* the tick of its last lease, for eviction */
};

/* The intermediates' allocator of each device and the graph it last planned. */
struct device_state {
  ggml_gallocr_t galloc;
  struct entry *planned;
};

/* Lock order: the device lock (1), then cache_lock (2), then the deferred list's lock (3). Pins and the pin count of an
   entry are under cache_lock alone; the memory of a tensor that the last holder lets go of is queued (level 3) and freed at
   the next hold of the device lock. */
CC_LOCK_LEVEL(2) CC_NO_SYSCALL_UNDER() static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static struct entry *cache[MAX_ENTRIES];
static int64_t compile_count = 0, tick = 0;
static struct device_state devices[9]; /* under the device lock */

/* Call under the device lock. */
static void destroy(struct entry *e) {
  int64_t d = e->device + 1;

  int64_t i;

  if (d >= 0 && d < 9 && devices[d].planned == e) devices[d].planned = NULL;
  for (i = 0; e->leaves != NULL && i < e->leaf_count; i++) {
    if (e->leaves[i].own != NULL) ggml_backend_buffer_free(e->leaves[i].own);
  }
  for (i = 0; i < e->output_count; i++) {
    if (e->outs[i].own != NULL) ggml_backend_buffer_free(e->outs[i].own);
  }
  if (e->fill_buffer != NULL) ggml_backend_buffer_free(e->fill_buffer);
  if (e->fill_ctx != NULL) ggml_free(e->fill_ctx);
  if (e->port_ctx != NULL) ggml_free(e->port_ctx);
  if (e->ctx != NULL) ggml_free(e->ctx);
  free(e->nodes);
  free(e->leaves);
  free(e->inner);
  free(e->key);
  free(e);
}

/* Points every leaf and output at nothing and lets go of their memory, so no
   call's tensor stays reachable from the cache. */
static void unbind(struct port *port) {
  port->t->data = NULL;
  port->t->buffer = NULL;
  port->conv = NULL;
  port->set = false;
  if (port->held != NULL) tg_storage_release(port->held);
  port->held = NULL;
}

static void release(struct entry *e) {
  int64_t i;

  for (i = 0; e->leaves != NULL && i < e->leaf_count; i++) unbind(&e->leaves[i]);
  for (i = 0; i < e->output_count; i++) unbind(&e->outs[i]);
}

static void destroy_later(void *e) { destroy((struct entry *)e); }

/* Gives the graph back to the cache, or drops it when it never compiled or
   found no slot. A hook: it never waits for the device lock. */
CC_SHORT void ggml_Lease_free(void *state) {
  struct ggml_Lease_state *self = (struct ggml_Lease_state *)state;
  int i, slot = -1;
  bool drop;

  if (self->e == NULL) return;
  release(self->e); /* the lease's graph: no one else touches its ports */
  pthread_mutex_lock(&cache_lock);
  self->e->in_use = false;
  for (i = 0; i < MAX_ENTRIES; i++) {
    if (cache[i] == self->e) slot = i;
  }
  drop = !self->e->compiled || slot < 0;
  if (drop && slot >= 0) cache[slot] = NULL;
  pthread_mutex_unlock(&cache_lock);
  if (drop) {
    self->e->link.fn = destroy_later;
    self->e->link.arg = self->e;
    tg_defer(&self->e->link);
  }
  self->e = NULL;
}

static struct ggml_context *new_context(size_t bytes) {
  struct ggml_init_params params;

  params.mem_size = bytes;
  params.mem_buffer = NULL;
  params.no_alloc = true;
  return ggml_init(params);
}

/* Room for the graph: a traced node lowers to a handful of ggml tensors at
   most (a matmul's transpose and cont, a reduction's reshapes; a `gather` or
   `scatter_add` is the largest, about 14 with its index guard); `node` and
   `compile` check the room is there. */
static size_t graph_size(int64_t capacity) { return (size_t)(capacity * 16 + 8 * MAX_OUTPUTS + 64); }

/* True when the graph's context has room for `tensors` more tensor headers
   and, before `compile`, for the graph itself. */
static bool room(struct entry *p, size_t tensors) {
  size_t graph = p->graph != NULL ? 0 : ggml_graph_overhead_custom(graph_size(p->capacity), false);

  return ggml_used_mem(p->ctx) + graph + tensors * ggml_tensor_overhead() <= ggml_get_mem_size(p->ctx);
}

/* The most nodes a graph takes. Its metadata is allocated up front, about
   16 tensor headers (6 KB) per node, so this bounds one graph at about
   400 MB of metadata; a traced MLP step is about a hundred nodes. */
#define MAX_NODES 65536

/* A free cache slot, unlinking the least recently used graph nobody holds
   into `victim` for the caller to destroy; -1 when every slot is held. Call
   under cache_lock. */
static int free_slot(struct entry **victim) {
  int i, oldest = -1;

  *victim = NULL;
  for (i = 0; i < MAX_ENTRIES; i++) {
    if (cache[i] == NULL) return i;
    if (!cache[i]->in_use && cache[i]->pins == 0 && (oldest < 0 || cache[i]->used < cache[oldest]->used)) oldest = i;
  }
  if (oldest >= 0) {
    *victim = cache[oldest];
    cache[oldest] = NULL;
  }
  return oldest;
}

/* The compiled graph cached under `key` on `device` that no one holds, marked
   held, or NULL. */
static struct entry *cached(int64_t device, ng_str key) {
  struct entry *e = NULL;
  int i;

  pthread_mutex_lock(&cache_lock);
  for (i = 0; i < MAX_ENTRIES && e == NULL; i++) {
    struct entry *c = cache[i];

    if (c != NULL && c->compiled && !c->in_use && c->device == device && strlen(c->key) == key.len &&
        memcmp(c->key, key.data, key.len) == 0) {
      e = c;
      e->in_use = true;
      e->used = ++tick;
    }
  }
  pthread_mutex_unlock(&cache_lock);
  return e;
}

an_term ggml_acquire(an_env *env, int64_t device, ng_str key, int64_t capacity, bool keep) {
  an_term term;
  struct ggml_Lease_state *lease = ggml_Lease_new(env, &term);
  struct entry *e, *victim;
  ggml_backend_t backend;
  char why[160];
  size_t size;
  int i;

  if (capacity < 1 || capacity > MAX_NODES) return tg_error(env, "a graph takes 1..65536 nodes");
  e = cached(device, key);
  if (e != NULL) {
    lease->e = e;
    return ggml_result_Lease_ok(env, term);
  }
  tg_lock_device();
  backend = tg_backend(device, why, sizeof why);
  if (backend == NULL) {
    tg_unlock_device();
    return tg_error(env, why);
  }
  e = (struct entry *)calloc(1, sizeof *e);
  size = graph_size(capacity);
  if (e != NULL) {
    e->device = device;
    e->backend = backend;
    e->capacity = capacity;
    e->key = (char *)malloc(key.len + 1);
    e->nodes = (struct gnode *)calloc((size_t)capacity, sizeof(struct gnode));
    e->leaves = (struct port *)calloc((size_t)capacity, sizeof(struct port));
    e->ctx = new_context(ggml_tensor_overhead() * size + ggml_graph_overhead_custom(size, false));
    e->fill_ctx = new_context(ggml_tensor_overhead() * (size_t)(capacity + 2 * MAX_OUTPUTS + 8));
    e->port_ctx = new_context(ggml_tensor_overhead() * (size_t)(capacity + MAX_OUTPUTS + 8));
  }
  if (e == NULL || e->key == NULL || e->nodes == NULL || e->leaves == NULL || e->ctx == NULL ||
      e->fill_ctx == NULL || e->port_ctx == NULL) {
    if (e != NULL) destroy(e);
    tg_unlock_device();
    return tg_error(env, "out of memory making a graph");
  }
  memcpy(e->key, key.data, key.len);
  e->key[key.len] = '\0';
  e->in_use = true;
  pthread_mutex_lock(&cache_lock);
  e->used = ++tick;
  victim = NULL;
  i = keep ? free_slot(&victim) : -1;
  if (i >= 0) cache[i] = e; /* no slot: the graph lives for this lease only */
  pthread_mutex_unlock(&cache_lock);
  if (victim != NULL) destroy(victim);
  tg_unlock_device();
  lease->e = e;
  return ggml_result_Lease_ok(env, term);
}

an_term ggml_ready(an_env *env, struct ggml_Lease_state *lease) {
  return an_bool(env, lease->e != NULL && lease->e->compiled);
}

an_term ggml_compiles(an_env *env) {
  int64_t n;

  pthread_mutex_lock(&cache_lock);
  n = compile_count;
  pthread_mutex_unlock(&cache_lock);
  return an_int(env, n);
}

static int64_t volume(const int64_t ne[4]) { return ne[0] * ne[1] * ne[2] * ne[3]; }

static int same_ne(const struct ggml_tensor *t, const int64_t ne[4]) {
  return t->ne[0] == ne[0] && t->ne[1] == ne[1] && t->ne[2] == ne[2] && t->ne[3] == ne[3];
}

/* The node's tensor at its full recorded shape. */
static struct ggml_tensor *full_of(struct entry *p, const struct gnode *n) {
  if (!n->lazy) return n->t;
  return ggml_repeat_4d(p->ctx, n->t, n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
}

static struct ggml_tensor *contiguous(struct entry *p, struct ggml_tensor *t) {
  return ggml_is_contiguous(t) ? t : ggml_cont(p->ctx, t);
}

static struct ggml_tensor *dense(struct entry *p, const struct gnode *n) {
  struct gnode *node = (struct gnode *)n; /* the nodes are `p->nodes`; the copy is a cache */

  if (node->dense == NULL) node->dense = contiguous(p, full_of(p, n));
  return node->dense;
}

/* The node as an operand of ggml_mul_mat. ggml calls a tensor contiguous when its strides match wherever an extent is
   above 1, but ggml-vulkan wants every stride to match (it copies the operand through F16 otherwise), so a view with an
   extent 1 axis (one head) is copied too. */
static struct ggml_tensor *operand(struct entry *p, const struct gnode *n) {
  struct ggml_tensor *t = dense(p, n);
  int i;

  if (ggml_is_quantized(t->type)) return t; /* a weight's blocks: its strides count bytes of blocks, and it is a leaf */
  for (i = 1; i < GGML_MAX_DIMS; i++) {
    if (t->nb[i] != t->nb[i - 1] * (size_t)t->ne[i - 1]) return ggml_cont(p->ctx, t);
  }
  return t;
}

/* The node's own tensor, copied when it is a view (a permute that kept axis 0): for the ops that take `->t` raw. */
static struct ggml_tensor *plain(struct entry *p, const struct gnode *n) { return contiguous(p, n->t); }

static struct ggml_tensor *reshape(struct entry *p, struct ggml_tensor *t, const int64_t ne[4]) {
  if (same_ne(t, ne)) return t;
  return ggml_reshape_4d(p->ctx, contiguous(p, t), ne[0], ne[1], ne[2], ne[3]);
}

typedef struct ggml_tensor *(*binary_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *);

/* `a op b` at `a`'s recorded shape, with `b` broadcast by ggml when lazy. */
static struct ggml_tensor *binary(struct entry *p, binary_fn fn, int commutes, const struct gnode *a,
                                  const struct gnode *b) {
  if (a->lazy && !b->lazy && commutes) return fn(p->ctx, plain(p, b), a->t);
  return fn(p->ctx, full_of(p, a), plain(p, b));
}

/* `1 - m` for a 0/1 mask. */
static struct ggml_tensor *flip(struct entry *p, struct ggml_tensor *m) {
  return ggml_scale_bias(p->ctx, contiguous(p, m), -1.0f, 1.0f);
}

/* A max over one row at least this long is found with argmax, not pool_1d. */
#define LONG_ROW 1024

/* `x` as a 2-D tensor whose rows (ne[0]) are the reduced block: reshaped to
   [inner, outer] for trailing axes, transposed from [outer, inner] for
   leading ones. NULL for axes that are neither block. */
static struct ggml_tensor *rows_to_reduce(struct entry *p, const struct gnode *x, int rank, const int64_t *axes,
                                          int naxes) {
  int64_t inner = 1, outer = 1, d;
  int i, trailing = 1, leading = 1;
  struct ggml_tensor *t = dense(p, x);

  for (i = 0; i < naxes; i++) {
    if (axes[i] != rank - naxes + i) trailing = 0;
    if (axes[i] != i) leading = 0;
  }
  if (!trailing && !leading) return NULL;
  for (i = 0; i < rank; i++) {
    d = x->ne[rank - 1 - i];
    if (trailing ? i >= rank - naxes : i < naxes) inner *= d;
    else outer *= d;
  }
  if (trailing) return ggml_reshape_2d(p->ctx, t, inner, outer);
  return ggml_cont(p->ctx, ggml_transpose(p->ctx, ggml_reshape_2d(p->ctx, t, outer, inner)));
}

/* The window of `x` that starts at the row-major `starts` and is `ne` wide
   (ggml order), made contiguous. */
static struct ggml_tensor *slice(struct entry *p, const struct gnode *x, const int64_t *starts, int rank,
                                 const int64_t ne[4]) {
  struct ggml_tensor *t = dense(p, x);
  size_t offset = 0;
  int i;

  for (i = 0; i < rank; i++) offset += (size_t)starts[i] * t->nb[rank - 1 - i];
  return ggml_cont(p->ctx, ggml_view_4d(p->ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], offset));
}

/* The indices as I32 row ids of a table of `rows` rows. ggml reads out of
   bounds (or aborts) on a bad index, so the indices are clamped to the rows
   first and `guard` counts the ones that were outside: `run` refuses its
   result unless that count is 0. The indices are F32 here, exact below 2^24,
   so `node` refuses a table with more rows. */
static struct ggml_tensor *row_ids(struct entry *p, const struct gnode *indices, int64_t rows,
                                   struct ggml_tensor **guard) {
  struct ggml_tensor *at = ggml_reshape_1d(p->ctx, dense(p, indices), volume(indices->ne));
  float top = (float)(rows - 1);
  /* ggml_clamp works in place: clamp a copy */
  struct ggml_tensor *clamped = ggml_clamp(p->ctx, ggml_cont(p->ctx, at), 0.0f, top);
  /* step(-i - 0.5) + step(i - top - 0.5): 1 for an index below 0 or past top */
  struct ggml_tensor *below = ggml_step(p->ctx, ggml_scale_bias(p->ctx, at, -1.0f, -0.5f));
  struct ggml_tensor *above = ggml_step(p->ctx, ggml_scale_bias(p->ctx, at, 1.0f, -top - 0.5f));

  *guard = ggml_sum(p->ctx, ggml_add(p->ctx, below, above));
  return ggml_cast(p->ctx, clamped, GGML_TYPE_I32);
}

/* The rows of `source` (its first row-major axis) that `indices` name, as
   ggml_get_rows over a [width, rows] view. */
static struct ggml_tensor *gather(struct entry *p, const struct gnode *source, int rank, const struct gnode *indices,
                                  struct ggml_tensor **guard) {
  int64_t rows = source->ne[rank - 1];
  struct ggml_tensor *table = ggml_reshape_2d(p->ctx, dense(p, source), volume(source->ne) / rows, rows);

  return ggml_get_rows(p->ctx, table, row_ids(p, indices, rows, guard));
}

/* The inverse of `gather`: the rows of `updates` ([width, k] once flat)
   added into `rows` zero rows, at the row each of the k indices names, as
   ggml_get_rows_back. The reference tensor only gives the answer its shape;
   it is not part of the graph. */
static struct ggml_tensor *scatter_add(struct entry *p, const struct gnode *indices, const struct gnode *updates,
                                       int64_t rows, struct ggml_tensor **guard) {
  int64_t k = volume(indices->ne);
  struct ggml_tensor *flat = ggml_reshape_2d(p->ctx, dense(p, updates), volume(updates->ne) / k, k);
  struct ggml_tensor *shape = ggml_new_tensor_2d(p->ctx, GGML_TYPE_F32, flat->ne[0], rows);

  return ggml_get_rows_back(p->ctx, flat, row_ids(p, indices, rows, guard), shape);
}

/* Whether `t` is the permute (0, 2, 1, 3) of its source: permuting it again gives the source back, so a norm that
   ran on the source feeds a rope with no view between them, which ggml-vulkan needs to fuse norm, mul and rope. */
static bool swapped_back(const struct ggml_tensor *t) {
  const struct ggml_tensor *s = t->src[0];

  return t->op == GGML_OP_PERMUTE && s->ne[1] == t->ne[2] && s->ne[2] == t->ne[1] && s->nb[1] == t->nb[2] &&
         s->nb[2] == t->nb[1] && s->ne[0] == t->ne[0] && s->ne[3] == t->ne[3];
}

/* The node as an operand of flash attention: its rows must be dense, its other strides may be a view's. */
static struct ggml_tensor *rows_of(struct entry *p, const struct gnode *n) {
  struct ggml_tensor *t = full_of(p, n);

  return t->nb[0] == ggml_type_size(t->type) ? t : ggml_cont(p->ctx, t);
}

static bool on_vulkan(const struct entry *p) {
  return ggml_backend_dev_type(ggml_backend_get_device(p->backend)) != GGML_BACKEND_DEVICE_TYPE_CPU;
}

/* The mask as F16 for flash attention, once for every layer that reads it. The lowering lets flash attention take a
   mask that holds 0 and large negative fills, so F16 keeps it exactly; the large ones are clamped to F16's lowest,
   since -1e9 would become -inf and a row with every key hidden would not stay the mean of v. */
static struct ggml_tensor *half_mask(struct entry *p, const struct gnode *mask) {
  struct gnode *node = (struct gnode *)mask; /* the nodes are `p->nodes`; the conversion is a cache */
  struct ggml_tensor *t = dense(p, mask);

  if (node->half == NULL) {
    if (t->type == GGML_TYPE_F16) node->half = t;
    else node->half = ggml_cast(p->ctx, ggml_clamp(p->ctx, ggml_cont(p->ctx, t), -65504.0f, 65504.0f), GGML_TYPE_F16);
  }
  return node->half;
}

/* softmax(scale * q kᵀ + mask) v over q [B, H, Sq, D], k [B, Hkv, Sk, D] and v [B, Hkv, Sk, Dv] (ne [D, Sq, H, B]
   and so on), as one ggml_flash_attn_ext that reads the heads as the views they are: no copy of q, k or v, no
   scores in memory. It takes the mask in F16 (so a mask value is rounded to 11 bits; the causal mask is 0 and
   -inf, exact) and answers [Dv, H, Sq, B], which is permuted back to [Dv, Sq, H, B] as a view. NULL when the
   backend has no kernel for the shapes (Vulkan wants widths that are multiples of 8). */
static struct ggml_tensor *flash(struct entry *p, const struct gnode *q, const struct gnode *k, const struct gnode *v,
                                 const struct gnode *mask, float scale) {
  struct ggml_tensor *bias, *out;

  /* Vulkan multiplies in fp16 by default, and then the two products beat the F32 kernel of flash attention over
     many queries (3.07 s against 3.12 s for ten prompts of 128 tokens); one query is faster flash. */
  if (on_vulkan(p) && tg_fp16() && q->ne[1] > 1) return NULL;
  bias = mask != NULL ? half_mask(p, mask) : NULL;
  out = ggml_flash_attn_ext(p->ctx, rows_of(p, q), rows_of(p, k), rows_of(p, v), bias, scale, 0.0f, 0.0f);

  ggml_prec_set_acc(out, GGML_PREC_F32);
  if (!ggml_backend_supports_op(p->backend, out)) return NULL;
  return ggml_permute(p->ctx, out, 0, 2, 1, 3);
}

/* The same as two mul_mat and one soft_max_ext, for what `flash` cannot take. mul_mat repeats k and v over the query
   heads of a group (ne[2] divides), which is grouped-query attention, and soft_max_ext takes the scale and the
   additive mask (its ne[2] and ne[3] repeat). The caller has checked every extent these asserts of ggml read. */
static struct ggml_tensor *attention(struct entry *p, const struct gnode *q, const struct gnode *k, const struct gnode *v,
                                     const struct gnode *mask, float scale, bool may_flash) {
  struct ggml_tensor *fused = may_flash ? flash(p, q, k, v, mask, scale) : NULL;
  struct ggml_tensor *scores, *weights, *rows;

  if (fused != NULL) return fused;
  scores = ggml_mul_mat(p->ctx, operand(p, k), operand(p, q)); /* [Sk, Sq, H, B] */
  weights = ggml_soft_max_ext(p->ctx, scores, mask != NULL ? dense(p, mask) : NULL, scale, 0.0f);
  rows = ggml_cont(p->ctx, ggml_permute(p->ctx, dense(p, v), 1, 0, 2, 3)); /* [Sk, Dv, Hkv, B] */
  return ggml_mul_mat(p->ctx, rows, weights); /* [Dv, Sq, H, B] */
}

/* The double whose bits `lower.agc` sent as an int. */
static double from_bits(int64_t bits) {
  double x;

  memcpy(&x, &bits, sizeof x);
  return x;
}

/* Whether an attention's arguments hold the shapes `attention` reads. */
static bool attends(const struct gnode *q, const struct gnode *k, const struct gnode *v, const struct gnode *mask) {
  if (q->rank != 4 || k->rank != 4 || v->rank != 4) return false;
  if (k->ne[0] != q->ne[0] || k->ne[3] != q->ne[3] || v->ne[3] != q->ne[3]) return false;
  if (v->ne[1] != k->ne[1] || v->ne[2] != k->ne[2] || q->ne[2] % k->ne[2] != 0) return false;
  if (mask == NULL) return true;
  return (mask->rank == 2 || mask->rank == 4) && mask->ne[0] == k->ne[1] && mask->ne[1] == q->ne[1] &&
         (mask->ne[2] == 1 || mask->ne[2] == q->ne[2]) && (mask->ne[3] == 1 || mask->ne[3] == q->ne[3]);
}

/* Rotary embedding of x [..., S, D]. ggml wants the rows along ne[2] ([D, heads, S, batch]), so the
   sequence axis is swapped with the one before it, rotated by ggml_rope_ext, and swapped back. The positions are F32
   here and ggml takes I32; `inverse` negates them, which turns every pair the other way. */
static struct ggml_tensor *rope(struct entry *p, const struct gnode *x, const struct gnode *positions, int dims,
                                bool neox, float base, bool inverse) {
  struct ggml_tensor *seen = full_of(p, x);
  struct ggml_tensor *rows = contiguous(p, swapped_back(seen) ? seen->src[0] : ggml_permute(p->ctx, seen, 0, 2, 1, 3));
  struct gnode *source = (struct gnode *)positions; /* the nodes are `p->nodes`; the cast is a cache */
  struct ggml_tensor *at = source->at;
  struct ggml_tensor *turned;

  if (inverse) at = ggml_cast(p->ctx, ggml_neg(p->ctx, dense(p, positions)), GGML_TYPE_I32);
  else if (at == NULL) at = source->at = ggml_cast(p->ctx, dense(p, positions), GGML_TYPE_I32);
  turned = ggml_rope_ext(p->ctx, rows, at, NULL, dims,
                         neox ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL, 4096, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
  return ggml_permute(p->ctx, turned, 0, 2, 1, 3); /* a view: the next op copies it only if it must */
}

/* Whether the args, of the output's rank (one less for a stack), match `ne`
   off the ggml axis `dim` and their extents on it add up to `ne[dim]`. */
static bool joins(const struct entry *p, const int64_t *args, int nargs, int rank, int dim, bool stacked,
                  const int64_t ne[4]) {
  int64_t total = 0;
  int i, d;

  for (i = 0; i < nargs; i++) {
    const struct gnode *x = &p->nodes[args[i]];

    if (x->rank != (stacked ? rank - 1 : rank)) return false;
    for (d = 0; d < 4; d++) {
      if (d == dim) continue;
      if (stacked ? x->ne[d < dim ? d : d - 1] != ne[d] : x->ne[d] != ne[d]) return false;
    }
    total += stacked ? 1 : x->ne[dim];
  }
  return total == ne[dim];
}

/* The args joined along the ggml axis `dim` into `ne`; a stacked arg gains
   that axis (extent 1) first. */
static struct ggml_tensor *concat(struct entry *p, const int64_t *args, int nargs, int dim, bool stacked,
                                  const int64_t ne[4]) {
  struct ggml_tensor *out = NULL;
  int i;

  for (i = 0; i < nargs; i++) {
    const struct gnode *x = &p->nodes[args[i]];
    int64_t part[4] = {ne[0], ne[1], ne[2], ne[3]};
    struct ggml_tensor *t;

    part[dim] = stacked ? 1 : x->ne[dim];
    t = reshape(p, dense(p, x), part);
    out = out == NULL ? t : ggml_concat(p->ctx, out, t, dim);
  }
  return out;
}

/* `input` ([C_in, L] or [B, C_in, L] row-major) convolved with `weight`
   ([C_out, C_in, K]): im2col in F32 (ggml_conv_1d's own path makes it F16),
   then one mul_mat; [L_out * B, C_out] is permuted to [B, C_out, L_out]. */
static struct ggml_tensor *conv1d(struct entry *p, const struct gnode *input, const struct gnode *weight, int stride,
                                  int padding) {
  struct ggml_tensor *kernel = dense(p, weight);
  struct ggml_tensor *cols = ggml_im2col(p->ctx, kernel, dense(p, input), stride, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
  int64_t out_len = cols->ne[1], batch = cols->ne[2], out_channels = kernel->ne[2];
  struct ggml_tensor *t = ggml_mul_mat(p->ctx, ggml_reshape_2d(p->ctx, cols, cols->ne[0], out_len * batch),
                                       ggml_reshape_2d(p->ctx, kernel, kernel->ne[0] * kernel->ne[1], out_channels));

  t = ggml_reshape_3d(p->ctx, t, out_len, batch, out_channels);
  if (batch > 1) t = ggml_cont(p->ctx, ggml_permute(p->ctx, t, 0, 2, 1, 3));
  else t = ggml_reshape_3d(p->ctx, t, out_len, out_channels, 1);
  return t;
}

/* Reads up to `max` ints; -2 when there are more, -1 when the cursor failed
   (the error is raised already), 0 when done. */
static int read_ints(an_env *env, ggml_list_Int *list, int64_t *out, int max, int *n) {
  int64_t v;
  int next;

  *n = 0;
  while ((next = ggml_list_Int_next(env, list, &v)) == 1) {
    if (*n >= max) return -2;
    out[(*n)++] = v;
  }
  return next;
}

/* The answer of a list that did not read: the cursor's error, or `text`. */
static an_term bad_list(an_env *env, int r, const char *text) { return r == -1 ? an_void(env) : tg_error(env, text); }

/* Where a packed operand (a quantized weight, or an F16 one) may go, as one closed list for every op: the leaf itself
   as the argument of a transpose or the table of a gather, and the second argument of a matmul (the leaf, or the
   transpose of one: a view, read in place). Any other use would copy or compute on blocks that ggml's ops do not take
   (`ggml_cont` of a quantized view sizes its copy by rows and writes past its buffer), so it is an error before any
   ggml call. `x` is argument `i` of a node of `op`; NULL when it fits. */
static const char *packed_misuse(int64_t op, int i, const struct gnode *x) {
  if (!ggml_is_quantized(x->t->type) && x->t->type != GGML_TYPE_F16) return NULL;
  if (op == OP_MATMUL && i == 1) return NULL;
  if (x->t->type == GGML_TYPE_F16) {
    /* an F16 cache: written in place by set_rows, read by attention, widened by a cast */
    if (op == OP_SET_ROWS && i == 0 && x->op == OP_LEAF) return NULL;
    if (op == OP_ATTENTION && (i == 1 || i == 2)) return NULL;
    if (op == OP_CAST && x->op == OP_LEAF) return NULL;
  }
  if (x->op == OP_LEAF && (op == OP_TRANSPOSE || (op == OP_GATHER && i == 0))) return NULL;
  return "a quantized or F16 weight goes only into the weight of a matmul, a transpose of the weight itself or the table of a gather";
}

an_term ggml_node(an_env *env, struct ggml_Lease_state *lease, int64_t op, ggml_list_Int args_list,
                  ggml_list_Int shape_list, ggml_list_Int ints_list) {
  struct entry *p = lease->e;
  int64_t args[MAX_ARGS], shape[4], ints[8];
  int nargs, rank, nints, i, r;
  struct gnode *n;
  const struct gnode *a = NULL, *b = NULL, *c = NULL;
  struct ggml_tensor *t = NULL;
  enum ggml_type type;
  const char *why;
  char text[256];

  if (p == NULL || p->compiled || p->graph != NULL) return tg_error(env, "the graph is compiled already, or its compile failed");
  if (p->count >= p->capacity) return tg_error(env, "the graph is full");
  if (op < 0 || op >= OP_COUNT) return tg_error(env, "unknown op code");
  if ((r = read_ints(env, &args_list, args, MAX_ARGS, &nargs)) != 0) return bad_list(env, r, "a node takes at most 256 args");
  if ((r = read_ints(env, &shape_list, shape, 4, &rank)) != 0) return bad_list(env, r, "ggml tensors have rank <= 4");
  if ((r = read_ints(env, &ints_list, ints, 8, &nints)) != 0) return bad_list(env, r, "a node takes at most 8 ints");
  for (i = 0; i < nargs; i++) {
    if (args[i] < 0 || args[i] >= p->count) return tg_error(env, "an arg is not an earlier node");
    why = packed_misuse(op, i, &p->nodes[args[i]]);
    if (why != NULL) return tg_error(env, why);
  }
  if (!room(p, 24 + 4 * (size_t)nargs)) return tg_error(env, "the graph ran out of room: it lowers to more ggml tensors than its capacity allows");
  {
    int64_t elements = 1;

    for (i = 0; i < rank; i++) {
      if (shape[i] <= 0) return tg_error(env, "a tensor with no elements is not lowered to ggml");
      if (elements > INT64_MAX / 8 / shape[i]) return tg_error(env, "the tensor is too large");
      elements *= shape[i];
    }
  }
  if (nargs > 0) a = &p->nodes[args[0]];
  if (nargs > 1) b = &p->nodes[args[1]];
  if (nargs > 2) c = &p->nodes[args[2]];
  if ((op >= OP_ADD && op <= OP_DIV) || op == OP_MATMUL || op == OP_EQ || op == OP_LT || op == OP_GATHER ||
      op == OP_SCATTER_ADD || op == OP_CONV1D || op == OP_ROPE || op == OP_BATCH_MATMUL) {
    if (nargs != 2) return tg_error(env, "a binary op takes 2 args");
  } else if (op == OP_WHERE || op == OP_SET_ROWS) {
    if (nargs != 3) return tg_error(env, "where and set_rows take 3 args");
  } else if (op == OP_ATTENTION) {
    if (nargs != 3 && nargs != 4) return tg_error(env, "attention takes q, k, v and an optional mask");
  } else if (op == OP_CONCAT) {
    if (nargs < 1) return tg_error(env, "concat takes at least 1 arg");
  } else if (op > OP_FULL && op != OP_ARANGE && nargs != 1) {
    return tg_error(env, "a unary op takes 1 arg");
  }

  n = &p->nodes[p->count];
  memset(n, 0, sizeof *n);
  n->op = (int)op;
  n->arg = -1;
  n->rank = rank;
  for (i = 0; i < 4; i++) n->ne[i] = i < rank ? shape[rank - 1 - i] : 1;

  switch (op) {
    case OP_LEAF:
      /* ints[0]: the type the leaf is bound to, as `Types.held_code`: 0 F32, 1 F16, 2 Q8_0 */
      if (nints > 0 && (ints[0] < 0 || ints[0] > 2)) return tg_error(env, "a leaf is F32, F16 or Q8_0");
      type = nints > 0 && ints[0] == 2 ? GGML_TYPE_Q8_0 : (nints > 0 && ints[0] == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32);
      if (type == GGML_TYPE_Q8_0 && (rank != 2 || n->ne[0] % ggml_blck_size(type) != 0)) {
        return tg_error(env, "a Q8_0 leaf is a rank-2 weight whose rows hold whole blocks of 32 values");
      }
      t = ggml_new_tensor_4d(p->port_ctx, type, n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
      ggml_set_input(t);
      p->leaves[p->leaf_count++].t = t;
      break;
    case OP_FULL:
      if (p->fill_store == NULL) p->fill_store = ggml_new_tensor_1d(p->fill_ctx, GGML_TYPE_F32, p->capacity);
      t = ggml_view_1d(p->fill_ctx, p->fill_store, 1, (size_t)p->fill_count * sizeof(float));
      n->zero = nints > 0 && ints[0] != 0;
      n->lazy = volume(n->ne) != 1;
      p->fill_count++;
      break;
    case OP_ADD: t = binary(p, ggml_add, 1, a, b); break;
    case OP_SUB: t = binary(p, ggml_sub, 0, a, b); break;
    case OP_MUL: t = binary(p, ggml_mul, 1, a, b); break;
    case OP_DIV: t = binary(p, ggml_div, 0, a, b); break;
    case OP_NEG: t = ggml_neg(p->ctx, dense(p, a)); break;
    case OP_EXP: t = ggml_exp(p->ctx, dense(p, a)); break;
    case OP_LOG: t = ggml_log(p->ctx, dense(p, a)); break;
    case OP_RELU: t = ggml_relu(p->ctx, dense(p, a)); break;
    case OP_SQRT: t = ggml_sqrt(p->ctx, dense(p, a)); break;
    case OP_SIGMOID: t = ggml_sigmoid(p->ctx, dense(p, a)); break;
    case OP_TANH: t = ggml_tanh(p->ctx, dense(p, a)); break;
    case OP_MATMUL:
      /* ggml_mul_mat(x, y) is y · xᵀ in row-major terms, so a · b is
         mul_mat(bᵀ, a). When b is a transpose (the backward's ct · wᵀ), bᵀ
         is its argument and no copy is made. */
      if (b->op == OP_TRANSPOSE) {
        t = ggml_mul_mat(p->ctx, operand(p, &p->nodes[b->arg]), operand(p, a));
      } else if (ggml_is_quantized(b->t->type)) {
        return tg_error(env, "a quantized weight is read as [out, in]: use linear with out_in, not a matmul by the weight");
      } else {
        t = ggml_mul_mat(p->ctx, ggml_cont(p->ctx, ggml_transpose(p->ctx, dense(p, b))), operand(p, a));
      }
      break;
    case OP_TRANSPOSE:
      /* ggml_transpose swaps ne0 and ne1 only */
      if (rank > 2) {
        snprintf(text, sizeof text, "transpose of a rank-%d tensor is not lowered to ggml", rank);
        return tg_error(env, text);
      }
      /* a quantized weight is only read in place, by the matmul that takes its transpose: the copy would need
         rows of whole blocks along the other axis */
      t = ggml_is_quantized(a->t->type) ? ggml_transpose(p->ctx, a->t) : ggml_cont(p->ctx, ggml_transpose(p->ctx, dense(p, a)));
      n->arg = (int)args[0];
      break;
    case OP_RESHAPE:
      if (a->lazy && ggml_nelements(a->t) == 1) {
        t = a->t;
        n->lazy = volume(n->ne) != 1;
        n->zero = a->zero;
      } else {
        t = reshape(p, dense(p, a), n->ne);
      }
      break;
    case OP_BROADCAST:
      for (i = 0; i < 4; i++) {
        if (a->t->ne[i] == 0 || n->ne[i] % a->t->ne[i] != 0) {
          return tg_error(env, "broadcast_to: the source does not repeat to the shape");
        }
      }
      t = a->t;
      n->lazy = !same_ne(t, n->ne);
      if (n->lazy) t = plain(p, a);
      n->zero = a->zero;
      break;
    case OP_SUM:
    case OP_MEAN:
    case OP_MAX: {
      struct ggml_tensor *rows;

      if (nints < 1) return tg_error(env, "a reduction takes the input's rank and its axes");
      if (nints == 1) { /* no axes: the value itself */
        t = reshape(p, dense(p, a), n->ne);
        break;
      }
      rows = rows_to_reduce(p, a, (int)ints[0], ints + 1, nints - 1);
      if (rows == NULL) {
        snprintf(text, sizeof text, "%s over axes that are not a leading or trailing block is not lowered to ggml",
                 op_names[op]);
        return tg_error(env, text);
      }
      if (op == OP_MAX && rows->ne[1] == 1 && rows->ne[0] >= LONG_ROW) {
        /* One thread walks a pooled row: 14 ms for 51,865 floats on Vulkan. argmax reduces it in a workgroup, and
           the value is read back with get_rows; which of several equal maxima it finds does not matter. */
        t = ggml_get_rows(p->ctx, ggml_reshape_2d(p->ctx, rows, 1, rows->ne[0]), ggml_argmax(p->ctx, rows));
      } else if (op == OP_MAX) {
        t = ggml_pool_1d(p->ctx, rows, GGML_OP_POOL_MAX, (int)rows->ne[0], (int)rows->ne[0], 0);
      } else {
        t = ggml_sum_rows(p->ctx, rows);
        if (op == OP_MEAN) t = ggml_scale(p->ctx, t, 1.0f / (float)rows->ne[0]);
      }
      t = reshape(p, t, n->ne);
      break;
    }
    case OP_IDENTITY:
      t = a->t;
      n->lazy = a->lazy;
      n->zero = a->zero;
      if (a->op == OP_TRANSPOSE) {
        n->op = OP_TRANSPOSE;
        n->arg = a->arg;
      }
      break;
    case OP_ARANGE:
      if (nints != 3) return tg_error(env, "arange takes start, stop and step");
      if (ints[2] <= 0 || ints[1] <= ints[0]) return tg_error(env, "arange with a step <= 0 or no values is not lowered to ggml");
      if (ints[0] < -16777216 || ints[1] > 16777216) return tg_error(env, "arange beyond 2^24 is not exact in F32");
      t = ggml_arange(p->ctx, (float)ints[0], (float)ints[1], (float)ints[2]);
      break;
    case OP_EQ:
      /* 1 - step(|a - b|): exact for integers below 2^24 */
      t = flip(p, ggml_step(p->ctx, ggml_abs(p->ctx, binary(p, ggml_sub, 0, a, b))));
      break;
    case OP_LT:
      t = ggml_step(p->ctx, binary(p, ggml_sub, 0, b, a));
      break;
    case OP_WHERE:
      /* c·x + (1-c)·y, exact for a 0/1 mask and finite branches; the zero
         branch of relu's and cross-entropy's rules drops out */
      if (c->zero) {
        t = binary(p, ggml_mul, 1, b, a);
      } else if (b->zero) {
        t = ggml_mul(p->ctx, flip(p, dense(p, a)), dense(p, c));
      } else {
        t = ggml_add(p->ctx, binary(p, ggml_mul, 1, b, a), ggml_mul(p->ctx, flip(p, dense(p, a)), dense(p, c)));
      }
      break;
    case OP_SLICE:
      if (nints != rank || a->rank != rank) return tg_error(env, "slice takes one start per axis of its source");
      for (i = 0; i < rank; i++) {
        int d = rank - 1 - i;

        if (ints[i] < 0 || ints[i] + n->ne[d] > a->ne[d]) return tg_error(env, "slice: a window outside its source");
      }
      t = slice(p, a, ints, rank, n->ne);
      break;
    case OP_GATHER:
      if (nints != 1 || ints[0] != a->rank || ints[0] < 1) return tg_error(env, "gather takes the source's rank");
      if (a->ne[a->rank - 1] > 16777216) {
        return tg_error(env, "gather from more than 2^24 rows is not lowered to ggml: F32 row ids are exact below that");
      }
      if (volume(n->ne) != volume(b->ne) * (volume(a->ne) / a->ne[a->rank - 1])) {
        return tg_error(env, "gather: the recorded shape is not the indices' shape followed by the source's rows");
      }
      t = reshape(p, gather(p, a, (int)ints[0], b, &n->guard), n->ne);
      break;
    case OP_SCATTER_ADD:
      if (n->rank < 1 || nints != 1 || ints[0] < 1 || ints[0] != n->ne[n->rank - 1]) {
        return tg_error(env, "scatter_add takes the rows of its answer");
      }
      if (ints[0] > 16777216) {
        return tg_error(env, "scatter_add into more than 2^24 rows is not lowered to ggml: F32 row ids are exact below that");
      }
      if (b->rank < a->rank || volume(b->ne) % volume(a->ne) != 0 ||
          volume(b->ne) / volume(a->ne) * ints[0] != volume(n->ne)) {
        return tg_error(env, "scatter_add takes updates that hold a row for each index, and answers the rows of them");
      }
      t = reshape(p, scatter_add(p, a, b, ints[0], &n->guard), n->ne);
      break;
    case OP_GELU_ERF: t = ggml_gelu_erf(p->ctx, dense(p, a)); break;
    case OP_SET_ROWS: {
      struct ggml_tensor *rows;
      int copy;

      if (nints != 1 || (ints[0] != 0 && ints[0] != 1)) return tg_error(env, "set_rows takes a flag, 1 to write a copy of its cache");
      copy = (int)ints[0];
      if (a->ne[1] > 16777216) {
        return tg_error(env, "set_rows into more than 2^24 rows is not lowered to ggml: F32 row ids are exact below that");
      }
      if (a->ne[0] != b->ne[0] || a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3] || volume(c->ne) != b->ne[1]) {
        return tg_error(env, "set_rows takes rows of the cache's width, one for each position");
      }
      if (copy) {
        if (a->t->type != GGML_TYPE_F32) return tg_error(env, "set_rows writes an F32 cache");
        rows = row_ids(p, c, a->ne[1], &n->guard);
        t = ggml_set_rows(p->ctx, ggml_dup(p->ctx, dense(p, a)), dense(p, b), rows);
      } else {
        struct port *leaf = NULL;
        int64_t k;

        if (a->op != OP_LEAF) return tg_error(env, "set_rows writes a leaf, an input of the graph, in place");
        if (b == a || c == a) return tg_error(env, "set_rows reads the cache it writes in place");
        for (k = 0; k < p->leaf_count; k++) {
          if (p->leaves[k].t == a->t) leaf = &p->leaves[k];
        }
        if (leaf == NULL || (a->t->type != GGML_TYPE_F32 && a->t->type != GGML_TYPE_F16)) {
          return tg_error(env, "set_rows writes an F32 or F16 input in place");
        }
        if (leaf->donated) return tg_error(env, "set_rows writes a leaf in place twice");
        leaf->donated = true;
        n->in_place = 1;
        t = ggml_set_rows(p->ctx, a->t, dense(p, b), row_ids(p, c, a->ne[1], &n->guard));
      }
      break;
    }
    case OP_CAST: {
      enum ggml_type want = nints == 1 && ints[0] == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32;

      if (nints != 1 || (ints[0] != 0 && ints[0] != 1)) return tg_error(env, "cast takes 1 for F16 or 0 for F32");
      if (a->t->type == want) {
        t = a->t;
        n->lazy = a->lazy;
        n->zero = a->zero;
      } else {
        t = ggml_cast(p->ctx, dense(p, a), want);
      }
      break;
    }
    case OP_TOP_K:
      if (a->rank != 1 || nints != 1 || ints[0] < 1 || ints[0] > a->ne[0]) {
        return tg_error(env, "top_k takes a vector and a k from 1 to its length");
      }
      if (a->ne[0] > 16777216) {
        return tg_error(env, "top_k of more than 2^24 values is not lowered to ggml: F32 ids are exact below that");
      }
      /* the indices are I32 in ggml and F32 in this layer, as every integer tensor */
      t = ggml_cast(p->ctx, ggml_top_k(p->ctx, dense(p, a), (int)ints[0]), GGML_TYPE_F32);
      break;
    case OP_NORM: {
      uint32_t bits;
      float eps;

      if (nints != 1 || ints[0] < 0 || ints[0] > UINT32_MAX) return tg_error(env, "a norm takes eps as the bits of an F32");
      bits = (uint32_t)ints[0];
      memcpy(&eps, &bits, sizeof eps);
      if (!(eps > 0.0f)) return tg_error(env, "a norm takes an eps above 0");
      t = ggml_norm(p->ctx, dense(p, a), eps);
      break;
    }
    case OP_RMS_NORM: {
      uint32_t bits;
      float eps;

      if (nints != 1 || ints[0] < 0 || ints[0] > UINT32_MAX) return tg_error(env, "an rms norm takes eps as the bits of an F32");
      bits = (uint32_t)ints[0];
      memcpy(&eps, &bits, sizeof eps);
      if (!(eps > 0.0f)) return tg_error(env, "an rms norm takes an eps above 0");
      t = ggml_rms_norm(p->ctx, dense(p, a), eps);
      break;
    }
    case OP_CONCAT:
      if (nints != 2 || ints[0] < 0 || ints[0] >= rank) return tg_error(env, "concat takes an axis and the stack flag");
      if (!joins(p, args, nargs, rank, rank - 1 - (int)ints[0], ints[1] != 0, n->ne)) {
        return tg_error(env, "concat: the args do not join into the recorded shape");
      }
      t = concat(p, args, nargs, rank - 1 - (int)ints[0], ints[1] != 0, n->ne);
      break;
    case OP_PERMUTE: {
      int order[4] = {0, 1, 2, 3};
      bool seen[4] = {false, false, false, false};

      if (nints != rank || a->rank != rank) return tg_error(env, "permute takes one axis per axis of its source");
      for (i = 0; i < rank; i++) {
        if (ints[i] < 0 || ints[i] >= rank || seen[ints[i]]) return tg_error(env, "permute takes each axis once");
        seen[ints[i]] = true;
        order[rank - 1 - (int)ints[i]] = rank - 1 - i; /* source ggml axis -> its place in the answer */
      }
      /* Axis 0 kept (the width of a head, say): a view, and whoever needs it contiguous (`dense`) copies it, or
         permutes it back and finds the source again. Any other permute is a copy now. */
      if (order[0] == 0 && order[1] == 2 && order[2] == 1 && order[3] == 3 && swapped_back(full_of(p, a))) {
        t = full_of(p, a)->src[0]; /* the permute of a permute: the tensor it came from */
      } else if (order[0] == 0) {
        t = ggml_permute(p->ctx, full_of(p, a), order[0], order[1], order[2], order[3]);
      } else {
        t = ggml_cont(p->ctx, ggml_permute(p->ctx, dense(p, a), order[0], order[1], order[2], order[3]));
      }
      break;
    }
    case OP_ATTENTION: {
      const struct gnode *mask = nargs == 4 ? &p->nodes[args[3]] : NULL;

      if (nints != 2 || !isfinite(from_bits(ints[0])) || fabs(from_bits(ints[0])) > FLT_MAX) {
        return tg_error(env, "attention takes a finite scale that a float holds");
      }
      if (!attends(a, b, c, mask)) {
        return tg_error(env, "attention takes q [B, H, Sq, D], k, v [B, Hkv, Sk, D | Dv] with Hkv dividing H, and a mask [Sq, Sk] or [B|1, H|1, Sq, Sk]");
      }
      t = attention(p, a, b, c, mask, (float)from_bits(ints[0]), ints[1] != 0);
      break;
    }
    case OP_ROPE:
      if (nints != 4 || a->rank < 2 || a->rank > 4 || ints[0] < 2 || ints[0] % 2 != 0 || ints[0] > a->ne[0]) {
        return tg_error(env, "rope takes a tensor of rank 2 to 4, an even dims within its width, the layout and the base");
      }
      if (b->rank != 1 || b->ne[0] != a->ne[1]) return tg_error(env, "rope takes one position per row");
      if (!(from_bits(ints[2]) > 0.0) || from_bits(ints[2]) > FLT_MAX) return tg_error(env, "rope takes a base in (0, a float's maximum]");
      t = rope(p, a, b, (int)ints[0], ints[1] != 0, (float)from_bits(ints[2]), ints[3] != 0);
      break;
    case OP_BATCH_MATMUL:
      if (a->rank != rank || b->rank != rank || rank < 2 || a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3] ||
          a->ne[0] != b->ne[1]) {
        return tg_error(env, "batch_matmul takes [B..., m, k] and [B..., k, n]");
      }
      /* like matmul: a · b is mul_mat(bᵀ, a), per batch */
      t = ggml_mul_mat(p->ctx, ggml_cont(p->ctx, ggml_permute(p->ctx, dense(p, b), 1, 0, 2, 3)), operand(p, a));
      break;
    case OP_CONV1D:
      if (nints != 2 || ints[0] < 1 || ints[1] < 0) return tg_error(env, "conv1d takes a stride >= 1 and a padding >= 0");
      if ((a->rank != 2 && a->rank != 3) || b->rank != 3 || a->ne[1] != b->ne[1]) {
        return tg_error(env, "conv1d takes an input [C, L] or [B, C, L] and a weight [O, C, K]");
      }
      t = conv1d(p, a, b, (int)ints[0], (int)ints[1]);
      break;
  }
  if (t == NULL) {
    snprintf(text, sizeof text, "ggml made no tensor for %s", op_names[op]);
    return tg_error(env, text);
  }
  if (!n->lazy && !same_ne(t, n->ne)) {
    snprintf(text, sizeof text,
             "%s made ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], the trace recorded [%" PRId64 ", %" PRId64
             ", %" PRId64 ", %" PRId64 "]",
             op_names[op], t->ne[0], t->ne[1], t->ne[2], t->ne[3], n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    return tg_error(env, text);
  }
  n->t = t;
  return ggml_result_Int_ok(env, p->count++);
}

/* `t` is `store` or a view into it; a graph without fills has no store. */
static bool in(const struct ggml_tensor *t, const struct ggml_tensor *store) {
  return store != NULL && (t == store || t->view_src == store);
}

an_term ggml_compile(an_env *env, struct ggml_Lease_state *lease, ggml_list_Int outputs) {
  struct entry *p = lease->e;
  int64_t outs[MAX_OUTPUTS];
  int nouts, r, i, j;
  char text[256];

  if (p == NULL || p->compiled || p->graph != NULL) return tg_error(env, "the graph is compiled already, or its compile failed");
  if ((r = read_ints(env, &outputs, outs, MAX_OUTPUTS, &nouts)) != 0) return bad_list(env, r, "at most 256 outputs");
  if (nouts < 1) return tg_error(env, "a graph has at least one output");
  for (i = 0; i < nouts; i++) {
    if (outs[i] < 0 || outs[i] >= p->count) return tg_error(env, "an output is not a node");
  }

  for (i = 0; i < p->count; i++) {
    if (p->nodes[i].guard != NULL) p->guard_count++;
  }
  if (!room(p, (size_t)(nouts * 8 + p->guard_count + 1))) return tg_error(env, "the graph ran out of room for its outputs");
  p->graph = ggml_new_graph_custom(p->ctx, graph_size(p->capacity), false);
  p->checks = ggml_new_tensor_1d(p->fill_ctx, GGML_TYPE_F32, nouts + p->guard_count);
  /* the positions of the ropes go first: a cast between a rope's norm and the rope keeps ggml-vulkan from fusing them */
  for (i = 0; i < p->count; i++) {
    if (p->nodes[i].at != NULL) ggml_build_forward_expand(p->graph, p->nodes[i].at);
  }
  for (i = 0; i < nouts; i++) {
    struct gnode *n = &p->nodes[outs[i]];
    struct ggml_tensor *value;
    struct ggml_tensor *check;

    p->outs[i].t = ggml_new_tensor_4d(p->port_ctx, n->t->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                      n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    if (n->in_place) {
      /* the output is the leaf itself: nothing to copy or to check, the node only has to run */
      p->outs[i].donated = true;
      ggml_build_forward_expand(p->graph, n->t);
      continue;
    }
    value = dense(p, n);
    ggml_set_output(p->outs[i].t);
    ggml_build_forward_expand(p->graph, ggml_cpy(p->ctx, value, p->outs[i].t));
    /* 0 * x is NaN for an inf or a NaN; sum_rows first, since one ggml_sum
       of a large tensor is a single slow workgroup on Vulkan */
    check = ggml_sum(p->ctx, ggml_sum_rows(p->ctx, ggml_scale(p->ctx, value->type == GGML_TYPE_F32 ? value : ggml_cast(p->ctx, value, GGML_TYPE_F32), 0.0f)));
    ggml_build_forward_expand(p->graph, ggml_cpy(p->ctx, check, ggml_view_1d(p->fill_ctx, p->checks, 1, i * sizeof(float))));
  }
  p->output_count = nouts;
  for (i = 0, j = nouts; i < p->count; i++) {
    struct ggml_tensor *guard = p->nodes[i].guard;

    if (guard != NULL) {
      ggml_build_forward_expand(p->graph, ggml_cpy(p->ctx, guard, ggml_view_1d(p->fill_ctx, p->checks, 1, j++ * sizeof(float))));
    }
  }
  for (i = 0; i < ggml_graph_n_nodes(p->graph); i++) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);

    if (!ggml_backend_supports_op(p->backend, node)) {
      snprintf(text, sizeof text, "the backend does not support graph node %d: %s", i, ggml_op_desc(node));
      return tg_error(env, text);
    }
  }

  /* Every node and source that is neither a port nor in the fill buffer:
     what the shared allocator places, and forgets after each run. */
  p->inner = (struct ggml_tensor **)calloc((size_t)ggml_graph_n_nodes(p->graph) * (GGML_MAX_SRC + 1), sizeof(struct ggml_tensor *));
  if (p->inner == NULL) return tg_error(env, "out of memory");
  for (i = 0; i < ggml_graph_n_nodes(p->graph); i++) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);

    for (j = -1; j < GGML_MAX_SRC; j++) {
      struct ggml_tensor *t = j < 0 ? node : node->src[j];

      if (t != NULL && !(t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) && !in(t, p->fill_store) &&
          !in(t, p->checks)) {
        p->inner[p->inner_count++] = t;
      }
    }
  }

  tg_lock_device();
  p->fill_buffer = ggml_backend_alloc_ctx_tensors(p->fill_ctx, p->backend);
  if (p->fill_buffer != NULL) ggml_backend_buffer_clear(p->fill_buffer, 0);
  tg_unlock_device();
  if (p->fill_buffer != NULL) {
    pthread_mutex_lock(&cache_lock);
    p->compiled = true;
    compile_count++;
    pthread_mutex_unlock(&cache_lock);
  }
  if (!p->compiled) return tg_error(env, "allocating the graph's fills failed");
  return ggml_result_Void_ok(env, an_void(env));
}

/* Points `port` at the memory of `h` when it has the port's type, or at its own memory
   (made once) for a converting copy, and holds `h`'s memory until `release`. Call
   under the device lock. */
static bool attach(struct entry *p, struct port *port, struct ggml_TensorHandle_state *h) {
  if (port->held != NULL) tg_storage_release(port->held);
  port->held = h->store;
  if (port->held != NULL) tg_storage_retain(port->held);
  if (h->t->type == port->t->type) {
    port->t->data = h->t->data;
    port->t->buffer = h->t->buffer;
    port->conv = NULL;
  } else {
    if (port->own == NULL) {
      port->own = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(p->backend), ggml_nbytes(port->t));
      if (port->own == NULL) return false;
    }
    port->t->data = ggml_backend_buffer_get_base(port->own);
    port->t->buffer = port->own;
    port->conv = h->t;
  }
  port->set = true;
  return true;
}

an_term ggml_bind(an_env *env, struct ggml_Lease_state *lease, int64_t slot, struct ggml_TensorHandle_state *t) {
  struct entry *p = lease->e;
  struct port *leaf;
  const char *why = NULL;

  if (p == NULL || !p->compiled) return tg_error(env, "bind a compiled graph");
  if (slot < 0 || slot >= p->leaf_count) return tg_error(env, "no such leaf");
  if (t->t == NULL) return tg_error(env, "bind a tensor that was made");
  leaf = &p->leaves[slot];
  if (ggml_nelements(t->t) != ggml_nelements(leaf->t)) return tg_error(env, "a bound tensor has another element count than its leaf");
  if (!ggml_backend_supports_buft(p->backend, ggml_backend_buffer_get_type(t->buffer))) {
    return tg_error(env, "a bound tensor lives on another device than its graph");
  }
  tg_lock_device();
  if (!attach(p, leaf, t)) why = "out of memory";
  else if (leaf->conv != NULL) why = tg_copy(t->t, leaf->t);
  tg_unlock_device();
  if (why != NULL) return tg_error(env, why);
  return ggml_result_Void_ok(env, an_void(env));
}

an_term ggml_output(an_env *env, struct ggml_Lease_state *lease, int64_t output, bool int32) {
  struct entry *p = lease->e;
  an_term term, failure;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  bool attached;

  if (p == NULL || !p->compiled) return tg_error(env, "an output of a compiled graph");
  if (output < 0 || output >= p->output_count) return tg_error(env, "no such output");
  if (!tg_new_tensor(env, p->device, p->outs[output].t->ne, int32 ? GGML_TYPE_I32 : p->outs[output].t->type, state, &failure)) {
    return failure;
  }
  tg_lock_device();
  attached = attach(p, &p->outs[output], state);
  tg_unlock_device();
  if (!attached) return tg_error(env, "out of memory");
  return ggml_result_TensorHandle_ok(env, term);
}

an_term ggml_output_into(an_env *env, struct ggml_Lease_state *lease, int64_t output, struct ggml_TensorHandle_state *t) {
  struct entry *p = lease->e;
  bool attached = false, alias = false;
  char why[96];
  int64_t i;

  if (p == NULL || !p->compiled) return tg_error(env, "an output of a compiled graph");
  if (output < 0 || output >= p->output_count) return tg_error(env, "no such output");
  if (t->t == NULL) return tg_error(env, "an output tensor that was made");
  if (ggml_nelements(t->t) != ggml_nelements(p->outs[output].t)) {
    return tg_error(env, "an output tensor has another element count than its output");
  }
  if (!ggml_backend_supports_buft(p->backend, ggml_backend_buffer_get_type(t->buffer))) {
    return tg_error(env, "an output tensor lives on another device than its graph");
  }
  tg_lock_device();
  for (i = 0; i < p->leaf_count; i++) {
    if (p->leaves[i].held == t->store) alias = true;
  }
  for (i = 0; i < p->output_count; i++) {
    if (i != output && p->outs[i].held == t->store) alias = true;
  }
  if (!alias) attached = attach(p, &p->outs[output], t);
  tg_unlock_device();
  if (alias) {
    snprintf(why, sizeof why, "output %" PRId64 ": the tensor given is an input of the run or another output", output);
    return tg_error(env, why);
  }
  if (!attached) return tg_error(env, "out of memory");
  return ggml_result_Void_ok(env, an_void(env));
}

an_term ggml_pin(an_env *env, struct ggml_Lease_state *lease) {
  struct entry *p = lease->e;
  an_term term;
  struct ggml_Pin_state *pin;
  bool cached = false;
  int i;

  if (p == NULL || !p->compiled) return tg_error(env, "pin a compiled graph");
  pin = ggml_Pin_new(env, &term);
  pthread_mutex_lock(&cache_lock);
  for (i = 0; i < MAX_ENTRIES; i++) {
    if (cache[i] == p) cached = true;
  }
  if (cached) p->pins++;
  pthread_mutex_unlock(&cache_lock);
  pin->e = cached ? p : NULL;
  return ggml_result_Pin_ok(env, term);
}

/* A pin holds its graph in the cache; a graph that has no slot (every one is in use or pinned) gets a pin that holds
   nothing, and runs as an ordinary graph. The free hook below ends a pin: it never waits for the device lock, and the
   graph cannot have been freed, since a pinned one is never a victim and is dropped only when it never compiled. */
CC_SHORT void ggml_Pin_free(void *state) {
  struct ggml_Pin_state *self = (struct ggml_Pin_state *)state;

  if (self->e == NULL) return;
  pthread_mutex_lock(&cache_lock);
  self->e->pins--;
  pthread_mutex_unlock(&cache_lock);
  self->e = NULL;
}

an_term ggml_run(an_env *env, struct ggml_Lease_state *lease, ggml_list_Float fills) {
  struct entry *p = lease->e;
  struct device_state *d;
  float *values, *checks;
  double v;
  int64_t i = 0, j;
  int next;
  const char *why = NULL;
  char text[160];

  if (p == NULL || !p->compiled) return tg_error(env, "run a compiled graph");
  for (i = 0; i < p->leaf_count; i++) {
    if (!p->leaves[i].set) return tg_error(env, "run with every leaf bound");
  }
  for (i = 0; i < p->output_count; i++) {
    if (!p->outs[i].set && !p->outs[i].donated) return tg_error(env, "run with every output made");
  }
  for (i = 0; i < p->leaf_count; i++) {
    int64_t k;

    if (!p->leaves[i].donated) continue;
    if (p->leaves[i].conv != NULL) return tg_error(env, "an input a set_rows writes in place must be F32");
    for (k = 0; k < p->leaf_count; k++) {
      if (k != i && p->leaves[k].held != NULL && p->leaves[k].held == p->leaves[i].held) {
        return tg_error(env, "an input a set_rows writes in place is bound to another input too");
      }
    }
  }
  i = 0;
  values = (float *)malloc((size_t)(p->fill_count + p->output_count + p->guard_count) * sizeof(float));
  if (values == NULL) return tg_error(env, "out of memory");
  checks = values + p->fill_count;
  while ((next = ggml_list_Float_next(env, &fills, &v)) == 1 && i < p->fill_count) values[i++] = (float)v;
  if (next < 0) {
    free(values);
    return an_void(env);
  }
  if (i != p->fill_count || next == 1) {
    free(values);
    return tg_error(env, "run needs one value per fill");
  }

  tg_lock_device();
  if (p->fill_count > 0) ggml_backend_tensor_set(p->fill_store, values, 0, (size_t)p->fill_count * sizeof(float));
  d = &devices[p->device + 1];
  if (d->galloc == NULL) d->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p->backend));
  if (d->galloc == NULL) why = "making the graph allocator failed";
  if (why == NULL) {
    for (i = 0; i < p->inner_count; i++) {
      p->inner[i]->data = NULL;
      p->inner[i]->buffer = NULL;
    }
    if (d->planned != p) {
      d->planned = NULL;
      if (!ggml_gallocr_reserve(d->galloc, p->graph)) why = "allocating the graph failed";
    }
  }
  if (why == NULL && !ggml_gallocr_alloc_graph(d->galloc, p->graph)) why = "allocating the graph failed";
  if (why == NULL) {
    d->planned = p;
    if (ggml_backend_graph_compute(p->backend, p->graph) != GGML_STATUS_SUCCESS) why = "ggml_backend_graph_compute failed";
  }
  if (why == NULL) {
    ggml_backend_tensor_get(p->checks, checks, 0, (size_t)(p->output_count + p->guard_count) * sizeof(float));
  }
  for (i = 0, j = p->output_count; why == NULL && i < p->count; i++) {
    if (p->nodes[i].guard != NULL && checks[j++] != 0.0f) {
      snprintf(text, sizeof text, "an index of %s is outside the rows of its table", op_names[p->nodes[i].op]);
      why = text;
    }
  }
  for (i = 0; why == NULL && i < p->output_count; i++) {
    if (!isfinite(checks[i])) {
      snprintf(text, sizeof text, "output %" PRId64 " of the graph holds a value that is not finite", i);
      why = text;
    } else if (p->outs[i].conv != NULL) {
      why = tg_copy(p->outs[i].t, p->outs[i].conv);
    }
  }
  release(p);
  tg_unlock_device();
  free(values);
  if (why != NULL) return tg_error(env, why);
  return ggml_result_Void_ok(env, an_void(env));
}
