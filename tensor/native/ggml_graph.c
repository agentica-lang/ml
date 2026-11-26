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
   nothing, except the constants a pinned program keeps bound (see `struct
   port`). The allocator keeps the largest buffer any graph needed for the
   rest of the process.

   Two locks, in this order: the device lock (1) and `cache_lock` (2).
   `cache_lock` (the slots, who holds a graph, the pins, the kept leaves of an
   idle graph, the counters) is held briefly and is all a free hook takes; the
   device lock is held through builds and runs. A graph the hook must drop is
   destroyed under the device lock later (`tg_defer`, which takes no lock).

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
  OP_ADAM_STEP,   /* args: p, grad, m, v, the 7 numbers of ggml's AdamW; ints: 1 to update m and v in place; the node is its new p */
  OP_ADAM_PART,   /* args: an adam_step; ints: 0, 1 or 2, the new p, m or v of it */
  OP_CONV2D,      /* ints: stride, padding, groups */
  OP_MAX_POOL2D,  /* ints: kernel, stride, padding */
  OP_AVG_POOL2D,  /* ints: kernel, stride, padding */
  OP_GATED_DELTA_RULE, /* args: q, k, v [T, H, S], g, beta [T, H], the state [H, S, S]; the answers [T, H * S] then the state [S, H * S] */
  OP_EXPERT_MATMUL, /* args: the weights [E, rows, inner], x [T, uses or 1, inner], the expert ids [T, uses] */
  OP_COUNT
};

static const char *op_names[OP_COUNT] = {
  "leaf", "full", "add", "sub", "mul", "div", "neg", "exp", "log", "relu", "sqrt", "sigmoid", "tanh",
  "matmul", "transpose", "reshape", "broadcast_to", "sum", "mean", "max", "identity", "arange", "eq", "lt", "where",
  "slice", "gather", "gelu_erf", "concat", "conv1d", "scatter_add", "permute", "attention", "rope", "batch_matmul",
  "norm", "rms_norm", "set_rows", "top_k", "cast", "adam_step", "adam_part", "conv2d", "max_pool2d", "avg_pool2d",
  "gated_delta_rule", "expert_matmul",
};

/* A leaf or an output: the tensor of the graph, its own memory (made on the
   first call that needs it) and, for this call, the tensor it converts to or
   from.

   Kept leaves. A run of a pinned program (`Trace.compile`) leaves its constants bound, so the next run binds only its
   inputs (`ggml_resume`, `ggml_keep`). What keeps a stale binding from ever being read:

   1. A bound port holds a count on the memory it points at (`held`), kept or not, so that memory is never freed under
      it. Whatever else goes wrong, no run reads freed memory.
   2. Only a leaf that reads its tensor in place is kept, and never a donated one: an input a `set_rows` writes and a
      converted copy (an I32 tensor), which would not follow its tensor, are bound by every run. The caller names the
      leaves of the set (`ggml_keep`), and the set is those leaves or none: a list with a leaf that cannot be kept
      keeps nothing, so what the caller leaves out of its next bind is what the graph holds.
   3. The kept leaves of an entry are one set, the constants of the program of one pin, `kept_for`. `ggml_resume`
      answers true to that pin alone; for any other program, or once a kept tensor is closed, it ends the set and the
      caller binds every leaf. So two programs that share a graph take turns and never read each other's weights. A
      kept leaf that is bound again ends the set too (at the end of the lease), and so does a run of a lease that did
      not ask (`resumed`): that run has the kept leaves unbound.
   4. The set ends, and its counts go, when its pin ends, when the handle of one of its tensors is closed
      (`tg_closed`), and when the entry is destroyed (an evicted graph has no pin, so it keeps nothing). An entry in
      use is left to its holder: the end of its lease ends the set if the pin is gone or a kept tensor is closed.
   5. The ports and `kept_leaves` of an entry in use belong to the holder of its lease; those of an idle entry are
      touched under `cache_lock` alone. `kept_for` and `in_use` are under `cache_lock`. */
struct port {
  struct ggml_tensor *t;
  ggml_backend_buffer_t own;
  struct ggml_tensor *conv; /* the bound tensor, converted to or from `own`; else NULL. Its memory is `held` */
  bool set;                               /* bound or prepared for the next run */
  bool donated;                           /* a leaf a `set_rows` writes in place or an `adam_step` moment, or the output that is that leaf: no copy in or out */
  bool kept;                              /* a leaf that stays bound after the run; `held->kept` counts it */
  bool wanted;                            /* inside `ggml_keep`: a leaf of the list it was given */
  struct tg_storage *held;                /* the memory of the bound tensor, kept until the run is over */
  struct ggml_tensor *writer;             /* an output: the node that writes it, which `place` puts on a side */
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
  struct ggml_tensor *ids; /* the node as the row ids of `ids_rows` rows for `ids_op`, made and guarded once */
  int64_t ids_rows;
  int ids_op;
  struct ggml_tensor *half; /* the node as the F16 mask of a flash attention, made once */
  struct ggml_tensor *packed; /* the node as an operand of mul_mat when `dense` is not packed, made once */
  struct ggml_tensor *window; /* a slice from the first element as the view it is: what flash attention reads */
  struct ggml_tensor *adam[3]; /* an adam_step: its new weight, first and second moment, each a tensor of its own */
  int adam_donates;            /* an adam_step that updates its moments, two input leaves, in place */
  int part;                    /* an adam_part: 0 the new weight, 1 the first moment, 2 the second */
};

#define MAX_OUTPUTS 512
#define MAX_ARGS 256
#define MAX_ENTRIES 32

/* A live pin of an entry, by its id: the state of a `Pin` is the VM's, and no pointer to it is kept. */
struct pin_id {
  int64_t id;
  struct pin_id *next;
};

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
  struct ggml_tensor *checks; /* the sum of each output (inf or NaN unless finite), then the guards */
  int64_t output_count, guard_count;
  struct ggml_tensor **inner; /* what the shared allocator places: valid while the device's plan is this graph's */
  int64_t inner_count;
  struct ggml_tensor **views; /* the views of leaves and outputs: pointed at them for a run, at nothing after it */
  int64_t view_count;
  bool compiled, in_use;
  int pins;     /* pins on it: never evicted while above 0 (under cache_lock) */
  struct pin_id *pinned; /* the ids of those pins (under cache_lock) */
  int64_t kept_for;    /* the pin whose constants the kept leaves are; 0 for none */
  int64_t kept_leaves; /* how many leaves are kept */
  int64_t used; /* the tick of its last lease, for eviction */
  /* Layers across the device and the CPU (`place`). A graph whose leaves are all in device memory never uses any of
     this and runs on its backend alone. */
  signed char *leaf_side;        /* per leaf: 1 when its tensor was in host memory at the last `place` */
  struct ggml_tensor **save_node; /* every node of the graph as the first mixed run found it, with its sources in */
  struct ggml_tensor **save_src;  /* `save_src`: the scheduler rewrites the sources and its backends reorder the nodes */
  struct ggml_tensor **side_key; /* node -> side, for the sources of the next node */
  signed char *side_val;
  size_t side_cap;
  size_t sched_need;             /* the nodes of the graph and an upper bound of its leaves, for the scheduler's tables */
  bool placed;                   /* `leaf_side` and the sides of the nodes are of the leaves now bound */
  bool mixed;                    /* some leaf is in host memory under a device graph: the scheduler runs it */
};

/* The intermediates' allocator of each device and the graph it last planned. */
struct device_state {
  ggml_gallocr_t galloc;
  struct entry *planned;
  ggml_backend_sched_t sched; /* the device and the CPU, for the graphs that have leaves on both */
  size_t sched_size;
  struct entry *sched_planned; /* the graph that the scheduler last reserved its memory for */
};
static int64_t mixed_split_count; /* the splits of the scheduler in the last mixed run, atomic */
static int64_t mixed_run_count; /* the runs through a scheduler, for the tests: atomic, so reading it never waits */

/* Lock order: the device lock (1), then cache_lock (2). Pins and the pin count of an entry are under cache_lock alone;
   the memory of a tensor that the last holder lets go of is queued (`tg_defer`, no lock) and freed at the next hold of
   the device lock. */
CC_LOCK_LEVEL(2) CC_NO_SYSCALL_UNDER() static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static struct entry *cache[MAX_ENTRIES];
static int64_t compile_count = 0, tick = 0, pin_count = 0;
static struct device_state devices[9]; /* under the device lock */

/* Points a leaf or an output at nothing and lets go of its memory, so no call's tensor stays reachable from it. */
static void unbind(struct entry *e, struct port *port) {
  port->t->data = NULL;
  port->t->buffer = NULL;
  port->conv = NULL;
  port->set = false;
  if (port->kept) {
    __atomic_fetch_sub(&port->held->kept, 1, __ATOMIC_SEQ_CST);
    e->kept_leaves--;
    port->kept = false;
  }
  if (port->held != NULL) tg_storage_release(port->held);
  port->held = NULL;
}

/* Ends the kept set of `e`: for the holder of its lease, or under cache_lock when `e` is idle. The caller clears
   `kept_for` under cache_lock. */
static void drop_kept(struct entry *e) {
  int64_t i;

  for (i = 0; e->kept_leaves > 0 && i < e->leaf_count; i++) {
    if (e->leaves[i].kept) unbind(e, &e->leaves[i]);
  }
}

/* True when the handle of a tensor that `e` keeps bound is closed. */
static bool kept_closed(const struct entry *e) {
  int64_t i;

  for (i = 0; e->kept_leaves > 0 && i < e->leaf_count; i++) {
    if (e->leaves[i].kept && __atomic_load_n(&e->leaves[i].held->closed, __ATOMIC_SEQ_CST) != 0) return true;
  }
  return false;
}

/* What the end of a run or of a lease lets go of: every output and every leaf that is not kept. */
static void release(struct entry *e) {
  int64_t i;

  for (i = 0; e->leaves != NULL && i < e->leaf_count; i++) {
    if (!e->leaves[i].kept) unbind(e, &e->leaves[i]);
  }
  for (i = 0; i < e->output_count; i++) unbind(e, &e->outs[i]);
}

/* A handle closes a tensor that graphs keep bound: every idle graph that keeps a closed tensor ends its set, so the
   memory goes as it did before leaves were kept (an entry in use does it at the end of its lease). The hook set
   `closed` before this, and `ggml_Lease_free` reads it under the same lock after it clears `in_use`: one of the two
   sees the other. */
CC_SHORT void tg_closed(void) {
  int i;

  pthread_mutex_lock(&cache_lock);
  for (i = 0; i < MAX_ENTRIES; i++) {
    struct entry *e = cache[i];

    if (e != NULL && !e->in_use && kept_closed(e)) {
      e->kept_for = 0;
      drop_kept(e);
    }
  }
  pthread_mutex_unlock(&cache_lock);
}

/* Call under the device lock, on an entry nothing else reaches. */
static void destroy(struct entry *e) {
  int64_t d = e->device + 1;

  int64_t i;

  if (d >= 0 && d < 9 && devices[d].planned == e) devices[d].planned = NULL;
  if (d >= 0 && d < 9 && devices[d].sched_planned == e) devices[d].sched_planned = NULL;
  drop_kept(e);
  for (i = 0; e->leaves != NULL && i < e->leaf_count; i++) {
    if (e->leaves[i].own != NULL) ggml_backend_buffer_free(e->leaves[i].own);
  }
  for (i = 0; i < e->output_count; i++) {
    if (e->outs[i].own != NULL) ggml_backend_buffer_free(e->outs[i].own);
  }
  if (e->fill_buffer != NULL) ggml_backend_buffer_free(e->fill_buffer);
  free(e->save_node); /* the block of `make_side_tables` */
  if (e->fill_ctx != NULL) ggml_free(e->fill_ctx);
  if (e->port_ctx != NULL) ggml_free(e->port_ctx);
  if (e->ctx != NULL) ggml_free(e->ctx);
  while (e->pinned != NULL) {
    struct pin_id *live = e->pinned;

    e->pinned = live->next;
    free(live);
  }
  free(e->nodes);
  free(e->leaves);
  free(e->inner);
  free(e->views);
  free(e->key);
  free(e);
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
  if (self->e->kept_leaves > 0 && (self->e->kept_for == 0 || kept_closed(self->e))) {
    self->e->kept_for = 0;
    drop_kept(self->e);
  }
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
   most (a matmul's transpose and cont, a reduction's reshapes; a `gather` is
   the largest, about 14 with its index guard); `node` and `compile` check the
   room is there. A masked `scatter_add` is more (about 12 for each of its
   chunks) and checks the room for them itself. */
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
   held, or NULL. A pinned one goes first: it is the copy that keeps the
   constants of its program bound. A program without a pin (`Trace.eval`) with
   the same key gets that copy too and ends its set, which costs the pinned
   program one whole bind and nothing else. */
static struct entry *cached(int64_t device, ng_str key) {
  struct entry *e = NULL;
  int i;

  pthread_mutex_lock(&cache_lock);
  for (i = 0; i < MAX_ENTRIES && (e == NULL || e->pins == 0); i++) {
    struct entry *c = cache[i];

    if (c != NULL && c->compiled && !c->in_use && c->device == device && (e == NULL || c->pins > 0) &&
        strlen(c->key) == key.len && memcmp(c->key, key.data, key.len) == 0) {
      e = c;
    }
  }
  if (e != NULL) {
    e->in_use = true;
    e->used = ++tick;
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

/* `t` as a tensor that starts at the beginning of its buffer: a view at an offset (a value of the fill store, a slice) is
   copied. */
static struct ggml_tensor *aligned(struct entry *p, struct ggml_tensor *t) {
  return t->view_src != NULL && t->view_offs != 0 ? ggml_cont(p->ctx, t) : t;
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
  struct gnode *node = (struct gnode *)n; /* the nodes are `p->nodes`; the copy is a cache */
  struct ggml_tensor *t = dense(p, n);
  int i;

  if (ggml_is_quantized(t->type)) return t; /* a weight's blocks: its strides count bytes of blocks, and it is a leaf */
  for (i = 1; i < GGML_MAX_DIMS; i++) {
    if (t->nb[i] == t->nb[i - 1] * (size_t)t->ne[i - 1]) continue;
    if (node->packed == NULL) node->packed = ggml_cont(p->ctx, t);
    return node->packed;
  }
  return t;
}

/* The node's own tensor, copied when it is a view (a permute that kept axis 0): for the ops that take `->t` raw. */
static struct ggml_tensor *plain(struct entry *p, const struct gnode *n) {
  return n->lazy ? contiguous(p, n->t) : dense(p, n);
}

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

/* `silu(x) * u` as one ggml_swiglu_split when `a` and `b` are `x * sigmoid(x)` and `u` (either way round), all F32 at
   one shape with dense rows, as an MLP's gate and up are; NULL otherwise. The node `x * sigmoid(x)` drops out of the
   graph unless something else reads it. */
static struct ggml_tensor *swiglu(struct entry *p, const struct gnode *a, const struct gnode *b) {
  const struct gnode *pair[2] = {a, b};
  int i, j;

  if (a->lazy || b->lazy) return NULL;
  for (i = 0; i < 2; i++) {
    struct ggml_tensor *silu = pair[i]->t, *up = pair[1 - i]->t;

    if (silu->op != GGML_OP_MUL) continue;
    for (j = 0; j < 2; j++) {
      struct ggml_tensor *x = silu->src[j], *sigmoid = silu->src[1 - j];

      if (sigmoid->op == GGML_OP_UNARY && ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID && sigmoid->src[0] == x &&
          x->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && ggml_are_same_shape(x, up) &&
          ggml_are_same_shape(x, silu) && ggml_is_contiguous_1(x) && ggml_is_contiguous_1(up)) {
        return ggml_swiglu_split(p->ctx, x, up);
      }
    }
  }
  return NULL;
}

/* `1 - m` for a 0/1 mask. */
static struct ggml_tensor *flip(struct entry *p, struct ggml_tensor *m) {
  return ggml_scale_bias(p->ctx, contiguous(p, m), -1.0f, 1.0f);
}

/* A max over one row at least this long is found with argmax, not pool_1d. */
#define LONG_ROW 1024

/* The largest of the values in `row` [n, 1] (contiguous) at its row: argmax
   finds it in one workgroup and get_rows reads it. */
static struct ggml_tensor *row_max(struct entry *p, struct ggml_tensor *row) {
  return ggml_get_rows(p->ctx, ggml_reshape_2d(p->ctx, row, 1, row->ne[0]), ggml_argmax(p->ctx, row));
}

/* The largest value of each row of `t` [m, k] (contiguous), as [k, 1]. */
static struct ggml_tensor *rows_max(struct entry *p, struct ggml_tensor *t) {
  struct ggml_tensor *at = ggml_reshape_2d(p->ctx, ggml_argmax(p->ctx, t), 1, t->ne[1]);
  struct ggml_tensor *picked = ggml_get_rows(p->ctx, ggml_reshape_3d(p->ctx, t, 1, t->ne[0], t->ne[1]), at);

  return ggml_reshape_2d(p->ctx, picked, t->ne[1], 1);
}

/* argmax reads its source at the start of a storage buffer binding, which Vulkan aligns to at most 256 bytes
   (`minStorageBufferOffsetAlignment`): ggml-vulkan aborts on a view that starts in between. */
#define PIECE_ALIGN_BYTES 256

/* The largest value of a long row [n, 1] (contiguous). One workgroup over the whole row is slow (1.2 ms for 151,936
   floats on Vulkan), so the row is cut into k pieces of m values, m about the square root of n rounded up to 256 bytes
   (64 floats), and a tail of the n - k * m left; the largest of the pieces' maxima is the answer. Every piece and the
   tail start at a multiple of 256 bytes from the row, and a row that is a view starting in between is copied first. */
static struct ggml_tensor *long_row_max(struct entry *p, struct ggml_tensor *row) {
  int64_t align = PIECE_ALIGN_BYTES / (int64_t)row->nb[0];
  int64_t n = row->ne[0], m = ((int64_t)ceil(sqrt((double)n)) + align - 1) / align * align, k = n / m;
  size_t step = (size_t)m * row->nb[0];
  struct ggml_tensor *maxima;

  if (row->view_offs % PIECE_ALIGN_BYTES != 0) row = ggml_cont(p->ctx, row);
  GGML_ASSERT(row->view_offs % PIECE_ALIGN_BYTES == 0 && step % PIECE_ALIGN_BYTES == 0);
  maxima = rows_max(p, ggml_view_2d(p->ctx, row, m, k, step, 0));
  if (k * m < n) {
    struct ggml_tensor *tail = ggml_view_2d(p->ctx, row, n - k * m, 1, step, (size_t)k * step);

    maxima = ggml_concat(p->ctx, maxima, rows_max(p, tail), 0);
  }
  return row_max(p, maxima);
}

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
   (ggml order), as a view: its strides are those of `x`. */
static struct ggml_tensor *slice(struct entry *p, const struct gnode *x, const int64_t *starts, int rank,
                                 const int64_t ne[4]) {
  struct ggml_tensor *t = dense(p, x);
  size_t offset = 0;
  int i;

  for (i = 0; i < rank; i++) offset += (size_t)starts[i] * t->nb[rank - 1 - i];
  return ggml_view_4d(p->ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], offset);
}

/* The indices as I32 row ids of a table of `rows` rows for an `op`. ggml reads
   out of bounds (or aborts) on a bad index, so the indices are clamped to the
   rows first and `guard` counts the ones that were outside: `run` refuses its
   result unless that count is 0. The indices are F32 here, exact below 2^24,
   so `node` refuses a table with more rows. The ids are made once per indices,
   rows and op (a decode step writes every layer's cache at the same
   positions): a later node gets them with no guard of its own, since the
   first one's guard already checks them. One slot per indices node: caches of
   alternating sizes at the same positions make the ids again for each. */
static struct ggml_tensor *row_ids(struct entry *p, const struct gnode *indices, int64_t rows, int op,
                                   struct ggml_tensor **guard) {
  struct gnode *source = (struct gnode *)indices; /* the nodes are `p->nodes`; the ids are a cache */
  struct ggml_tensor *at;
  float top = (float)(rows - 1);
  struct ggml_tensor *clamped, *below, *above;

  if (source->ids != NULL && source->ids_rows == rows && source->ids_op == op) {
    *guard = NULL;
    return source->ids;
  }
  at = ggml_reshape_1d(p->ctx, dense(p, indices), volume(indices->ne));
  /* ggml_clamp works in place: clamp a copy */
  clamped = ggml_clamp(p->ctx, ggml_cont(p->ctx, at), 0.0f, top);
  /* step(-i - 0.5) + step(i - top - 0.5): 1 for an index below 0 or past top */
  below = ggml_step(p->ctx, ggml_scale_bias(p->ctx, at, -1.0f, -0.5f));
  above = ggml_step(p->ctx, ggml_scale_bias(p->ctx, at, 1.0f, -top - 0.5f));
  *guard = ggml_sum(p->ctx, ggml_add(p->ctx, below, above));
  source->ids = ggml_cast(p->ctx, clamped, GGML_TYPE_I32);
  source->ids_rows = rows;
  source->ids_op = op;
  return source->ids;
}

/* The rows of `source` (its first row-major axis) that `indices` name, as
   ggml_get_rows over a [width, rows] view. */
static struct ggml_tensor *gather(struct entry *p, const struct gnode *source, int rank, const struct gnode *indices,
                                  struct ggml_tensor **guard) {
  int64_t rows = source->ne[rank - 1];
  struct ggml_tensor *table = ggml_reshape_2d(p->ctx, dense(p, source), volume(source->ne) / rows, rows);

  return ggml_get_rows(p->ctx, table, row_ids(p, indices, rows, OP_GATHER, guard));
}

static bool on_vulkan(const struct entry *p);

/* The most floats of one chunk of the mask `scatter_add_masked` makes (64 MB), the most chunks it makes (so k is at
   most 32768), and the ggml tensors it makes for each chunk and besides them. */
#define MASK_CELLS (1 << 24)
#define MASK_CHUNKS 64
#define MASK_TENSORS_PER_CHUNK 12
#define MASK_TENSORS_BESIDES 8

/* `scatter_add` on Vulkan: its `get_rows_back` kernel has a thread for each cell of the answer, each one reading all
   k indices, so it costs rows * k * width. Here every row of `updates` is first given the sum of the rows that name
   its own row, with one matmul by the k x k mask `ids[i] == ids[j]` (k * k * width, exact: it adds only the rows of one
   index), and those totals are then written with set_rows into the zero rows (the rows that share an index write the
   same total). The mask is made, and multiplied, kc columns at a time. */
static struct ggml_tensor *scatter_add_masked(struct entry *p, struct ggml_tensor *flat, struct ggml_tensor *ids,
                                              int64_t rows, int64_t kc) {
  int64_t k = flat->ne[1], at;
  struct ggml_tensor *ids_f32 = ggml_cast(p->ctx, ids, GGML_TYPE_F32);
  struct ggml_tensor *by_index = ggml_cont(p->ctx, ggml_transpose(p->ctx, flat)); /* [k, width] */
  struct ggml_tensor *answer = ggml_fill(p->ctx, ggml_new_tensor_2d(p->ctx, GGML_TYPE_F32, flat->ne[0], rows), 0.0f);

  for (at = 0; at < k; at += kc) {
    int64_t n = k - at < kc ? k - at : kc;
    struct ggml_tensor *all = ggml_reshape_2d(p->ctx, ids_f32, k, 1);
    struct ggml_tensor *mine =
        ggml_reshape_2d(p->ctx, ggml_view_1d(p->ctx, ids_f32, n, (size_t)at * sizeof(float)), 1, n);
    struct ggml_tensor *grid = ggml_new_tensor_2d(p->ctx, GGML_TYPE_F32, k, n); /* a shape for the repeat */
    struct ggml_tensor *apart = ggml_abs(p->ctx, ggml_sub(p->ctx, ggml_repeat(p->ctx, all, grid), mine));
    /* the ids are whole numbers: 1 - |difference| is 1 for a pair that shares its row and at most 0 for any other */
    struct ggml_tensor *same = ggml_relu(p->ctx, ggml_scale_bias(p->ctx, apart, -1.0f, 1.0f));
    struct ggml_tensor *totals = ggml_mul_mat(p->ctx, by_index, same); /* [width, n] */

    answer = ggml_set_rows(p->ctx, answer, totals, ggml_view_1d(p->ctx, ids, n, (size_t)at * sizeof(int32_t)));
  }
  return answer;
}

/* The inverse of `gather`: the rows of `updates` ([width, k] once flat)
   added into `rows` zero rows, at the row each of the k indices names, as
   ggml_get_rows_back. The reference tensor only gives the answer its shape;
   it is not part of the graph. On Vulkan in F32 mode (the fp16 mode would round
   the updates in the matmul), when k is below the rows and the graph has room for
   the chunks, the masked sums of `scatter_add_masked` do it with less work. */
static struct ggml_tensor *scatter_add(struct entry *p, const struct gnode *indices, const struct gnode *updates,
                                       int64_t rows, struct ggml_tensor **guard) {
  int64_t k = volume(indices->ne);
  struct ggml_tensor *flat = ggml_reshape_2d(p->ctx, dense(p, updates), volume(updates->ne) / k, k);
  struct ggml_tensor *ids = row_ids(p, indices, rows, OP_SCATTER_ADD, guard);
  int64_t kc = MASK_CELLS / k > 0 ? MASK_CELLS / k : 1;
  int64_t chunks;
  bool masked;

  if (kc > k) kc = k;
  chunks = (k + kc - 1) / kc;
  masked = on_vulkan(p) && !tg_fp16() && tg_masked_scatter() && k < rows && chunks <= MASK_CHUNKS &&
           room(p, (size_t)(MASK_TENSORS_PER_CHUNK * chunks + MASK_TENSORS_BESIDES));
  if (masked) return scatter_add_masked(p, flat, ids, rows, kc);
  return ggml_get_rows_back(p->ctx, flat, ids, ggml_new_tensor_2d(p->ctx, GGML_TYPE_F32, flat->ne[0], rows));
}

/* Whether `t` is the permute (0, 2, 1, 3) of its source: permuting it again gives the source back, so a norm that
   ran on the source feeds a rope with no view between them, which ggml-vulkan needs to fuse norm, mul and rope. */
static bool swapped_back(const struct ggml_tensor *t) {
  const struct ggml_tensor *s = t->src[0];

  return t->op == GGML_OP_PERMUTE && s->ne[1] == t->ne[2] && s->ne[2] == t->ne[1] && s->nb[1] == t->nb[2] &&
         s->nb[2] == t->nb[1] && s->ne[0] == t->ne[0] && s->ne[3] == t->ne[3] && s->nb[0] == t->nb[0] &&
         s->nb[3] == t->nb[3];
}

/* The node as an operand of flash attention: its rows must be dense, its other strides may be a view's. The first
   rows of a cache (a slice from its first element) are read where they are, as llama.cpp reads the rows in use. */
static struct ggml_tensor *rows_of(struct entry *p, const struct gnode *n) {
  struct ggml_tensor *t = n->window != NULL ? n->window : full_of(p, n);

  return t->nb[0] == ggml_type_size(t->type) ? t : ggml_cont(p->ctx, t);
}

static bool on_vulkan(const struct entry *p) {
  return ggml_backend_dev_type(ggml_backend_get_device(p->backend)) != GGML_BACKEND_DEVICE_TYPE_CPU;
}

/* The mask as F16 for flash attention, once for every layer that reads it. The lowering lets flash attention take a
   mask that holds 0 and large negative fills (-1e9, say), which F16 keeps exactly once clamped to its lowest, -65504:
   -1e9 itself would become -inf, and a row with every key hidden would not stay finite. The clamp works in place, so
   it runs on a copy: the mask may be read by the products of another layer. */
static struct ggml_tensor *half_mask(struct entry *p, const struct gnode *mask) {
  struct gnode *node = (struct gnode *)mask; /* the nodes are `p->nodes`; the conversion is a cache */

  if (node->half == NULL) {
    node->half = ggml_cast(p->ctx, ggml_clamp(p->ctx, ggml_cont(p->ctx, dense(p, mask)), -65504.0f, 65504.0f), GGML_TYPE_F16);
  }
  return node->half;
}

/* softmax(scale * q kᵀ + mask) v over q [B, H, Sq, D], k [B, Hkv, Sk, D] and v [B, Hkv, Sk, Dv] (ne [D, Sq, H, B]
   and so on), as one ggml_flash_attn_ext that reads the heads as the views they are: no copy of q, k or v, no
   scores in memory. It takes the mask in F16 (the fills it gets, 0 and -1e9 clamped to -65504, are exact) and answers [Dv, H, Sq, B], which is permuted back to [Dv, Sq, H, B] as a view. NULL when the
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

/* `input` ([B, C, H, W] row-major) convolved with `weight` ([O, C / groups, KH, KW]) by ggml's direct kernel, which
   computes in F32 on the CPU and on Vulkan. Each of the `groups` runs of output channels reads its own run of input
   channels, and the answers join along the channels. */
static struct ggml_tensor *conv2d(struct entry *p, const struct gnode *input, const struct gnode *weight, int stride,
                                  int padding, int groups) {
  struct ggml_tensor *x = dense(p, input), *w = dense(p, weight), *out = NULL;
  int64_t in_per = x->ne[2] / groups, out_per = w->ne[3] / groups;
  int g;

  if (groups == 1) return ggml_conv_2d_direct(p->ctx, w, x, stride, stride, padding, padding, 1, 1);
  for (g = 0; g < groups; g++) {
    struct ggml_tensor *xg = ggml_cont(p->ctx, ggml_view_4d(p->ctx, x, x->ne[0], x->ne[1], in_per, x->ne[3], x->nb[1],
                                                           x->nb[2], x->nb[3], (size_t)(g * in_per) * x->nb[2]));
    struct ggml_tensor *wg = ggml_view_4d(p->ctx, w, w->ne[0], w->ne[1], w->ne[2], out_per, w->nb[1], w->nb[2],
                                          w->nb[3], (size_t)(g * out_per) * w->nb[3]);
    struct ggml_tensor *y = ggml_conv_2d_direct(p->ctx, wg, xg, stride, stride, padding, padding, 1, 1);

    out = out == NULL ? y : ggml_concat(p->ctx, out, y, 2);
  }
  return out;
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
   transpose of one: a view, read in place). An F16 cache goes only where `transformers` puts it: the cache leaf of a
   set_rows, the key or value of an attention (a leaf, the result of a set_rows or a cast, or a slice of one of
   those: the rows in use), and a cast of a leaf to F32.
   Any other use would copy or compute on blocks that ggml's ops do not take
   (`ggml_cont` of a quantized view sizes its copy by rows and writes past its buffer), so it is an error before any
   ggml call. `x` is argument `i` of a node of `op`; NULL when it fits. */
static const char *packed_misuse(int64_t op, int i, const struct gnode *x) {
  if (!ggml_is_quantized(x->t->type) && x->t->type != GGML_TYPE_F16) return NULL;
  if (op == OP_MATMUL && i == 1) return NULL;
  if (x->t->type == GGML_TYPE_F16) {
    /* an F16 cache: written in place by set_rows, read by attention, widened by a cast */
    if (op == OP_SET_ROWS && i == 0 && x->op == OP_LEAF) return NULL;
    if (op == OP_ATTENTION && (i == 1 || i == 2) &&
        (x->op == OP_LEAF || x->op == OP_SET_ROWS || x->op == OP_CAST || x->op == OP_SLICE)) {
      return NULL;
    }
    if (op == OP_SLICE && (x->op == OP_LEAF || x->op == OP_SET_ROWS || x->op == OP_CAST)) return NULL;
    if (op == OP_CAST && x->op == OP_LEAF) return NULL;
  }
  if (x->op == OP_LEAF && (op == OP_TRANSPOSE || (op == OP_GATHER && i == 0))) return NULL;
  return "a quantized or F16 weight goes only into the weight of a matmul, a transpose of the weight itself or the table of a gather; an F16 cache only into a set_rows (a leaf), the key or value of an attention, a slice that an attention reads, or a cast (a leaf)";
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
      op == OP_SCATTER_ADD || op == OP_CONV1D || op == OP_CONV2D || op == OP_ROPE || op == OP_BATCH_MATMUL) {
    if (nargs != 2) return tg_error(env, "a binary op takes 2 args");
  } else if (op == OP_WHERE || op == OP_SET_ROWS || op == OP_EXPERT_MATMUL) {
    if (nargs != 3) return tg_error(env, "where, set_rows and expert_matmul take 3 args");
  } else if (op == OP_ATTENTION) {
    if (nargs != 3 && nargs != 4) return tg_error(env, "attention takes q, k, v and an optional mask");
  } else if (op == OP_CONCAT) {
    if (nargs < 1) return tg_error(env, "concat takes at least 1 arg");
  } else if (op == OP_GATED_DELTA_RULE) {
    if (nargs != 6) return tg_error(env, "gated_delta_rule takes q, k, v, g, beta and the state");
  } else if (op == OP_ADAM_STEP) {
    if (nargs != 5) return tg_error(env, "adam_step takes p, grad, m, v and the hyper-parameters");
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
      /* ints[0]: the type the leaf is bound to, as `Types.held_code`: 0 F32, 1 F16, 2 Q8_0, 3 Q4_K, 4 Q6_K */
      if (nints > 0 && (ints[0] < 0 || ints[0] > 4)) return tg_error(env, "a leaf is F32, F16, Q8_0, Q4_K or Q6_K");
      type = nints > 0 ? held_type(ints[0]) : GGML_TYPE_F32;
      if (ggml_is_quantized(type) && (rank != 2 || n->ne[0] % ggml_blck_size(type) != 0)) {
        return tg_error(env, "a quantized leaf is a rank-2 weight whose rows hold whole blocks");
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
    case OP_MUL:
      t = swiglu(p, a, b);
      if (t == NULL) t = binary(p, ggml_mul, 1, a, b);
      break;
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
        /* One thread walks a pooled row: 14 ms for 51,865 floats on Vulkan. argmax reduces it in workgroups, and
           the value is read back with get_rows; which of several equal maxima it finds does not matter. */
        t = long_row_max(p, rows);
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
      for (i = 0; i < rank && ints[i] == 0; i++) {
      }
      if (i == rank) n->window = t;
      t = ggml_cont(p->ctx, t);
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
        rows = row_ids(p, c, a->ne[1], OP_SET_ROWS, &n->guard);
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
        t = ggml_set_rows(p->ctx, a->t, dense(p, b), row_ids(p, c, a->ne[1], OP_SET_ROWS, &n->guard));
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
      if (a->rank < 1 || a->rank > 2 || nints != 1 || ints[0] < 1 || ints[0] > a->ne[0]) {
        return tg_error(env, "top_k takes a vector or a matrix and a k from 1 to the length of a row");
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
    case OP_GATED_DELTA_RULE: {
      const struct gnode *g = &p->nodes[args[3]], *beta = &p->nodes[args[4]], *state = &p->nodes[args[5]];
      const int64_t width = a->ne[0], heads = a->ne[1], tokens = a->ne[2];
      int64_t ne_gate[4] = {1, heads, tokens, 1};
      struct ggml_tensor *operands[6];

      if (nints != 0 || a->rank != 3 || b->rank != 3 || c->rank != 3 || memcmp(b->ne, a->ne, sizeof a->ne) != 0 || memcmp(c->ne, a->ne, sizeof a->ne) != 0) {
        return tg_error(env, "gated_delta_rule takes q, k and v of one shape [tokens, heads, width]");
      }
      if (g->rank != 2 || beta->rank != 2 || g->ne[0] != heads || g->ne[1] != tokens || beta->ne[0] != heads || beta->ne[1] != tokens) {
        return tg_error(env, "gated_delta_rule takes g and beta of [tokens, heads]");
      }
      if (state->rank != 3 || state->ne[0] != width || state->ne[1] != width || state->ne[2] != heads) {
        return tg_error(env, "gated_delta_rule takes a state of [heads, width, width]");
      }
      if (rank != 2 || n->ne[0] != width * heads || n->ne[1] != tokens + width) {
        return tg_error(env, "gated_delta_rule answers [tokens + width, heads * width]");
      }
      operands[0] = dense(p, a);
      operands[1] = dense(p, b);
      operands[2] = dense(p, c);
      operands[3] = dense(p, g);
      operands[4] = dense(p, beta);
      operands[5] = dense(p, state);
      for (i = 0; i < 6; i++) {
        if (operands[i]->type != GGML_TYPE_F32) return tg_error(env, "gated_delta_rule takes F32 tensors");
      }
      /* ggml takes the gate and beta as [1, heads, tokens]: the scale of q is its own, 1 / sqrt(width) */
      t = ggml_gated_delta_net(p->ctx, operands[0], operands[1], operands[2], reshape(p, operands[3], ne_gate),
                               reshape(p, operands[4], ne_gate), operands[5], 1);
      break;
    }
    case OP_EXPERT_MATMUL: {
      struct ggml_tensor *weights, *inputs, *ids;

      if (nints != 0 || a->rank != 3 || b->rank != 3 || c->rank != 2 || b->ne[0] != a->ne[0] || c->ne[1] != b->ne[2] ||
          (b->ne[1] != c->ne[0] && b->ne[1] != 1)) {
        return tg_error(env, "expert_matmul takes weights [experts, rows, inner], x [tokens, uses or 1, inner] and ids [tokens, uses]");
      }
      if (rank != 3 || n->ne[0] != a->ne[1] || n->ne[1] != c->ne[0] || n->ne[2] != c->ne[1]) {
        return tg_error(env, "expert_matmul answers [tokens, uses, rows]");
      }
      weights = operand(p, a);
      inputs = operand(p, b);
      if (weights->type != GGML_TYPE_F32 || inputs->type != GGML_TYPE_F32) return tg_error(env, "expert_matmul takes F32 tensors");
      /* the ids are F32 in this layer; an id outside the experts is counted by the guard and refused by `run` */
      ids = ggml_reshape_2d(p->ctx, row_ids(p, c, a->ne[2], OP_EXPERT_MATMUL, &n->guard), c->ne[0], c->ne[1]);
      t = ggml_mul_mat_id(p->ctx, weights, inputs, ids);
      break;
    }
    case OP_ADAM_STEP: {
      struct gnode *g = &p->nodes[args[1]], *m = &p->nodes[args[2]], *v = &p->nodes[args[3]], *hyper = &p->nodes[args[4]];
      struct ggml_tensor *weight, *moments[2], *step;
      int axis = a->rank, k;

      if (nints != 1 || (ints[0] != 0 && ints[0] != 1) || rank != a->rank + 1 || rank > 4 || n->ne[axis] != 3) {
        return tg_error(env, "adam_step takes a tensor of rank 3 or less, and 1 to update its moments in place");
      }
      for (k = 0; k < axis; k++) {
        if (n->ne[k] != a->ne[k] || g->ne[k] != a->ne[k] || m->ne[k] != a->ne[k] || v->ne[k] != a->ne[k]) {
          return tg_error(env, "adam_step takes p, grad, m and v of one shape");
        }
      }
      if (g->rank != a->rank || m->rank != a->rank || v->rank != a->rank || volume(hyper->ne) != 7) {
        return tg_error(env, "adam_step takes p, grad, m and v of one shape, and 7 hyper-parameters");
      }
      if (dense(p, a)->type != GGML_TYPE_F32 || dense(p, g)->type != GGML_TYPE_F32 || dense(p, hyper)->type != GGML_TYPE_F32 ||
          (ints[0] == 0 && (dense(p, m)->type != GGML_TYPE_F32 || dense(p, v)->type != GGML_TYPE_F32))) {
        return tg_error(env, "adam_step takes F32 tensors");
      }
      /* The op updates its weight and moments where they lie. The weight is read by the rest of the graph (the loss
         and the gradients), and every argument may be a value other nodes or the caller read, so the weight is always
         a copy. The moments are copies too, unless the step updates them in place (`ints[0]`, which the lowering sets
         only for two inputs that nothing else reads, whose results are outputs): the caller's moments are then the
         new ones. Each of the three is a tensor of its own, so that the backend binds it at an aligned offset. */
      weight = ggml_cpy(p->ctx, dense(p, a), ggml_dup_tensor(p->ctx, dense(p, a)));
      /* ggml asserts that the weight is a parameter (a leaf marked for training); `ggml_set_param` refuses a tensor
         that is not a leaf, as a copy is. Only that assert reads the flag here: no backward pass is built. */
      weight->flags |= GGML_TENSOR_FLAG_PARAM;
      for (k = 0; k < 2; k++) {
        struct gnode *moment = k == 0 ? m : v;

        if (ints[0] == 0) {
          moments[k] = ggml_cpy(p->ctx, dense(p, moment), ggml_dup_tensor(p->ctx, dense(p, moment)));
        } else {
          struct port *leaf = NULL;
          int64_t l;

          if (moment->op != OP_LEAF || moment->t->type != GGML_TYPE_F32) return tg_error(env, "adam_step updates F32 inputs in place");
          for (l = 0; l < p->leaf_count; l++) {
            if (p->leaves[l].t == moment->t) leaf = &p->leaves[l];
          }
          if (leaf == NULL || leaf->donated || m == v) return tg_error(env, "adam_step updates two inputs, each in place once");
          leaf->donated = true;
          moments[k] = moment->t;
        }
      }
      /* the kernel binds its gradient and its numbers as buffers: a view at an offset (one value of the fill store,
         a slice) is not aligned for every backend, so such a tensor is copied first */
      step = ggml_opt_step_adamw(p->ctx, weight, aligned(p, dense(p, g)), moments[0], moments[1], aligned(p, dense(p, hyper)));
      /* The three results, each a view that depends on the step, so that whoever reads one runs after it. */
      n->adam[0] = step;
      for (k = 0; k < 2; k++) {
        n->adam[k + 1] = ggml_view_4d(p->ctx, moments[k], a->ne[0], a->ne[1], a->ne[2], a->ne[3], moments[k]->nb[1], moments[k]->nb[2],
                                      moments[k]->nb[3], 0);
        n->adam[k + 1]->src[1] = step;
      }
      n->adam_donates = (int)ints[0];
      /* the node itself is its weight; its recorded shape is the stack of the three, which only the trace holds */
      n->t = step;
      n->ne[axis] = 1;
      n->rank = a->rank;
      t = step;
      break;
    }
    case OP_ADAM_PART:
      if (a->op != OP_ADAM_STEP || nints != 1 || ints[0] < 0 || ints[0] > 2 || a->adam[0] == NULL) {
        return tg_error(env, "adam_part takes 0, 1 or 2 of an adam_step");
      }
      t = a->adam[ints[0]];
      /* the moments of a step that updates them in place: the output is the input itself, nothing to copy. The first
         moment is not checked (a non-finite one reaches the weight, which is); the second is, since an infinite one
         leaves the weight finite and frozen. */
      n->part = (int)ints[0];
      if (a->adam_donates && ints[0] > 0) n->in_place = 1;
      break;
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
    case OP_CONV2D:
      if (nints != 3 || ints[0] < 1 || ints[1] < 0 || ints[2] < 1) {
        return tg_error(env, "conv2d takes a stride >= 1, a padding >= 0 and groups >= 1");
      }
      if (a->rank != 4 || b->rank != 4 || b->ne[2] * ints[2] != a->ne[2] || b->ne[3] % ints[2] != 0) {
        return tg_error(env, "conv2d takes an input [B, C, H, W] and a weight [O, C / groups, KH, KW]");
      }
      /* ggml's direct kernel on the CPU works in 16 MiB (GGML_IM2COL_WORK_SIZE) and aborts when one patch of
         KW * KH * C_in floats and its C_out answers do not fit. Vulkan has no such limit. */
      if (!on_vulkan(p) && ((size_t)(b->ne[0] * b->ne[1] * b->ne[2] + b->ne[3] / ints[2]) * sizeof(float)) > 16u * 1024 * 1024) {
        return tg_error(env, "conv2d on the ggml CPU backend needs (KW * KH * C_in per group + C_out per group) * 4 bytes to fit in 16 MiB");
      }
      if (!room(p, 6 * (size_t)ints[2])) return tg_error(env, "conv2d: the groups lower to more ggml tensors than the graph has room for");
      t = conv2d(p, a, b, (int)ints[0], (int)ints[1], (int)ints[2]);
      break;
    case OP_MAX_POOL2D:
    case OP_AVG_POOL2D:
      if (nints != 3 || ints[0] < 1 || ints[1] < 1 || ints[2] < 0 || ints[2] > ints[0] / 2) {
        return tg_error(env, "a 2-D pool takes a kernel >= 1, a stride >= 1 and a padding of at most half the kernel");
      }
      if (a->rank != 4) return tg_error(env, "a 2-D pool takes an input [B, C, H, W]");
      t = ggml_pool_2d(p->ctx, dense(p, a), op == OP_MAX_POOL2D ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG, (int)ints[0],
                       (int)ints[0], (int)ints[1], (int)ints[1], (float)ints[2], (float)ints[2]);
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

/* The tensor that an output can be: the memory of the value itself, so that no copy is made. That is the value when it
   is a contiguous F32 result of an op, or what it is a whole view of (an `adam_step` result views its weight's copy).
   NULL when the value must be copied: a leaf, a view of something else, a half, a constant made on the way, or a
   tensor that is another output already. */
static struct ggml_tensor *own_output(const struct entry *p, const struct gnode *n, const struct ggml_tensor *value) {
  struct ggml_tensor *root = n->t->view_src != NULL ? n->t->view_src : n->t;

  if (n->op == OP_LEAF || value != n->t || n->lazy || root->view_src != NULL || root->type != GGML_TYPE_F32) return NULL;
  if (root->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_PARAM)) return NULL;
  if (in(root, p->fill_store) || in(root, p->checks)) return NULL;
  if (!ggml_is_contiguous(root) || !same_ne(root, n->ne)) return NULL;
  if (n->t != root && (n->t->view_offs != 0 || !ggml_are_same_shape(n->t, root) || !ggml_are_same_stride(n->t, root))) return NULL;
  return root;
}

/* Non-finite values of `value` as the sum of its rows: an inf or a NaN stays one, so the run refuses it. A finite
   output whose sum overflows F32 is refused too (the CPU adds in double and Vulkan in F32, so the edge differs). */
static struct ggml_tensor *finite_check(struct entry *p, struct ggml_tensor *value) {
  if (value->type != GGML_TYPE_F32) value = ggml_cast(p->ctx, value, GGML_TYPE_F32);
  return ggml_sum(p->ctx, ggml_sum_rows(p->ctx, value));
}

an_term ggml_compile(an_env *env, struct ggml_Lease_state *lease, ggml_list_Int outputs) {
  struct ggml_tensor *finite[MAX_OUTPUTS]; /* the finiteness check of each output; its copy to `checks` is made after all of them */
  struct entry *p = lease->e;
  int64_t outs[MAX_OUTPUTS];
  struct ggml_tensor **inner, **views;
  size_t slots;
  int nouts, r, i, j;
  char text[256];

  if (p == NULL || p->compiled || p->graph != NULL) return tg_error(env, "the graph is compiled already, or its compile failed");
  if ((r = read_ints(env, &outputs, outs, MAX_OUTPUTS, &nouts)) != 0) return bad_list(env, r, "at most 512 outputs");
  if (nouts < 1) return tg_error(env, "a graph has at least one output");
  for (i = 0; i < nouts; i++) {
    if (outs[i] < 0 || outs[i] >= p->count) return tg_error(env, "an output is not a node");
  }

  for (i = 0; i < p->count; i++) {
    if (p->nodes[i].guard != NULL) p->guard_count++;
  }
  if (!room(p, (size_t)(nouts * 8 + p->guard_count + 1))) return tg_error(env, "the graph ran out of room for its outputs");
  memset(finite, 0, sizeof finite);
  p->graph = ggml_new_graph_custom(p->ctx, graph_size(p->capacity), false);
  p->checks = ggml_new_tensor_1d(p->fill_ctx, GGML_TYPE_F32, nouts + p->guard_count);
  /* the positions of the ropes go first: a cast between a rope's norm and the rope keeps ggml-vulkan from fusing them */
  for (i = 0; i < p->count; i++) {
    if (p->nodes[i].at != NULL) ggml_build_forward_expand(p->graph, p->nodes[i].at);
  }
  for (i = 0; i < nouts; i++) {
    struct gnode *n = &p->nodes[outs[i]];
    struct ggml_tensor *value;
    struct ggml_tensor *own = NULL;

    if (!n->in_place && !n->lazy) own = own_output(p, n, dense(p, n));
    p->outs[i].writer = own;
    p->outs[i].t = own != NULL ? own
                               : ggml_new_tensor_4d(p->port_ctx, n->t->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                                    n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    if (n->in_place) {
      /* the output is the leaf itself: nothing to copy, the node only has to run */
      p->outs[i].donated = true;
      ggml_build_forward_expand(p->graph, n->t);
      if (n->op == OP_ADAM_PART && n->part == 2) {
        finite[i] = finite_check(p, n->t);
        ggml_build_forward_expand(p->graph, finite[i]);
      }
      continue;
    }
    value = dense(p, n);
    ggml_set_output(p->outs[i].t);
    if (own == NULL) {
      p->outs[i].writer = ggml_cpy(p->ctx, value, p->outs[i].t);
      ggml_build_forward_expand(p->graph, p->outs[i].writer);
    }
    /* sum_rows first, since one ggml_sum of a large tensor is a single slow workgroup on Vulkan; the check reads
       the value, so it runs after the node that makes it */
    finite[i] = finite_check(p, value);
    ggml_build_forward_expand(p->graph, finite[i]);
  }
  p->output_count = nouts;
  /* The sums are made where each output is (so each ends its value's life there), and their 1-float copies into
     `checks` (device memory) come after all of them: on a graph that runs on two backends, a copy between two layers on
     the CPU would be a split of the scheduler of its own. */
  for (i = 0; i < nouts; i++) {
    if (finite[i] != NULL) {
      ggml_build_forward_expand(p->graph, ggml_cpy(p->ctx, finite[i], ggml_view_1d(p->fill_ctx, p->checks, 1, i * sizeof(float))));
    }
  }
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

  /* Every node and source that is neither a port nor in the fill buffer: a view of a port follows its port, which
     every run binds (`views`); the rest is what the shared allocator places (`inner`). */
  slots = (size_t)ggml_graph_n_nodes(p->graph) * (GGML_MAX_SRC + 1);
  inner = (struct ggml_tensor **)calloc(slots, sizeof(struct ggml_tensor *));
  views = (struct ggml_tensor **)calloc(slots, sizeof(struct ggml_tensor *));
  if (inner == NULL || views == NULL) {
    free(inner);
    free(views);
    return tg_error(env, "out of memory");
  }
  p->inner = inner;
  p->views = views;
  for (i = 0; i < ggml_graph_n_nodes(p->graph); i++) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);

    for (j = -1; j < GGML_MAX_SRC; j++) {
      struct ggml_tensor *t = j < 0 ? node : node->src[j];

      if (t == NULL || (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) || in(t, p->fill_store) ||
          in(t, p->checks)) {
        continue;
      }
      if (t->view_src != NULL && (t->view_src->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT))) {
        p->views[p->view_count++] = t;
      } else {
        p->inner[p->inner_count++] = t;
      }
    }
  }

  p->sched_need = 0;
  for (i = 0; i < ggml_graph_n_nodes(p->graph); i++) {
    for (j = 0; j < GGML_MAX_SRC; j++) {
      const struct ggml_tensor *src = ggml_graph_node(p->graph, i)->src[j];

      if (src != NULL && src->op == GGML_OP_NONE) p->sched_need++;
    }
  }
  p->sched_need = 2 * (size_t)ggml_graph_n_nodes(p->graph) + 2 * p->sched_need + 2 * MAX_OUTPUTS + 64;

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

/* Layers across a device and the CPU. A leaf in host memory under a graph of a Vulkan device (a weight or a KV cache
   that the loader put on the CPU) makes the graph "mixed": `place` gives every node a side, 0 the device and 1 the CPU,
   from where its operands are, and `compute_mixed` runs the graph through a scheduler over the two backends, which
   copies the activations where the sides meet. A graph whose leaves are all in device memory is never touched by any of
   this. */
#define DEVICE_SIDE 0
#define CPU_SIDE 1

static void forget(struct entry *e);

static bool is_view_op(enum ggml_op op) {
  return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static int buffer_side(const struct ggml_backend_buffer *b) { return ggml_backend_buffer_is_host((ggml_backend_buffer_t)b) ? CPU_SIDE : DEVICE_SIDE; }

static size_t side_slot(const struct entry *p, const struct ggml_tensor *t) {
  return ((size_t)((uintptr_t)t >> 4) * 11400714819323198485ull >> 20) & (p->side_cap - 1);
}

static void side_put(struct entry *p, const struct ggml_tensor *t, int side) {
  size_t i = side_slot(p, t);

  while (p->side_key[i] != NULL && p->side_key[i] != t) i = (i + 1) & (p->side_cap - 1);
  p->side_key[i] = (struct ggml_tensor *)t;
  p->side_val[i] = (signed char)side;
}

/* The value kept for `t`: 0 or 1 is its side, 2 and 3 are the sides that its consumers ask for (device, CPU) and 4 that
   they ask for both, while `place` goes through the graph from the end; -1 is nothing. */
static int side_raw(const struct entry *p, const struct ggml_tensor *t) {
  size_t i = side_slot(p, t);

  while (p->side_key[i] != NULL) {
    if (p->side_key[i] == t) return p->side_val[i];
    i = (i + 1) & (p->side_cap - 1);
  }
  return -1;
}

static int side_get(const struct entry *p, const struct ggml_tensor *t) {
  int raw = side_raw(p, t);

  return raw < 2 ? raw : -1;
}

/* The side of a tensor that an earlier node or a leaf has: its own memory, else what `place` decided for it or for
   the tensor it is a view of; -1 when nothing says. */
static int side_of(const struct entry *p, const struct ggml_tensor *t) {
  const struct ggml_tensor *root = t->view_src != NULL ? t->view_src : t;
  int side;

  if (t->buffer != NULL) return buffer_side(t->buffer);
  if (root->buffer != NULL) return buffer_side(root->buffer);
  side = side_get(p, t);
  return side >= 0 ? side : side_get(p, root);
}

/* A device leaf below this many bytes (a norm weight, a position, a scalar, a mask) does not pull a node to the device by
   its bytes; a matrix that a product or a row lookup reads (a weight, however small) pins the node there. */
#define SMALL_LEAF 65536

/* The side of `node`, or -1 when nothing about its operands says (`place` then follows its consumers): the side of the
   memory it writes when that is made (a view of a cache); the CPU when it reads a tensor that lives there (a weight, a
   cache), whatever else it reads (a long prompt's activations are bigger than a weight, and copying a weight on every
   run is what a layer on the CPU is there to avoid); else the device when a product or a row lookup reads a matrix that lives there (a weight);
   else the side with most operand bytes, a tie going to the side of the product of a matrix (the layer's output: the
   residual stream follows the layer it passes through, whatever the order of the operands) and else to the CPU when an
   operand was computed there. */
static int decide(const struct entry *p, const struct ggml_tensor *node) {
  int64_t bytes[2] = {0, 0};
  int j, side, product = -1;
  bool host = false, weight = false, on_cpu = false, known = false, mixed_products = false;

  if (node->buffer != NULL) return buffer_side(node->buffer);
  if (node->view_src != NULL && node->view_src->buffer != NULL) return buffer_side(node->view_src->buffer);
  for (j = 0; j < GGML_MAX_SRC; j++) {
    const struct ggml_tensor *src = node->src[j];
    const struct ggml_tensor *root;

    if (src == NULL) continue;
    root = src->view_src != NULL ? src->view_src : src;
    if (root->buffer != NULL) {
      side = buffer_side(root->buffer);
      /* a view is as large as the tensor it reads from: a row of a large table is a large leaf */
      if (side == CPU_SIDE) host = true;
      else if (j == 0 && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_GET_ROWS) &&
               (root->ne[0] > 1 && root->ne[1] > 1)) weight = true;
      else if (ggml_nbytes(root) >= SMALL_LEAF) bytes[DEVICE_SIDE] += (int64_t)ggml_nbytes(src);
      continue;
    }
    side = side_of(p, src);
    if (side < 0) continue;
    bytes[side] += (int64_t)ggml_nbytes(src);
    if (side == CPU_SIDE) on_cpu = true;
    known = true;
    if (root->op == GGML_OP_MUL_MAT || root->op == GGML_OP_MUL_MAT_ID) {
      if (product >= 0 && product != side) mixed_products = true;
      product = side;
    }
  }
  if (host) return CPU_SIDE;
  if (weight) return DEVICE_SIDE;
  if (bytes[CPU_SIDE] != bytes[DEVICE_SIDE]) return bytes[CPU_SIDE] > bytes[DEVICE_SIDE] ? CPU_SIDE : DEVICE_SIDE;
  if (!known) return -1;
  if (product >= 0 && !mixed_products) return product;
  return on_cpu ? CPU_SIDE : DEVICE_SIDE;
}

/* Gives `node` the side `want`, or the other when its backend does not run the op. Returns false when neither does. */
static bool put_side(struct entry *p, struct ggml_tensor *node, int want, ggml_backend_t cpu) {
  if (!ggml_backend_supports_op(want == CPU_SIDE ? cpu : p->backend, node)) {
    want = 1 - want;
    if (!ggml_backend_supports_op(want == CPU_SIDE ? cpu : p->backend, node)) return false;
  }
  side_put(p, node, want);
  return true;
}

/* The arrays of a mixed graph, made by the first `place` that finds a leaf in host memory (a graph that has none never
   pays for them): one block, whose first part is pointers (`save_node`, `save_src`, `side_key`) and the rest bytes
   (`leaf_side`, `side_val`), freed with the entry. */
static bool make_side_tables(struct entry *p) {
  size_t nodes = (size_t)ggml_graph_n_nodes(p->graph), side_cap = 16, pointers;

  while (side_cap < nodes * 2 + 16) side_cap *= 2;
  pointers = nodes + nodes * GGML_MAX_SRC + side_cap;
  p->save_node = (struct ggml_tensor **)calloc(1, pointers * sizeof(struct ggml_tensor *) + (size_t)p->leaf_count + side_cap);
  if (p->save_node == NULL) return false;
  p->save_src = p->save_node + nodes;
  p->side_key = p->save_node + nodes + nodes * GGML_MAX_SRC;
  p->leaf_side = (signed char *)(p->save_node + pointers);
  p->side_val = p->leaf_side + p->leaf_count;
  p->side_cap = side_cap;
  return true;
}

static void save_sources(struct entry *p) {
  int64_t i, j, n = ggml_graph_n_nodes(p->graph);

  for (i = 0; i < n; i++) {
    p->save_node[i] = ggml_graph_node(p->graph, i);
    for (j = 0; j < GGML_MAX_SRC; j++) p->save_src[i * GGML_MAX_SRC + j] = p->save_node[i]->src[j];
  }
}

/* Sets `mixed` and, for a mixed graph, the side of every node that is not a view (in `side_key`), from the leaves bound
   now. The sides are kept until a leaf changes sides. Call under the device lock. Returns NULL, or why it could not. */
static const char *place(struct entry *p) {
  ggml_backend_t cpu;
  char text[160];
  int64_t i, n = ggml_graph_n_nodes(p->graph);
  bool changed = false, any = false;

  if (p->device < 0) return NULL;
  for (i = 0; i < p->leaf_count && !any; i++) {
    any = p->leaves[i].t->buffer != NULL && buffer_side(p->leaves[i].t->buffer) == CPU_SIDE;
  }
  if (!any) {
    p->placed = false;
    p->mixed = false;
    return NULL;
  }
  if (p->leaf_side == NULL) {
    if (!make_side_tables(p)) return "out of memory";
    save_sources(p);
  }
  for (i = 0; i < p->leaf_count; i++) {
    signed char side = p->leaves[i].t->buffer != NULL ? (signed char)buffer_side(p->leaves[i].t->buffer) : DEVICE_SIDE;

    if (side != p->leaf_side[i]) changed = true;
    p->leaf_side[i] = side;
  }
  if (!changed && p->placed) return NULL;
  p->placed = false;
  p->mixed = false;
  if ((cpu = tg_backend(-1, text, sizeof text)) == NULL) return "the CPU backend did not open";
  /* the intermediates of an earlier plain run must not look like memory a node writes */
  if (devices[p->device + 1].planned == p) devices[p->device + 1].planned = NULL;
  forget(p);
  memset(p->side_key, 0, p->side_cap * sizeof(struct ggml_tensor *));
  for (i = 0; i < n; i++) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);
    int side;

    if (is_view_op(node->op)) continue;
    side = decide(p, node);
    if (side >= 0 && !put_side(p, node, side, cpu)) return "no backend runs a node of the graph";
  }
  /* A node that nothing decided (a table made from constants) goes where its consumers are, so that it does not make
     a trip of its own to the other backend: the device when they are on both sides, and when it has none. The graph is
     gone through from its end, so a node comes after all its consumers. */
  for (i = n - 1; i >= 0; i--) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);
    int side, raw, j;

    if (is_view_op(node->op)) continue;
    raw = side_raw(p, node);
    if (raw < 0 || raw >= 2) {
      if (!put_side(p, node, raw == 3 ? CPU_SIDE : DEVICE_SIDE, cpu)) return "no backend runs a node of the graph";
    }
    side = side_get(p, node);
    for (j = 0; j < GGML_MAX_SRC; j++) {
      struct ggml_tensor *src = node->src[j];

      if (src == NULL) continue;
      if (src->view_src != NULL) src = src->view_src;
      if (src->buffer != NULL || src->op == GGML_OP_NONE || is_view_op(src->op)) continue;
      raw = side_raw(p, src);
      if (raw == 0 || raw == 1) continue;
      side_put(p, src, raw < 0 || raw == 2 + side ? 2 + side : 4);
    }
  }
  p->placed = true;
  p->mixed = true;
  return NULL;
}

/* Points `port` at the memory of `h` when it has the port's type, or notes `h` for a converting copy through the
   port's own memory (`own_memory`), and holds `h`'s memory until `release`. It touches no device, so the holder of the
   lease calls it without the device lock. */
static void attach(struct entry *p, struct port *port, struct ggml_TensorHandle_state *h) {
  if (port->kept) {
    /* what is kept is no longer the whole set of its pin: it ends with this lease */
    pthread_mutex_lock(&cache_lock);
    p->kept_for = 0;
    pthread_mutex_unlock(&cache_lock);
  }
  if (h->store != NULL) tg_storage_retain(h->store);
  unbind(p, port);
  port->held = h->store;
  if (h->t->type == port->t->type) {
    port->t->data = h->t->data;
    port->t->buffer = h->t->buffer;
  } else {
    port->conv = h->t;
  }
  port->set = true;
}

/* Points a port that converts at its own memory, made once. Call under the device lock. */
static bool own_memory(struct entry *p, struct port *port) {
  if (port->own == NULL) {
    port->own = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(p->backend), ggml_nbytes(port->t));
    if (port->own == NULL) return false;
  }
  port->t->data = ggml_backend_buffer_get_base(port->own);
  port->t->buffer = port->own;
  return true;
}

an_term ggml_bind(an_env *env, struct ggml_Lease_state *lease, int64_t slot, struct ggml_TensorHandle_state *t) {
  struct entry *p = lease->e;
  struct port *leaf;

  if (p == NULL || !p->compiled) return tg_error(env, "bind a compiled graph");
  if (slot < 0 || slot >= p->leaf_count) return tg_error(env, "no such leaf");
  if (t->t == NULL) return tg_error(env, "bind a tensor that was made");
  leaf = &p->leaves[slot];
  if (ggml_nelements(t->t) != ggml_nelements(leaf->t)) return tg_error(env, "a bound tensor has another element count than its leaf");
  if (!ggml_backend_supports_buft(p->backend, ggml_backend_buffer_get_type(t->buffer)) &&
      !(p->device >= 0 && !t->vulkan && ggml_backend_buffer_is_host(t->buffer))) {
    return tg_error(env, "a bound tensor lives on another device than its graph");
  }
  if (leaf->t->type != t->t->type && (leaf->t->type != GGML_TYPE_F32 || t->t->type != GGML_TYPE_I32)) {
    return tg_error(env, "an F16 or quantized tensor is not converted to or from another type");
  }
  attach(p, leaf, t);
  return ggml_result_Void_ok(env, an_void(env));
}

an_term ggml_output(an_env *env, struct ggml_Lease_state *lease, int64_t output, bool int32) {
  struct entry *p = lease->e;
  const char *why;
  an_term term, failure;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  bool attached;
  int64_t device = p != NULL ? p->device : 0;

  if (p == NULL || !p->compiled) return tg_error(env, "an output of a compiled graph");
  if (output < 0 || output >= p->output_count) return tg_error(env, "no such output");
  tg_lock_device();
  why = place(p);
  if (why == NULL && p->mixed && p->outs[output].writer != NULL && side_get(p, p->outs[output].writer) == CPU_SIDE) {
    device = -1;
  }
  tg_unlock_device();
  if (why != NULL) return tg_error(env, why);
  if (!tg_new_tensor(env, device, p->outs[output].t->ne, int32 ? GGML_TYPE_I32 : p->outs[output].t->type, state, &failure)) {
    return failure;
  }
  tg_lock_device();
  attach(p, &p->outs[output], state);
  attached = p->outs[output].conv == NULL || own_memory(p, &p->outs[output]);
  tg_unlock_device();
  if (!attached) return tg_error(env, "out of memory");
  return ggml_result_TensorHandle_ok(env, term);
}

an_term ggml_output_into(an_env *env, struct ggml_Lease_state *lease, int64_t output, struct ggml_TensorHandle_state *t) {
  struct entry *p = lease->e;
  bool attached = false, alias = false;
  const char *why;
  char text[96];
  int64_t i;

  if (p == NULL || !p->compiled) return tg_error(env, "an output of a compiled graph");
  if (output < 0 || output >= p->output_count) return tg_error(env, "no such output");
  if (t->t == NULL) return tg_error(env, "an output tensor that was made");
  if (ggml_nelements(t->t) != ggml_nelements(p->outs[output].t)) {
    return tg_error(env, "an output tensor has another element count than its output");
  }
  tg_lock_device();
  why = place(p);
  if (why == NULL && ggml_backend_buffer_is_host(t->buffer) && !t->vulkan && p->device >= 0) {
    /* a tensor on the CPU is the output of a node that runs there */
    if (!p->mixed || p->outs[output].writer == NULL || side_get(p, p->outs[output].writer) != CPU_SIDE) {
      why = "an output tensor lives on another device than its graph";
    }
  } else if (why == NULL && !ggml_backend_supports_buft(p->backend, ggml_backend_buffer_get_type(t->buffer))) {
    why = "an output tensor lives on another device than its graph";
  } else if (why == NULL && p->mixed && p->outs[output].writer != NULL && side_get(p, p->outs[output].writer) == CPU_SIDE) {
    why = "an output tensor lives on another device than the layer that writes it";
  }
  if (why != NULL) {
    tg_unlock_device();
    return tg_error(env, why);
  }
  for (i = 0; i < p->leaf_count; i++) {
    if (p->leaves[i].held == t->store) alias = true;
  }
  for (i = 0; i < p->output_count; i++) {
    if (i != output && p->outs[i].held == t->store) alias = true;
  }
  if (!alias) {
    /* the run writes `t`: what `upload` kept of it on the host is not its value any more */
    if (t->store != NULL) {
      free(t->store->mirror);
      t->store->mirror = NULL;
    }
    attach(p, &p->outs[output], t);
    attached = p->outs[output].conv == NULL || own_memory(p, &p->outs[output]);
  }
  tg_unlock_device();
  if (alias) {
    snprintf(text, sizeof text, "output %" PRId64 ": the tensor given is an input of the run or another output", output);
    return tg_error(env, text);
  }
  if (!attached) return tg_error(env, "out of memory");
  return ggml_result_Void_ok(env, an_void(env));
}

an_term ggml_mixed_runs(an_env *env) {
  return an_int(env, __atomic_load_n(&mixed_run_count, __ATOMIC_RELAXED));
}

an_term ggml_mixed_splits(an_env *env) { return an_int(env, __atomic_load_n(&mixed_split_count, __ATOMIC_RELAXED)); }

an_term ggml_pin(an_env *env, struct ggml_Lease_state *lease) {
  struct entry *p = lease->e;
  an_term term;
  struct ggml_Pin_state *pin;
  struct pin_id *live;
  bool cached = false;
  int i;

  if (p == NULL || !p->compiled) return tg_error(env, "pin a compiled graph");
  live = (struct pin_id *)calloc(1, sizeof *live);
  if (live == NULL) return tg_error(env, "out of memory");
  pin = ggml_Pin_new(env, &term);
  pthread_mutex_lock(&cache_lock);
  for (i = 0; i < MAX_ENTRIES; i++) {
    if (cache[i] == p) cached = true;
  }
  if (cached) {
    p->pins++;
    pin->id = live->id = ++pin_count;
    live->next = p->pinned;
    p->pinned = live;
  }
  pthread_mutex_unlock(&cache_lock);
  if (!cached) free(live);
  pin->e = cached ? p : NULL;
  return ggml_result_Pin_ok(env, term);
}

an_term ggml_on_cpu(an_env *env, struct ggml_TensorHandle_state *t) { return an_bool(env, !t->vulkan); }

/* A pin that holds nothing has id 0, which keeps nothing. */
an_term ggml_pin_id(an_env *env, struct ggml_Pin_state *pin) { return an_int(env, pin->id); }

an_term ggml_resume(an_env *env, struct ggml_Lease_state *lease, int64_t pin) {
  struct entry *p = lease->e;
  bool kept;

  if (p == NULL || !p->compiled) return an_bool(env, false);
  pthread_mutex_lock(&cache_lock);
  kept = pin != 0 && p->kept_leaves > 0 && p->kept_for == pin && !kept_closed(p);
  if (!kept) p->kept_for = 0;
  pthread_mutex_unlock(&cache_lock);
  if (!kept) drop_kept(p); /* the lease's graph: its ports are the holder's */
  lease->resumed = kept;
  return an_bool(env, kept);
}

an_term ggml_keep(an_env *env, struct ggml_Lease_state *lease, int64_t pin, ggml_list_Int slots) {
  struct entry *p = lease->e;
  const struct pin_id *live;
  int64_t i, slot;
  bool whole = true, fresh;
  int next;

  if (p == NULL || !p->compiled) return an_void(env);
  while ((next = ggml_list_Int_next(env, &slots, &slot)) == 1) {
    struct port *leaf = slot >= 0 && slot < p->leaf_count ? &p->leaves[slot] : NULL;

    if (leaf == NULL || !leaf->set || leaf->donated || leaf->conv != NULL || leaf->held == NULL) whole = false;
    else leaf->wanted = true;
  }
  pthread_mutex_lock(&cache_lock);
  for (live = p->pinned; live != NULL && live->id != pin; live = live->next) {
  }
  whole = whole && next == 0 && live != NULL;
  if (!whole || p->kept_for != pin) drop_kept(p); /* another program's, or a set that would not be the list */
  p->kept_for = whole ? pin : 0;
  fresh = whole && p->kept_leaves == 0; /* a set this pin has already stays as it is */
  for (i = 0; i < p->leaf_count; i++) {
    struct port *leaf = &p->leaves[i];

    if (fresh && leaf->wanted) {
      leaf->kept = true;
      __atomic_fetch_add(&leaf->held->kept, 1, __ATOMIC_SEQ_CST);
      p->kept_leaves++;
    }
    leaf->wanted = false;
  }
  pthread_mutex_unlock(&cache_lock);
  if (whole) lease->resumed = true;
  return an_void(env);
}

an_term ggml_kept(an_env *env, struct ggml_Lease_state *lease) {
  return an_int(env, lease->e != NULL ? lease->e->kept_leaves : 0);
}

/* A pin holds its graph in the cache; a graph that has no slot (every one is in use or pinned) gets a pin that holds
   nothing, and runs as an ordinary graph. The free hook below ends a pin: it never waits for the device lock, and the
   graph cannot have been freed, since a pinned one is never a victim and is dropped only when it never compiled. */
CC_SHORT void ggml_Pin_free(void *state) {
  struct ggml_Pin_state *self = (struct ggml_Pin_state *)state;

  struct pin_id **link, *live = NULL;

  if (self->e == NULL) return;
  pthread_mutex_lock(&cache_lock);
  self->e->pins--;
  for (link = &self->e->pinned; *link != NULL; link = &(*link)->next) {
    if ((*link)->id == self->id) {
      live = *link;
      *link = live->next;
      break;
    }
  }
  if (self->e->kept_for == self->id) {
    self->e->kept_for = 0;
    if (!self->e->in_use) drop_kept(self->e); /* else its holder does, at the end of the lease */
  }
  pthread_mutex_unlock(&cache_lock);
  free(live);
  self->e = NULL;
}

/* The intermediates of `e` point at nothing: the allocator's plan is not this graph's any more, or is made again. */
static void forget(struct entry *e) {
  int64_t i;

  for (i = 0; i < e->inner_count; i++) {
    e->inner[i]->data = NULL;
    e->inner[i]->buffer = NULL;
  }
}

static void set_backends(struct entry *p, ggml_backend_sched_t sched, ggml_backend_t backends[2]) {
  int64_t i, n = ggml_graph_n_nodes(p->graph);

  for (i = 0; i < n; i++) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);
    int side = side_get(p, node);

    if (side >= 0) ggml_backend_sched_set_tensor_backend(sched, node, backends[side]);
  }
}

static void restore_sources(struct entry *p) {
  struct ggml_tensor **nodes = ggml_graph_nodes(p->graph);
  int64_t i, j, n = ggml_graph_n_nodes(p->graph);

  for (i = 0; i < n; i++) {
    nodes[i] = p->save_node[i];
    for (j = 0; j < GGML_MAX_SRC; j++) p->save_node[i]->src[j] = p->save_src[i * GGML_MAX_SRC + j];
  }
}

/* Runs a mixed graph (`place`) through the scheduler of its device: every node goes to the backend that `place`
   chose, the scheduler splits the graph where the sides meet and copies what crosses, and the intermediates of both
   sides are its own. It rewrites the sources of the nodes it copies for, so they are put back at the end. Call under
   the device lock, with the leaves bound. Returns NULL, or why it failed. */
static const char *compute_mixed(struct entry *p, struct device_state *d) {
  ggml_backend_t backends[2];
  const char *why = NULL;
  char text[160];
  int64_t i;
  size_t need = p->sched_need;

  __atomic_fetch_add(&mixed_run_count, 1, __ATOMIC_RELAXED);
  backends[0] = p->backend;
  if ((backends[1] = tg_backend(-1, text, sizeof text)) == NULL) return "the CPU backend did not open";
  for (i = 0; i < p->output_count; i++) {
    int side;

    if (p->outs[i].donated || p->outs[i].writer == NULL || p->outs[i].t->buffer == NULL) continue;
    side = side_get(p, p->outs[i].writer);
    if (side >= 0 && side != buffer_side(p->outs[i].t->buffer)) return "an output lives on the other side than the layer that writes it";
  }
  if (d->sched != NULL && d->sched_size < need) {
    ggml_backend_sched_free(d->sched);
    d->sched = NULL;
    d->sched_planned = NULL;
  }
  if (d->sched == NULL) {
    d->sched_size = need < 4096 ? 4096 : need;
    d->sched = ggml_backend_sched_new(backends, NULL, 2, d->sched_size, false, false);
    if (d->sched == NULL) return "making the scheduler failed";
  }
  if (d->planned == p) d->planned = NULL;
  forget(p);
  ggml_backend_sched_reset(d->sched);
  set_backends(p, d->sched, backends);
  if (d->sched_planned != p) {
    /* one scheduler serves every mixed graph of the device: its memory is planned for this graph, or it would reuse
       the offsets of another one (reserve resets the backends it was given, so they are set again) */
    if (!ggml_backend_sched_reserve(d->sched, p->graph)) why = "reserving the scheduler failed";
    restore_sources(p);
    forget(p);
    ggml_backend_sched_reset(d->sched);
    set_backends(p, d->sched, backends);
    d->sched_planned = why == NULL ? p : NULL;
  }
  if (why == NULL && !ggml_backend_sched_alloc_graph(d->sched, p->graph)) why = "allocating the graph failed";
  if (why == NULL && ggml_backend_sched_graph_compute(d->sched, p->graph) != GGML_STATUS_SUCCESS) why = "ggml_backend_sched_graph_compute failed";
  if (why == NULL) __atomic_store_n(&mixed_split_count, ggml_backend_sched_get_n_splits(d->sched), __ATOMIC_RELAXED);
  restore_sources(p);
  forget(p);
  ggml_backend_sched_reset(d->sched);
  return why;
}

/* Runs a graph whose leaves are all in the memory of its device, on its backend alone, with the allocator that every
   graph of the device shares. Call under the device lock, with the leaves bound. Returns NULL, or why it failed. */
static const char *compute_plain(struct entry *p, struct device_state *d) {
  const char *why = NULL;
  int64_t i;

  if (d->galloc == NULL) d->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p->backend));
  if (d->galloc == NULL) return "making the graph allocator failed";
  if (d->planned != p) {
    /* The allocator is planned for one graph at a time, and a new plan may move its buffer: the graph that had the
       plan forgets where its intermediates were, and this one is placed again. */
    if (d->planned != NULL) forget(d->planned);
    d->planned = NULL;
    forget(p);
    if (!ggml_gallocr_reserve(d->galloc, p->graph) || !ggml_gallocr_alloc_graph(d->galloc, p->graph)) {
      why = "allocating the graph failed";
    } else {
      d->planned = p;
    }
  } else {
    /* The same graph ran last and nothing planned the allocator since: its intermediates are where that run left
       them. Only the views of the leaves and outputs follow what this run bound. */
    for (i = 0; i < p->view_count; i++) {
      if (p->views[i]->buffer == NULL) ggml_backend_view_init(p->views[i]);
    }
  }
  if (why == NULL && ggml_backend_graph_compute(p->backend, p->graph) != GGML_STATUS_SUCCESS) {
    why = "ggml_backend_graph_compute failed";
  }
  return why;
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
  if (p->kept_leaves > 0 && !lease->resumed) {
    pthread_mutex_lock(&cache_lock);
    p->kept_for = 0;
    pthread_mutex_unlock(&cache_lock);
    drop_kept(p);
  }
  for (i = 0; i < p->leaf_count; i++) {
    if (!p->leaves[i].set) return tg_error(env, "run with every leaf bound");
  }
  for (i = 0; i < p->output_count; i++) {
    if (!p->outs[i].set && !p->outs[i].donated) return tg_error(env, "run with every output made");
  }
  for (i = 0; i < p->leaf_count; i++) {
    int64_t k;

    if (!p->leaves[i].donated) continue;
    if (p->leaves[i].conv != NULL) return tg_error(env, "an input written in place (a set_rows cache, an adam_step moment) must be the tensor the graph was lowered with");
    for (k = 0; k < p->leaf_count; k++) {
      if (k != i && p->leaves[k].held != NULL && p->leaves[k].held == p->leaves[i].held) {
        return tg_error(env, "an input written in place is bound to another input too");
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
  for (i = 0; why == NULL && i < p->leaf_count; i++) {
    if (p->leaves[i].conv == NULL) continue;
    if (!own_memory(p, &p->leaves[i])) why = "out of memory";
    else why = tg_copy(p->leaves[i].conv, p->leaves[i].t, p->leaves[i].held != NULL ? p->leaves[i].held->mirror : NULL);
  }
  d = &devices[p->device + 1];
  if (why == NULL) why = place(p);
  if (why == NULL) why = p->mixed ? compute_mixed(p, d) : compute_plain(p, d);
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
      snprintf(text, sizeof text, "output %" PRId64 " of the graph holds a value that is not finite, or its sum overflows F32", i);
      why = text;
    } else if (p->outs[i].conv != NULL) {
      why = tg_copy(p->outs[i].t, p->outs[i].conv, NULL);
    }
  }
  for (i = 0; i < p->view_count; i++) {
    p->views[i]->data = NULL;
    p->views[i]->buffer = NULL;
  }
  release(p);
  tg_unlock_device();
  free(values);
  if (why != NULL) return tg_error(env, why);
  return ggml_result_Void_ok(env, an_void(env));
}
