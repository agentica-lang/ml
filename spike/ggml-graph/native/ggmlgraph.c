/* The thin C layer of `ggml_graph.agc`: it turns the nodes `lower.agc` sends
   into one ggml cgraph, allocates it once, and runs it once per step.

   Every value is F32 here. A `Bool` mask is 0/1 and an integer tensor (the
   labels, `arange`) holds exact integers below 2^24, which is all an MLP
   classifier's step needs.

   Layout: a row-major shape [d0, ..., dk] is a ggml tensor with ne[0] = dk,
   ne[1] = dk-1, ..., so trailing-axis broadcasting (numpy's, and §3.1's
   `broadcast_to`) is ggml's own leading-ne repeat.

   A node may be "lazy": its ggml tensor holds fewer elements than its recorded
   shape and repeats to it (a fill, a `broadcast_to`). Binary ops take a lazy
   right operand as ggml's own broadcast; anything else materializes it with
   `ggml_repeat_4d`. This keeps a `full` of [4096, 512] zeros from being 8 MiB
   of zeros, and a bias `broadcast_to` from being a copy.

   This file is compiled as C or as C++ (the torch module of `tensor/` sets
   CC to g++), so it casts every `malloc`. */
#define GGMLGRAPH_NATIVE_IMPLEMENTATION
#include "ggmlgraph_native.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <inttypes.h>
#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The op codes `lower.agc` sends; keep both lists in one order. */
enum {
  OP_INPUT = 0, /* a leaf uploaded per step */
  OP_STATE,     /* a leaf kept across steps (a parameter, a moment) */
  OP_FIXED,     /* a leaf set once (a lifted constant) */
  OP_FULL,      /* a fill with `value`: one element, lazy */
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
  OP_COUNT
};

static const char *op_names[OP_COUNT] = {
  "input", "state", "fixed", "full", "add", "sub", "mul", "div", "neg", "exp", "log", "relu", "sqrt", "sigmoid",
  "tanh", "matmul", "transpose", "reshape", "broadcast_to", "sum", "mean", "max", "identity", "arange", "eq", "lt",
  "where",
};

struct ggmlgraph_Backend_state {
  ggml_backend_t backend;
};

static void ggmlgraph_Backend_free(void *state) {
  struct ggmlgraph_Backend_state *self = (struct ggmlgraph_Backend_state *)state;

  if (self->backend != NULL) {
    ggml_backend_free(self->backend);
    self->backend = NULL;
  }
}
#define GGMLGRAPH_Backend_FREE ggmlgraph_Backend_free

struct gnode {
  struct ggml_tensor *t;
  int64_t ne[4];
  int op;
  int lazy;    /* t has fewer elements than ne and repeats to it */
  int zero;    /* a fill of 0.0, possibly broadcast or reshaped */
  int arg;     /* a transpose's argument, for matmul's layout */
  float value; /* a fill's value, set once the leaves are allocated */
};

#define MAX_OUTPUTS 16

struct ggmlgraph_Program_state {
  ggml_backend_t backend;        /* borrowed from the Backend resource */
  struct ggml_context *ctx;      /* the op tensors and the graph; no data */
  struct ggml_context *leaf_ctx; /* the leaves, whose data is leaf_buffer */
  ggml_backend_buffer_t leaf_buffer;
  ggml_gallocr_t galloc;
  struct ggml_cgraph *graph;
  struct gnode *nodes;
  int64_t count, capacity;
  int64_t outputs[MAX_OUTPUTS];
  int64_t output_count;
  float *xs, *ys; /* a benchmark's host batches */
  int64_t x_node, y_node, batch_count;
  int64_t compile_ns, runs, upload_ns, compute_ns;
  int compiled; /* compile succeeded; a failed compile leaves 0 for good */
};

static void ggmlgraph_Program_free(void *state) {
  struct ggmlgraph_Program_state *self = (struct ggmlgraph_Program_state *)state;

  if (self->galloc != NULL) ggml_gallocr_free(self->galloc);
  if (self->leaf_buffer != NULL) ggml_backend_buffer_free(self->leaf_buffer);
  if (self->leaf_ctx != NULL) ggml_free(self->leaf_ctx);
  if (self->ctx != NULL) ggml_free(self->ctx);
  free(self->nodes);
  free(self->xs);
  free(self->ys);
  memset(self, 0, sizeof *self);
}
#define GGMLGRAPH_Program_FREE ggmlgraph_Program_free

static int64_t now_ns(void) {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static an_term error_text(an_env *env, const char *message) { return an_error(env, message, strlen(message)); }

static const char *device_type_name(enum ggml_backend_dev_type type) {
  switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
    default: return "unknown";
  }
}

an_term ggmlgraph_backend_cpu(an_env *env, int64_t threads) {
  an_term term;
  struct ggmlgraph_Backend_state *state;
  ggml_backend_t backend;

  if (threads < 1 || threads > 256) return error_text(env, "backend_cpu takes 1..256 threads");
  backend = ggml_backend_cpu_init();
  if (backend == NULL) return error_text(env, "ggml_backend_cpu_init returned no backend");
  ggml_backend_cpu_set_n_threads(backend, (int)threads);
  state = ggmlgraph_Backend_new(env, &term);
  state->backend = backend;
  return ggmlgraph_result_Backend_ok(env, term);
}

/* ggml-Vulkan converts F32 `mul_mat` inputs to fp16 on a device that has
   fp16, unless GGML_VK_DISABLE_F16 is set when the device is first opened.
   No per-op switch works here: `ggml_prec_set_acc(F32)` changes nothing, and
   `ggml_prec_set_src(F32)` makes the backend refuse the op. So the precision
   is chosen once per process, by the first `backend_vulkan` that opens a
   device; the lock keeps two tasks from choosing at once. */
static int vulkan_fp16 = -1;
static pthread_mutex_t vulkan_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *open_vulkan(bool fp16, ggml_backend_t *out) {
  ggml_backend_t backend;
  ggml_backend_dev_t device;

  if (vulkan_fp16 != -1 && vulkan_fp16 != (int)fp16)
    return vulkan_fp16 ? "this process opened Vulkan with fp16 mul_mat already"
                       : "this process opened Vulkan with F32 mul_mat already";
  if (vulkan_fp16 == -1) {
    if (fp16) unsetenv("GGML_VK_DISABLE_F16");
    else setenv("GGML_VK_DISABLE_F16", "1", 1);
  }
  if (ggml_backend_vk_get_device_count() < 1) return "no Vulkan device";
  backend = ggml_backend_vk_init(0);
  if (backend == NULL) return "ggml_backend_vk_init(0) returned no backend";
  device = ggml_backend_get_device(backend);
  if (!ggml_backend_is_vk(backend) || device == NULL || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
    ggml_backend_free(backend);
    return "the Vulkan backend fell back to a CPU";
  }
  vulkan_fp16 = fp16;
  *out = backend;
  return NULL;
}

an_term ggmlgraph_backend_vulkan(an_env *env, bool fp16) {
  an_term term;
  struct ggmlgraph_Backend_state *state;
  ggml_backend_t backend = NULL;
  const char *why;

  pthread_mutex_lock(&vulkan_lock);
  why = open_vulkan(fp16, &backend);
  pthread_mutex_unlock(&vulkan_lock);
  if (why != NULL) return error_text(env, why);
  state = ggmlgraph_Backend_new(env, &term);
  state->backend = backend;
  return ggmlgraph_result_Backend_ok(env, term);
}

an_term ggmlgraph_describe(an_env *env, struct ggmlgraph_Backend_state *backend) {
  ggml_backend_dev_t device = ggml_backend_get_device(backend->backend);
  char text[512];

  if (device == NULL) return error_text(env, "the backend has no ggml device");
  snprintf(text, sizeof text, "%s — %s (%s)", ggml_backend_dev_name(device), ggml_backend_dev_description(device),
           device_type_name(ggml_backend_dev_type(device)));
  return ggmlgraph_result_String_ok(env, (ng_str){ text, strlen(text) });
}

static struct ggml_context *new_context(size_t bytes) {
  struct ggml_init_params params;

  params.mem_size = bytes;
  params.mem_buffer = NULL;
  params.no_alloc = true;
  return ggml_init(params);
}

/* Room for the graph: a traced node lowers to a handful of ggml tensors at
   most (a matmul's transpose and cont, a reduction's reshapes). */
static size_t graph_size(int64_t capacity) { return (size_t)(capacity * 8 + 64); }

an_term ggmlgraph_program(an_env *env, struct ggmlgraph_Backend_state *backend, int64_t capacity) {
  an_term term;
  struct ggmlgraph_Program_state *p;
  size_t size;

  if (capacity < 1 || capacity > 1000000) return error_text(env, "program takes 1..1000000 nodes");
  size = graph_size(capacity);
  p = ggmlgraph_Program_new(env, &term);
  p->backend = backend->backend;
  p->capacity = capacity;
  p->nodes = (struct gnode *)calloc((size_t)capacity, sizeof(struct gnode));
  p->ctx = new_context(ggml_tensor_overhead() * size + ggml_graph_overhead_custom(size, false));
  p->leaf_ctx = new_context(ggml_tensor_overhead() * (size_t)(capacity + 8));
  p->x_node = p->y_node = -1;
  if (p->nodes == NULL || p->ctx == NULL || p->leaf_ctx == NULL) return error_text(env, "out of memory making a program");
  return ggmlgraph_result_Program_ok(env, term);
}

static int64_t volume(const int64_t ne[4]) { return ne[0] * ne[1] * ne[2] * ne[3]; }

static int same_ne(const struct ggml_tensor *t, const int64_t ne[4]) {
  return t->ne[0] == ne[0] && t->ne[1] == ne[1] && t->ne[2] == ne[2] && t->ne[3] == ne[3];
}

/* The node's tensor at its full recorded shape. */
static struct ggml_tensor *full_of(struct ggmlgraph_Program_state *p, const struct gnode *n) {
  if (!n->lazy) return n->t;
  return ggml_repeat_4d(p->ctx, n->t, n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
}

static struct ggml_tensor *contiguous(struct ggmlgraph_Program_state *p, struct ggml_tensor *t) {
  return ggml_is_contiguous(t) ? t : ggml_cont(p->ctx, t);
}

static struct ggml_tensor *dense(struct ggmlgraph_Program_state *p, const struct gnode *n) {
  return contiguous(p, full_of(p, n));
}

static struct ggml_tensor *reshape(struct ggmlgraph_Program_state *p, struct ggml_tensor *t, const int64_t ne[4]) {
  if (same_ne(t, ne)) return t;
  return ggml_reshape_4d(p->ctx, contiguous(p, t), ne[0], ne[1], ne[2], ne[3]);
}

typedef struct ggml_tensor *(*binary_fn)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *);

/* `a op b` at `a`'s recorded shape, with `b` broadcast by ggml when lazy. */
static struct ggml_tensor *binary(struct ggmlgraph_Program_state *p, binary_fn fn, int commutes, const struct gnode *a,
                                  const struct gnode *b) {
  if (a->lazy && !b->lazy && commutes) return fn(p->ctx, b->t, a->t);
  return fn(p->ctx, full_of(p, a), b->t);
}

/* `1 - m` for a 0/1 mask. */
static struct ggml_tensor *flip(struct ggmlgraph_Program_state *p, struct ggml_tensor *m) {
  return ggml_scale_bias(p->ctx, contiguous(p, m), -1.0f, 1.0f);
}

/* `x` as a 2-D tensor whose rows (ne[0]) are the reduced block: reshaped to
   [inner, outer] for trailing axes, transposed from [outer, inner] for
   leading ones. NULL for axes that are neither block. */
static struct ggml_tensor *rows_to_reduce(struct ggmlgraph_Program_state *p, const struct gnode *x, int rank,
                                          const int64_t *axes, int naxes) {
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

/* Reads up to `max` ints; -2 when there are more, -1 when the cursor failed
   (the error is raised already), 0 when done. */
static int read_ints(an_env *env, ggmlgraph_list_Int *list, int64_t *out, int max, int *n) {
  int64_t v;
  int next;

  *n = 0;
  while ((next = ggmlgraph_list_Int_next(env, list, &v)) == 1) {
    if (*n >= max) return -2;
    out[(*n)++] = v;
  }
  return next;
}

an_term ggmlgraph_node(an_env *env, struct ggmlgraph_Program_state *p, int64_t op, ggmlgraph_list_Int args_list,
                       ggmlgraph_list_Int shape_list, ggmlgraph_list_Int ints_list, double value) {
  int64_t args[3], shape[4], ints[8];
  int nargs, rank, nints, i, r;
  struct gnode *n;
  const struct gnode *a = NULL, *b = NULL, *c = NULL;
  struct ggml_tensor *t = NULL;
  char text[256];

  if (p->graph != NULL) return error_text(env, "the program is compiled already, or its compile failed");
  if (p->count >= p->capacity) return error_text(env, "the program is full");
  if (op < 0 || op >= OP_COUNT) return error_text(env, "unknown op code");
  if ((r = read_ints(env, &args_list, args, 3, &nargs)) != 0) return r == -1 ? an_void(env) : error_text(env, "a node takes at most 3 args");
  if ((r = read_ints(env, &shape_list, shape, 4, &rank)) != 0) return r == -1 ? an_void(env) : error_text(env, "ggml tensors have rank <= 4");
  if ((r = read_ints(env, &ints_list, ints, 8, &nints)) != 0) return r == -1 ? an_void(env) : error_text(env, "a node takes at most 8 ints");
  for (i = 0; i < nargs; i++) {
    if (args[i] < 0 || args[i] >= p->count) return error_text(env, "an arg is not an earlier node");
  }
  if (nargs > 0) a = &p->nodes[args[0]];
  if (nargs > 1) b = &p->nodes[args[1]];
  if (nargs > 2) c = &p->nodes[args[2]];
  if ((op >= OP_ADD && op <= OP_DIV) || op == OP_MATMUL || op == OP_EQ || op == OP_LT) {
    if (nargs != 2) return error_text(env, "a binary op takes 2 args");
  } else if (op == OP_WHERE) {
    if (nargs != 3) return error_text(env, "where takes 3 args");
  } else if (op > OP_FULL && op != OP_ARANGE && nargs != 1) {
    return error_text(env, "a unary op takes 1 arg");
  }

  n = &p->nodes[p->count];
  memset(n, 0, sizeof *n);
  n->op = (int)op;
  n->arg = -1;
  for (i = 0; i < 4; i++) n->ne[i] = i < rank ? shape[rank - 1 - i] : 1;

  switch (op) {
    case OP_INPUT:
    case OP_STATE:
    case OP_FIXED:
      t = ggml_new_tensor_4d(p->leaf_ctx, GGML_TYPE_F32, n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
      if (op == OP_INPUT) ggml_set_input(t);
      break;
    case OP_FULL:
      t = ggml_new_tensor_1d(p->leaf_ctx, GGML_TYPE_F32, 1);
      n->value = (float)value;
      n->zero = value == 0.0;
      n->lazy = volume(n->ne) != 1;
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
        t = ggml_mul_mat(p->ctx, dense(p, &p->nodes[b->arg]), dense(p, a));
      } else {
        t = ggml_mul_mat(p->ctx, ggml_cont(p->ctx, ggml_transpose(p->ctx, dense(p, b))), dense(p, a));
      }
      break;
    case OP_TRANSPOSE:
      t = ggml_cont(p->ctx, ggml_transpose(p->ctx, dense(p, a)));
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
        if (n->ne[i] % a->t->ne[i] != 0) return error_text(env, "broadcast_to: the source does not repeat to the shape");
      }
      t = a->t;
      n->lazy = !same_ne(t, n->ne);
      n->zero = a->zero;
      break;
    case OP_SUM:
    case OP_MEAN:
    case OP_MAX: {
      struct ggml_tensor *rows;

      if (nints < 2) return error_text(env, "a reduction takes the input's rank and its axes");
      rows = rows_to_reduce(p, a, (int)ints[0], ints + 1, nints - 1);
      if (rows == NULL) {
        snprintf(text, sizeof text, "%s over axes that are not a leading or trailing block", op_names[op]);
        return error_text(env, text);
      }
      if (op == OP_MAX) {
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
      if (nints != 3) return error_text(env, "arange takes start, stop and step");
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
        t = ggml_mul(p->ctx, flip(p, dense(p, a)), full_of(p, c));
      } else {
        t = ggml_add(p->ctx, binary(p, ggml_mul, 1, b, a), ggml_mul(p->ctx, flip(p, dense(p, a)), full_of(p, c)));
      }
      break;
  }
  if (t == NULL) {
    snprintf(text, sizeof text, "ggml made no tensor for %s", op_names[op]);
    return error_text(env, text);
  }
  if (!n->lazy && !same_ne(t, n->ne)) {
    snprintf(text, sizeof text,
             "%s made ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], the trace recorded [%" PRId64 ", %" PRId64
             ", %" PRId64 ", %" PRId64 "]",
             op_names[op], t->ne[0], t->ne[1], t->ne[2], t->ne[3], n->ne[0], n->ne[1], n->ne[2], n->ne[3]);
    return error_text(env, text);
  }
  n->t = t;
  return ggmlgraph_result_Int_ok(env, p->count++);
}

an_term ggmlgraph_share(an_env *env, struct ggmlgraph_Program_state *p, int64_t node,
                        struct ggmlgraph_Program_state *other, int64_t source) {
  struct ggml_tensor *mine, *theirs;

  if (p->graph != NULL) return error_text(env, "share before compile");
  if (!other->compiled) return error_text(env, "share from a compiled program");
  if (p->backend != other->backend) return error_text(env, "shared leaves live on one backend");
  if (node < 0 || node >= p->count || source < 0 || source >= other->count) return error_text(env, "no such node");
  if (p->nodes[node].op != OP_STATE && p->nodes[node].op != OP_INPUT) return error_text(env, "only a leaf is shared");
  mine = p->nodes[node].t;
  theirs = other->nodes[source].t;
  if (theirs->data == NULL || !same_ne(mine, theirs->ne)) return error_text(env, "a shared leaf needs one allocated shape");
  /* the op tensors made from this leaf already point at its struct, so the
     struct takes the other program's data instead of being replaced */
  mine->data = theirs->data;
  mine->buffer = theirs->buffer;
  return ggmlgraph_result_Void_ok(env, an_void(env));
}

an_term ggmlgraph_compile(an_env *env, struct ggmlgraph_Program_state *p, ggmlgraph_list_Int outputs,
                          ggmlgraph_list_Int writebacks) {
  int64_t start = now_ns(), outs[MAX_OUTPUTS], pairs[2 * 64];
  int nouts, npairs, r, i;
  char text[256];
  struct ggml_tensor *leaf;

  if (p->graph != NULL) return error_text(env, "the program is compiled already, or its compile failed");
  if ((r = read_ints(env, &outputs, outs, MAX_OUTPUTS, &nouts)) != 0) return r == -1 ? an_void(env) : error_text(env, "at most 16 outputs");
  if ((r = read_ints(env, &writebacks, pairs, 2 * 64, &npairs)) != 0) return r == -1 ? an_void(env) : error_text(env, "at most 64 writebacks");
  if (npairs % 2 != 0) return error_text(env, "writebacks come in pairs");
  for (i = 0; i < nouts; i++) {
    if (outs[i] < 0 || outs[i] >= p->count) return error_text(env, "an output is not a node");
  }
  for (i = 0; i < npairs; i++) {
    if (pairs[i] < 0 || pairs[i] >= p->count) return error_text(env, "a writeback names no node");
    if (i % 2 == 0 && p->nodes[pairs[i]].op != OP_STATE) return error_text(env, "a writeback goes to a state leaf");
  }
  for (i = 0; i < npairs; i += 2) {
    int j;

    if (volume(p->nodes[pairs[i]].ne) != volume(p->nodes[pairs[i + 1]].ne)) {
      return error_text(env, "a writeback's value has another element count than its state");
    }
    /* a value that is a written state would read it after its own write
       (a swap): the result would depend on the order of the copies */
    for (j = 0; j < npairs; j += 2) {
      if (pairs[i + 1] == pairs[j]) return error_text(env, "a writeback's value is a state another writeback replaces");
      if (j != i && pairs[i] == pairs[j]) return error_text(env, "a state takes two writebacks");
    }
  }

  p->graph = ggml_new_graph_custom(p->ctx, graph_size(p->capacity), false);
  for (i = 0; i < nouts; i++) {
    struct gnode *n = &p->nodes[outs[i]];

    n->t = full_of(p, n);
    n->lazy = 0;
    ggml_set_output(n->t);
    ggml_build_forward_expand(p->graph, n->t);
    p->outputs[p->output_count++] = outs[i];
  }
  /* every update is computed before the first writeback, so no node reads a
     state after it was replaced */
  for (i = 0; i < npairs; i += 2) {
    struct gnode *update = &p->nodes[pairs[i + 1]];

    update->t = dense(p, update);
    update->lazy = 0;
    ggml_build_forward_expand(p->graph, update->t);
  }
  for (i = 0; i < npairs; i += 2) {
    ggml_build_forward_expand(p->graph, ggml_cpy(p->ctx, p->nodes[pairs[i + 1]].t, p->nodes[pairs[i]].t));
  }

  /* the leaves live in one persistent buffer the graph allocator never
     touches; a shared leaf points into another program's buffer already */
  for (leaf = ggml_get_first_tensor(p->leaf_ctx); leaf != NULL; leaf = ggml_get_next_tensor(p->leaf_ctx, leaf)) {
    if (leaf->data == NULL) break;
  }
  if (leaf != NULL) {
    p->leaf_buffer = ggml_backend_alloc_ctx_tensors(p->leaf_ctx, p->backend);
    if (p->leaf_buffer == NULL) return error_text(env, "allocating the leaves failed");
    ggml_backend_buffer_clear(p->leaf_buffer, 0);
  }
  for (i = 0; i < p->count; i++) {
    if (p->nodes[i].op == OP_FULL) ggml_backend_tensor_set(p->nodes[i].t, &p->nodes[i].value, 0, sizeof(float));
  }
  for (i = 0; i < ggml_graph_n_nodes(p->graph); i++) {
    struct ggml_tensor *node = ggml_graph_node(p->graph, i);

    if (!ggml_backend_supports_op(p->backend, node)) {
      snprintf(text, sizeof text, "the backend does not support graph node %d: %s", i, ggml_op_desc(node));
      return error_text(env, text);
    }
  }
  p->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p->backend));
  if (p->galloc == NULL || !ggml_gallocr_alloc_graph(p->galloc, p->graph)) return error_text(env, "allocating the graph failed");
  p->compile_ns = now_ns() - start;
  p->compiled = 1;
  return ggmlgraph_result_Void_ok(env, an_void(env));
}

static struct gnode *leaf_of(struct ggmlgraph_Program_state *p, int64_t node) {
  if (node < 0 || node >= p->count) return NULL;
  if (p->nodes[node].op != OP_INPUT && p->nodes[node].op != OP_STATE && p->nodes[node].op != OP_FIXED) return NULL;
  return &p->nodes[node];
}

an_term ggmlgraph_set(an_env *env, struct ggmlgraph_Program_state *p, int64_t node, ggmlgraph_list_Float values) {
  struct gnode *n = leaf_of(p, node);
  int64_t count, i = 0;
  float *host;
  double v;
  int next;

  if (!p->compiled) return error_text(env, "set after a successful compile");
  if (n == NULL) return error_text(env, "set takes a leaf");
  count = volume(n->ne);
  host = (float *)malloc((size_t)count * sizeof(float));
  if (host == NULL) return error_text(env, "out of memory");
  while ((next = ggmlgraph_list_Float_next(env, &values, &v)) == 1 && i < count) host[i++] = (float)v;
  if (next < 0) {
    free(host);
    return an_void(env);
  }
  if (i != count || next == 1) {
    free(host);
    return error_text(env, "set needs exactly one value per element");
  }
  ggml_backend_tensor_set(n->t, host, 0, (size_t)count * sizeof(float));
  free(host);
  return ggmlgraph_result_Void_ok(env, an_void(env));
}

/* A small LCG, so a benchmark's data costs no FFI and no Agentica lists. */
static float next_uniform(uint64_t *s) {
  *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
  return (float)((*s >> 40) & 0xFFFFFF) / 16777216.0f;
}

an_term ggmlgraph_fill(an_env *env, struct ggmlgraph_Program_state *p, int64_t node, int64_t seed, double scale) {
  struct gnode *n = leaf_of(p, node);
  uint64_t s = (uint64_t)seed;
  int64_t count, i;
  float *host;

  if (!p->compiled) return error_text(env, "fill after a successful compile");
  if (n == NULL) return error_text(env, "fill takes a leaf");
  count = volume(n->ne);
  host = (float *)malloc((size_t)count * sizeof(float));
  if (host == NULL) return error_text(env, "out of memory");
  for (i = 0; i < count; i++) host[i] = (next_uniform(&s) * 2.0f - 1.0f) * (float)scale;
  ggml_backend_tensor_set(n->t, host, 0, (size_t)count * sizeof(float));
  free(host);
  return ggmlgraph_result_Void_ok(env, an_void(env));
}

/* The node's values on the host, or NULL with `*why` set. */
static float *host_copy(struct ggmlgraph_Program_state *p, int64_t node, int64_t *count, const char **why);

an_term ggmlgraph_gap(an_env *env, struct ggmlgraph_Program_state *p, int64_t node,
                      struct ggmlgraph_Program_state *other, int64_t other_node) {
  int64_t n, m, i;
  const char *why = NULL;
  float *a = host_copy(p, node, &n, &why), *b = NULL;
  double scale = 1e-6, diff = 0.0;

  if (a != NULL) b = host_copy(other, other_node, &m, &why);
  if (a == NULL || b == NULL) {
    free(a);
    return error_text(env, why);
  }
  if (n != m) why = "gap compares nodes of different sizes";
  for (i = 0; why == NULL && i < n; i++) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      why = "gap found a value that is not finite";
      break;
    }
    if (fabs(b[i]) > scale) scale = fabs(b[i]);
    if (fabs((double)a[i] - b[i]) > diff) diff = fabs((double)a[i] - b[i]);
  }
  free(a);
  free(b);
  if (why != NULL) return error_text(env, why);
  return ggmlgraph_result_Float_ok(env, diff / scale);
}

an_term ggmlgraph_batches(an_env *env, struct ggmlgraph_Program_state *p, int64_t x, int64_t y, int64_t classes,
                          int64_t count, int64_t seed) {
  struct gnode *xn = leaf_of(p, x), *yn = leaf_of(p, y);
  int64_t xsize, ysize, i;
  uint64_t s = (uint64_t)seed;

  if (xn == NULL || yn == NULL || xn->op != OP_INPUT || yn->op != OP_INPUT) return error_text(env, "batches takes two input leaves");
  if (count < 1 || classes < 1) return error_text(env, "batches takes count >= 1 and classes >= 1");
  xsize = volume(xn->ne);
  ysize = volume(yn->ne);
  free(p->xs);
  free(p->ys);
  p->xs = (float *)malloc((size_t)(xsize * count) * sizeof(float));
  p->ys = (float *)malloc((size_t)(ysize * count) * sizeof(float));
  if (p->xs == NULL || p->ys == NULL) {
    p->batch_count = 0;
    return error_text(env, "out of memory making batches");
  }
  for (i = 0; i < xsize * count; i++) p->xs[i] = next_uniform(&s) * 2.0f - 1.0f;
  for (i = 0; i < ysize * count; i++) p->ys[i] = (float)((int64_t)(next_uniform(&s) * (float)classes) % classes);
  p->x_node = x;
  p->y_node = y;
  p->batch_count = count;
  return ggmlgraph_result_Void_ok(env, an_void(env));
}

an_term ggmlgraph_run(an_env *env, struct ggmlgraph_Program_state *p, int64_t batch) {
  int64_t t0, t1, t2;
  float loss = 0.0f;

  if (!p->compiled) return error_text(env, "run a compiled program");
  if (batch < 0) return error_text(env, "run takes a batch index >= 0");
  t0 = now_ns();
  if (p->batch_count > 0) {
    struct gnode *xn = &p->nodes[p->x_node], *yn = &p->nodes[p->y_node];
    int64_t k = batch % p->batch_count, xsize = volume(xn->ne), ysize = volume(yn->ne);

    ggml_backend_tensor_set(xn->t, p->xs + k * xsize, 0, (size_t)xsize * sizeof(float));
    ggml_backend_tensor_set(yn->t, p->ys + k * ysize, 0, (size_t)ysize * sizeof(float));
  }
  t1 = now_ns();
  if (ggml_backend_graph_compute(p->backend, p->graph) != GGML_STATUS_SUCCESS) {
    return error_text(env, "ggml_backend_graph_compute failed");
  }
  ggml_backend_synchronize(p->backend);
  if (p->output_count > 0) ggml_backend_tensor_get(p->nodes[p->outputs[0]].t, &loss, 0, sizeof(float));
  t2 = now_ns();
  p->runs++;
  p->upload_ns += t1 - t0;
  p->compute_ns += t2 - t1;
  return ggmlgraph_result_Float_ok(env, (double)loss);
}

/* An output, a state or an input: the nodes whose memory the graph
   allocator does not reuse. */
static int readable(const struct ggmlgraph_Program_state *p, int64_t node) {
  int64_t i;

  if (p->nodes[node].op == OP_STATE || p->nodes[node].op == OP_INPUT) return 1;
  for (i = 0; i < p->output_count; i++) {
    if (p->outputs[i] == node) return 1;
  }
  return 0;
}

static float *host_copy(struct ggmlgraph_Program_state *p, int64_t node, int64_t *count, const char **why) {
  float *host;

  if (!p->compiled) {
    *why = "read a compiled program";
    return NULL;
  }
  if (node < 0 || node >= p->count || !readable(p, node)) {
    *why = "read an output, a state or an input";
    return NULL;
  }
  *count = ggml_nelements(p->nodes[node].t);
  host = (float *)malloc((size_t)*count * sizeof(float));
  if (host == NULL) {
    *why = "out of memory";
    return NULL;
  }
  ggml_backend_tensor_get(p->nodes[node].t, host, 0, (size_t)*count * sizeof(float));
  return host;
}

an_term ggmlgraph_read(an_env *env, struct ggmlgraph_Program_state *p, int64_t node) {
  ggmlgraph_list_Float_builder list;
  struct ggml_tensor *t;
  int64_t count, i;
  float *host;

  if (!p->compiled) return error_text(env, "read a compiled program");
  if (node < 0 || node >= p->count) return error_text(env, "no such node");
  if (!readable(p, node)) return error_text(env, "read an output, a state or an input");
  t = p->nodes[node].t;
  count = ggml_nelements(t);
  host = (float *)malloc((size_t)count * sizeof(float));
  if (host == NULL) return error_text(env, "out of memory");
  ggml_backend_tensor_get(t, host, 0, (size_t)count * sizeof(float));
  ggmlgraph_list_Float_begin(env, &list, (size_t)count);
  for (i = 0; i < count; i++) ggmlgraph_list_Float_push(env, &list, (double)host[i]);
  free(host);
  return ggmlgraph_result_list_Float_ok(env, ggmlgraph_list_Float_end(env, &list));
}

an_term ggmlgraph_stats(an_env *env, struct ggmlgraph_Program_state *p) {
  ggmlgraph_list_Int_builder list;

  ggmlgraph_list_Int_begin(env, &list, 7);
  ggmlgraph_list_Int_push(env, &list, p->compiled ? ggml_graph_n_nodes(p->graph) : 0);
  ggmlgraph_list_Int_push(env, &list, p->leaf_buffer != NULL ? (int64_t)ggml_backend_buffer_get_size(p->leaf_buffer) : 0);
  ggmlgraph_list_Int_push(env, &list, p->galloc != NULL ? (int64_t)ggml_gallocr_get_buffer_size(p->galloc, 0) : 0);
  ggmlgraph_list_Int_push(env, &list, p->compile_ns);
  ggmlgraph_list_Int_push(env, &list, p->runs);
  ggmlgraph_list_Int_push(env, &list, p->upload_ns);
  ggmlgraph_list_Int_push(env, &list, p->compute_ns);
  return ggmlgraph_result_list_Int_ok(env, ggmlgraph_list_Int_end(env, &list));
}

GGMLGRAPH_NATIVE_MODULE(NULL)
