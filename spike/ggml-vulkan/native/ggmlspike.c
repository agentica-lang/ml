/* The thin C layer of `ggml_spike.agc`, on the header that native-gen makes
   from its `extern module` block. It calls the pinned ggml submodule
   (`third_party/ggml`, built by `tools/build-ggml.sh`) and does nothing else:
   one ggml call per function, the checks ggml would otherwise do silently
   wrong (ggml's `add` broadcasts), and the readback.

   ggml is a system dependency of this spike, never a copy: its headers and
   static libraries arrive through `C_INCLUDE_PATH` and `LIBRARY_PATH`, which
   `tools/build-ggml.sh` prints. The module is `optional` in
   `agentica.toml`, so without those the programs say the module is unbuilt
   instead of measuring nothing.

   The one thing this file must not get wrong: `backend_vulkan` fails unless
   `ggml_backend_vk_init(0)` returns a backend that `ggml_backend_is_vk`
   accepts and whose device is not a CPU. A silent CPU fallback would make
   every Vulkan number on the README a CPU number.

   Do not edit the generated header: change the .agc and make the header
   again. */
#define GGMLSPIKE_NATIVE_IMPLEMENTATION
#include "ggmlspike_native.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* An open backend. The state owns the `ggml_backend_t`; `threads` is what
   `backend_cpu` set, kept only to name it in `describe`. */
struct ggmlspike_Backend_state {
  ggml_backend_t backend;
  int32_t threads;
};

static void ggmlspike_Backend_free(void *state) {
  struct ggmlspike_Backend_state *self = (struct ggmlspike_Backend_state *)state;

  if (self->backend != NULL) {
    ggml_backend_free(self->backend);
    self->backend = NULL;
  }
}
#define GGMLSPIKE_Backend_FREE ggmlspike_Backend_free

/* A tensor, and the ggml context and backend buffer it lives in. Both are
   this resource's: the buffer is freed before the context it was allocated
   from. `backend` is borrowed from the `Backend` the tensor was made on,
   which the Agentica side keeps open for the whole run. */
struct ggmlspike_Tensor_state {
  ggml_backend_t backend;
  struct ggml_context *ctx;
  struct ggml_backend_buffer *buffer;
  struct ggml_tensor *tensor;
};

static void ggmlspike_Tensor_free(void *state) {
  struct ggmlspike_Tensor_state *self = (struct ggmlspike_Tensor_state *)state;

  if (self->buffer != NULL) {
    ggml_backend_buffer_free(self->buffer);
    self->buffer = NULL;
  }
  if (self->ctx != NULL) {
    ggml_free(self->ctx);
    self->ctx = NULL;
  }
  self->tensor = NULL;
}
#define GGMLSPIKE_Tensor_FREE ggmlspike_Tensor_free

/* An op that fails returns `error(text)`, as the reference backend does; an
   op that cannot even be attempted is a `Result` error too, never a halt. */
static an_term error_text(an_env *env, const char *message) {
  return an_error(env, message, strlen(message));
}

/* The device types of `ggml_backend_dev_type`, as the README names them. */
static const char *device_type_name(enum ggml_backend_dev_type type) {
  switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU: return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU: return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU: return "IGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
    default: return "unknown";
  }
}

/* A shape as the reference prints one: "[4, 3]". */
static void shape_text(const struct ggml_tensor *t, char *out, size_t size) {
  size_t at = 0;

  out[0] = '\0';
  at += (size_t)snprintf(out, size, "[");
  for (int i = (int)ggml_n_dims(t) - 1; i >= 0 && at < size; i--) {
    at += (size_t)snprintf(out + at, size - at, "%s%" PRId64, i == (int)ggml_n_dims(t) - 1 ? "" : ", ", t->ne[i]);
  }
  snprintf(out + at, size - at, "]");
}

/* The deterministic values of `tensor/tools/bench.agc`, so every backend in
   this spike gets the same input and a checksum can be compared across them:
   no random source, and no data crosses the boundary to fill a tensor. */
static float pattern_value(size_t i) {
  return (float)(((int64_t)i * 7 + 3) % 11) / 11.0f - 0.4f;
}

/* A ggml context that owns tensor structs only: the data of every tensor is
   allocated by the backend buffer of the op that makes it. */
static struct ggml_context *new_context(size_t bytes) {
  struct ggml_init_params params = {
    .mem_size = bytes,
    .mem_buffer = NULL,
    .no_alloc = true,
  };

  return ggml_init(params);
}

/* The graph of one op: allocate the tensors the context holds on `backend`,
   compute it, and wait for it. `*buffer_out` receives the backend buffer the
   result's data lives in. Returns NULL on success and an error text
   otherwise. This is what an op of a ggml-backed `tensor/` would do per call,
   and the reason `empty_graph` exists: it measures the part that is not the
   arithmetic. */
static const char *run_graph(ggml_backend_t backend, struct ggml_context *ctx, struct ggml_tensor *out,
                             struct ggml_backend_buffer **buffer_out) {
  struct ggml_cgraph *graph;
  struct ggml_backend_buffer *buffer;
  enum ggml_status status;

  if (!ggml_backend_supports_op(backend, out)) {
    return "the backend does not support this op";
  }
  /* ggml_backend_alloc_ctx_tensors gives every tensor of the context its
     data on the backend's default buffer type; the buffer is the caller's. */
  buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buffer == NULL) {
    return "ggml_backend_alloc_ctx_tensors failed";
  }
  graph = ggml_new_graph(ctx);
  if (graph == NULL) {
    ggml_backend_buffer_free(buffer);
    return "ggml_new_graph failed";
  }
  ggml_build_forward_expand(graph, out);

  status = ggml_backend_graph_compute(backend, graph);
  if (status != GGML_STATUS_SUCCESS) {
    ggml_backend_buffer_free(buffer);
    return "ggml_backend_graph_compute failed";
  }
  /* Vulkan computes asynchronously; every op of this module is timed after it
     has finished, not after it was submitted. */
  ggml_backend_synchronize(backend);
  *buffer_out = buffer;
  return NULL;
}

/* Wraps a computed tensor (its context and backend buffer) as the resource
   the signature returns. The buffer holds every tensor of the context, so
   `Native.close` frees the result's data with it. */
static an_term finish(an_env *env, ggml_backend_t backend, struct ggml_context *ctx,
                      struct ggml_backend_buffer *buffer, struct ggml_tensor *tensor) {
  an_term term;
  struct ggmlspike_Tensor_state *state = ggmlspike_Tensor_new(env, &term);

  state->backend = backend;
  state->ctx = ctx;
  state->buffer = buffer;
  state->tensor = tensor;
  return ggmlspike_result_Tensor_ok(env, term);
}

static void discard(struct ggml_context *ctx, struct ggml_backend_buffer *buffer) {
  if (buffer != NULL) {
    ggml_backend_buffer_free(buffer);
  }
  if (ctx != NULL) {
    ggml_free(ctx);
  }
}

/* One op: a context holding the result, `make` builds the graph in it, and
   the backend buffer `run_graph` allocates is the result's. */
typedef struct ggml_tensor *(*make_op_t)(struct ggml_context *ctx, void *left, void *right);

an_term ggmlspike_devices(an_env *env) {
  ggmlspike_list_String_builder list;
  size_t count = ggml_backend_dev_count();
  size_t i;

  ggmlspike_list_String_begin(env, &list, count);
  for (i = 0; i < count; i++) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    char text[512];

    snprintf(text, sizeof text, "%s — %s (%s)", ggml_backend_dev_name(device),
             ggml_backend_dev_description(device), device_type_name(ggml_backend_dev_type(device)));
    ggmlspike_list_String_push(env, &list, (ng_str){ text, strlen(text) });
  }
  return ggmlspike_result_list_String_ok(env, ggmlspike_list_String_end(env, &list));
}

an_term ggmlspike_backend_cpu(an_env *env, int64_t threads) {
  an_term term;
  struct ggmlspike_Backend_state *state;
  ggml_backend_t backend;

  if (threads < 1 || threads > 256) {
    return error_text(env, "backend_cpu takes 1..256 threads");
  }
  backend = ggml_backend_cpu_init();
  if (backend == NULL) {
    return error_text(env, "ggml_backend_cpu_init returned no backend");
  }
  ggml_backend_cpu_set_n_threads(backend, (int)threads);

  state = ggmlspike_Backend_new(env, &term);
  state->backend = backend;
  state->threads = (int32_t)threads;
  return ggmlspike_result_Backend_ok(env, term);
}

an_term ggmlspike_backend_vulkan(an_env *env) {
  an_term term;
  struct ggmlspike_Backend_state *state;
  ggml_backend_t backend;
  ggml_backend_dev_t device;

  if (ggml_backend_vk_get_device_count() < 1) {
    return error_text(env, "ggml_backend_vk_get_device_count() reports no Vulkan device");
  }
  backend = ggml_backend_vk_init(0);
  if (backend == NULL) {
    return error_text(env, "ggml_backend_vk_init(0) returned no backend");
  }
  if (!ggml_backend_is_vk(backend)) {
    ggml_backend_free(backend);
    return error_text(env, "ggml_backend_vk_init(0) returned a backend that is not Vulkan");
  }
  device = ggml_backend_get_device(backend);
  if (device == NULL || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
    ggml_backend_free(backend);
    return error_text(env, "the Vulkan backend's device is a CPU: the run fell back");
  }

  state = ggmlspike_Backend_new(env, &term);
  state->backend = backend;
  state->threads = 0;
  return ggmlspike_result_Backend_ok(env, term);
}

an_term ggmlspike_describe(an_env *env, struct ggmlspike_Backend_state *backend) {
  ggml_backend_dev_t device = ggml_backend_get_device(backend->backend);
  char text[512];

  if (device == NULL) {
    return error_text(env, "the backend has no ggml device");
  }
  if (backend->threads > 0) {
    snprintf(text, sizeof text, "%s — %s (%s), %d threads", ggml_backend_dev_name(device),
             ggml_backend_dev_description(device), device_type_name(ggml_backend_dev_type(device)),
             (int)backend->threads);
  } else {
    snprintf(text, sizeof text, "%s — %s (%s), vulkan=%d", ggml_backend_dev_name(device),
             ggml_backend_dev_description(device), device_type_name(ggml_backend_dev_type(device)),
             (int)ggml_backend_is_vk(backend->backend));
  }
  return ggmlspike_result_String_ok(env, (ng_str){ text, strlen(text) });
}

an_term ggmlspike_create(an_env *env, struct ggmlspike_Backend_state *backend, ggmlspike_list_Int shape) {
  int64_t rows = 0, cols = 0;
  int rank = 0;
  struct ggml_context *ctx;
  struct ggml_tensor *tensor;
  struct ggml_backend_buffer *buffer;
  float *host;
  size_t count, i;

  for (;;) {
    int64_t value = 0;
    int next = ggmlspike_list_Int_next(env, &shape, &value);

    if (next < 0) return an_void(env);
    if (next == 0) break;
    if (rank == 0) rows = value;
    else if (rank == 1) cols = value;
    rank++;
  }
  if (rank != 2) {
    return error_text(env, "create takes a shape of two dimensions, [rows, cols]");
  }
  if (rows < 1 || cols < 1) {
    return error_text(env, "create takes a positive shape");
  }

  ctx = new_context(ggml_tensor_overhead() * 4);
  if (ctx == NULL) {
    return error_text(env, "ggml_init failed");
  }
  /* ggml indexes by column first: a row-major `[rows, cols]` matrix is a
     tensor of ne[0] = cols and ne[1] = rows. */
  tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
  if (tensor == NULL) {
    discard(ctx, NULL);
    return error_text(env, "ggml_new_tensor_2d failed");
  }

  buffer = ggml_backend_alloc_ctx_tensors(ctx, backend->backend);
  if (buffer == NULL) {
    discard(ctx, NULL);
    return error_text(env, "ggml_backend_alloc_ctx_tensors failed");
  }

  count = (size_t)ggml_nelements(tensor);
  host = (float *)malloc(count * sizeof(float));
  if (host == NULL) {
    discard(ctx, buffer);
    return error_text(env, "out of memory filling the tensor");
  }
  for (i = 0; i < count; i++) {
    host[i] = pattern_value(i);
  }
  ggml_backend_tensor_set(tensor, host, 0, count * sizeof(float));
  free(host);

  return finish(env, backend->backend, ctx, buffer, tensor);
}

static struct ggml_tensor *make_add(struct ggml_context *ctx, void *left, void *right) {
  struct ggml_tensor *a = (struct ggml_tensor *)left;
  struct ggml_tensor *b = (struct ggml_tensor *)right;

  if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) {
    return NULL;
  }
  return ggml_add(ctx, a, b);
}

static struct ggml_tensor *make_relu(struct ggml_context *ctx, void *left, void *right) {
  (void)right;
  return ggml_relu(ctx, (struct ggml_tensor *)left);
}

static struct ggml_tensor *make_mul_mat(struct ggml_context *ctx, void *left, void *right) {
  struct ggml_tensor *a = (struct ggml_tensor *)left;
  struct ggml_tensor *b = (struct ggml_tensor *)right;

  if (a->ne[0] != b->ne[0]) {
    return NULL;
  }
  return ggml_mul_mat(ctx, a, b);
}

/* One op of the module: two input tensors of one backend, a result of the
   same backend. The shape checks name the shapes, because ggml's `add`
   broadcasts instead of refusing. */
static an_term op_two(an_env *env, struct ggmlspike_Tensor_state *a, struct ggmlspike_Tensor_state *b,
                      make_op_t make, const char *name) {
  struct ggml_context *ctx;
  struct ggml_tensor *out;
  struct ggml_backend_buffer *buffer = NULL;
  char left[64], right[64];
  const char *failure;

  if (a->backend != b->backend) {
    return error_text(env, "the two tensors live on different backends");
  }
  shape_text(a->tensor, left, sizeof left);
  shape_text(b->tensor, right, sizeof right);

  ctx = new_context(ggml_tensor_overhead() * 4 + ggml_graph_overhead());
  if (ctx == NULL) {
    return error_text(env, "ggml_init failed");
  }
  out = make(ctx, a->tensor, b->tensor);
  if (out == NULL) {
    char text[256];

    discard(ctx, NULL);
    snprintf(text, sizeof text, "%s takes mutually compatible tensors, got %s and %s", name, left, right);
    return error_text(env, text);
  }
  failure = run_graph(a->backend, ctx, out, &buffer);
  if (failure != NULL) {
    discard(ctx, buffer);
    return error_text(env, failure);
  }
  return finish(env, a->backend, ctx, buffer, out);
}

an_term ggmlspike_add(an_env *env, struct ggmlspike_Tensor_state *a, struct ggmlspike_Tensor_state *b) {
  return op_two(env, a, b, make_add, "add");
}

an_term ggmlspike_relu(an_env *env, struct ggmlspike_Tensor_state *a) {
  return op_two(env, a, a, make_relu, "relu");
}

an_term ggmlspike_mul_mat(an_env *env, struct ggmlspike_Tensor_state *a, struct ggmlspike_Tensor_state *b) {
  return op_two(env, a, b, make_mul_mat, "mul_mat");
}

an_term ggmlspike_empty_graph(an_env *env, struct ggmlspike_Backend_state *backend) {
  struct ggml_context *ctx = new_context(ggml_graph_overhead() * 2);
  struct ggml_cgraph *graph;
  enum ggml_status status;

  if (ctx == NULL) {
    return error_text(env, "ggml_init failed");
  }
  graph = ggml_new_graph(ctx);
  if (graph == NULL) {
    discard(ctx, NULL);
    return error_text(env, "ggml_new_graph failed");
  }
  status = ggml_backend_graph_compute(backend->backend, graph);
  if (status != GGML_STATUS_SUCCESS) {
    discard(ctx, NULL);
    return error_text(env, "ggml_backend_graph_compute failed on an empty graph");
  }
  ggml_backend_synchronize(backend->backend);
  discard(ctx, NULL);
  return ggmlspike_result_Void_ok(env, an_void(env));
}

an_term ggmlspike_checksum(an_env *env, struct ggmlspike_Tensor_state *t) {
  size_t count = (size_t)ggml_nelements(t->tensor);
  size_t bytes = count * sizeof(float);
  float *host = (float *)malloc(bytes);
  double total = 0.0;
  size_t i;

  if (host == NULL) {
    return error_text(env, "out of memory reading the tensor back");
  }
  ggml_backend_synchronize(t->backend);
  ggml_backend_tensor_get(t->tensor, host, 0, bytes);
  for (i = 0; i < count; i++) {
    total += host[i];
  }
  free(host);
  return ggmlspike_result_Float_ok(env, total);
}

an_term ggmlspike_readback(an_env *env, struct ggmlspike_Tensor_state *t) {
  size_t count = (size_t)ggml_nelements(t->tensor);
  size_t bytes = count * sizeof(float);
  float *host = (float *)malloc(bytes);
  ggmlspike_list_Float_builder list;
  size_t i;

  if (host == NULL) {
    return error_text(env, "out of memory reading the tensor back");
  }
  ggml_backend_synchronize(t->backend);
  ggml_backend_tensor_get(t->tensor, host, 0, bytes);
  ggmlspike_list_Float_begin(env, &list, count);
  for (i = 0; i < count; i++) {
    ggmlspike_list_Float_push(env, &list, (double)host[i]);
  }
  free(host);
  return ggmlspike_result_list_Float_ok(env, ggmlspike_list_Float_end(env, &list));
}

an_term ggmlspike_allocated(an_env *env, struct ggmlspike_Tensor_state *t) {
  return ggmlspike_result_Int_ok(env, (int64_t)ggml_backend_buffer_get_size(t->buffer));
}

an_term ggmlspike_memory(an_env *env, struct ggmlspike_Backend_state *backend) {
  ggml_backend_dev_t device = ggml_backend_get_device(backend->backend);
  ggmlspike_list_Int_builder list;
  size_t free_bytes = 0, total_bytes = 0;

  if (device == NULL) {
    return error_text(env, "the backend has no ggml device");
  }
  ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
  ggmlspike_list_Int_begin(env, &list, 2);
  ggmlspike_list_Int_push(env, &list, (int64_t)free_bytes);
  ggmlspike_list_Int_push(env, &list, (int64_t)total_bytes);
  return ggmlspike_result_list_Int_ok(env, ggmlspike_list_Int_end(env, &list));
}

GGMLSPIKE_NATIVE_MODULE(NULL)
