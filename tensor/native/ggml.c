/* The thin C layer of `src/ggml_native.agc`, over the pinned ggml submodule
   (`third_party/ggml`, built by `tools/build-engine.sh`).

   It creates tensors in a backend buffer of a device and reads them back;
   nothing here computes. Devices come from ggml's backend registry, so the
   same file builds against the CPU-only tree (CI) and the Vulkan tree: there
   the registry has no "Vulkan" entry and every Vulkan device is an error.

   Layout: a row-major shape [d0, ..., dk] is a ggml tensor with ne[0] = dk,
   ne[1] = dk-1, ..., so the bytes are the same row-major order.

   This file is compiled as C or as C++ (`tools/build-torch.sh` sets CC to
   g++), so it casts every `malloc`. */
#if !defined(__cplusplus) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* setenv, unsetenv */
#endif
#define GGML_NATIVE_IMPLEMENTATION
#include "ggml_native.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ggml_TensorHandle_state {
  struct ggml_context *ctx;
  struct ggml_tensor *t;
  ggml_backend_buffer_t buffer;
};

static void ggml_TensorHandle_free(void *state) {
  struct ggml_TensorHandle_state *self = (struct ggml_TensorHandle_state *)state;

  if (self->buffer != NULL) ggml_backend_buffer_free(self->buffer);
  if (self->ctx != NULL) ggml_free(self->ctx);
  memset(self, 0, sizeof *self);
}
#define GGML_TensorHandle_FREE ggml_TensorHandle_free

/* ggml-Vulkan converts F32 `mul_mat` inputs to fp16 on a device that has
   fp16, unless GGML_VK_DISABLE_F16 is set when the device is first opened,
   and no per-op switch exists (spike/ggml-graph/README.md). So the precision
   belongs to the process: `prefer_fp16` records the choice, the first Vulkan
   open sets the variable from it and fixes it, and the lock keeps two tasks
   from choosing or opening at once. The registry calls run under it too. */
static pthread_mutex_t registry_lock = PTHREAD_MUTEX_INITIALIZER;
static bool want_fp16 = false;
static bool vulkan_opened = false;

static an_term error_text(an_env *env, const char *message) { return an_error(env, message, strlen(message)); }

/* The Vulkan registry entry, or NULL in a CPU-only build. Call under the lock. */
static ggml_backend_reg_t vulkan_reg(void) { return ggml_backend_reg_by_name("Vulkan"); }

an_term ggml_vulkan_devices(an_env *env) {
  ggml_backend_reg_t reg;
  size_t n;

  pthread_mutex_lock(&registry_lock);
  reg = vulkan_reg();
  n = reg == NULL ? 0 : ggml_backend_reg_dev_count(reg);
  pthread_mutex_unlock(&registry_lock);
  return an_int(env, (int64_t)n);
}

an_term ggml_prefer_fp16(an_env *env, bool enabled) {
  bool conflict;
  bool fixed;

  pthread_mutex_lock(&registry_lock);
  conflict = vulkan_opened && want_fp16 != enabled;
  fixed = want_fp16;
  if (!conflict) want_fp16 = enabled;
  pthread_mutex_unlock(&registry_lock);
  if (conflict) {
    return error_text(env, fixed ? "this process opened Vulkan with fp16 mul_mat already; choose before the first Vulkan tensor"
                                 : "this process opened Vulkan with F32 mul_mat already; choose before the first Vulkan tensor");
  }
  return ggml_result_Void_ok(env, an_void(env));
}

/* The buffer type of `device` (-1: CPU, i >= 0: Vulkan device i), or false
   with the reason in `why`. Opening a Vulkan device fixes the precision. */
static bool buffer_type_of(int64_t device, ggml_backend_buffer_type_t *out, char *why, size_t len) {
  ggml_backend_reg_t reg;
  ggml_backend_dev_t dev = NULL;
  size_t n;

  pthread_mutex_lock(&registry_lock);
  if (device == -1) {
    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == NULL) snprintf(why, len, "this ggml build has no CPU backend");
  } else if (device < 0) {
    snprintf(why, len, "a Vulkan device index is >= 0, got %lld", (long long)device);
  } else if ((reg = vulkan_reg()) == NULL) {
    snprintf(why, len, "this ggml build has no Vulkan backend (build it with `tools/build-engine.sh vulkan`)");
  } else if ((size_t)device >= (n = ggml_backend_reg_dev_count(reg))) {
    snprintf(why, len, "no Vulkan device %lld: ggml sees %zu", (long long)device, n);
  } else {
    dev = ggml_backend_reg_dev_get(reg, (size_t)device);
    if (dev != NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
      snprintf(why, len, "Vulkan device %lld is a CPU", (long long)device);
      dev = NULL;
    } else if (dev != NULL && !vulkan_opened) {
      if (want_fp16) unsetenv("GGML_VK_DISABLE_F16");
      else setenv("GGML_VK_DISABLE_F16", "1", 1);
      vulkan_opened = true;
    }
  }
  *out = dev == NULL ? NULL : ggml_backend_dev_buffer_type(dev);
  pthread_mutex_unlock(&registry_lock);
  if (dev != NULL && *out == NULL) snprintf(why, len, "the device has no buffer type");
  return *out != NULL;
}

static bool failed_with(an_term *failure, an_term term) {
  *failure = term;
  return false;
}

static bool failed(an_env *env, an_term *failure, const char *message) {
  return failed_with(failure, error_text(env, message));
}

/* A new tensor of `shape` and `type` in its own buffer on `device`, into
   `state`; false with the error term in `failure`. The free hook releases a
   half-made state. */
static bool make_tensor(an_env *env, int64_t device, ggml_list_Int *shape, enum ggml_type type,
                        struct ggml_TensorHandle_state *state, an_term *failure) {
  int64_t dims[GGML_MAX_DIMS], ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  int64_t dim, count = 1;
  int rank = 0, code, i;
  char why[160];
  ggml_backend_buffer_type_t buft;
  struct ggml_init_params params;
  size_t size;

  while ((code = ggml_list_Int_next(env, shape, &dim)) == 1) {
    if (dim < 0) return failed(env, failure, "a shape holds no negative dimension");
    if (rank == GGML_MAX_DIMS) return failed(env, failure, "the ggml backend holds tensors of rank 4 at most");
    if (dim > 0 && count > INT64_MAX / 8 / dim) return failed(env, failure, "the tensor is too large");
    count *= dim;
    dims[rank++] = dim;
  }
  if (code < 0) return failed_with(failure, an_void(env));
  for (i = 0; i < rank; i++) ne[i] = dims[rank - 1 - i];

  if (!buffer_type_of(device, &buft, why, sizeof why)) return failed(env, failure, why);

  params.mem_size = ggml_tensor_overhead();
  params.mem_buffer = NULL;
  params.no_alloc = true;
  state->ctx = ggml_init(params);
  if (state->ctx == NULL) return failed(env, failure, "out of memory");
  state->t = ggml_new_tensor_4d(state->ctx, type, ne[0], ne[1], ne[2], ne[3]);

  size = ggml_backend_buft_get_alloc_size(buft, state->t);
  if (size > ggml_backend_buft_get_max_size(buft)) {
    return failed(env, failure, "the tensor is larger than the device's largest buffer");
  }
  if (size < ggml_backend_buft_get_alignment(buft)) size = ggml_backend_buft_get_alignment(buft);
  state->buffer = ggml_backend_buft_alloc_buffer(buft, size);
  if (state->buffer == NULL) return failed(env, failure, "the device could not allocate the tensor");
  if (ggml_backend_tensor_alloc(state->buffer, state->t, ggml_backend_buffer_get_base(state->buffer)) !=
      GGML_STATUS_SUCCESS) {
    return failed(env, failure, "the device could not place the tensor in its buffer");
  }
  return true;
}

/* `count` host values as the bytes of `type` (F32 or I32), in `bytes`. */
static void *host_bytes(const double *values, int64_t count, enum ggml_type type) {
  void *bytes = malloc((size_t)(count > 0 ? count : 1) * 4);
  int64_t i;

  if (bytes == NULL) return NULL;
  for (i = 0; i < count; i++) {
    if (type == GGML_TYPE_I32) ((int32_t *)bytes)[i] = (int32_t)values[i];
    else ((float *)bytes)[i] = (float)values[i];
  }
  return bytes;
}

static an_term store(an_env *env, struct ggml_TensorHandle_state *state, const double *values, an_term term) {
  int64_t count = ggml_nelements(state->t);
  void *bytes = host_bytes(values, count, state->t->type);

  if (bytes == NULL) return error_text(env, "out of memory");
  ggml_backend_tensor_set(state->t, bytes, 0, ggml_nbytes(state->t));
  free(bytes);
  return ggml_result_TensorHandle_ok(env, term);
}

an_term ggml_upload(an_env *env, int64_t device, ggml_list_Int shape, ggml_list_Float data, bool int32) {
  an_term term;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  an_term failure;
  int64_t count, i = 0;
  double *values, v;
  int code;

  if (!make_tensor(env, device, &shape, int32 ? GGML_TYPE_I32 : GGML_TYPE_F32, state, &failure)) return failure;
  count = ggml_nelements(state->t);
  values = (double *)malloc((size_t)(count > 0 ? count : 1) * sizeof(double));
  if (values == NULL) return error_text(env, "out of memory");
  while ((code = ggml_list_Float_next(env, &data, &v)) == 1 && i < count) values[i++] = v;
  if (code < 0) {
    free(values);
    return an_void(env);
  }
  if (i != count || code == 1) {
    free(values);
    return error_text(env, "upload needs exactly one value per element");
  }
  failure = store(env, state, values, term);
  free(values);
  return failure;
}

an_term ggml_filled(an_env *env, int64_t device, ggml_list_Int shape, double value, bool int32) {
  an_term term;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  an_term failure;
  int64_t count, i;
  double *values;

  if (!make_tensor(env, device, &shape, int32 ? GGML_TYPE_I32 : GGML_TYPE_F32, state, &failure)) return failure;
  count = ggml_nelements(state->t);
  values = (double *)malloc((size_t)(count > 0 ? count : 1) * sizeof(double));
  if (values == NULL) return error_text(env, "out of memory");
  for (i = 0; i < count; i++) values[i] = value;
  failure = store(env, state, values, term);
  free(values);
  return failure;
}

an_term ggml_readback(an_env *env, struct ggml_TensorHandle_state *t) {
  int64_t count, i;
  void *bytes;
  ggml_list_Float_builder out;
  an_term list;

  if (t->t == NULL) return error_text(env, "readback of a tensor that was never made");
  count = ggml_nelements(t->t);
  bytes = malloc(ggml_nbytes(t->t) > 0 ? ggml_nbytes(t->t) : 1);
  if (bytes == NULL) return error_text(env, "out of memory");
  ggml_backend_tensor_get(t->t, bytes, 0, ggml_nbytes(t->t));
  ggml_list_Float_begin(env, &out, (size_t)count);
  for (i = 0; i < count; i++) {
    if (t->t->type == GGML_TYPE_I32) ggml_list_Float_push(env, &out, (double)((int32_t *)bytes)[i]);
    else ggml_list_Float_push(env, &out, (double)((float *)bytes)[i]);
  }
  free(bytes);
  list = ggml_list_Float_end(env, &out);
  return ggml_result_list_Float_ok(env, list);
}

GGML_NATIVE_MODULE(NULL)
