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

/* One lock for the registry and for every Vulkan buffer operation (allocate,
   set, get, free): compute tasks may touch Vulkan buffers at once, and one
   lock is the simple way to keep ggml-Vulkan's transfers from overlapping.
   CPU buffers are plain memory and need none. */
static pthread_mutex_t device_lock = PTHREAD_MUTEX_INITIALIZER;

struct ggml_TensorHandle_state {
  struct ggml_context *ctx;
  struct ggml_tensor *t;
  ggml_backend_buffer_t buffer;
  bool vulkan;
};

static void lock_for(bool vulkan) {
  if (vulkan) pthread_mutex_lock(&device_lock);
}

static void unlock_for(bool vulkan) {
  if (vulkan) pthread_mutex_unlock(&device_lock);
}

static void ggml_TensorHandle_free(void *state) {
  struct ggml_TensorHandle_state *self = (struct ggml_TensorHandle_state *)state;

  if (self->buffer != NULL) {
    lock_for(self->vulkan);
    ggml_backend_buffer_free(self->buffer);
    unlock_for(self->vulkan);
  }
  if (self->ctx != NULL) ggml_free(self->ctx);
  memset(self, 0, sizeof *self);
}
#define GGML_TensorHandle_FREE ggml_TensorHandle_free

static an_term error_text(an_env *env, const char *message) { return an_error(env, message, strlen(message)); }

/* The Vulkan device `index`, counting only devices that are not a CPU (a
   software rasterizer), or NULL; `count` gets how many there are. NULL in a
   CPU-only build. Call under the lock. */
static ggml_backend_dev_t vulkan_device(size_t index, size_t *count) {
  ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
  ggml_backend_dev_t found = NULL, dev;
  size_t i, n = reg == NULL ? 0 : ggml_backend_reg_dev_count(reg);

  *count = 0;
  for (i = 0; i < n; i++) {
    dev = ggml_backend_reg_dev_get(reg, i);
    if (dev == NULL || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
    if (*count == index) found = dev;
    (*count)++;
  }
  return found;
}

an_term ggml_vulkan_devices(an_env *env) {
  size_t n;

  pthread_mutex_lock(&device_lock);
  vulkan_device(SIZE_MAX, &n);
  pthread_mutex_unlock(&device_lock);
  return an_int(env, (int64_t)n);
}

/* The buffer type of `device` (-1: CPU, i >= 0: Vulkan device i), or false
   with the reason in `why`. */
static bool buffer_type_of(int64_t device, ggml_backend_buffer_type_t *out, char *why, size_t len) {
  ggml_backend_dev_t dev = NULL;
  size_t n;

  pthread_mutex_lock(&device_lock);
  if (device == -1) {
    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == NULL) snprintf(why, len, "this ggml build has no CPU backend");
  } else if (device < 0) {
    snprintf(why, len, "a Vulkan device index is >= 0, got %lld", (long long)device);
  } else if ((dev = vulkan_device((size_t)device, &n)) == NULL) {
    if (ggml_backend_reg_by_name("Vulkan") == NULL) {
      snprintf(why, len, "no Vulkan device %lld: this ggml build has no Vulkan backend (`tools/build-engine.sh vulkan`)",
               (long long)device);
    } else {
      snprintf(why, len, "no Vulkan device %lld: ggml sees %zu", (long long)device, n);
    }
  }
  *out = dev == NULL ? NULL : ggml_backend_dev_buffer_type(dev);
  pthread_mutex_unlock(&device_lock);
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
  enum ggml_status placed;

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
  state->vulkan = device >= 0;

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
  lock_for(state->vulkan);
  state->buffer = ggml_backend_buft_alloc_buffer(buft, size);
  placed = state->buffer == NULL ? GGML_STATUS_ALLOC_FAILED
                                 : ggml_backend_tensor_alloc(state->buffer, state->t,
                                                             ggml_backend_buffer_get_base(state->buffer));
  unlock_for(state->vulkan);
  if (state->buffer == NULL) return failed(env, failure, "the device could not allocate the tensor");
  if (placed != GGML_STATUS_SUCCESS) return failed(env, failure, "the device could not place the tensor in its buffer");
  return true;
}

/* Host bytes for every element of `t` (4 per element: F32 or I32). */
static void *host_bytes(struct ggml_tensor *t) {
  size_t n = ggml_nbytes(t);
  return malloc(n > 0 ? n : 1);
}

static void put(void *bytes, int64_t i, enum ggml_type type, double v) {
  if (type == GGML_TYPE_I32) ((int32_t *)bytes)[i] = (int32_t)v;
  else ((float *)bytes)[i] = (float)v;
}

/* Copies `bytes` into the tensor and answers it. */
static an_term store(an_env *env, struct ggml_TensorHandle_state *state, void *bytes, an_term term) {
  lock_for(state->vulkan);
  ggml_backend_tensor_set(state->t, bytes, 0, ggml_nbytes(state->t));
  unlock_for(state->vulkan);
  free(bytes);
  return ggml_result_TensorHandle_ok(env, term);
}

an_term ggml_upload(an_env *env, int64_t device, ggml_list_Int shape, ggml_list_Float data, bool int32) {
  an_term term, failure;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  enum ggml_type type = int32 ? GGML_TYPE_I32 : GGML_TYPE_F32;
  int64_t count, i = 0;
  void *bytes;
  double v;
  int code;

  if (!make_tensor(env, device, &shape, type, state, &failure)) return failure;
  count = ggml_nelements(state->t);
  bytes = host_bytes(state->t);
  if (bytes == NULL) return error_text(env, "out of memory");
  while ((code = ggml_list_Float_next(env, &data, &v)) == 1 && i < count) put(bytes, i++, type, v);
  if (code < 0) {
    free(bytes);
    return an_void(env);
  }
  if (i != count || code == 1) {
    free(bytes);
    return error_text(env, "upload needs exactly one value per element");
  }
  return store(env, state, bytes, term);
}

an_term ggml_filled(an_env *env, int64_t device, ggml_list_Int shape, double value, bool int32) {
  an_term term, failure;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  enum ggml_type type = int32 ? GGML_TYPE_I32 : GGML_TYPE_F32;
  int64_t count, i;
  void *bytes;

  if (!make_tensor(env, device, &shape, type, state, &failure)) return failure;
  count = ggml_nelements(state->t);
  bytes = host_bytes(state->t);
  if (bytes == NULL) return error_text(env, "out of memory");
  for (i = 0; i < count; i++) put(bytes, i, type, value);
  return store(env, state, bytes, term);
}

an_term ggml_readback(an_env *env, struct ggml_TensorHandle_state *t) {
  int64_t count, i;
  void *bytes;
  ggml_list_Float_builder out;
  an_term list;

  if (t->t == NULL) return error_text(env, "readback of a tensor that was never made");
  count = ggml_nelements(t->t);
  bytes = host_bytes(t->t);
  if (bytes == NULL) return error_text(env, "out of memory");
  lock_for(t->vulkan);
  ggml_backend_tensor_get(t->t, bytes, 0, ggml_nbytes(t->t));
  unlock_for(t->vulkan);
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
