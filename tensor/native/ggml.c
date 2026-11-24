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
#define _POSIX_C_SOURCE 200809L /* setenv, sysconf */
#endif
#if !defined(__cplusplus) && defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE /* sched_getaffinity, CPU_COUNT */
#endif
#define GGML_NATIVE_IMPLEMENTATION
#include "ggml_shared.h"
#include <math.h>
#ifdef __linux__
#include <sched.h>
#endif

#include "float_file.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

pthread_mutex_t tg_device_lock = PTHREAD_MUTEX_INITIALIZER;

/* Frees that wait for the device lock. A free hook runs on the scheduler, so
   it must never wait for a graph to finish: it queues its work here, and the
   next holder of the device lock does it. */
CC_LOCK_LEVEL(3) CC_NO_SYSCALL_UNDER() static pthread_mutex_t deferred_lock = PTHREAD_MUTEX_INITIALIZER;
static struct deferred *deferred_head = NULL;

void tg_defer(struct deferred *d) {
  pthread_mutex_lock(&deferred_lock);
  d->next = deferred_head;
  deferred_head = d;
  pthread_mutex_unlock(&deferred_lock);
}

void tg_lock_device(void) {
  struct deferred *d, *next;

  pthread_mutex_lock(&tg_device_lock);
  pthread_mutex_lock(&deferred_lock);
  d = deferred_head;
  deferred_head = NULL;
  pthread_mutex_unlock(&deferred_lock);
  for (; d != NULL; d = next) {
    next = d->next; /* `fn` may free the node itself */
    d->fn(d->arg);
  }
}

void tg_unlock_device(void) { pthread_mutex_unlock(&tg_device_lock); }

/* A CPU buffer is plain memory: Vulkan buffer operations take the lock, CPU
   uploads and readbacks do not, so CPU tensors move while a graph runs. Making
   a tensor (`tg_new_tensor`) waits for the lock even on the CPU, to look the
   device up. */
CC_ACQUIRES_IF(tg_device_lock, *held) static void lock_for(bool vulkan, bool *held) {
  *held = vulkan;
  if (vulkan) tg_lock_device();
}

CC_RELEASES_IF(tg_device_lock, held) static void unlock_for(bool held) {
  if (held) tg_unlock_device();
}

static void free_storage(void *arg) {
  struct tg_storage *s = (struct tg_storage *)arg;

  if (s->buffer != NULL) ggml_backend_buffer_free(s->buffer);
  if (s->ctx != NULL) ggml_free(s->ctx);
  free(s);
}

void tg_storage_retain(struct tg_storage *s) { __atomic_fetch_add(&s->refs, 1, __ATOMIC_RELAXED); }

void tg_storage_release(struct tg_storage *s) {
  if (__atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL) == 0) {
    s->link.fn = free_storage;
    s->link.arg = s;
    tg_defer(&s->link);
  }
}

/* A free hook runs on the scheduler, so it never waits for a graph: the memory goes when the last holder (this
   handle, or a graph bound to it) lets go, at the next hold of the device lock. */
static void ggml_TensorHandle_free(void *state) {
  struct ggml_TensorHandle_state *self = (struct ggml_TensorHandle_state *)state;

  if (self->store != NULL) tg_storage_release(self->store);
  memset(self, 0, sizeof *self);
}
#define GGML_TensorHandle_FREE ggml_TensorHandle_free
#define GGML_Lease_FREE ggml_Lease_free
#define GGML_Pin_FREE ggml_Pin_free

/* ggml-Vulkan converts F32 `mul_mat` inputs to fp16 on a device that has
   fp16, unless GGML_VK_DISABLE_F16 is set when ggml first enumerates its
   devices (any registry call does), and no per-op switch exists
   (spike/ggml-graph/README.md). So the precision belongs to the process:
   `ggml_vulkan_fp16` records the choice in `want_fp16` (fp16 unless asked
   otherwise), and the first registry call of this file sets the variable
   from it (set only for F32) and fixes it. Without a `ggml_vulkan_fp16` call, a
   GGML_VK_DISABLE_F16 the user exported wins: the process is F32. Call under
   the device lock. */
static bool want_fp16 = true;
static bool chosen = false;
static bool precision_fixed = false;

static void fix_precision(void) {
  if (precision_fixed) return;
  /* This runs on a compute-pool thread, under the device lock, before ggml
     reads the variable; nothing else of this module reads the environment. */
  if (!chosen && getenv("GGML_VK_DISABLE_F16") != NULL) want_fp16 = false;
  if (want_fp16) unsetenv("GGML_VK_DISABLE_F16");
  else setenv("GGML_VK_DISABLE_F16", "1", 1);
  precision_fixed = true;
}

/* The CPU and up to eight Vulkan devices, opened once for the process. */
#define MAX_DEVICES 9
static ggml_backend_t backends[MAX_DEVICES];

an_term tg_error(an_env *env, const char *message) { return an_error(env, message, strlen(message)); }

/* The Vulkan device `index`, counting only devices that are not a CPU (a
   software rasterizer), or NULL; `count` gets how many there are. NULL in a
   CPU-only build. Call under the lock. */
static ggml_backend_dev_t vulkan_device(size_t index, size_t *count) {
  ggml_backend_reg_t reg;
  ggml_backend_dev_t found = NULL, dev;
  size_t i, n;

  fix_precision();
  reg = ggml_backend_reg_by_name("Vulkan");
  n = reg == NULL ? 0 : ggml_backend_reg_dev_count(reg);
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

  tg_lock_device();
  vulkan_device(SIZE_MAX, &n);
  tg_unlock_device();
  return an_int(env, (int64_t)n);
}

/* The ggml device of `device` (-1: CPU, i >= 0: Vulkan device i), or NULL
   with the reason in `why`. Call under the device lock. */
static ggml_backend_dev_t device_of(int64_t device, char *why, size_t len) {
  ggml_backend_dev_t dev = NULL;
  size_t n = 0;

  fix_precision();
  if (device == -1) {
    dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == NULL) snprintf(why, len, "this ggml build has no CPU backend");
  } else if (device < 0 || (dev = vulkan_device((size_t)device, &n)) == NULL) {
    if (ggml_backend_reg_by_name("Vulkan") == NULL) {
      snprintf(why, len, "no Vulkan device %lld: this ggml build has no Vulkan backend (`tools/build-engine.sh vulkan`)",
               (long long)device);
    } else {
      snprintf(why, len, "no Vulkan device %lld: ggml sees %zu", (long long)device, n);
    }
  } else if (device >= MAX_DEVICES - 1) {
    snprintf(why, len, "Vulkan device %lld: this backend opens devices 0...%d", (long long)device, MAX_DEVICES - 1);
    dev = NULL;
  }
  return dev;
}

ggml_backend_t tg_backend(int64_t device, char *why, size_t len) {
  ggml_backend_dev_t dev;
  ggml_backend_t backend;
  long cores;

  if (device >= -1 && device < MAX_DEVICES - 1 && backends[device + 1] != NULL) return backends[device + 1];
  if ((dev = device_of(device, why, len)) == NULL) return NULL;
  backend = ggml_backend_dev_init(dev, NULL);
  if (backend == NULL) {
    snprintf(why, len, "ggml could not open %s", ggml_backend_dev_name(dev));
    return NULL;
  }
  if (device == -1) {
    cores = sysconf(_SC_NPROCESSORS_ONLN);
    ggml_backend_cpu_set_n_threads(backend, cores > 0 ? (int)cores : 4);
  }
  backends[device + 1] = backend;
  return backend;
}

/* Read once a device is open, when the choice is fixed: it never changes after the first registry call. */
bool tg_fp16(void) { return want_fp16; }

an_term ggml_vulkan_fp16(an_env *env, bool enabled) {
  bool conflict, fixed;

  tg_lock_device();
  conflict = precision_fixed && want_fp16 != enabled;
  fixed = want_fp16;
  if (!conflict) {
    want_fp16 = enabled;
    chosen = true;
  }
  tg_unlock_device();
  if (conflict) {
    return tg_error(env, fixed ? "this process chose fp16 mul_mat for Vulkan already; choose before the first ggml call"
                               : "this process chose F32 mul_mat for Vulkan already; choose before the first ggml call");
  }
  return ggml_result_Void_ok(env, an_void(env));
}

static bool failed_with(an_term *failure, an_term term) {
  *failure = term;
  return false;
}

static bool failed(an_env *env, an_term *failure, const char *message) {
  return failed_with(failure, tg_error(env, message));
}

bool tg_new_tensor(an_env *env, int64_t device, const int64_t ne[4], enum ggml_type type,
                   struct ggml_TensorHandle_state *state, an_term *failure) {
  char why[160];
  ggml_backend_dev_t dev;
  ggml_backend_buffer_type_t buft = NULL;
  struct ggml_init_params params;
  size_t size;
  enum ggml_status placed;
  bool held;

  tg_lock_device();
  dev = device_of(device, why, sizeof why);
  if (dev != NULL) buft = ggml_backend_dev_buffer_type(dev);
  tg_unlock_device();
  if (dev == NULL) return failed(env, failure, why);
  if (buft == NULL) return failed(env, failure, "the device has no buffer type");
  state->vulkan = device >= 0;
  state->store = (struct tg_storage *)calloc(1, sizeof *state->store);
  if (state->store == NULL) return failed(env, failure, "out of memory");
  state->store->refs = 1;

  params.mem_size = ggml_tensor_overhead();
  params.mem_buffer = NULL;
  params.no_alloc = true;
  state->ctx = ggml_init(params);
  state->store->ctx = state->ctx;
  if (state->ctx == NULL) return failed(env, failure, "out of memory");
  state->t = ggml_new_tensor_4d(state->ctx, type, ne[0], ne[1], ne[2], ne[3]);

  size = ggml_backend_buft_get_alloc_size(buft, state->t);
  if (size > ggml_backend_buft_get_max_size(buft)) {
    return failed(env, failure, "the tensor is larger than the device's largest buffer");
  }
  if (size < ggml_backend_buft_get_alignment(buft)) size = ggml_backend_buft_get_alignment(buft);
  lock_for(state->vulkan, &held);
  state->buffer = ggml_backend_buft_alloc_buffer(buft, size);
  state->store->buffer = state->buffer;
  placed = state->buffer == NULL ? GGML_STATUS_ALLOC_FAILED
                                 : ggml_backend_tensor_alloc(state->buffer, state->t,
                                                             ggml_backend_buffer_get_base(state->buffer));
  unlock_for(held);
  if (state->buffer == NULL) return failed(env, failure, "the device could not allocate the tensor");
  if (placed != GGML_STATUS_SUCCESS) return failed(env, failure, "the device could not place the tensor in its buffer");
  return true;
}

/* A new tensor of the row-major `shape`, as `tg_new_tensor`. */
static bool make_tensor(an_env *env, int64_t device, ggml_list_Int *shape, enum ggml_type type,
                        struct ggml_TensorHandle_state *state, an_term *failure) {
  int64_t dims[GGML_MAX_DIMS], ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  int64_t dim, count = 1;
  int rank = 0, code, i;

  while ((code = ggml_list_Int_next(env, shape, &dim)) == 1) {
    if (dim < 0) return failed(env, failure, "a shape holds no negative dimension");
    if (rank == GGML_MAX_DIMS) return failed(env, failure, "the ggml backend holds tensors of rank 4 at most");
    if (dim > 0 && count > INT64_MAX / 8 / dim) return failed(env, failure, "the tensor is too large");
    count *= dim;
    dims[rank++] = dim;
  }
  if (code < 0) return failed_with(failure, an_void(env));
  for (i = 0; i < rank; i++) ne[i] = dims[rank - 1 - i];
  if (ggml_is_quantized(type) && ne[0] % ggml_blck_size(type) != 0) {
    return failed(env, failure, "a quantized tensor holds whole blocks of 32 values in each row");
  }
  return tg_new_tensor(env, device, ne, type, state, failure);
}

/* 2^24: the integers F32 holds exactly. */
#define EXACT 16777216.0f

const char *tg_copy(struct ggml_tensor *src, struct ggml_tensor *dst) {
  int64_t n = ggml_nelements(src), i;
  void *bytes;
  const char *why = NULL;

  if (n != ggml_nelements(dst)) return "the tensors have different element counts";
  if (src->type == dst->type) {
    ggml_backend_tensor_copy(src, dst);
    return NULL;
  }
  if (src->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F16 || ggml_is_quantized(src->type) ||
      ggml_is_quantized(dst->type)) {
    return "an F16 or quantized tensor is not converted to or from another type";
  }
  bytes = malloc(ggml_nbytes(src) > 0 ? ggml_nbytes(src) : 1);
  if (bytes == NULL) return "out of memory";
  ggml_backend_tensor_get(src, bytes, 0, ggml_nbytes(src));
  for (i = 0; i < n && why == NULL; i++) {
    if (src->type == GGML_TYPE_I32) {
      int32_t v = ((int32_t *)bytes)[i];

      if (v > (int32_t)EXACT || v < -(int32_t)EXACT) why = "an I32 value beyond 2^24 is not exact in F32";
      ((float *)bytes)[i] = (float)v;
    } else {
      float v = ((float *)bytes)[i];

      if (!(v <= EXACT && v >= -EXACT)) why = "an I32 result beyond 2^24 (or not finite) is not exact in F32";
      else ((int32_t *)bytes)[i] = (int32_t)v;
    }
  }
  if (why == NULL) ggml_backend_tensor_set(dst, bytes, 0, ggml_nbytes(dst));
  free(bytes);
  return why;
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
  bool held;

  lock_for(state->vulkan, &held);
  ggml_backend_tensor_set(state->t, bytes, 0, ggml_nbytes(state->t));
  unlock_for(held);
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
  if (bytes == NULL) return tg_error(env, "out of memory");
  while ((code = ggml_list_Float_next(env, &data, &v)) == 1 && i < count) put(bytes, i++, type, v);
  if (code < 0) {
    free(bytes);
    return an_void(env);
  }
  if (i != count || code == 1) {
    free(bytes);
    return tg_error(env, "upload needs exactly one value per element");
  }
  return store(env, state, bytes, term);
}

/* Where the floats of a tensor come from: a byte range of a file that C reads, or the bytes Agentica read. */
struct load_source {
  const char *path;
  size_t path_len;
  int64_t offset;
  const uint8_t *data;
  size_t len;
};

/* The F16 value of every float of a chunk, into the tensor's memory: a value up to 65504 in magnitude, the largest F16,
   is kept (rounded), a larger finite one is an error naming its element, and an infinity passes through. */
#define HALF_MAX 65504.0f

struct half_sink {
  ggml_fp16_t *half;
};

/* Threads call it at once, each for its own `first` and `n`: it writes only that part of `half`. */
static bool half_chunk(void *ctx, const float *values, int64_t first, int64_t n, char *why, size_t why_len) {
  struct half_sink *sink = (struct half_sink *)ctx;
  unsigned bad = 0;
  int64_t i;

  /* no branch, so that the compiler can vectorize the scan; only a chunk that has a bad value is searched */
  for (i = 0; i < n; i++) bad |= (unsigned)(fabsf(values[i]) > HALF_MAX) & (unsigned)(fabsf(values[i]) < INFINITY);
  if (bad != 0) {
    for (i = 0; i < n; i++) {
      if (fabsf(values[i]) > HALF_MAX && fabsf(values[i]) < INFINITY) {
        snprintf(why, why_len, "the value %g at element %lld is outside what F16 holds (up to 65504)", (double)values[i],
                 (long long)first + (long long)i);
        return false;
      }
    }
  }
  ggml_fp32_to_fp16_row(values, sink->half + first, n);
  return true;
}

/* The threads that convert a file to F16: the cores this process may run on (the cpuset of a container), else the
   online ones, at most 8 (the reads are of one file on one disk). */
#define HALF_MAX_THREADS 8

static int half_threads(void) {
  long cores = sysconf(_SC_NPROCESSORS_ONLN);

#ifdef __linux__
  cpu_set_t allowed;

  if (sched_getaffinity(0, sizeof allowed, &allowed) == 0 && CPU_COUNT(&allowed) > 0) cores = CPU_COUNT(&allowed);
#endif
  if (cores < 1) return 1;
  return cores > HALF_MAX_THREADS ? HALF_MAX_THREADS : (int)cores;
}

/* What a loaded tensor holds, the code of `Types.held_code`: F32 or F16 converted from the floats of the source, or
   the Q8_0 blocks of a Q8_0 source as they are. */
enum { HELD_F32 = 0, HELD_F16 = 1, HELD_Q8_0 = 2 };

/* The tensor is filled where the source is read: in place on a host buffer (the CPU), through a host copy for
   a device that has its own memory (Vulkan). */
static an_term load_from(an_env *env, int64_t device, ggml_list_Int shape, const struct load_source *src,
                         int64_t format, int64_t held) {
  an_term term, failure;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  char why[FF_PATH_MAX + 160];
  bool in_place, locked, read;
  enum ggml_type type = held == HELD_F16 ? GGML_TYPE_F16 : (held == HELD_Q8_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
  void *host;

  if (held < HELD_F32 || held > HELD_Q8_0) return tg_error(env, "unknown held type code");
  if ((held == HELD_Q8_0) != (format == FF_Q8_0) && held != HELD_F32) {
    return tg_error(env, "a Q8_0 tensor is read from a Q8_0 range, and a Q8_0 range is read as F32 or kept as Q8_0");
  }
  if (!make_tensor(env, device, &shape, type, state, &failure)) return failure;
  in_place = !state->vulkan && ggml_backend_buffer_is_host(state->buffer);
  host = in_place ? state->t->data : host_bytes(state->t);
  if (host == NULL) return tg_error(env, "out of memory");
  if (held == HELD_Q8_0) {
    int64_t bytes = (int64_t)ggml_nbytes(state->t);

    if (src->data != NULL) {
      read = src->len == (size_t)bytes;
      if (read) memcpy(host, src->data, src->len);
      else snprintf(why, sizeof why, "%zu bytes are not the %lld of the tensor", src->len, (long long)bytes);
    } else {
      read = ff_read_bytes(src->path, src->path_len, src->offset, bytes, (uint8_t *)host, why, sizeof why);
    }
  } else if (held == HELD_F16) {
    struct half_sink sink = {(ggml_fp16_t *)host};

    if (src->data != NULL) {
      read = ff_each_bytes(src->data, src->len, ggml_nelements(state->t), format, half_chunk, &sink, why, sizeof why);
    } else {
      read = ff_each(src->path, src->path_len, src->offset, ggml_nelements(state->t), format, half_chunk, &sink,
                     half_threads(), why, sizeof why);
    }
  } else if (src->data != NULL) {
    read = ff_from_bytes(src->data, src->len, ggml_nelements(state->t), format, (float *)host, why, sizeof why);
  } else {
    read = ff_read(src->path, src->path_len, src->offset, ggml_nelements(state->t), format, (float *)host, why,
                   sizeof why);
  }
  if (!read) {
    if (!in_place) free(host);
    return tg_error(env, why);
  }
  if (!in_place) {
    lock_for(state->vulkan, &locked);
    ggml_backend_tensor_set(state->t, host, 0, ggml_nbytes(state->t));
    unlock_for(locked);
    free(host);
  }
  return ggml_result_TensorHandle_ok(env, term);
}

an_term ggml_load(an_env *env, int64_t device, ng_str path, int64_t offset, ggml_list_Int shape, int64_t format,
                  int64_t held) {
  struct load_source src = {path.data, path.len, offset, NULL, 0};

  return load_from(env, device, shape, &src, format, held);
}

an_term ggml_from_bytes(an_env *env, int64_t device, ng_buf data, ggml_list_Int shape, int64_t format, int64_t held) {
  struct load_source src = {NULL, 0, 0, data.data, data.len};

  return load_from(env, device, shape, &src, format, held);
}

/* What `ggml_convert_file` writes: an open file, where the values go, and the format they take. */
struct convert_sink {
  int fd;
  int64_t offset;
  int64_t format;
};

static bool pwrite_all(int fd, const void *from, size_t len, int64_t at) {
  const uint8_t *p = (const uint8_t *)from;
  ssize_t put;

  while (len > 0) {
    put = pwrite(fd, p, len, (off_t)at);
    if (put < 0 && errno == EINTR) continue;
    if (put <= 0) return false;
    p += put;
    at += put;
    len -= (size_t)put;
  }
  return true;
}

/* Writes the values of a chunk at their place in the file as F32, F16 (a value beyond 65504 is an error, as for a half
   weight) or BF16 (round to nearest even). Threads call it at once, each for its own `first` and `n`. */
static bool convert_chunk(void *ctx, const float *values, int64_t first, int64_t n, char *why, size_t why_len) {
  const struct convert_sink *sink = (const struct convert_sink *)ctx;
  size_t width = ff_width(sink->format);
  uint16_t *out = NULL;
  bool ok;

  if (sink->format == FF_F32) {
    ok = pwrite_all(sink->fd, values, (size_t)n * 4, sink->offset + first * 4);
    if (!ok) snprintf(why, why_len, "the file could not be written (no space left?)");
    return ok;
  }
  if (sink->format == FF_F16) {
    int64_t i;

    for (i = 0; i < n; i++) {
      if (fabsf(values[i]) > HALF_MAX && fabsf(values[i]) < INFINITY) {
        snprintf(why, why_len, "the value %g at element %lld is outside what F16 holds (up to 65504)",
                 (double)values[i], (long long)first + (long long)i);
        return false;
      }
    }
  }
  out = (uint16_t *)calloc((size_t)n + 1, width);
  if (out == NULL) {
    snprintf(why, why_len, "out of memory");
    return false;
  }
  if (sink->format == FF_F16) ggml_fp32_to_fp16_row(values, (ggml_fp16_t *)out, n);
  else ggml_fp32_to_bf16_row_ref(values, (ggml_bf16_t *)out, n);
  ok = pwrite_all(sink->fd, out, (size_t)n * width, sink->offset + first * (int64_t)width);
  if (!ok) snprintf(why, why_len, "the file could not be written (no space left?)");
  free(out);
  return ok;
}

/* The same format on both sides: the bytes are copied as they are, so a NaN keeps its payload and nothing is rounded. */
static bool copy_range(const char *src, size_t src_len, int64_t src_offset, int64_t count, int fd, int64_t offset,
                       size_t width, char *why, size_t why_len) {
  int64_t done = 0, n;
  uint8_t *raw;
  int in = ff_open_range(src, src_len, src_offset, count, width, why, why_len);

  if (in < 0) return false;
  raw = (uint8_t *)calloc(FF_CHUNK, 1);
  if (raw == NULL) {
    close(in);
    snprintf(why, why_len, "out of memory");
    return false;
  }
  while (done < count * (int64_t)width) {
    n = count * (int64_t)width - done;
    if (n > (int64_t)FF_CHUNK) n = (int64_t)FF_CHUNK;
    if (!ff_pread(in, raw, (size_t)n, src_offset + done)) {
      snprintf(why, why_len, "%.*s ended while it was read (it shrank?)", (int)src_len, src);
      break;
    }
    if (!pwrite_all(fd, raw, (size_t)n, offset + done)) {
      snprintf(why, why_len, "the file could not be written (no space left?)");
      break;
    }
    done += n;
  }
  free(raw);
  close(in);
  return done == count * (int64_t)width;
}

an_term ggml_convert_file(an_env *env, ng_str src, int64_t src_offset, int64_t src_format, int64_t count, ng_str dst,
                          int64_t dst_offset, int64_t dst_format) {
  struct convert_sink sink;
  char why[FF_PATH_MAX + 160];
  char name[FF_PATH_MAX];
  struct stat info;
  bool ok;
  int fd;

  if (ff_width(dst_format) == 0 || ff_width(src_format) == 0) return tg_error(env, "unknown float format code");
  if (dst_offset < 0 || count < 0 || count > (INT64_MAX - dst_offset) / 4) {
    return tg_error(env, "the range to write is not inside what a file holds");
  }
  if (dst.len == 0 || dst.len >= sizeof name || memchr(dst.data, 0, dst.len) != NULL) {
    return tg_error(env, "not a usable file path");
  }
  memcpy(name, dst.data, dst.len);
  name[dst.len] = '\0';
  fd = open(name, O_WRONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    snprintf(why, sizeof why, "cannot open %s: %s", name, ff_open_why(errno));
    return tg_error(env, why);
  }
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    close(fd);
    snprintf(why, sizeof why, "%s is not a regular file", name);
    return tg_error(env, why);
  }
  sink.fd = fd;
  sink.offset = dst_offset;
  sink.format = dst_format;
  if (src_format == dst_format) {
    ok = copy_range(src.data, src.len, src_offset, count, fd, dst_offset, ff_width(src_format), why, sizeof why);
  } else {
    ok = ff_each(src.data, src.len, src_offset, count, src_format, convert_chunk, &sink, half_threads(), why,
                 sizeof why);
  }
  close(fd);
  if (!ok) return tg_error(env, why);
  return ggml_result_Void_ok(env, an_void(env));
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
  if (bytes == NULL) return tg_error(env, "out of memory");
  for (i = 0; i < count; i++) put(bytes, i, type, value);
  return store(env, state, bytes, term);
}

an_term ggml_transfer(an_env *env, struct ggml_TensorHandle_state *t, int64_t device) {
  an_term term, failure;
  struct ggml_TensorHandle_state *state = ggml_TensorHandle_new(env, &term);
  bool vulkan, held;

  if (t->t == NULL) return tg_error(env, "transfer of a tensor that was never made");
  if (!tg_new_tensor(env, device, t->t->ne, t->t->type, state, &failure)) return failure;
  vulkan = t->vulkan || state->vulkan;
  lock_for(vulkan, &held);
  ggml_backend_tensor_copy(t->t, state->t);
  unlock_for(held);
  return ggml_result_TensorHandle_ok(env, term);
}

an_term ggml_readback(an_env *env, struct ggml_TensorHandle_state *t) {
  int64_t count, i;
  void *bytes;
  bool held;
  ggml_list_Float_builder out;
  an_term list;

  if (t->t == NULL) return tg_error(env, "readback of a tensor that was never made");
  count = ggml_nelements(t->t);
  bytes = host_bytes(t->t);
  if (bytes == NULL) return tg_error(env, "out of memory");
  lock_for(t->vulkan, &held);
  ggml_backend_tensor_get(t->t, bytes, 0, ggml_nbytes(t->t));
  unlock_for(held);
  if (ggml_is_quantized(t->t->type)) {
    float *values = (float *)malloc((size_t)(count > 0 ? count : 1) * sizeof(float));

    if (values == NULL) {
      free(bytes);
      return tg_error(env, "out of memory");
    }
    ggml_get_type_traits(t->t->type)->to_float(bytes, values, count);
    free(bytes);
    bytes = values;
  }
  ggml_list_Float_begin(env, &out, (size_t)count);
  for (i = 0; i < count; i++) {
    if (t->t->type == GGML_TYPE_I32) ggml_list_Float_push(env, &out, (double)((int32_t *)bytes)[i]);
    else if (t->t->type == GGML_TYPE_F16) ggml_list_Float_push(env, &out, (double)ggml_fp16_to_fp32(((ggml_fp16_t *)bytes)[i]));
    else ggml_list_Float_push(env, &out, (double)((float *)bytes)[i]);
  }
  free(bytes);
  list = ggml_list_Float_end(env, &out);
  return ggml_result_list_Float_ok(env, list);
}

GGML_NATIVE_MODULE(NULL)
