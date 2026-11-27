/* The thin C layer of `src/xla_native.agc`: XLA on the CPU through the PJRT C API of a plugin loaded with `dlopen`
   (`tensor/DESIGN.md` §7). Nothing of XLA is linked at build time; `pjrt_c_api.h` is the header of the API the
   plugin speaks, vendored with its notice: the header is PJRT 0.81 and a plugin may speak a later minor (0.117 in the
   release xla-c9276de87e15), which `load_plugin` accepts.

   One client and its first device serve the process. A buffer and a program are refcounted, so a run (`Call`) holds
   what it reads even when its handle is closed meanwhile. */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#define XLA_NATIVE_IMPLEMENTATION
#include "cchecker.h"
#include "xla_native.h"

#include "float_file.h"
#include "pjrt_c_api.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The dtype codes of `src/xla.agc` (`code_of`), in order. */
enum { DT_F32, DT_F64, DT_I32, DT_I64, DT_I16, DT_I8, DT_U8, DT_BOOL, DT_COUNT };
static const PJRT_Buffer_Type TYPES[DT_COUNT] = {
    PJRT_Buffer_Type_F32, PJRT_Buffer_Type_F64, PJRT_Buffer_Type_S32, PJRT_Buffer_Type_S64,
    PJRT_Buffer_Type_S16, PJRT_Buffer_Type_S8,  PJRT_Buffer_Type_U8,  PJRT_Buffer_Type_PRED,
};
static const size_t SIZES[DT_COUNT] = {4, 8, 4, 8, 2, 1, 1, 1};

#define MAX_RANK 16

/* Set once by `xla_open`, under `open_lock`; `api` is published last (release), so a reader that sees it sees the
   client and the device. */
CC_LOCK_LEVEL(1) static pthread_mutex_t open_lock = PTHREAD_MUTEX_INITIALIZER;
static const PJRT_Api *api; /* atomic */
static PJRT_Client *client;
static PJRT_Device *device;
static char *opened_path;
static const PJRT_Api *initialized; /* the plugin whose Plugin_Initialize ran, under `open_lock` */
static int64_t live_buffers; /* atomic: for a test (`xla_buffers`) */

struct buffer {
  int refs; /* atomic */
  PJRT_Buffer *b;
};

/* A compiled program. The cache holds one count while it lists it, and each `Program` and `Call` one more. */
struct program {
  int refs; /* atomic */
  PJRT_LoadedExecutable *exe;
  size_t outputs;
  char *text; /* the key, with `len` bytes */
  size_t len;
  int64_t *donated; /* the inputs the program updates in place, `ndonated` of them */
  size_t ndonated;
  struct program *next; /* the cache's list, most recently used first, under `cache_lock` */
};

/* The programs compiled lately, by their text: a trace compiled again (`Trace.eval` per step, a `Trace.compile` after
   `Trace.close`) finds its program here. At most CACHE_SIZE; the least recently used one that only the cache holds
   goes first. */
#define CACHE_SIZE 32
CC_LOCK_LEVEL(2) CC_NO_SYSCALL_UNDER() static pthread_mutex_t cache_lock = PTHREAD_MUTEX_INITIALIZER;
static struct program *cache_head;
static int cache_count;
static int64_t compiled_count; /* atomic: for a test (`xla_compiles`) */

struct xla_Buffer_state {
  struct buffer *buffer;
};

struct xla_Program_state {
  struct program *program;
};

struct xla_Call_state {
  bool ran;
  struct program *program;
  struct buffer **args;
  size_t count, cap;
};

static const PJRT_Api *loaded(void) { return __atomic_load_n(&api, __ATOMIC_ACQUIRE); }

/* Copies the message of `err` into `out` and destroys it. */
static void message_of(const PJRT_Api *a, PJRT_Error *err, char *out, size_t size) {
  PJRT_Error_Message_Args m = {.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, .error = err};
  PJRT_Error_Destroy_Args d = {.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, .error = err};

  a->PJRT_Error_Message(&m);
  snprintf(out, size, "%.*s", (int)(m.message_size < 1000 ? m.message_size : 1000), m.message);
  a->PJRT_Error_Destroy(&d);
}

static an_term failed(an_env *env, const char *message) { return an_error(env, message, strlen(message)); }

/* The error of a PJRT call, `what: message`. */
static an_term pjrt_failed(an_env *env, PJRT_Error *err, const char *what) {
  char text[1024], line[1200];

  message_of(loaded(), err, text, sizeof text);
  snprintf(line, sizeof line, "%s: %s", what, text);
  return failed(env, line);
}

/* Waits for `ev` and destroys it; the error of the event, or NULL. */
static PJRT_Error *await_event(PJRT_Event *ev) {
  const PJRT_Api *a = loaded();
  PJRT_Error *err;

  if (ev == NULL) return NULL;
  PJRT_Event_Await_Args w = {.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = ev};
  err = a->PJRT_Event_Await(&w);
  PJRT_Event_Destroy_Args d = {.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = ev};
  a->PJRT_Event_Destroy(&d);
  return err;
}

static void buffer_release(struct buffer *b) {
  if (b == NULL || __atomic_sub_fetch(&b->refs, 1, __ATOMIC_ACQ_REL) > 0) return;
  PJRT_Buffer_Destroy_Args d = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = b->b};
  PJRT_Error *err = loaded()->PJRT_Buffer_Destroy(&d);
  if (err != NULL) {
    char ignored[8];
    message_of(loaded(), err, ignored, sizeof ignored);
  }
  free(b);
  __atomic_sub_fetch(&live_buffers, 1, __ATOMIC_RELAXED);
}

/* A new buffer of one count over `pb`, or NULL. */
static struct buffer *buffer_new(PJRT_Buffer *pb) {
  struct buffer *b = (struct buffer *)malloc(sizeof *b);

  if (b == NULL) return NULL;
  b->refs = 1;
  b->b = pb;
  __atomic_add_fetch(&live_buffers, 1, __ATOMIC_RELAXED);
  return b;
}

static void program_release(struct program *p) {
  if (p == NULL || __atomic_sub_fetch(&p->refs, 1, __ATOMIC_ACQ_REL) > 0) return;
  PJRT_LoadedExecutable_Destroy_Args d = {.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
                                          .executable = p->exe};
  PJRT_Error *err = loaded()->PJRT_LoadedExecutable_Destroy(&d);
  if (err != NULL) {
    char ignored[8];
    message_of(loaded(), err, ignored, sizeof ignored);
  }
  free(p->text);
  free(p->donated);
  free(p);
}

/* The listed program of `text` and `outputs` under `cache_lock`, moved to the front, or NULL. */
static struct program *cache_lookup(const char *text, size_t len, size_t outputs) {
  struct program **at, *p;

  for (at = &cache_head; (p = *at) != NULL; at = &p->next) {
    if (p->len == len && p->outputs == outputs && memcmp(p->text, text, len) == 0) {
      *at = p->next;
      p->next = cache_head;
      cache_head = p;
      return p;
    }
  }
  return NULL;
}

/* The cached program of `text` and `outputs` with one more count, or NULL. */
static struct program *cache_find(const char *text, size_t len, size_t outputs) {
  struct program *p;

  pthread_mutex_lock(&cache_lock);
  p = cache_lookup(text, len, outputs);
  if (p != NULL) __atomic_add_fetch(&p->refs, 1, __ATOMIC_RELAXED);
  pthread_mutex_unlock(&cache_lock);
  return p;
}

/* Lists `p` (which has its caller's count) in the cache, which takes one more count, and returns it; when another
   task listed the same text first, `p` goes and the listed one comes back with one more count. While the cache is
   over CACHE_SIZE, the least recently used programs that only the cache holds leave it; their counts go after the
   lock. */
static struct program *cache_add(struct program *p) {
  struct program **at, **oldest, *listed, *gone = NULL;

  pthread_mutex_lock(&cache_lock);
  listed = cache_lookup(p->text, p->len, p->outputs);
  if (listed != NULL) {
    __atomic_add_fetch(&listed->refs, 1, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&cache_lock);
    program_release(p);
    return listed;
  }
  __atomic_add_fetch(&p->refs, 1, __ATOMIC_RELAXED);
  p->next = cache_head;
  cache_head = p;
  cache_count++;
  while (cache_count > CACHE_SIZE) {
    oldest = NULL;
    for (at = &p->next; (listed = *at) != NULL; at = &listed->next) {
      if (__atomic_load_n(&listed->refs, __ATOMIC_ACQUIRE) == 1) oldest = at;
    }
    if (oldest == NULL) break;
    listed = *oldest;
    *oldest = listed->next;
    listed->next = gone;
    gone = listed;
    cache_count--;
  }
  pthread_mutex_unlock(&cache_lock);
  while (gone != NULL) {
    listed = gone;
    gone = gone->next;
    program_release(listed);
  }
  return p;
}

CC_SHORT static void xla_Buffer_free(void *state) {
  struct xla_Buffer_state *self = (struct xla_Buffer_state *)state;

  buffer_release(self->buffer);
  self->buffer = NULL;
}

CC_SHORT static void xla_Program_free(void *state) {
  struct xla_Program_state *self = (struct xla_Program_state *)state;

  program_release(self->program);
  self->program = NULL;
}

/* Lets go of what `c` holds: a call runs once, and the outputs of its run keep it as their origin, so it must hold
   nothing after the run. */
static void call_clear(struct xla_Call_state *c) {
  size_t i;

  for (i = 0; i < c->count; i++) buffer_release(c->args[i]);
  free(c->args);
  program_release(c->program);
  memset(c, 0, sizeof *c);
}

CC_SHORT static void xla_Call_free(void *state) { call_clear((struct xla_Call_state *)state); }
#define XLA_Buffer_FREE xla_Buffer_free
#define XLA_Program_FREE xla_Program_free
#define XLA_Call_FREE xla_Call_free

an_term xla_arch(an_env *env) {
#if defined(__x86_64__)
  return an_string(env, "x86_64", 6);
#elif defined(__aarch64__)
  return an_string(env, "aarch64", 7);
#else
  return an_string(env, "other", 5);
#endif
}

/* Loads the plugin and makes the client; a message in `why` on failure, which leaves nothing loaded. */
static bool load_plugin(const char *path, char *why, size_t size) {
  void *lib;
  const PJRT_Api *(*get_api)(void);
  const PJRT_Api *a;
  PJRT_Error *err = NULL;

#if defined(__x86_64__)
  /* The plugin is built for AVX: loading it on a CPU without would end the process (SIGILL). */
  if (!__builtin_cpu_supports("avx")) {
    snprintf(why, size, "the XLA plugin needs a CPU with AVX");
    return false;
  }
#endif
  lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (lib == NULL) {
    snprintf(why, size, "cannot load the XLA plugin %s: %s", path, dlerror());
    return false;
  }
  *(void **)(&get_api) = dlsym(lib, "GetPjrtApi");
  if (get_api == NULL || (a = get_api()) == NULL) {
    snprintf(why, size, "%s has no GetPjrtApi: it is not a PJRT plugin", path);
    dlclose(lib);
    return false;
  }
  if (a->pjrt_api_version.major_version != PJRT_API_MAJOR || a->pjrt_api_version.minor_version < PJRT_API_MINOR) {
    snprintf(why, size, "the XLA plugin %s speaks PJRT %d.%d, this module needs %d.%d or a later minor", path,
             a->pjrt_api_version.major_version, a->pjrt_api_version.minor_version, PJRT_API_MAJOR, PJRT_API_MINOR);
    dlclose(lib);
    return false;
  }
  /* A failure after this keeps the library open (dlopen counts it), so a later try finds it initialized. */
  if (initialized != a) {
    PJRT_Plugin_Initialize_Args init = {.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
    err = a->PJRT_Plugin_Initialize(&init);
    if (err == NULL) initialized = a;
  }
  if (err == NULL) {
    /* Use the plugin's default options: cpu_device_count is CPU-only, and we use only the first device anyway. */
    PJRT_Client_Create_Args cc = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
    err = a->PJRT_Client_Create(&cc);
    if (err == NULL) {
      PJRT_Client_AddressableDevices_Args ad = {.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
                                                .client = cc.client};
      err = a->PJRT_Client_AddressableDevices(&ad);
      if (err == NULL && ad.num_addressable_devices > 0) {
        client = cc.client;
        device = ad.addressable_devices[0];
        __atomic_store_n(&api, a, __ATOMIC_RELEASE);
        return true;
      }
      PJRT_Client_Destroy_Args cd = {.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE, .client = cc.client};
      PJRT_Error *dropped = a->PJRT_Client_Destroy(&cd);
      if (dropped != NULL) {
        char ignored[8];
        message_of(a, dropped, ignored, sizeof ignored);
      }
      if (err == NULL) {
        snprintf(why, size, "the XLA plugin %s has no device", path);
        return false;
      }
    }
  }
  {
    char text[1024];

    message_of(a, err, text, sizeof text);
    snprintf(why, size, "the XLA plugin %s did not start: %s", path, text);
  }
  return false;
}

an_term xla_open(an_env *env, ng_str path) {
  char why[1400], *copy;
  bool ok;

  copy = (char *)malloc(path.len + 1);
  if (copy == NULL) return failed(env, "out of memory");
  memcpy(copy, path.data, path.len);
  copy[path.len] = '\0';
  pthread_mutex_lock(&open_lock);
  if (opened_path != NULL) {
    ok = strcmp(opened_path, copy) == 0;
    if (!ok) snprintf(why, sizeof why, "the XLA plugin %s is loaded already, not %s", opened_path, copy);
    free(copy);
  } else {
    ok = load_plugin(copy, why, sizeof why);
    if (ok) opened_path = copy;
    else free(copy);
  }
  pthread_mutex_unlock(&open_lock);
  if (!ok) return failed(env, why);
  return xla_result_Void_ok(env, an_void(env));
}

/* The dimensions of `shape` into `dims`, their element count into `count`; an error term on failure. */
static bool dims_of(an_env *env, xla_list_Int *shape, int64_t *dims, size_t *rank, int64_t *count, an_term *failure) {
  int64_t dim;
  int code;

  *rank = 0;
  *count = 1;
  while ((code = xla_list_Int_next(env, shape, &dim)) == 1) {
    if (dim < 0) return *failure = failed(env, "a shape holds no negative dimension"), false;
    if (*rank == MAX_RANK) return *failure = failed(env, "the XLA backend holds tensors of rank 16 at most"), false;
    if (dim > 0 && *count > INT64_MAX / 8 / dim) return *failure = failed(env, "the tensor is too large"), false;
    *count *= dim;
    dims[(*rank)++] = dim;
  }
  if (code < 0) return *failure = an_void(env), false;
  return true;
}

static void store(void *bytes, int dtype, int64_t i, double x) {
  switch (dtype) {
  case DT_F32: ((float *)bytes)[i] = (float)x; break;
  case DT_F64: ((double *)bytes)[i] = x; break;
  case DT_I32: ((int32_t *)bytes)[i] = (int32_t)x; break;
  case DT_I64: ((int64_t *)bytes)[i] = (int64_t)x; break;
  case DT_I16: ((int16_t *)bytes)[i] = (int16_t)x; break;
  case DT_I8: ((int8_t *)bytes)[i] = (int8_t)x; break;
  case DT_U8: ((uint8_t *)bytes)[i] = (uint8_t)x; break;
  default: ((uint8_t *)bytes)[i] = x != 0.0; break;
  }
}

static double load(const void *bytes, int dtype, int64_t i) {
  switch (dtype) {
  case DT_F32: return (double)((const float *)bytes)[i];
  case DT_F64: return ((const double *)bytes)[i];
  case DT_I32: return (double)((const int32_t *)bytes)[i];
  case DT_I64: return (double)((const int64_t *)bytes)[i];
  case DT_I16: return (double)((const int16_t *)bytes)[i];
  case DT_I8: return (double)((const int8_t *)bytes)[i];
  case DT_U8: return (double)((const uint8_t *)bytes)[i];
  default: return ((const uint8_t *)bytes)[i] ? 1.0 : 0.0;
  }
}

/* A new buffer resource from the host `bytes`. */
static an_term put(an_env *env, const void *bytes, int dtype, const int64_t *dims, size_t rank) {
  const PJRT_Api *a = loaded();
  struct xla_Buffer_state *state;
  struct buffer *b;
  an_term term;
  PJRT_Error *err;

  PJRT_Client_BufferFromHostBuffer_Args args = {
      .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
      .client = client,
      .data = bytes,
      .type = TYPES[dtype],
      .dims = dims,
      .num_dims = rank,
      .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
      .device = device,
  };
  err = a->PJRT_Client_BufferFromHostBuffer(&args);
  if (err == NULL) err = await_event(args.done_with_host_buffer);
  if (err != NULL) {
    if (args.buffer != NULL) {
      PJRT_Buffer_Destroy_Args d = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = args.buffer};
      PJRT_Error *ignored = a->PJRT_Buffer_Destroy(&d);
      if (ignored != NULL) {
        char text[8];
        message_of(a, ignored, text, sizeof text);
      }
    }
    return pjrt_failed(env, err, "the upload failed");
  }
  b = buffer_new(args.buffer);
  if (b == NULL) {
    PJRT_Buffer_Destroy_Args d = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = args.buffer};
    PJRT_Error *ignored = a->PJRT_Buffer_Destroy(&d);
    if (ignored != NULL) {
      char text[8];
      message_of(a, ignored, text, sizeof text);
    }
    return failed(env, "out of memory");
  }
  state = xla_Buffer_new(env, &term);
  state->buffer = b;
  return xla_result_Buffer_ok(env, term);
}

/* Host memory for `count` elements of `dtype`, or NULL. */
static void *host(int64_t count, int dtype) { return malloc((size_t)(count > 0 ? count : 1) * SIZES[dtype]); }

an_term xla_upload(an_env *env, xla_list_Int shape, xla_list_Float data, int64_t dtype) {
  int64_t dims[MAX_RANK], count, i = 0;
  size_t rank;
  double x;
  int code;
  void *bytes;
  an_term failure, out;

  if (loaded() == NULL) return failed(env, "the XLA plugin is not loaded");
  if (dtype < 0 || dtype >= DT_COUNT) return failed(env, "an unknown dtype code");
  if (!dims_of(env, &shape, dims, &rank, &count, &failure)) return failure;
  if ((int64_t)data.len != count) return failed(env, "the values do not match the shape");
  bytes = host(count, (int)dtype);
  if (bytes == NULL) return failed(env, "out of memory");
  while ((code = xla_list_Float_next(env, &data, &x)) == 1) store(bytes, (int)dtype, i++, x);
  if (code < 0) {
    free(bytes);
    return an_void(env);
  }
  out = put(env, bytes, (int)dtype, dims, rank);
  free(bytes);
  return out;
}

/* Whether the `need` bytes at `offset` of the file `path` are inside it: a cheap check before the host buffer is
   allocated; `ff_read` still opens and checks the file itself. */
static bool range_in_file(const char *path, size_t path_len, int64_t offset, int64_t need) {
  char name[FF_PATH_MAX];
  struct stat st;

  if (path_len >= sizeof name) return false;
  memcpy(name, path, path_len);
  name[path_len] = 0;
  if (stat(name, &st) != 0 || offset < 0) return false;
  return offset <= (int64_t)st.st_size && need <= (int64_t)st.st_size - offset;
}

/* An F32 buffer of `shape` from the range at `offset` of the file `path`, or from `data` (`path` NULL), which `format`
   holds (a float format or the blocks of a quantized one): widened straight into the host buffer, with no list in
   between. The file is checked against the range and `data` against the length of the shape before a value is read. */
static an_term loaded_floats(an_env *env, xla_list_Int *shape, const char *path, size_t path_len, int64_t offset,
                             const uint8_t *data, size_t len, int64_t format) {
  int64_t dims[MAX_RANK], count;
  size_t rank;
  char why[FF_PATH_MAX + 160];
  an_term failure, out;
  float *values;
  int64_t need;
  bool read;

  if (loaded() == NULL) return failed(env, "the XLA plugin is not loaded");
  if (!dims_of(env, shape, dims, &rank, &count, &failure)) return failure;
  need = ff_bytes(format, count);
  if (need >= 0 && path == NULL && (size_t)need != len) return failed(env, "the bytes are not the length of the shape");
  if (need >= 0 && path != NULL && !range_in_file(path, path_len, offset, need)) {
    return failed(env, "the range reaches past the end of the file");
  }
  values = (float *)host(count, DT_F32);
  if (values == NULL) return failed(env, "out of memory");
  read = path != NULL ? ff_read(path, path_len, offset, count, format, values, why, sizeof why)
                      : ff_from_bytes(data, len, count, format, values, why, sizeof why);
  if (!read) {
    free(values);
    return failed(env, why);
  }
  out = put(env, values, DT_F32, dims, rank);
  free(values);
  return out;
}

an_term xla_load(an_env *env, ng_str path, int64_t offset, xla_list_Int shape, int64_t format) {
  return loaded_floats(env, &shape, path.data, path.len, offset, NULL, 0, format);
}

an_term xla_from_bytes(an_env *env, ng_buf data, xla_list_Int shape, int64_t format) {
  return loaded_floats(env, &shape, NULL, 0, 0, data.data, data.len, format);
}

an_term xla_filled(an_env *env, xla_list_Int shape, double value, int64_t dtype) {
  int64_t dims[MAX_RANK], count, i;
  size_t rank;
  void *bytes;
  an_term failure, out;

  if (loaded() == NULL) return failed(env, "the XLA plugin is not loaded");
  if (dtype < 0 || dtype >= DT_COUNT) return failed(env, "an unknown dtype code");
  if (!dims_of(env, &shape, dims, &rank, &count, &failure)) return failure;
  bytes = host(count, (int)dtype);
  if (bytes == NULL) return failed(env, "out of memory");
  for (i = 0; i < count; i++) store(bytes, (int)dtype, i, value);
  out = put(env, bytes, (int)dtype, dims, rank);
  free(bytes);
  return out;
}

an_term xla_readback(an_env *env, struct xla_Buffer_state *b) {
  const PJRT_Api *a = loaded();
  int64_t count = 1, i;
  size_t k;
  int dtype;
  void *bytes;
  PJRT_Error *err;
  xla_list_Float_builder out;

  if (b->buffer == NULL) return failed(env, "readback of a closed buffer");
  PJRT_Buffer_ElementType_Args et = {.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE, .buffer = b->buffer->b};
  err = a->PJRT_Buffer_ElementType(&et);
  if (err != NULL) return pjrt_failed(env, err, "readback");
  for (dtype = 0; dtype < DT_COUNT && TYPES[dtype] != et.type; dtype++) {
  }
  if (dtype == DT_COUNT) return failed(env, "readback of a buffer of an element type the backend does not hold");
  PJRT_Buffer_Dimensions_Args dm = {.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE, .buffer = b->buffer->b};
  err = a->PJRT_Buffer_Dimensions(&dm);
  if (err != NULL) return pjrt_failed(env, err, "readback");
  for (k = 0; k < dm.num_dims; k++) count *= dm.dims[k];
  bytes = host(count, dtype);
  if (bytes == NULL) return failed(env, "out of memory");
  PJRT_Buffer_ToHostBuffer_Args th = {.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
                                      .src = b->buffer->b,
                                      .dst = bytes,
                                      .dst_size = (size_t)count * SIZES[dtype]};
  err = a->PJRT_Buffer_ToHostBuffer(&th);
  if (err == NULL) err = await_event(th.event);
  if (err != NULL) {
    free(bytes);
    return pjrt_failed(env, err, "readback");
  }
  xla_list_Float_begin(env, &out, (size_t)count);
  for (i = 0; i < count; i++) xla_list_Float_push(env, &out, load(bytes, dtype, i));
  free(bytes);
  return xla_result_list_Float_ok(env, xla_list_Float_end(env, &out));
}

/* A new program of `text` compiled by the plugin, or NULL with the error term in `failure`. */
static struct program *compile_text(an_env *env, ng_str text, size_t outputs, an_term *failure) {
  /* xla.CompileOptionsProto { executable_build_options (3) { num_replicas (4): 1, num_partitions (5): 1 } }: the
     plugin aborts on an empty one. */
  static const char options[] = {0x1a, 0x04, 0x20, 0x01, 0x28, 0x01};
  const PJRT_Api *a = loaded();
  struct program *p;
  size_t got;
  PJRT_Error *err;

  PJRT_Program program = {.struct_size = PJRT_Program_STRUCT_SIZE,
                          .code = (char *)text.data,
                          .code_size = text.len,
                          .format = "mlir",
                          .format_size = 4};
  PJRT_Client_Compile_Args cc = {.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                                 .client = client,
                                 .program = &program,
                                 .compile_options = options,
                                 .compile_options_size = sizeof options};
  err = a->PJRT_Client_Compile(&cc);
  if (err != NULL) return *failure = pjrt_failed(env, err, "XLA did not compile the program"), NULL;
  p = (struct program *)calloc(1, sizeof *p);
  if (p != NULL) p->text = (char *)malloc(text.len ? text.len : 1);
  if (p == NULL || p->text == NULL) {
    free(p);
    PJRT_LoadedExecutable_Destroy_Args d = {.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
                                            .executable = cc.executable};
    err = a->PJRT_LoadedExecutable_Destroy(&d);
    if (err != NULL) {
      char ignored[8];
      message_of(a, err, ignored, sizeof ignored);
    }
    return *failure = failed(env, "out of memory"), NULL;
  }
  p->refs = 1;
  p->exe = cc.executable;
  p->outputs = outputs;
  memcpy(p->text, text.data, text.len);
  p->len = text.len;
  __atomic_add_fetch(&compiled_count, 1, __ATOMIC_RELAXED);
  PJRT_LoadedExecutable_GetExecutable_Args ge = {.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE,
                                                 .loaded_executable = cc.executable};
  err = a->PJRT_LoadedExecutable_GetExecutable(&ge);
  if (err == NULL) {
    PJRT_Executable_NumOutputs_Args no = {.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE,
                                          .executable = ge.executable};
    err = a->PJRT_Executable_NumOutputs(&no);
    got = no.num_outputs;
    PJRT_Executable_Destroy_Args ed = {.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE,
                                       .executable = ge.executable};
    PJRT_Error *dropped = a->PJRT_Executable_Destroy(&ed);
    if (dropped != NULL) {
      char ignored[8];
      message_of(a, dropped, ignored, sizeof ignored);
    }
  }
  if (err != NULL) {
    program_release(p);
    return *failure = pjrt_failed(env, err, "XLA compiled a program it cannot describe"), NULL;
  }
  if (got != outputs) {
    program_release(p);
    return *failure = failed(env, "XLA compiled a program with another number of outputs"), NULL;
  }
  return p;
}

an_term xla_compile(an_env *env, ng_str text, int64_t outputs, xla_list_Int donated) {
  struct program *p;
  int64_t *inputs, x;
  size_t n = 0;
  int code;
  an_term term, failure;

  if (loaded() == NULL) return failed(env, "the XLA plugin is not loaded");
  inputs = (int64_t *)malloc((donated.len ? donated.len : 1) * sizeof *inputs);
  if (inputs == NULL) return failed(env, "out of memory");
  while ((code = xla_list_Int_next(env, &donated, &x)) == 1) inputs[n++] = x;
  if (code < 0) {
    free(inputs);
    return an_void(env);
  }
  p = cache_find(text.data, text.len, (size_t)outputs);
  if (p == NULL) {
    p = compile_text(env, text, (size_t)outputs, &failure);
    if (p == NULL) {
      free(inputs);
      return failure;
    }
    p->donated = inputs;
    p->ndonated = n;
    inputs = NULL;
    p = cache_add(p);
  }
  free(inputs);
  xla_Program_new(env, &term)->program = p;
  return xla_result_Program_ok(env, term);
}

an_term xla_compiles(an_env *env) { return an_int(env, __atomic_load_n(&compiled_count, __ATOMIC_RELAXED)); }

an_term xla_call(an_env *env, struct xla_Program_state *p) {
  struct xla_Call_state *state;
  an_term term;

  state = xla_Call_new(env, &term);
  if (p->program != NULL) {
    __atomic_add_fetch(&p->program->refs, 1, __ATOMIC_RELAXED);
    state->program = p->program;
  }
  return term;
}

an_term xla_arg(an_env *env, struct xla_Call_state *c, struct xla_Buffer_state *b) {
  struct buffer **grown;
  size_t cap;

  if (b->buffer == NULL) return failed(env, "an argument of a run is a closed buffer");
  if (c->ran) return failed(env, "an argument of a call that ran already");
  if (c->count == c->cap) {
    cap = c->cap ? c->cap * 2 : 8;
    grown = (struct buffer **)realloc(c->args, cap * sizeof *grown);
    if (grown == NULL) return failed(env, "out of memory");
    c->args = grown;
    c->cap = cap;
  }
  __atomic_add_fetch(&b->buffer->refs, 1, __ATOMIC_RELAXED);
  c->args[c->count++] = b->buffer;
  return xla_result_Void_ok(env, an_void(env));
}

an_term xla_execute(an_env *env, struct xla_Call_state *c) {
  const PJRT_Api *a = loaded();
  PJRT_Buffer **args, **outs;
  PJRT_Event *done[1] = {NULL};
  PJRT_Error *err;
  xla_list_Buffer_builder list;
  int64_t *kept;
  size_t i, j, n, nkept = 0;
  bool short_of_memory = false, donated;
  an_term term;

  if (c->ran) return failed(env, "a call runs once");
  if (c->program == NULL) return failed(env, "the program of the call is closed");
  n = c->program->outputs;
  args = (PJRT_Buffer **)malloc((c->count ? c->count : 1) * sizeof *args);
  outs = (PJRT_Buffer **)calloc(n ? n : 1, sizeof *outs);
  kept = (int64_t *)malloc((c->count ? c->count : 1) * sizeof *kept);
  if (args == NULL || outs == NULL || kept == NULL) {
    free(args);
    free(outs);
    free(kept);
    return failed(env, "out of memory");
  }
  /* PJRT gives up only the inputs aliased to an output (`tf.aliasing_output`); every other input is listed as kept. */
  for (i = 0; i < c->count; i++) {
    args[i] = c->args[i]->b;
    donated = false;
    for (j = 0; j < c->program->ndonated; j++) donated = donated || c->program->donated[j] == (int64_t)i;
    if (!donated) kept[nkept++] = (int64_t)i;
  }
  PJRT_Buffer *const *arg_lists[1] = {args};
  PJRT_Buffer **out_lists[1] = {outs};
  PJRT_ExecuteOptions options = {.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
                                 .non_donatable_input_indices = kept,
                                 .num_non_donatable_input_indices = nkept};
  PJRT_LoadedExecutable_Execute_Args ex = {.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
                                           .executable = c->program->exe,
                                           .options = &options,
                                           .argument_lists = arg_lists,
                                           .num_devices = 1,
                                           .num_args = c->count,
                                           .output_lists = out_lists,
                                           .device_complete_events = done};
  err = a->PJRT_LoadedExecutable_Execute(&ex);
  free(args);
  free(kept);
  if (err == NULL) err = await_event(done[0]);
  call_clear(c);
  c->ran = true;
  xla_list_Buffer_begin(env, &list, n);
  for (i = 0; i < n; i++) {
    struct buffer *b;

    if (outs[i] == NULL) continue;
    b = buffer_new(outs[i]);
    if (b == NULL) {
      PJRT_Buffer_Destroy_Args d = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = outs[i]};
      PJRT_Error *ignored = a->PJRT_Buffer_Destroy(&d);
      if (ignored != NULL) {
        char text[8];
        message_of(a, ignored, text, sizeof text);
      }
      short_of_memory = true;
      continue;
    }
    xla_Buffer_new(env, &term)->buffer = b;
    xla_list_Buffer_push(env, &list, term);
  }
  free(outs);
  if (err != NULL) return pjrt_failed(env, err, "the run failed");
  if (short_of_memory) return failed(env, "out of memory");
  return xla_result_list_Buffer_ok(env, xla_list_Buffer_end(env, &list));
}

an_term xla_buffers(an_env *env) { return an_int(env, __atomic_load_n(&live_buffers, __ATOMIC_RELAXED)); }

XLA_NATIVE_MODULE(NULL)
