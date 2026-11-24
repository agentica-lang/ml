/* What `ggml.c` (tensors, devices) and `ggml_graph.c` (compiled graphs)
   share. Not generated: native-gen makes `ggml_native.h`. */
#ifndef TENSOR_GGML_SHARED_H
#define TENSOR_GGML_SHARED_H

#include "cchecker.h"
#include "ggml_native.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <pthread.h>

/* A queued `fn(arg)`, embedded in what it frees so queuing needs no memory. */
struct deferred {
  void (*fn)(void *);
  void *arg;
  struct deferred *next;
};

/* The memory of one tensor, shared by the handle that made it and by every graph that is bound to it: it is freed
   (under the device lock, deferred) when the last of them lets go, so a handle that is closed while a run holds it
   leaves the run its data. */
struct tg_storage {
  int refs; /* atomic */
  int kept; /* atomic: the leaves of cached graphs that stay bound to it between runs (`ggml_graph.c`) */
  int closed; /* atomic: its handle is closed */
  int32_t *mirror; /* the values of a small I32 tensor as `upload` got them, NULL once a graph writes the tensor; read
                      and dropped under the device lock */
  struct ggml_context *ctx;
  ggml_backend_buffer_t buffer;
  struct deferred link;
};

struct ggml_TensorHandle_state {
  struct ggml_context *ctx;
  struct ggml_tensor *t;
  ggml_backend_buffer_t buffer;
  bool vulkan;
  struct tg_storage *store; /* owns ctx and buffer; NULL for a handle that was never made */
};

/* One more holder of `s`. */
CC_SHORT void tg_storage_retain(struct tg_storage *s);
/* One holder less: the last one queues the free of the memory. Never waits (a free hook calls it). */
CC_SHORT void tg_storage_release(struct tg_storage *s);
/* The handle of a tensor that graphs keep bound is closed (`closed` is set): the idle graphs let go of it
   (`ggml_graph.c`). Never waits (a free hook calls it). */
CC_SHORT void tg_closed(void);

/* The device lock: the device table, the registry, every graph build and
   run and every Vulkan buffer operation. Taking it first does the frees that
   free hooks deferred (`tg_defer`), since a hook must never wait for a graph.
   CPU and Vulkan graphs share it, so they run one at a time. */
CC_LOCK_LEVEL(1) extern pthread_mutex_t tg_device_lock;
CC_ACQUIRES(tg_device_lock) void tg_lock_device(void);
CC_RELEASES(tg_device_lock) void tg_unlock_device(void);

/* Runs `d->fn(d->arg)` under the device lock, the next time anyone takes it. */
void tg_defer(struct deferred *d);

an_term tg_error(an_env *env, const char *message);

/* The backend of `device` (-1: CPU, i >= 0: Vulkan device i), opened once
   for the process, or NULL with the reason in `why`. Call under the device
   lock. */
ggml_backend_t tg_backend(int64_t device, char *why, size_t len);

/* Whether the process chose fp16 products on Vulkan (`ggml_vulkan_fp16`, true by default). */
bool tg_fp16(void);

/* A fresh tensor of `ne` and `type` on `device` into `state`; false with the
   error term in `failure`. Takes the device lock itself. */
bool tg_new_tensor(an_env *env, int64_t device, const int64_t ne[4], enum ggml_type type,
                   struct ggml_TensorHandle_state *state, an_term *failure);

/* Copies `src` into `dst`, converting between F32 and I32 through the host
   when their types differ; NULL, or why it did not (an element count that
   differs, an integer beyond 2^24). `mirror` is NULL or the values of an I32
   `src` on the host, which then is not read back from its device. Call under
   the device lock. */
const char *tg_copy(struct ggml_tensor *src, struct ggml_tensor *dst, const int32_t *mirror);

/* A held graph of the cache (`ggml_graph.c`). */
struct entry;
struct ggml_Lease_state {
  struct entry *e;
};

void ggml_Lease_free(void *state);

/* A pin on a graph of the cache (`ggml_pin`). `id` names it among every pin of the process (0 for a pin that holds
   nothing): the graph keeps the constants of one pin bound. */
struct ggml_Pin_state {
  struct entry *e;
  int64_t id;
};

void ggml_Pin_free(void *state);

#endif
