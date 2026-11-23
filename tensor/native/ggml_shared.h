/* What `ggml.c` (tensors, devices) and `ggml_graph.c` (compiled graphs)
   share. Not generated: native-gen makes `ggml_native.h`. */
#ifndef TENSOR_GGML_SHARED_H
#define TENSOR_GGML_SHARED_H

#include "ggml_native.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <pthread.h>

struct ggml_TensorHandle_state {
  struct ggml_context *ctx;
  struct ggml_tensor *t;
  ggml_backend_buffer_t buffer;
  bool vulkan;
};

/* The one lock of the device table, the registry, the graph cache, every
   graph run and every Vulkan buffer operation. */
extern pthread_mutex_t tg_lock;

an_term tg_error(an_env *env, const char *message);

/* The backend of `device` (-1: CPU, i >= 0: Vulkan device i), opened once
   for the process, or NULL with the reason in `why`. Call under tg_lock. */
ggml_backend_t tg_backend(int64_t device, char *why, size_t len);

/* A fresh tensor of `ne` and `type` on `device` into `state`; false with the
   error term in `failure`. Takes tg_lock itself. */
bool tg_new_tensor(an_env *env, int64_t device, const int64_t ne[4], enum ggml_type type,
                   struct ggml_TensorHandle_state *state, an_term *failure);

/* Copies `src` into `dst` (same element count), converting between F32 and
   I32 through the host when their types differ. Call under tg_lock. */
bool tg_copy(struct ggml_tensor *src, struct ggml_tensor *dst);

/* A held graph of the cache (`ggml_graph.c`). */
struct entry;
struct ggml_Lease_state {
  struct entry *e;
};

void ggml_Lease_free(void *state);

#endif
