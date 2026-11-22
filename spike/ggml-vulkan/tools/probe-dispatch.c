/* The dispatch probe of the ggml-vulkan spike: what one `ggml_backend_graph_compute`
   costs on the iGPU when the graph is empty, when it is submitted
   asynchronously, and when several `mul_mat`s ride in one graph. `bench.agc`
   measures what the module pays per op; this program separates the part of
   that which is the submit from the part that is arithmetic.

   Build and run from `spike/ggml-vulkan/`, with ggml built by
   `tools/build-ggml.sh` (the same environment the module needs):

   ```sh
   eval "$(sh tools/build-ggml.sh)"
   cc -O2 tools/probe-dispatch.c -o build/probe-dispatch \
     -I../../third_party/ggml/include \
     build/vulkan/src/libggml.a build/vulkan/src/ggml-vulkan/libggml-vulkan.a \
     build/vulkan/src/libggml-cpu.a build/vulkan/src/libggml-base.a \
     -lvulkan -lstdc++ -lm -lpthread
   ./build/probe-dispatch
   ```

   It prints, on the Vulkan backend only:
   - the mean of 200 empty graphs, one submit + synchronize each (the number
     `bench.agc`'s "empty graph" row reports),
   - the mean of 100 empty graphs submitted asynchronously and waited for
     once at the end,
   - the best-of-5 wall time of a graph holding 1, 2, 4 and 8 `mul_mat`s of
     512^3, which is what a step-shaped graph would pay. */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(void) {
  ggml_backend_t backend = ggml_backend_vk_init(0);
  if (backend == NULL || !ggml_backend_is_vk(backend)) {
    fprintf(stderr, "probe-dispatch: no Vulkan backend on this machine\n");
    return 1;
  }

  /* Empty graphs, one submit and one wait each: what an op-by-op caller pays
     before any arithmetic. */
  struct ggml_init_params params = { (size_t)64 * 1024 * 1024, NULL, true };
  struct ggml_context *ctx = ggml_init(params);
  int rounds = 200;
  double start = now();
  for (int i = 0; i < rounds; i++) {
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_backend_graph_compute(backend, graph);
  }
  printf("empty graph, submit + synchronize each: %8.3f ms  (%d rounds, mean)\n",
         (now() - start) * 1e3 / rounds, rounds);

  /* The same 100 submissions without waiting for each: the cost of a submit
     when the caller does not need the result yet. */
  rounds = 100;
  start = now();
  for (int i = 0; i < rounds; i++) {
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_backend_graph_compute_async(backend, graph);
  }
  ggml_backend_synchronize(backend);
  printf("empty graph, 100 submits then one wait: %8.3f ms total, %6.3f ms each\n",
         (now() - start) * 1e3, (now() - start) * 1e3 / rounds);
  ggml_free(ctx);

  /* Several mul_mats in one graph: the amortized cost of an op when the whole
     step is one submit. */
  const int n = 512;
  for (int nodes = 1; nodes <= 8; nodes *= 2) {
    struct ggml_context *big = ggml_init((struct ggml_init_params){ (size_t)64 * 1024 * 1024, NULL, true });
    struct ggml_tensor *a = ggml_new_tensor_2d(big, GGML_TYPE_F32, n, n);
    struct ggml_tensor *b = ggml_new_tensor_2d(big, GGML_TYPE_F32, n, n);
    struct ggml_tensor *out[8];
    for (int i = 0; i < nodes; i++) {
      out[i] = ggml_mul_mat(big, a, b);
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(big, backend);
    float *host = (float *)malloc((size_t)n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) {
      host[i] = (float)((i * 7 + 3) % 11) / 11.0f - 0.4f;
    }
    ggml_backend_tensor_set(a, host, 0, (size_t)n * n * sizeof(float));
    ggml_backend_tensor_set(b, host, 0, (size_t)n * n * sizeof(float));

    double best = 1e9;
    for (int r = 0; r < 5; r++) {
      struct ggml_cgraph *graph = ggml_new_graph(big);
      for (int i = 0; i < nodes; i++) {
        ggml_build_forward_expand(graph, out[i]);
      }
      double begin = now();
      ggml_backend_graph_compute(backend, graph);
      double elapsed = now() - begin;
      if (elapsed < best) best = elapsed;
    }
    printf("%d mul_mat 512^3 in one graph: %8.3f ms best, %6.3f ms per op (%5.1f GFLOPS)\n",
           nodes, best * 1e3, best * 1e3 / nodes,
           2.0 * (double)n * (double)n * (double)n * (double)nodes / best / 1e9);
    free(host);
    ggml_backend_buffer_free(buffer);
    ggml_free(big);
  }
  ggml_backend_free(backend);
  return 0;
}
