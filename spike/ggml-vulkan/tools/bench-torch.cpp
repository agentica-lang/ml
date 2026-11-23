/* The reference point of the ggml-vulkan spike: libtorch CPU on the same
   four `mul_mat` shapes, with the same row-major `a[m, k] . b[n, k]^T` the
   ggml bench measures, and the same statistics (median and min of `rounds`
   calls after one warm-up).

   It is a standalone program, not a module, because the number wanted here is
   the library's own matmul with no FFI in it: ggml's numbers paid the
   boundary, so the reference says what a CPU library that is already linked
   in costs for the same arithmetic. libtorch is the system dependency
   `tensor/tools/build-torch.sh` finds, never a copy.

   Build and run from `spike/ggml-vulkan/`:

   ```sh
   eval "$(sh ../../tensor/tools/build-torch.sh)"
   g++ -O2 -std=c++17 tools/bench-torch.cpp -o build/bench-torch \
     -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 -lpthread
   ./build/bench-torch 1
   ./build/bench-torch 8
   ``` */
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct shape {
  const char *name;
  int64_t m, n, k;
};

/* The same four shapes as `bench.agc`, in the same order. */
static const struct shape shapes[] = {
  {"256^3", 256, 256, 256},
  {"512^3", 512, 512, 512},
  {"1024^3", 1024, 1024, 1024},
  {"512x2048, k512", 512, 2048, 512},
};

static int rounds_for(const struct shape &s) {
  double flops = 2.0 * (double)s.m * (double)s.n * (double)s.k;
  if (flops >= 2147483648.0) return 4;
  if (flops >= 268435456.0) return 7;
  return 11;
}

/* The deterministic pattern `native/ggmlspike.c` fills every ggml tensor
   with, so both sides multiply the same numbers. */
static torch::Tensor pattern(int64_t rows, int64_t cols) {
  auto t = torch::empty({rows, cols}, torch::dtype(torch::kFloat32));
  float *p = t.data_ptr<float>();
  for (int64_t i = 0; i < rows * cols; i++) {
    p[i] = (float)((i * 7 + 3) % 11) / 11.0f - 0.4f;
  }
  return t;
}

int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  if (threads < 1) threads = 1;

  // One thread does the inter-op work: the comparison is about the matmul.
  at::set_num_threads(threads);
  at::set_num_interop_threads(1);

  printf("# libtorch %s, %d intra-op threads, %d inter-op\n",
         TORCH_VERSION, at::get_num_threads(), at::get_num_interop_threads());
  printf("# one torch::matmul per row, median and min of `rounds` calls after one warm-up\n");
  printf("%-22s %-14s %9s %9s %8s %8s\n", "shape", "backend", "median ms", "min ms", "GFLOPS", "GFLOPS");
  printf("%-22s %-14s %9s %9s %8s %8s\n", "", "", "", "", "(median)", "(min)");

  for (const auto &s : shapes) {
    auto a = pattern(s.m, s.k);
    auto b = pattern(s.n, s.k);
    auto bt = b.transpose(0, 1);

    // One warm-up: first touch of the threading pool and of the allocator.
    (void)torch::matmul(a, bt);

    std::vector<double> samples;
    int rounds = rounds_for(s);
    for (int i = 0; i < rounds; i++) {
      auto start = std::chrono::steady_clock::now();
      auto c = torch::matmul(a, bt);
      auto elapsed = std::chrono::steady_clock::now() - start;
      samples.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
      (void)c;
    }
    std::sort(samples.begin(), samples.end());
    double median = samples[samples.size() / 2];
    double best = samples.front();
    double flops = 2.0 * (double)s.m * (double)s.n * (double)s.k;
    // GFLOPS = flops / (ms * 1e-3 s) / 1e9 = flops / (ms * 1e6).
    printf("%-22s %-14s %9.3f %9.3f %8.1f %8.1f\n", s.name,
           threads == 1 ? "torch, 1 thread" : "torch, 8 threads", median, best,
           flops / (median * 1e6), flops / (best * 1e6));
  }
  return 0;
}
