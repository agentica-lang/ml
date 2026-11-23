// The ceiling of the libtorch CPU column: the same MLP training step as
// `bench.agc` (784 -> 512 -> 512 -> 10, relu, softmax cross-entropy, SGD lr
// 0.01 or Adam lr 0.001) written in plain C++ with libtorch's own autograd
// and optimizers, no FFI and no Agentica. Four random batches are made up
// front and cycled, as `Raw.batches` does.
//
//   eval "$(sh ../../tensor/tools/build-torch.sh)"
//   g++ -O2 -std=c++17 tools/bench-torch-step.cpp -o "$TMPDIR/bench-torch-step" \
//     -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10
//   "$TMPDIR/bench-torch-step" 0 sgd      # 0: libtorch's default thread count
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  int threads = argc > 1 ? std::atoi(argv[1]) : 0;
  bool adam = argc > 2 && std::strcmp(argv[2], "adam") == 0;
  if (threads > 0) torch::set_num_threads(threads);
  std::printf("# libtorch C++, %s, %d threads\n", adam ? "adam" : "sgd", torch::get_num_threads());
  torch::manual_seed(1);
  for (int64_t batch : {64, 256, 1024, 4096}) {
    torch::nn::Sequential net(torch::nn::Linear(784, 512), torch::nn::ReLU(), torch::nn::Linear(512, 512),
                              torch::nn::ReLU(), torch::nn::Linear(512, 10));
    std::unique_ptr<torch::optim::Optimizer> opt;
    if (adam) opt = std::make_unique<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(0.001));
    else opt = std::make_unique<torch::optim::SGD>(net->parameters(), torch::optim::SGDOptions(0.01));
    std::vector<torch::Tensor> xs, ys;
    for (int i = 0; i < 4; i++) {
      xs.push_back(torch::rand({batch, 784}) * 2 - 1);
      ys.push_back(torch::randint(0, 10, {batch}));
    }
    auto step = [&](int i) {
      opt->zero_grad();
      auto loss = torch::nn::functional::cross_entropy(net->forward(xs[i % 4]), ys[i % 4]);
      loss.backward();
      opt->step();
      return loss.item<float>();
    };
    for (int i = 0; i < 3; i++) step(i);
    std::vector<double> times;
    double start = now_ms();
    for (int i = 3; times.size() < 5 || (now_ms() - start < 3000 && times.size() < 200); i++) {
      double t0 = now_ms();
      step(i);
      times.push_back(now_ms() - t0);
    }
    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    std::printf("batch %lld: %.2f ms/step (min %.2f, %zu steps), %.0f samples/s\n", (long long)batch, median, times[0],
                times.size(), batch / (median / 1000.0));
  }
  return 0;
}
