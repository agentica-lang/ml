# Spike ml#89: CUDA on a g4dn.xlarge (Tesla T4)

2026-11-25, about 1 h 45 min of instance time. Recipe: `site.yml` (Ansible,
idempotent; setup always runs, one tag per measurement, logs pulled into
`results/` at the end of each run):

```sh
.neuron/.venv/bin/ansible-playbook -i inventory.ini site.yml --tags engine,llama   # builds
.neuron/.venv/bin/ansible-playbook -i inventory.ini site.yml --tags models,gguf    # weights
.neuron/.venv/bin/ansible-playbook -i inventory.ini site.yml --tags test|train|xla|torch|qwen
```

`TREE` is a local worktree of ml main at dd2347d (with submodules) with
`cuda.patch` committed on a local branch; the tree is synced without `.git`, so
`build-engine.sh` takes the ggml commit from `ENGINE_REV`.

## Machine

- g4dn.xlarge, eu-west-1, spot: 4 vCPU (Xeon Platinum 8259CL), 15 GB RAM,
  Tesla T4 (sm_75, 15 GB, 14.9 GB usable, fp16, no bf16).
- AMI "Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)",
  ami-0f33a321d4b081423. Driver 595.112.07, CUDA 13.2 at /usr/local/cuda
  (nvcc, cuBLAS 13.3, cuDNN 9.23, NCCL 2.29 in /usr/local/cuda/lib).
- Agentica 0.38.0 linux-x86_64 release.
- The machine ran nothing else during a measurement. The llama.cpp build was
  paused (SIGSTOP) while the ggml-cuda and XLA rows ran.

## What the spike changed (`cuda.patch`, local only)

1. `tensor/tools/build-engine.sh cuda`: `-DGGML_CUDA=ON
   -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS:-75}`, target `ggml-cuda`; the
   linker script adds `libggml-cuda.a`, `libcudart.so`, `libcublas.so`,
   `libcublasLt.so`, `stubs/libcuda.so` and, because ggml found NCCL
   (`GGML_CUDA_NCCL` is ON by default), `lib/libnccl.so`. `ENGINE_REV`
   overrides the ggml commit.
2. `tensor/native/ggml.c`: the device lookup falls back to the "CUDA" registry
   when there is no "Vulkan" one, so `Device.Vulkan(0)` is the T4 in a CUDA
   build. A real PR adds `Device.Cuda(i)`.
3. `tensor/native/xla.c`: `PJRT_Client_Create` is retried without options:
   the CUDA plugin refuses `cpu_device_count` ("Unexpected option name").
4. `tensor/src/xla.agc`: `AGENTICA_ML_XLA_PLUGIN` (as in the Neuron spike).
5. `benchmarks/training/bench_torch.py`: `TORCH_DEVICE`.
6. `transformers/tools/bench_qwen.agc`: 128 new tokens (was 64).

## Builds

- ggml v1.0.0 (417a024) with CUDA for sm_75 only: about 27 min cold on the 4
  vCPU (no ccache). The engine needs `/usr/local/cuda/lib64` and
  `/usr/local/cuda/lib` (NCCL) on `LD_LIBRARY_PATH` at run time; without the
  second, `libnccl.so.2: cannot open shared object file`.
- llama.cpp b12262 (15759e0, 2026-11-10), `-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=75`, target llama-bench: about 30 min.

## Tensor tests on the T4 (`--tags test`)

`TENSOR_REQUIRE_GGML=1`, test/ggml*_test.agc and trace_parity: 90 pass, 3 fail.

- `ggml_fp16_test`: expects Vulkan's fp16 `mul_mat` by default; ggml-cuda keeps
  F32 `mul_mat` in F32. Vulkan-specific; a CUDA build must skip or adapt it.
- `ggml_half_test` "a big half load": an F16 weight times an F32 input over
  98304 rows is 1% off (-461312 for -465920). ggml-cuda runs F16 weights
  through cuBLAS with fp16 compute unless the op asks `GGML_PREC_F32`.
- `ggml_views_test` "binary ops ... as a broadcast source": a binary op whose
  broadcast source is a non-contiguous view differs (by 1.74). A ggml-cuda
  bug or a missing `contiguous` on our side; to look at in the PR.

XLA, `jax-cuda13-pjrt==0.12.0.1` (PyPI, 2026-10-21), `TENSOR_REQUIRE_XLA=1`:
`test/xla_test.agc` 13/13 pass. The plugin reports "Tesla T4, Compute
Capability 7.5 (Driver 13.2, Runtime 13.2, Toolkit 13.0, DNN 9.23.2)". It
uses the system CUDA 13.2 and cuDNN (no nvidia-* wheels), and it preallocates
75% of the device memory (10.9 GiB, BFC allocator).

## MLP train step, median ms (784-512-512-10, `train/tools/bench_train.agc`; 30 steps, 15 at 4096)

| stack | SGD 64 / 256 / 1024 / 4096 | Adam 64 / 256 / 1024 / 4096 | first step |
|---|---|---|---|
| ggml-cuda, compiled | 1.01 / 1.15 / 1.66 / 4.97 | 1.09 / 1.26 / 1.80 / 4.97 | 45–51 ms |
| ggml-cuda, `fit` | 1.20 / 1.31 / 2.30 / 6.62 | 1.33 / 1.47 / 2.42 / 6.77 | 49–58 ms |
| XLA CUDA plugin, compiled | 0.75 / 0.91 / 1.32 / 3.62 | 0.85 / 1.00 / 1.41 / 3.54 | 0.39–0.59 s |
| XLA CUDA plugin, `fit` | 1.12 / 1.35 / 2.15 / 5.43 | 1.23 / 1.49 / 2.26 / 5.49 | 0.44–0.62 s |
| PyTorch 2.6.0+cu124 eager | 0.84 / 0.84 / 1.00 / 3.13 | 1.09 / 1.15 / 1.14 / 3.52 | 0.12–0.18 s |
| PyTorch 2.6.0+cu124 `torch.compile` | 1.16 / 1.23 / 1.33 / 3.21 | 1.21 / 1.38 / 1.40 / 3.19 | 2.1–7.9 s |

The losses of the first ten steps agree across all stacks to about 1e-7.
All are fp32 (the T4 has no TF32).

## Qwen3 generation on the T4 (`--tags qwen`)

Prompt 128 token ids in one step, then 128 greedy one-token steps; median of 3
rounds. Ours: `transformers/tools/bench_qwen.agc` on ggml-cuda (a warm-up pass,
then a timed one). llama.cpp: `llama-bench -p 128 -n 128 -ngl 99 -r 1`
(pp128 and tg128 measured apart; flash attention auto). The Q8_0 rows load the
very same file on both sides: Qwen's official GGUFs.

| model | weights | ours prefill tok/s | llama.cpp prefill | ours decode tok/s | llama.cpp decode |
|---|---|---|---|---|---|
| Qwen3-1.7B | Q8_0 GGUF | 2844 | 3453 | 99.3 | 107.1 |
| Qwen3-1.7B | f16 from safetensors | 3282 | — | 65.5 | — |
| Qwen3-4B | Q8_0 GGUF | 1524 | 1810 | 47.2 | 49.0 |
| Qwen3-4B | f16 from safetensors | 1600 | — | 30.3 | — |
| Qwen3-8B | Q8_0 GGUF | 992 | 1115 | 28.5 | 29.4 |

Ours is 82–89% of llama.cpp's prefill and 93–97% of its decode on the same
file. Load (warm page cache): 1.7B Q8_0 3 s, 4B Q8_0 14 s, 8B Q8_0 16 s;
f16 from the bf16 safetensors 25 s (1.7B) and 60 s (4B). Host RSS 1.0–1.5 GB.

Weights:

- safetensors at pinned revisions: Qwen3-1.7B 70d244cc86cc, Qwen3-4B
  1cfa9a720891, Qwen3-8B b968826d9c46.
- GGUFs (official Qwen repos, Q8_0 only; they have no F16 file), sha256
  checked by `get_url`:
  - Qwen/Qwen3-1.7B-GGUF@90862c4b9d27, Qwen3-1.7B-Q8_0.gguf,
    061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a;
  - Qwen/Qwen3-4B-GGUF@bc640142c66e, Qwen3-4B-Q8_0.gguf,
    8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300;
  - Qwen/Qwen3-8B-GGUF@7c41481f57cb, Qwen3-8B-Q8_0.gguf,
    408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6.

Loading note: a bf16 safetensors loads as F32, or as F16 with `WEIGHTS=f16`;
`WEIGHTS=q8_0` keeps Q8_0 only for tensors a GGUF stores as Q8_0. So 8B from
the safetensors does not fit the T4: F32 runs out of device memory at layer 16
(12.6 GB used), F16 at `lm_head` (14.5 GB used). 8B runs from a Q8_0 GGUF made
by Python tools (here Qwen's own).

## Pitfalls

- `nvidia-smi --query-gpu=memory.used` reads 0 between processes; sample it
  while the process runs.
- The tree synced without `.git` breaks `build-engine.sh`'s `git rev-parse`
  (hence `ENGINE_REV`).
- Two `synchronize` of the same tree at once collide (`--delay-updates`); run
  the plays one after the other.
- `results/llama-build-*.log` ends with the pip freeze of a convert venv from
  a first version of the recipe; the owner chose the official GGUFs instead,
  and the conversion was stopped before it wrote a file.
