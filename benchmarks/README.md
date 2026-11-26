# Benchmarks

Local runs against the Python and C++ stacks, never in CI, all on the one machine below: Whisper against whisper.cpp, training against PyTorch and JAX, a DistilBERT fine-tune against PyTorch and PEFT, BERT and Qwen3 against llama.cpp, Qwen3.5-0.8B and LFM3-1.7B alone.

## Whisper-tiny, ours against whisper.cpp

Both engines run the same weights (the Hugging Face `openai/whisper-tiny` checkpoint, converted for whisper.cpp by [`convert_whisper_ggml.py`](convert_whisper_ggml.py)), the same two clips, greedy decoding, no timestamps, a named language and 8 threads. A run is the whole pipeline: log-mel, encoder and decoder. The load is timed on its own, one warm-up transcription runs first, and the number is the median of 5 timed runs.

| | |
|---|---|
| Machine | Ryzen 5 3550H (4 cores, 8 threads, AVX2 and FMA, no AVX512), Radeon Vega iGPU (Mesa RADV 25.0.7), 12 GB, Debian, Linux 6.12 |
| Ours | agentica-lang/ml `0e0bb8b` plus the device pick of ml#81 for the two ggml rows and the medium table (Agentica 0.37.0), `ac1b402` and Agentica 0.36.0 for the libtorch row; ggml `v1.0.0` (the `third_party/ggml` submodule), libtorch 2.4.1 CPU |
| whisper.cpp | `398997ed` (v1.9.4 plus 86 commits), its own ggml 0.24.0, built out of tree with `-DCMAKE_BUILD_TYPE=Release`, and `-DGGML_VULKAN=ON` for the Vulkan one |
| Clips | `jfk.wav` from whisper.cpp (11 s, English) and `whisper/live/fixtures/es-hola.wav` (2.6 s, Spanish) |
| Threads | ours on ggml CPU: every online core (8) in these rows, one per physical core (4) since ml#222; `-t 8` for whisper.cpp, libtorch's default |

The real-time factor is the run time over the clip length, so lower is better. The load is the time from the file to a model ready to transcribe.

| Row | RTF jfk | RTF es-hola | Load | Peak memory | Tokens against Hugging Face |
|---|---|---|---|---|---|
| ours, ggml CPU | 0.116 | 0.419 | 0.4 s | 0.50 GB | same |
| ours, ggml Vulkan | 0.069 | 0.238 | 0.45 s | 0.35 GB | same |
| ours, libtorch CPU | 0.619 | 1.987 | 0.3 s | 1.5 GB | same |
| whisper.cpp CPU, F32 | 0.080 | 0.287 | 0.10 s | 0.22 GB | one more token on jfk |
| whisper.cpp CPU, F32, no flash attention | 0.119 | 0.429 | 0.11 s | 0.29 GB | same |
| whisper.cpp CPU, F16 (its default) | 0.116 | 0.456 | 0.08 s | 0.15 GB | one more token on jfk |
| whisper.cpp Vulkan, F32 | 0.041 | 0.110 | 0.17 s | 0.16 GB | one more token on jfk |
| whisper.cpp Vulkan, F32, no flash attention | 0.041 | 0.107 | 0.17 s | 0.16 GB | one more token on jfk |
| whisper.cpp Vulkan, F16 (its default) | 0.038 | 0.106 | 0.13 s | 0.12 GB | one more token on jfk |

The two ggml `ours` rows are the `bench_whisper.sh` run of ml `e362042` (Agentica 0.37.0, F32 weights, `Options.timestamps` false, the median of 3 runs); the medium rows come from an earlier run. None of the whisper rows was timed again after ml#213 (llamafile's sgemm) and ml#222 (the thread default). The libtorch row, the whisper.cpp rows and the load and memory columns are from earlier runs of the same machine.

Whisper's window is 30 s whatever the clip, so a short clip costs the same encoder time as a long one: that is why es-hola has the larger factor.

Long audio (ml#35, with timestamps, language detected, tokens the same as Hugging Face's long-form `generate`): `jfk` six times with silence between (73.5 s, three windows, 152 ids) and es-hola 24 times (75.5 s, three windows, 353 ids), run after the five short cases in one process, on ml `84030b5` plus the seek. The log-mel of the whole clip takes about 0.15 s and cutting a window 0.01 to 0.05 s.

| Row | RTF jfk x6 | RTF es-hola x24 |
|---|---|---|
| ours, ggml CPU | 0.122 | 0.230 |
| ours, ggml Vulkan | 0.162 | 0.332 |

A 300 s clip (`jfk` six times with silence, four times, then 6 s of silence; 611 ids, at least 10 windows) with timestamps and a named language (`asr/tools/bench_transcribe.agc`, the model loaded once, a warm-up run before the timed one, `/usr/bin/time -v` for the peak resident memory of the whole process). Both backends give the same ids and text. The whole-clip log-mel (30,000 frames) ran on Vulkan too, with no failure. The CPU row is the quiet-machine run: with other jobs on the machine the same run took 75 s and then 120 s.

| Row | Time | RTF | Peak memory |
|---|---|---|---|
| ours, ggml CPU | 34.6 s | 0.116 | 0.67 GB |
| ours, ggml Vulkan | 46.6 s | 0.155 | 0.38 GB |

whisper.cpp turns flash attention on by default; ours has it since ml#161, so the default row is the like-for-like one. On Vulkan flash attention makes no difference. F16 is slower than F32 on this CPU (probably because it has no half-precision arithmetic) and faster on the iGPU only by a little.

### The tokens

The golden for the check is Hugging Face's `generate` on whisper-tiny (`asr/test/fixtures/transcribe_golden.json`). Every `asr/` row, on all three backends and both clips, gives exactly its token ids and text. whisper.cpp gives them on es-hola, and on jfk with flash attention off on the CPU. In the other jfk rows it writes one more token, a comma (id 11) after "for you": the text is the same sentence, `... can do for you, ask what you can do for your country.` It is the same in F32 and F16 and on both devices, and it goes away with flash attention off on the CPU, so it looks like a near tie in the logits and not a converter error.

### Where the time goes

The stages of `Asr.transcribe` on jfk, in milliseconds, measured before flash attention (ml#161) and the argmax change of the last section, and not timed again since (`asr/tools/bench_transcribe.agc`, the `phase` lines; the profile is `AGENTICA_VM_TRACE=profile agentica tools/bench_transcribe.agc ... ggml en -1` in `asr/`). The decoder row is one step, with the pick on the device (below).

| Stage | ours ggml CPU | ours ggml Vulkan | whisper.cpp CPU F32, no flash | whisper.cpp Vulkan F32 |
|---|---|---|---|---|
| log-mel (one graph, upload included) | 35 | 95 | 14 | 19 |
| encoder | 994 | 396 | 1054 | 158 |
| cross-attention cache | 66 | 37 | | |
| decoder step | 38.5 | 68.2 (about 22 after the argmax change) | 7 | 11 |

- **The encoder is on par on the CPU** (994 ms against 1054 ms without flash attention, 662 ms with it): similar ggml CPU kernels (ggml v1.0.0 against 0.24.0). On Vulkan ours is 2.5 times slower (396 ms against 158 ms).
- **The decoder is the gap**: a step was 38.5 ms on ggml CPU and 68.2 ms on Vulkan (about 22 ms after the argmax change), against 7 and 11 ms for whisper.cpp.
- **The log-mel is one graph of tensor ops** (the windowed DFT as a matmul, the mel filters as a second one) that `load` compiled, and its output stays on the device. It computes all 3000 frames, so its cost does not depend on the clip. Vulkan is slower than the CPU there because the matmul alone takes about 20 ms on this iGPU.
- **The load is 0.3 to 0.45 s and peaks under 0.5 GB**: C reads each tensor's byte range into the backend's buffer (`Tensor.from_file`, ml#63), so no float is an Agentica value. whisper.cpp reads the file in 0.1 s.
- On Vulkan a graph costs about 2 ms to dispatch (`note.md`), and a step has many small ones.
- libtorch runs the encoder op by op: 3.2 s, and 117.7 M VM steps.

### Where a decoder step goes

Whisper-tiny, jfk, ggml CPU and Vulkan. `asr/tools/bench_step.agc` loads the model, runs the log-mel, the encoder and the cross-attention cache once, then runs the prompt and the 23 tokens of jfk through `Asr.step_pick` (24 steps) PASSES times, closing each step's caches as `Asr.transcribe` does. The first pass is a warm-up; the per-step wall time is the mean of the other passes (with 5 passes, 96 timed steps out of 120).

- **VM steps** (exact) and **where the time goes** (profile on, so the VM runs slower): `AGENTICA_VM_TRACE=profile AGENTICA_VM_TRACE_FILE=REPORT agentica tools/bench_step.agc MODEL_DIR WAV ggml en PASSES - pick`, once with PASSES 0 and once with 5, then `python3 -I benchmarks/step_profile.py BASE FULL 120` (120 is the steps the 5-pass run has more than the 0-pass run).
- **Milliseconds**, with the profile off: `benchmarks/step_timers.sh apply` swaps the `System.Profile.measure` labels for `Time.monotonic_ns` timers (it edits the tree; `revert` undoes it) and `bench_step.agc` then prints `probe NAME NS` totals, again for PASSES 0 and 5; a section's ms per step is the difference over 120 steps. The same script makes `ggml_run` print its allocator and compute times when `TG_TIME` is set. The `ml.*` labels (in `Trace.eval_many`, `Lower.evaluate` and `Asr.step`) also count the encoder's and the cross-attention cache's evaluations, which the 0-pass run holds too, so the subtraction removes them.

Agentica 0.37.0, ml `b59c68c` plus the pick, before the argmax change of the last section (which took the Vulkan step to about 22 ms), the means of those 120 steps (the wall time is the mean of the 96 timed ones):

| Part of a step | VM steps | CPU, ms | Vulkan, ms |
|---|---|---|---|
| tracing (`traced(m)` and `stepped_graph`: about 1,400 nodes recorded) | 0.05 M | 0.5 | 0.9 |
| cache key and lowering (`Lower.plan`, `Lower.program`) | 0.06 M | 1.7 | 2.8 |
| lease lookup (`acquire` by the key string) | 0 | 0.04 | 0.05 |
| input upload (the id and the position, 8 bytes in 2 calls) | 0 | 0.1 | 0.6 |
| binding (147 leaves, weights and caches, read in place, no copy) | 0.005 M | 0.8 | 1.7 |
| native execution (`run`: allocator, compute, finiteness checks, copy-out) | 0 | 39.3 | 62.2 |
| read-back (`[id, logprob]`) | 0.0001 M | 0.03 | 0.3 |
| sum of the parts | 0.11 M | 42.4 | 68.5 |
| wall time of a step | | 38.5 | 68.2 |

The sum and the wall time come from two processes, and the warm-up pass, which runs each new graph for the first time, is slower than the others: that is why the CPU parts sum to 42.4 ms against a wall time of 38.5.

- **The step is compiled once for each pair of cache rows and capacity** (`Trace.compile` and `Trace.run`): 5 graphs for the 24 steps of jfk (8, 16 and 32 rows), so tracing and planning above are those compilations spread over the steps. The encoder blocks share one compiled graph, with the weights as inputs.
- **The pick and the timestamp rules run in the step's graph** (`Asr.step_pick`, `Asr.picked`, `Asr.scored`), so the 51,865 logits stay on the device and `[id, logprob]` is the only thing that crosses. Reading the logits instead costs 19 to 33 ms a step and 1.04 M VM steps. `Asr.step` and `Greedy.allowed` stay as the reference, and tests compare the graph with them rule by rule.
- **Nothing big is uploaded.** The step feeds 8 bytes. The weights and both caches are bound where they are (`bind` makes no copy for an F32 tensor). Each step makes 9 fresh output tensors: the logits or the pick and 8 self-attention caches of `capacity x 384` floats (393 KB at capacity 32), allocated and written by the graph.
- **The device part is what remains**: 39 ms on the CPU and 62 ms on Vulkan (before the argmax change) for a graph of about 1,400 ggml nodes. The Agentica work around it is 3 to 6 ms.

## Whisper-medium, ours only

whisper-medium (the Hugging Face `openai/whisper-medium` checkpoint, 3.06 GB of F32, `model.safetensors` sha256 `62f73550fa6db24b0c6f6c5962bd0dae80fa644e93cde9cd9c3792971b47fd28`) Same machine and clips, under a 6 GB memory cap, 2 timed runs after the warm-up (the median). No whisper.cpp row: only the tiny model was converted.

| Row | RTF jfk | RTF es-hola | Load | Peak memory of the load | Tokens |
|---|---|---|---|---|---|
| ours, ggml CPU | 7.04 | 18.2 | 7.7 s | 4.0 GB | same as the medium output of whisper.cpp's server (`whisper/live/fixtures/server-*-medium.json`) |
| ours, ggml Vulkan | 1.38 | 4.03 | 24 s | 0.49 GB | same |

The Vulkan weights sit in the device's own buffers, which on this iGPU are system memory outside the process's resident set, so its row shows the process only. The encoder takes 22 s on the CPU and 5 s on Vulkan, and a decoder step about 2.4 s on the CPU and 270 ms on Vulkan (the last section). The Vulkan load is slower than the CPU's (24 s against 7.7 s); where that time goes was not profiled.

## Training against PyTorch and JAX (ml#50)

One training step of an MLP classifier (784-512-512-10, relu, softmax cross-entropy, fp32) in every stack, on the same machine, with the same init, data and optimizer, and the threads of the Threads row (4 for ggml on the CPU, 8 for PyTorch). The unit is ms per step, the median of 30 steps after the compile step and three warm ones (15 at batch 4096). A step is the forward, the gradients, the optimizer update and reading the loss back to the host; the batch stays on the device. The peers are [`bench_torch.py`](training/bench_torch.py) and [`bench_jax.py`](training/bench_jax.py), the Agentica side is [`train/tools/bench_train.agc`](../train/tools/bench_train.agc), and [`training/run.sh`](training/run.sh) runs the grid and [`training/table.py`](training/table.py) prints the tables from [`training/results.jsonl`](training/results.jsonl), which holds every run.

| | |
|---|---|
| Machine | the one above (Ryzen 5 3550H, 8 threads; Vega iGPU for the Vulkan rows, which share it with the desktop): each of the two holds of the run started at a load average under 1.0 (0.88 and 0.95), but the rows are not on a quiet machine: every line records the load average at its start, which counts the busy rows before it (see below) |
| Ours | agentica-lang/ml `a1cdc89` plus ml#222 (run at `d30be87`, merged as `0811da343`; its review changed only the fallbacks and the cap of the thread count), Agentica 0.37.0, ggml `417a024b4fb6`; the Vulkan rows multiply in fp32 because the program calls `Tensor.vulkan_fp16(false)` before the first ggml call |
| PyTorch | 2.6.0+cpu, Python 3.13.5; `eager` is autograd and `torch.optim` (the default foreach implementation, not `fused=True`), `compile` is `torch.compile` of the loss and of a functional update, with one extra `copy_` of each parameter after it |
| JAX | 0.12.0.1 with jaxlib 0.12.0.1 (numpy 2.5.5, ml_dtypes 0.7.0, scipy 1.18.1, opt_einsum 3.4.0), CPU, one `jax.jit` of the whole step |
| Threads | ggml's CPU rows run on 4 threads, the physical cores of the machine (the default since ml#222; `TENSOR_THREADS` sets it); PyTorch and libtorch use `OMP_NUM_THREADS=8`, the online CPUs (`run.sh` refuses any other count), and JAX uses XLA's default CPU pool (all the cores; its size was not checked) |
| XLA | the `xla-cpu` rows: `Backend.Xla(Device.Cpu)` with the plugin of `xla-cpu-pjrt` 0.0.1 (PyPI; `tools/install_xla.agc` installed that wheel then, and installs our own build now), agentica-lang/ml `e33d3bc` plus ml#228's bench PR, Agentica 0.38.0; timed on another day than the other rows, with the machine quiet (see below); XLA's threads are not pinned like the other rows (4 or 8): the plugin uses its default CPU pool, whose size was not checked (the process had 37 threads at batch 64 to 1024 and 46 at 4096, the VM's included) |
| Data and init | the closed forms of `tensor/tools/bench-ggml.agc` (`values`), the labels `(i * 7) % 10`; SGD lr 0.01, Adam lr 0.001, betas 0.9 and 0.999, eps 1e-8 with the bias correction passed in as two scalars |

The rows: `ggml-cpu` and `ggml-vulkan` are `Backend.Ggml`, `torch-cpu` is `Backend.Torch` (libtorch 2.4.1+cpu, the library the native module links, not the PyTorch 2.6.0 of the peers). `-fit` is the steps of `Train.fit` and the batch is gathered from a one-batch dataset (on ggml it compiles the step once for each batch shape and gathers the batch inside the step, so its gap to `-compiled` is small; `torch-cpu-fit` still gathers outside it). `-compiled` is one `Trace.compile` of the whole step and a `run` per step, so the gap between the two is what `fit` adds around the step. `xla-cpu` is `Backend.Xla`: `-fit` and `-compiled` as above, each step one XLA program with the batch gathered in it. There is no PyTorch or JAX row on the iGPU: neither trains on Vulkan here.

Correctness comes first: every row prints the loss of its first ten steps, and `table.py` compares them with PyTorch in float64 from the same init and batch, and shows `WRONG` and no time for a row that does not follow. It compares the movement of the loss (loss at step i minus loss at step 0), not the loss: SGD at lr 0.01 moves it only by 4e-6 to 2e-3 in ten steps, so a loss within 1e-3 of the oracle's would pass for an optimizer that does nothing (`table_test.py` checks that a curve that does not move, or is not finite or short, is `WRONG`). The movement must be within 2e-6 (eight float32 steps of a loss near 2.3) plus 1% of the oracle's (SGD) or 10% (Adam). Every row of this run passes: its losses are within 8e-7 (SGD) and 7.9e-4 (Adam, which amplifies float32 rounding) of the oracle's. At SGD batch 4096 the movement is 3.6e-6, so only a learning rate off by more than about 57 percent would show there.

SGD, ms per step:

| row | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| ggml-cpu-fit | 5.94 | 11.97 | 37.15 | 192.35 |
| ggml-cpu-compiled | 5.71 | 12.26 | 35.97 | 180.15 |
| ggml-vulkan-fit | 9.04 | 9.33 | 22.41 | 74.96 |
| ggml-vulkan-compiled | 7.95* | 9.58 | 20.29 | 68.31 |
| torch-cpu-fit | 8.05 | 15.93 | 47.39 | 181.60 |
| xla-cpu-fit | 3.74* | 7.24 | 27.62 | 122.59 |
| xla-cpu-compiled | 3.59* | 8.43* | 26.12 | 115.49 |
| pytorch-eager | 3.69 | 9.49 | 33.80 | 146.77 |
| pytorch-compile | 4.25 | 8.90 | 32.09 | 153.11 |
| jax-jit | 3.37 | 7.39 | 26.45 | 120.02 |

SGD, the median over the median of `pytorch-eager` (below 1 is faster than PyTorch):

| row | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| ggml-cpu-fit | 1.61 | 1.26 | 1.10 | 1.31 |
| ggml-cpu-compiled | 1.55 | 1.29 | 1.06 | 1.23 |
| ggml-vulkan-fit | 2.45 | 0.98 | 0.66 | 0.51 |
| ggml-vulkan-compiled | 2.15 | 1.01 | 0.60 | 0.47 |
| torch-cpu-fit | 2.18 | 1.68 | 1.40 | 1.24 |
| xla-cpu-fit | 1.01 | 0.76 | 0.82 | 0.84 |
| xla-cpu-compiled | 0.97 | 0.89 | 0.77 | 0.79 |
| pytorch-eager | 1.00 | 1.00 | 1.00 | 1.00 |
| pytorch-compile | 1.15 | 0.94 | 0.95 | 1.04 |
| jax-jit | 0.91 | 0.78 | 0.78 | 0.82 |

Adam, ms per step:

| row | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| ggml-cpu-fit | 9.11 | 14.09 | 37.97 | 192.52 |
| ggml-cpu-compiled | 7.68 | 13.83 | 36.02 | 187.23 |
| ggml-vulkan-fit | 8.95* | 10.00 | 22.77 | 76.59 |
| ggml-vulkan-compiled | 8.69 | 9.08 | 20.93 | 70.05 |
| torch-cpu-fit | 22.90 | 28.99 | 72.00 | 183.84 |
| xla-cpu-fit | 4.07 | 9.03 | 28.48 | 128.42 |
| xla-cpu-compiled | 4.42* | 8.96 | 26.68 | 116.98 |
| pytorch-eager | 5.51 | 11.13 | 34.18 | 155.63 |
| pytorch-compile | 5.73* | 10.94 | 35.11 | 154.18 |
| jax-jit | 3.78 | 8.36 | 27.06 | 120.39 |

Adam, the median over the median of `pytorch-eager` (below 1 is faster than PyTorch):

| row | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| ggml-cpu-fit | 1.65 | 1.27 | 1.11 | 1.24 |
| ggml-cpu-compiled | 1.39 | 1.24 | 1.05 | 1.20 |
| ggml-vulkan-fit | 1.62 | 0.90 | 0.67 | 0.49 |
| ggml-vulkan-compiled | 1.58 | 0.82 | 0.61 | 0.45 |
| torch-cpu-fit | 4.16 | 2.61 | 2.11 | 1.18 |
| xla-cpu-fit | 0.74 | 0.81 | 0.83 | 0.83 |
| xla-cpu-compiled | 0.80 | 0.81 | 0.78 | 0.75 |
| pytorch-eager | 1.00 | 1.00 | 1.00 | 1.00 |
| pytorch-compile | 1.04 | 0.98 | 1.03 | 0.99 |
| jax-jit | 0.69 | 0.75 | 0.79 | 0.77 |

- On Vulkan the step beats PyTorch eager from batch 256 with Adam (`fit` takes 0.90 of its time, the compiled step 0.82) and from 1024 with SGD (0.66 and 0.60); SGD at 256 ties (0.98 and 1.01). At 4096 it takes 0.45 to 0.51. JAX on the CPU is ahead of Vulkan up to batch 256 and behind it from 1024 (26 to 27 ms against 20 to 23 ms there, 120 ms against 68 to 77 ms at 4096).
- At batch 64 PyTorch is faster than ggml: 1.6 to 2.5 times on Vulkan (`fit` 2.45 with SGD, 1.62 with Adam) and 1.4 to 1.65 times on the CPU. The step has a fixed cost of a few ms on the GPU side (the dispatches, about 2 ms on this iGPU) that a small batch does not hide.
- On the CPU, ggml takes 1.4 to 1.65 times PyTorch eager's time at batch 64, 1.24 to 1.29 at 256, 1.05 to 1.11 at 1024 and 1.2 to 1.3 at 4096, where it used to take about 2 times at every batch. Among the rows timed under load, JAX is the fastest CPU stack (0.69 to 0.91 of PyTorch). `torch-cpu` (the same kind of kernels as PyTorch, driven from the VM) costs 2.2 (SGD) to 4.2 (Adam) times at batch 64 and 1.2 at batch 4096: that gap is the VM and the tracing, with libtorch 2.4.1 instead of 2.6.0.
- What `fit` costs over the compiled step is small now: the loader's gather runs inside the step, and the step is compiled once for each batch shape. At batch 64 it is 0.2 ms (SGD) to 1.4 ms (Adam) on the CPU and 1.1 ms (SGD) to 0.3 ms (Adam) on Vulkan; at batch 4096 6 to 7 ms on Vulkan and 5 (Adam) to 12 ms (SGD) on the CPU. The Vulkan gaps at batch 64 are noisy: the cells they come from are marked `*` (over 1.15 times their minimum) or are reruns.
- On the CPU, XLA takes 0.75 to 0.89 of PyTorch eager's time from batch 256 (both optimizers) and at batch 64 with Adam (0.74 to 0.80), and ties it at batch 64 with SGD (0.97 to 1.01), with unequal load: the XLA rows ran later on a quiet machine, so a rerun of all rows on a quiet machine is needed to confirm. That puts it next to JAX (its compiled step is ahead of JAX at 1024 and 4096) and 1.3 to 2.2 times faster than ggml on the CPU. `fit` costs little over the compiled step on XLA: 0.2 ms or less at batch 64 and 256 (it is faster in two cells), 1.5 to 1.8 ms at 1024 and 7 to 11 ms at 4096.
- So for ggml, the owner's goal of equal or better than PyTorch holds on Vulkan from batch 256 (Adam) or 1024 (SGD), and on the CPU only at 1024 (within 5 to 11 percent); not on the CPU at the other batches, nor on Vulkan at 64. With XLA it holds on the CPU at every batch but SGD at 64, a tie. What is left for ggml: the elementwise ops and the step's fixed cost at small batches on the CPU, and the fixed cost of a step on the GPU.

How the cells were run. A cell is one process of one run. The driver of this run (a script outside the repository that calls `bench_train.agc` and the peers of this folder, one cell at a time) waited for a load average under 1.0 before each of its two holds (the first lines of the two holds started at 0.88 and 0.95), ran the grid in the order of `run.sh`, and ran a cell once more when its median was over 1.15 times its minimum: seven cells, whose both runs are in `results.jsonl` (`run` 1 and 2), and the tables show the second. That choice biases the rerun cells: only cells whose first run was high ran again and the second run is shown, so they look better than the cells that did not (the median of 6 of the 7 went down). `run.sh run` alone does not apply this rule; the driver did:

| cell | run 1 | run 2 |
|---|---|---|
| ggml-cpu-compiled sgd 64 | 6.74 (5.61) | 5.71 (5.38) |
| ggml-vulkan-compiled sgd 64 | 7.85 (6.79) | 7.95 (6.73) |
| ggml-vulkan-fit adam 64 | 9.18 (7.91) | 8.95 (7.73) |
| ggml-vulkan-fit adam 256 | 12.23 (9.41) | 10.00 (8.95) |
| jax-jit sgd 64 | 3.92 (2.94) | 3.37 (2.97) |
| pytorch-compile adam 64 | 6.30 (5.05) | 5.73 (4.87) |
| torch-cpu-fit adam 256 | 31.51 (26.59) | 28.99 (27.39) |

Three cells are still over the line after the second run (`*`): `ggml-vulkan-compiled` SGD at 64, `ggml-vulkan-fit` Adam at 64 and `pytorch-compile` Adam at 64. The load average of a line is the one of its start, so it counts the busy rows before it (the rows use 4 to 8 threads): 53 of the 71 lines of the earlier rows started above 1.5 (at most 3.71), and nothing else ran in the build lock meanwhile. The first step, which holds the compile, is in `results.jsonl` as `first_ms`: 0.3 to 0.5 s for JAX, 2.7 to 3 s for `torch.compile`, 0.01 to 0.3 s for ggml and libtorch (the large batches are the long ones).

The XLA rows (ml#228) ran on their own, later: a driver script (outside the repository) ran `run.sh cell` once per cell, in the order of `run.sh`, each started at a load average under 0.8 (0.55 to 0.77) with nothing else running, once and with no rerun; four cells are over the 1.15 line (`*`). They are quieter than the other rows, whose lines started at up to 3.71, so compare them with that in mind. Their first step, with the XLA compile, is 0.24 to 0.56 s.

Threads, measured the same way: `TENSOR_THREADS=8` against the default of 4 (the physical cores) gives the same median within noise on `ggml-cpu-fit` and `ggml-cpu-compiled` at the four batches (4 threads is about 5 percent faster at 1024 and 4096) and a slowest step of 62 ms at most against up to 123 ms at batch 64 to 1024; the table is in ml#222.

## Fine-tuning DistilBERT on STS-B, ours against PyTorch (ml#182)

One epoch of a sentence-transformers DistilBERT (`sentence-transformers/msmarco-distilbert-cos-v5`, 66 M weights, all trained) on the first 1000 pairs of STS-B, on the CPU, in Agentica ml ([`distilbert_stsb.agc`](../transformers/tools/distilbert_stsb.agc), `Train.fit_batches` on `Backend.Ggml`) and in PyTorch ([`distilbert_stsb.py`](stsb/distilbert_stsb.py)), with the same rules. The unit is ms per step, the median of the steps that are not the first of their batch shape (on our side that step compiles a program; the same steps are left out on both sides). A step is building the batch, the forward, the gradients, the AdamW update and reading the loss back to the host.

| | |
|---|---|
| Machine | the one above (Ryzen 5 3550H, 8 threads, 12 GB), the machine quiet: `run.sh` refuses to start a run at a load average above 1.0 |
| Ours | agentica-lang/ml `c70c4c3` plus ml#182, Agentica 0.37.0, ggml `417a024b4fb6` (its CPU build has llamafile's sgemm on, since ml#213) |
| PyTorch | 2.6.0+cpu with transformers 4.57.6 and tokenizers 0.22.1, Python 3.13.5; autograd and `torch.optim.AdamW`, eager attention, no `torch.compile` |
| Threads | 8 on both sides: ggml used every online CPU (since ml#222 `run.sh` sets `TENSOR_THREADS` to get the same), and the peer sets `torch.set_num_threads` to the CPUs of the process (`OMP_NUM_THREADS=8`) |
| Model | revision `c598d92e340e`, `model.safetensors` (never the pickle), `F32` |
| Data | `sentence-transformers/stsb` at revision `ab7a5ac0e35a`: the first 1000 pairs of `train` in order, no shuffle, and the 1500 pairs of `validation` for Spearman; `run.sh fetch` checks the sha256 of every file |
| Rules | batch 16 pairs (the last one 8), so 63 steps; WordPiece with a limit of 128 tokens; the two sentences of a batch go through the model as one batch of 32 rows, padded to its longest sentence rounded up to a multiple of `PAD_TO`; mean pooling; the cosine loss of sentence-transformers (the squared error of the cosine against the score); AdamW lr 2e-5, betas 0.9 and 0.999, eps 1e-8, weight decay 0.01 except on a bias or a layer-norm weight; no warmup, no clipping, no dropout |

These are not the defaults of sentence-transformers' trainer, which shuffles, warms the learning rate up and clips the gradient: the rules are the smallest set both sides can follow exactly, so that the two loss curves can be compared number by number.

| `PAD_TO` | batch shapes | ours, ms per step | PyTorch, ms per step | ours / PyTorch | the 63 steps, ours | the 63 steps, PyTorch | largest loss gap | peak memory, ours / PyTorch |
|---|---|---|---|---|---|---|---|---|
| 1 (the longest sentence) | 14 | 2035 | 1363 | 1.49 | 177 s | 91 s | 1.9e-7 | 2.73 GB / 1.94 GB |

Spearman of the cosine against the score on `validation` is 0.8150 before and 0.8340 after, the same on both sides. The loss goes from 0.037646 (step 1) to 0.029733 (step 63) on both sides; over the run it moves between 0.007 and 0.065 with the batch.

We are 1.49 times slower than PyTorch. The first step of a shape costs about 0.45 s more than a warm one on our side (median 2484 ms against 2035: the trace and the compile), and PyTorch's is slower too (1611 ms against 1363). Only `PAD_TO=1` was run again: the earlier `PAD_TO=8` row lost on both sides (the longer rows cost more than the ten compiles it saved) and is not repeated.

Memory is the peak resident size of the whole process (GNU time, the `memory` line of `run.sh`).

Correctness comes first: `summary.py` exits 1 when the two runs did not use the same thread count, the same decayed weights (their count and a SHA-256 of their names) or the same token ids (a SHA-256 of all of them), when they do not have the same steps and shapes, when a loss is not finite or the two losses of a step differ by more than 2.4e-7 (3e-7 for the LoRA runs), or when a Spearman value is missing or differs by more than 1e-4. Our thread count is not measured: ggml took the online CPUs (`TENSOR_THREADS`, which `run.sh` sets, gives the same since ml#222; `tensor/native/ggml.c`) and reports no count, so `run.sh` writes that number for our side and refuses a process kept to fewer CPUs.

The 1e-4 for Spearman is room for rounding, since the two sides do not add in the same order (our cosine is a float64 dot product on the host, PyTorch's a float32 one) and two near-equal cosines can change places. With n = 1500 pairs, one swap of two neighbours in the order moves rho by 12 d / (n (n² - 1)), d being the gap of their ranks in the other list: about 1.8e-6 for a typical d of 500 and at most 5.3e-6. So 1e-4 is twenty swaps at the worst case, and far below a real difference: the training itself moves rho by 0.019.

The 2.4e-7 is a guard for this run, not a proof, and it has less room than it had. In the first run it was 1.5e-7: PyTorch against itself at 8 and at 4 threads differed by up to 2.4e-8, ours against PyTorch by 6.0e-8, and PyTorch without the weight decay by 2.8e-7. With llamafile's sgemm on, our sums run in another order and a right run is 1.94e-7 away, so the guard moved to 2.4e-7: 1.24 times the gap of a right run and 0.72 times that of a PyTorch without the weight decay (3.35e-7 against this run of ours; at lr 2e-5 a decay of 0.01 moves the weights very little in 63 steps). Both margins are thin, and the loss and the Spearman agree to seven digits either way. The guard was set from these single runs (one run of each, no repeats), so it cannot prove the decay rule: the real protection is the hash of the decayed names, which `summary.py` compares, and `run.sh smoke`. It is tied to these rules, this checkpoint and this engine, and a different run needs its own. The check with room is `run.sh smoke`: four made-up pairs (one of two 221-word sentences, which the limit of 128 tokens cuts) with a weight decay of 1, where the two sides agree to 1.9e-9 and a peer without the decay is 2.4e-6 away, which the smoke run requires to fail. The LoRA smoke run (two steps, the same decay of 1) agrees to 5.2e-8, but a LoRA peer without the decay is 3e-8 away there, inside the tolerance, so the smoke run does not require it to fail; for the LoRA run the check is the full-size one above (1.1e-6) and the hash of the decayed names.

### LoRA fine-tune (ml#137), ours against PEFT

The same run with a LoRA adapter instead of the full model: the model is frozen and an adapter on `q_lin` and `v_lin` of the 6 layers (12 weights, so 24 trained tensors, r 8, alpha 16, no dropout) is trained, with AdamW at lr 2e-4 (the usual 10 times the full rate), a weight decay of 0.01 on every adapter tensor, and the rest of the rules above. `run.sh ours-lora` makes the adapter with `Lora.init` (PEFT's start: `A` uniform, `B` zero) and saves it in PEFT's files (`Lora.save`); `run.sh peer-lora` loads that folder with `PeftModel.from_pretrained(is_trainable=True)` and trains it, so both sides start from the same tensors. After the training our model has the adapter merged in (`Lora.merged`) and PEFT's runs with it in the forward; the Spearman is the same.

| | ours | PEFT (PyTorch) | ours / PEFT |
|---|---|---|---|
| ms per step, `PAD_TO=1` | 1022 | 758 | 1.35 |
| the 63 steps | 70 s | 51 s | 1.38 |
| peak memory | 0.96 GB | 0.94 GB | 1.02 |
| largest loss gap | 1.4e-7 | | |
| Spearman before / after | 0.8150 / 0.8326 | 0.8150 / 0.8326 | |

The loss goes from 0.037646 to 0.025698 on both sides. Against the full fine-tune above, on the same engine and in the same session: our step is 2.0 times shorter (1022 ms against 2035) and PyTorch's 1.8 times (758 ms against 1363); our peak memory is 2.8 times smaller (0.96 GB against 2.73), PyTorch's 2.1 times (0.94 GB against 1.94). The adapter reaches 0.8326 Spearman against 0.8340 for the full model in 63 steps.

The speed goal of ml#137 (from ml#50: training equal to or better than PyTorch) is not met: we are 1.35 times slower than PEFT on ggml CPU, and the work to close it is ml#236 (LoRA on XLA, and on the activations). The likely cost is how the adapter is trained here: in the training forward `Lora.applied` builds `W + (alpha / r) · B A` for each of the 12 weights, so the model's own forward runs on a merged weight, and the backward computes a gradient of the full `[out, in]` size for each merged weight before it gives `A` and `B` theirs. PEFT adds `(alpha / r) · (x Aᵀ) Bᵀ` to the output instead, which never forms a full-size gradient. We did not measure this split. Applying the adapter on the activations is a change in the model's forward and is left for a follow-up. The CPU training path is XLA (`Backend.Xla`, ml#228), not more tuning of ggml, which is for inference and Vulkan. The memory is the same on both sides.

The check of this run is the same as above, with a tolerance of 3e-7: a right run is 1.4e-7 away (the 4-pair smoke run, 5.2e-8) and a PEFT run without the weight decay is 1.1e-6 away, which `summary.py` reports as DIFFER. The runs: PEFT 0.21.2 with accelerate 1.15.0 and psutil 7.2.2 (pinned by `run.sh setup` into the shared `venv-torch`, which an older venv needs run again), and the load average at the start of the four runs was 0.80 (`ours-lora`), 0.71 (`peer-lora`), 0.69 (`ours`) and 0.77 (`peer`). It is one run per row, with about 10% of spread between runs, and ours ran first in each pair. The caveats: one epoch of 1000 pairs, no shuffle or warmup, no dropout (PEFT's `lora_dropout` is 0 here and ml#209 tracks dropout), the adapter starts from ours, and the learning rate of 2e-4 is a choice, not tuned.

## BERT (ml#78), ours against llama.cpp

`bert-base-uncased` (the Hugging Face checkpoint, `float32`), the forward over token ids, so the tokenizer is not in the time. Ours is `transformers/tools/bench_bert.agc` (a warm-up run, then the median of 5 runs, 3 for the batch of 8; each run traces and runs the whole graph) on ml `d14bfd6`'s `transformers/` (`bert.agc` before its memory fixes), same machine as above, load average under 2. llama.cpp (`4f5d1f64b`, build 2172, CPU only: this checkout has no Vulkan build) runs the GGUF that its own converter makes from the same weights (`convert_hf_to_gguf.py --outtype f32`), with `llama-bench -p 128,512 -n 0 -t 8 -r 5`. Every row is one sentence unless it says a batch.

| Row | 1 x 128 tokens | 1 x 512 tokens | 8 x 128 tokens | Load | Peak memory |
|---|---|---|---|---|---|
| ours, ggml CPU | 261 tokens/s | 275 | 342 | 0.65 s | 0.70 GB |
| ours, ggml Vulkan | 888 | 1137 | 1338 | 0.58 s | 0.39 GB |
| llama.cpp CPU, F32 | 615 | 615 | | | |

llama.cpp is about 2.3 times faster than ours on the CPU for one sentence; the gap was not profiled. Two things that could account for part of it, untested: ours traces and plans the whole graph at every call, and its attention is the unfused one. The Vulkan row has no llama.cpp counterpart on this machine. The numbers are the same weights in F32 everywhere; half weights would halve the memory (see [Half weights](#half-weights-ml86)) and were not measured for BERT. Correctness: the last hidden state and the pooler output match `BertModel` to 5.0e-5 on CPU and 2.8e-5 on Vulkan (`transformers/tools/bert_check.agc`).

## Qwen3-0.6B (ml#78), ours against llama.cpp

Since ml#148 Vulkan multiplies in fp16 by default, which is faster and does not keep the ids of Hugging Face's greedy `generate`; every check against Hugging Face or whisper.cpp runs with it off (exact `float32`). The tables below were measured before it: their Vulkan rows named `F32` ran exact (`bench_qwen.sh` row `ours-vulkan-exact` now), and the rows named `vulkan_fp16` ran what is now the default (`ours-vulkan`, `ours-vulkan-f16`). The rows of the script are renamed, not re-measured here.

`Qwen/Qwen3-0.6B`, greedy generation, one sequence. Ours is `transformers/tools/bench_qwen.agc` on ml `a72f56c` (`Generate` with the cache written in place, ml#133; the weight layout of ml#123): the load, one warm-up pass that compiles the steps, then a timed pass: a prompt of 128 token ids in one step, then 64 one-token steps after it (the cache grows from 128 to 192 tokens). llama.cpp (`e348f05cf`, built out of tree from a `git archive` of its HEAD, once with the CPU only and once with `-DGGML_VULKAN=ON`, `llama-bench` only) runs the GGUFs that its own `convert_hf_to_gguf.py` makes from the same weights, run as `python3 -I`, with `llama-bench -p 128 -n 64 -t 8 -r 1` (`-ngl 99` on Vulkan): `pp128` is the prompt and `tg64` the 64 tokens, starting from an empty cache. Ours always computes in `float32` (the `bfloat16` file is read as `F32`), so the like-for-like row is llama.cpp's F32; its F16 row reads half the memory per token and is what a user would run next. The Qwen3 rows of this section and of Half weights were taken with an older version of `bench_qwen.sh` (llama.cpp at its 8 default threads against ours at 4, `-p 128 -n 64` from an empty cache, an earlier llama.cpp checkout) and are not comparable to the equal-terms rows of [Q4_K_M](#q4_k_m-against-llamacpp-ml291); the F32 and F16 rows were not taken again, but the F16 pair is in that section.

[`benchmarks/bench_qwen.sh`](bench_qwen.sh) runs the six rows interleaved, one process per row and round (5 rounds, so a change of load hits all rows), with nothing else timing, and prints the medians. Same machine as above, the other jobs of the machine stopped, but the whisper.cpp container held 1.96 GB of the 3 GB of VRAM, so the F32 weights of the Vulkan rows were partly in system memory (the VRAM paragraph below has them with the VRAM free). The load average was 1.9 at the start, and the benchmark's own 8 threads are most of the rest. Peak memory is `/usr/bin/time`'s maximum resident set of the whole process: llama.cpp maps the GGUF, so on Vulkan its weights are not in the process, ours are loaded.

| Row | Prompt, 128 tokens | 64 tokens after it | Load | Peak memory |
|---|---|---|---|---|
| ours, ggml CPU, F32 | 61.9 tokens/s | 9.17 tokens/s | 2.65 s | 2.68 GB |
| llama.cpp CPU, F32 | 123.0 | 10.23 | | 2.89 GB |
| llama.cpp CPU, F16 | 90.0 | 18.09 | | 1.49 GB |
| ours, ggml Vulkan, F32, exact (fp16 off) | 215.1 | 10.34 | 1.71 s | 0.86 GB |
| llama.cpp Vulkan, F32 | 412.1 | 13.72 | | 0.13 GB |
| llama.cpp Vulkan, F16 | 455.9 | 26.33 | | 0.13 GB |

On the same F32 weights llama.cpp is 2.0 times faster on the CPU prompt and 1.1 times on its tokens, and on Vulkan 1.9 times on the prompt and 1.3 times on the tokens. Correctness: the ids match Hugging Face's greedy `generate` on five prompts on both backends (`transformers/tools/generate_check.agc`). The tokens on Vulkan were measured again after ml#180: [A Qwen3 token on Vulkan by the length of the cache](#a-qwen3-token-on-vulkan-by-the-length-of-the-cache-ml174).

### Where a Qwen3 step goes

`transformers/tools/profile_qwen.agc MODEL_DIR PREFILLS DECODES [cpu|vulkan|vulkan32]` loads the model, compiles the steps with one warm-up pass (a prompt of 128 ids, then 8 one-token steps), and then runs the timed prefills and decode steps. It writes `MARK prefill N`, `MARK decode N` and `MARK end` to stderr, so `python3 -I benchmarks/vk_perf.py STDERR.txt` splits the kernel log of `GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1` into a prefill table and a decode table (calls and GPU time per kernel, averaged per graph). With `AGENTICA_VM_TRACE=profile` and `AGENTICA_VM_TRACE_FILE`, `step_profile.py` subtracts a run with 0 steps from a full one, to get the host side (`ml.*` sections). The numbers below are on ml `a72f56c`, a quiet machine, llama.cpp `e348f05cf`.

One decode step on Vulkan with 128 tokens in the cache: 96.7 ms of wall time for ours against 74.7 ms for llama.cpp; the GPU time of the kernels is 91.0 against 74.7 ms. The matrix-vector products with a weight take 74.3 against 71.0 ms, so what is left is about 17 ms of other kernels against about 4. In ours the step has the copies (`CONT`) 4.2 ms, the two attention products 4.0 ms, the pick (`ARGMAX`) 2.4 ms and the rest under 1 ms each, and 5.7 ms outside the graph. The new cache rows are written in place by `set_rows` (56 calls, 0.6 ms a step; the cache is consumed by the step, `Trace.run`, `tensor/DESIGN.md`) and the RMS norm is one primitive (`ggml_rms_norm`).

A prefill of 128 tokens, with the VRAM free (5 rounds): 514 ms of kernels for ours against 282 to 307 ms for llama.cpp F32. The weight matmuls are 2.70 ms for a `m=3072 k=1024` product (151 ms for 56), 2.98 for `m=1024 k=3072`, 1.02 for 1024x1024, and 1.98 and 1.96 for the two 2048 ones; llama.cpp's are 1.59, 1.52, 0.51, 0.97 and 1.09 ms. The output head is 22.4 ms for ours and 21.2 for llama.cpp. Outside the products, ours has `CONT` 25.5 ms (478 calls) and the two attention products 25 ms; llama.cpp has flash attention and the fused rope, norm and `SET_ROWS` kernels in about 30 ms, and no `CONT`. That table was measured with fp16 products off, which was most of the gap (2.7 ms against 1.8 ms for the same product); since ml#148 `tensor/` runs with them on (see [Half weights](#half-weights-ml86) for the prompt with F16 weights).

**fp16.** ggml-vulkan runs F32 matmuls on a device with fp16 support in a faster mode, and `tensor/` turns it on by default (ml#148); `Tensor.vulkan_fp16(false)` is the exact `F32` choice (`profile_qwen.agc` mode `vulkan32`). llama.cpp has it on: its `pp128` is 404 tokens/s with it and 262 with `GGML_VK_DISABLE_F16=1`. ggml's own `test-backend-ops perf` (`m=4096 n=512 k=14336`, F32 activations) gives 124 ms a run (485 GFLOPS) with fp16 on and 220 ms (273 GFLOPS) with it off, and 81.5 ms (738 GFLOPS) with F16 weights.

**VRAM.** The card has 3 GB (`/sys/class/drm/card*/device/mem_info_vram_total`), and on this machine the whisper.cpp container holds about 1.96 GB of it, so the 2.4 GB F32 Qwen3 model spills about 1.2 GB of weights into system memory (GTT), which makes a product 2.5 to 2.8 ms instead of 1.5 (ml#130, ml#141). Every F32 Qwen3 row measured with the container up is the spill case, as the table above. With the container stopped (`mem_info_vram_used` 17 MB) ours on Vulkan F32 exact runs 237.7 tokens/s for the prompt and 10.82 for the tokens, against 457.5 and 14.44 for llama.cpp F32. Print `mem_info_vram_used` before a benchmark (`bench_qwen.sh` does). The GPU was busy (`gpu_busy_percent` 99 percent) and mostly in its top clock state in both our prefill loop and `llama-bench`, so the clock is not the gap.

### A Qwen3 token on Vulkan by the length of the cache (ml#174)

The tables above are from ml `a72f56c`. Since then the cache became `F16`, ml#177 and ml#179 took host time out of a step, and ml#180 made a one-token step read the rows of the cache in use and not its whole capacity. The tokens on Vulkan were measured again on 2026-11-24 on the same machine, the other jobs of the machine stopped (load average 0.6 for the 128 column and 1.3 to 1.9 for the 2048 one, `mem_info_vram_used` 17 MB): Qwen3-0.6B, `F32` weights, exact `float32` products (`Tensor.vulkan_fp16(false)`), the `F16` cache, one sequence.

Ours is a loop over `Generate.step` (a script that is not in the repository; `bench_qwen.agc` is the same loop with a prompt of 128 and 64 steps): a prompt of N token ids in one step, 8 one-token steps that compile the step, then 24 timed one-token steps. llama.cpp is `8ca5a80` on the F32 GGUF, `llama-bench -fa 1 -p 0 -n 32 -d N -r 1 -ctk f16 -ctv f16`. One process per run; ours before ml#180, ours with it and llama.cpp alternated, 3 runs each; the median and the range, in ms per token:

| | N = 128 | N = 2048 |
|---|---|---|
| ours, ml `a6092fb` (before ml#180) | 71.92 (71.75 to 74.83) | 84.71 (84.29 to 84.71) |
| ours, ml#180 | 72.71 (72.54 to 73.04) | 80.42 (79.75 to 80.42) |
| llama.cpp Vulkan, F32, `F16` cache | 70.62 (69.72 to 70.82) | 79.35 (77.93 to 79.61) |

That is 13.8 and 12.4 tokens/s for ours with ml#180 and 14.2 and 12.6 for llama.cpp: ours needs 3% more time for a token at 128 and 1.3% more at 2048. When ml#174 was opened it was 84.5 against 70.0 ms and 90.6 against 77.1. At 128 the step of ml#180 is the graph of main (the cache has 256 rows and the step reads them all), and the two rows there differ by less than their ranges. At 2048 the capacity is 4096 rows: main reads all of them, ml#180 reads 2560 and llama.cpp 2304, and the per-kernel timer gives flash attention 8.9 ms of a token, from 13.9.

Not measured again: the CPU rows, the prompt, the load, the peak memory and the half weights.

## Qwen3.5-0.8B (ml#150), ours only

`Qwen/Qwen3.5-0.8B` (revision and hash in `transformers/DESIGN.md`, `float32` weights), greedy generation, one sequence, `Generate` of `pkg/transformers/generate_qwen35`. Ours is `transformers/tools/bench_qwen35.agc`, measured with Agentica 0.39.0 on the code of ml#277 (on top of ml `c501dd3`), timed as the Qwen3 tool is: the load, one warm-up pass that compiles the steps, then a prompt of 128 token ids in one step and 64 one-token steps after it. The model has 18 Gated DeltaNet layers and 6 full-attention layers, so only 6 layers keep a cache that grows. Same machine as above, one run per row (the CPU row twice: 74.9 and 75.6 tokens/s for the prompt, 7.93 and 7.61 for the tokens), the load average between 1.3 and 3 (other jobs of the machine were running); peak memory is `/usr/bin/time`'s maximum resident set. There is no llama.cpp row: it was not run for this model. The checks against Hugging Face run with Vulkan exact (fp16 off), so that row is the one that was checked.

| Row | Prompt, 128 tokens | 64 tokens after it | Load | Peak memory |
|---|---|---|---|---|
| ours, ggml CPU | 75.2 tokens/s | 7.8 tokens/s | 2.7 s | 3.33 GB |
| ours, ggml Vulkan, exact (fp16 off) | 169.8 | 8.7 | 2.7 s | 1.31 GB |
| ours, ggml Vulkan, fp16 products | 238.8 | 8.7 | 2.6 s | 1.33 GB |

Correctness: the greedy ids match Hugging Face's `generate` on six prompts of 5 to 375 tokens on both backends (`transformers/tools/generate_qwen35_check.agc`).

```sh
agentica --max-steps 100000000000 transformers/tools/bench_qwen35.agc MODEL_DIR cpu                          # ggml CPU
VULKAN_FP16=0 agentica --max-steps 100000000000 transformers/tools/bench_qwen35.agc MODEL_DIR vulkan   # Vulkan, exact
VULKAN_FP16=1 agentica --max-steps 100000000000 transformers/tools/bench_qwen35.agc MODEL_DIR vulkan   # Vulkan, fp16 products
```

## LFM3-1.7B (ml#275), ours only

`LiquidAI/LFM3-1.7B` (revision in `transformers/DESIGN.md`, `F16` weights, 3.4 GB), greedy generation, one sequence, `Generate` of `pkg/transformers/generate_lfm3`. Ours is `transformers/tools/bench_lfm3.agc`, timed as the Qwen3.5 tool is (the load, one warm-up pass that compiles the steps, then a prompt of 128 token ids in one step and 64 one-token steps after it). Same machine as above, one run per row, with other jobs of the machine running (the load average was between 3.9 and 6.7); peak memory is `/usr/bin/time`'s maximum resident set. There is no llama.cpp row. The model has 13 convolution layers (a state that does not grow), 5 sliding-window layers and 6 global ones, which keep a cache that grows (the sliding ones read the whole cache under a window mask).

| Row | Prompt, 128 tokens | 64 tokens after it | Load | Peak memory |
|---|---|---|---|---|
| ours, ggml CPU | 21.1 tokens/s | 5.9 tokens/s | 6.3 s | 3.78 GB |
| ours, ggml Vulkan, exact (fp16 off) | 71.0 | 7.7 | 7.3 s | 0.85 GB |
| ours, ggml Vulkan, fp16 products | 115.9 | 8.5 | 6.7 s | 0.86 GB |

Correctness: the greedy ids match the official code's `generate` on prompts of 5 to 601 tokens (two are past the sliding window) on both backends (`transformers/tools/generate_lfm3_check.agc`).

```sh
agentica --max-steps 100000000000 transformers/tools/bench_lfm3.agc MODEL_DIR cpu                          # ggml CPU
VULKAN_FP16=0 agentica --max-steps 100000000000 transformers/tools/bench_lfm3.agc MODEL_DIR vulkan   # Vulkan, exact
VULKAN_FP16=1 agentica --max-steps 100000000000 transformers/tools/bench_lfm3.agc MODEL_DIR vulkan   # Vulkan, fp16 products
```

## Half weights (ml#86)

`WEIGHTS=f16` keeps the big matrices (rank 2, at least 100000 values) of the checkpoint as F16 weights on ggml (`Load`, hub/DESIGN.md); `VULKAN_FP16=1` lets Vulkan multiply in fp16 (since ml#148 that is the default and `VULKAN_FP16=0` the exact choice; `Tensor.vulkan_fp16`). The tools read both from the environment, and `bench_qwen.sh` and `bench_whisper.sh` have the rows: `ours-cpu-f16`, `ours-vulkan-f16` and `ours-vulkan-exact` for Qwen3, `agentica-ggml-cpu-f16`, `agentica-ggml-vulkan-f16` and `agentica-ggml-vulkan-exact` for whisper. The tokens of the Hugging Face goldens stay the same with F16 weights and exact products; fp16 products do not keep them (they also break them with F32 weights), so the checks run with `Tensor.vulkan_fp16(false)`.

Qwen3-0.6B, `bench_qwen.sh` as above, 2 rounds, on ml `e362042`; the load average was 3.5 at the start and 2.6 at the end, and the llama.cpp rows ran between ours, so the numbers carry a few percent of noise. The whisper.cpp container held 1.96 GB of the 3 GB of VRAM, so the Vulkan rows with F32 weights are the spill case of the VRAM paragraph above; with the VRAM free (5 rounds, medians) ours on Vulkan gives 231.0 tokens/s for the prompt and 10.81 for the tokens with F32 weights, 242.0 and 16.36 with F16 weights, and 502.0 and 16.43 with F16 weights and `vulkan_fp16`:

| Row | Prompt, 128 tokens | 64 tokens after it | Load | Peak memory |
|---|---|---|---|---|
| ours, ggml CPU, F32 | 59.2 tokens/s | 8.97 tokens/s | 2.13 s | 2.53 GB |
| ours, ggml CPU, F16 weights | 57.5 | 15.45 | 1.65 s | 1.43 GB |
| llama.cpp CPU, F32 | 130.6 | 10.48 | | 2.89 GB |
| llama.cpp CPU, F16 | 90.8 | 17.61 | | 1.49 GB |
| ours, ggml Vulkan, F32, exact (fp16 off) | 216.9 | 10.07 | 2.7 s | 0.85 GB |
| ours, ggml Vulkan, F16 weights | 243.5 | 16.23 | 2.28 s | 0.58 GB |
| ours, ggml Vulkan, F16 weights and `vulkan_fp16` | 372.8 | 16.16 | 2.03 s | 0.58 GB |
| llama.cpp Vulkan, F32 | 416.2 | 13.87 | | 0.13 GB |
| llama.cpp Vulkan, F16 | 450.7 | 26.25 | | 0.13 GB |

On Vulkan F16 weights make our tokens 1.6 times faster (llama.cpp's own gain there is 1.9 times), the prompt 12 percent faster, and 1.7 times faster with `vulkan_fp16` as well. On the CPU this window measured +72 percent in the tokens (llama.cpp's gain is 1.7 times) and no gain in the prompt, but the PR 2 window (ml#134) showed none (5.6 and 6.1 against 7.6 and 5.9 tokens/s, with other loads on the machine): a CPU gain is not claimed until a repeat. The Vulkan gain was the same in both (+39 percent then, +60 percent now, on a different F32 base).

whisper-tiny, `bench_whisper.sh` (`ONLY=agentica-ggml`, 3 runs, whisper-tiny at 384 wide, so its weights are small): the tokens equal Hugging Face's in all rows, including `vulkan_fp16`, on the two clips.

| Row | RTF jfk | RTF es-hola |
|---|---|---|
| ours, ggml CPU | 0.116 | 0.419 |
| ours, ggml CPU, F16 weights | 0.110 | 0.417 |
| ours, ggml Vulkan | 0.069 | 0.238 |
| ours, ggml Vulkan, F16 weights | 0.065 | 0.259 |
| ours, ggml Vulkan, F16 weights and `vulkan_fp16` | 0.056 | 0.217 |

Tiny gains 6 percent on jfk and nothing on es-hola from F16 weights alone: its decoder steps are short and its weights small, so the weights are not what it waits for.

The load. The conversion of a BF16 or F32 file to F16 runs on up to 8 threads (one run of whole chunks of the tensor each, `tensor/native/float_file.h`). Loading on the CPU takes 3.1 s for whisper-medium (2.3 to 2.6 s with F32 weights) and 1.5 s for Qwen3-0.6B (about 1.5 s with F32 weights).

## Q4_K_M against llama.cpp (ml#291)

Qwen3-0.6B as a Q4_K_M GGUF (456 MiB, Q4_K and Q6_K matrices, `F32` norms), read by both sides: ours with `weights: Dtype.Q4_K` (`WEIGHTS=q4_k`, ml#126: the blocks stay packed on ggml) and llama.cpp's `llama-bench`. The F16 rows are the same model in half weights.

- **Machine and tools.** The machine of the sections above (4 cores, 8 threads, the Vega iGPU; VRAM free, 17 MB used before the run), nothing else timing, Agentica 0.39.1. Ours is ml `9517f8d` (the code of every row; the scripts are the ones in the tree) over ggml `v1.0.0` (`417a024b4fb6`), built by `tensor/tools/build-engine.sh`: `GGML_NATIVE=OFF` (AVX2, FMA and F16C), `GGML_OPENMP=OFF`, `GGML_LLAMAFILE=ON`, no BLAS, fp16 products on Vulkan. llama.cpp is `711f9f4c117042570ef66ef58fba1f15835a331b` (official source, which vendors ggml 0.29.0, built out of tree with `cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF` and, for Vulkan, `-DGGML_VULKAN=ON`, so its defaults: `GGML_NATIVE=ON` for this CPU, `GGML_OPENMP=ON`, `GGML_LLAMAFILE=ON`, `GGML_CPU_REPACK=ON`, no BLAS; the recipe of the builds is under [Run it](#run-it)). The two ggml versions differ, and that was not isolated.
- **The file.** `llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-q4_k_m.gguf Q4_K_M` with that same commit; sha256 of the F16 file `72d3cea51a8d269bc842687649700042e4d9744d13b1eb545cf58965baf5e4cc`, of the Q4_K_M one `944461ec64c0a644b8c440f322d5347ccb031f3d4886f8c7510033e52d0db056`.
- **Equal terms.** 4 threads on the CPU rows of both sides (`THREADS`, the physical cores: `TENSOR_THREADS=4` and `llama-bench -t 4`; ours defaults to the physical cores, llama.cpp to every thread of the machine); the script prints ours' `cpu_threads=4` (`ggml_cpu_threads`) and llama-bench's `n_threads=4`. The Vulkan rows run the graph on the GPU and ours has no CPU thread pool there (`-t 4` is still passed to llama-bench). `llama-bench -fa 1 -ctk f16 -ctv f16` (flash attention and an `F16` cache, as ours); its two rows are `-p 128 -n 0` (the prompt from an empty cache) and `-p 0 -n 64 -d 128` (64 tokens after 128 of context), the same as ours: a prompt of 128 ids in one step, then 64 one-token steps (the cache grows from 128 to 192 rows). The csv of `llama-bench` says `n_threads=4 flash_attn=1 type_k=f16 type_v=f16`.
- **Method.** `ROWS=q4k,q4km,f16 benchmarks/bench_qwen.sh`, 5 rounds; every round runs each pair of ours and llama.cpp once, and the two swap places from one round to the next (ours first in odd rounds), alone on the machine, under a lock and a 5 GB memory cap (`flock LOCK systemd-run --user --scope -p MemoryMax=5G sh benchmarks/bench_qwen.sh ...`). The load average at the end of the rounds was 3.3 to 3.9 (the 4 threads of the row and a few more).

Medians of the 5 rounds:

| Row | Prompt, 128 tokens | 64 tokens after it | Load | Peak RSS | Peak GPU memory |
|---|---|---|---|---|---|
| ours, ggml CPU, Q4_K | 94.3 tokens/s | 33.63 tokens/s | 0.55 s | 0.74 GB | |
| llama.cpp CPU, Q4_K_M | 86.0 | 37.74 | | 0.75 GB | |
| ours, ggml CPU, F16 weights | 78.6 | 18.40 | 1.88 s | 1.42 GB | |
| llama.cpp CPU, F16 | 83.8 | 19.36 | | 1.51 GB | |
| ours, ggml Vulkan, Q4_K | 463.8 | 42.70 | 0.67 s | 0.44 GB | 565 MB |
| llama.cpp Vulkan, Q4_K_M | 553.9 | 50.74 | | 0.15 GB | 627 MB |
| ours, ggml Vulkan, F16 weights | 481.2 | 21.61 | 2.17 s | 0.63 GB | 1260 MB |
| llama.cpp Vulkan, F16 | 581.0 | 23.54 | | 0.15 GB | 1605 MB |

The ratio of ours to llama.cpp in each round (the rounds in order), then its minimum and maximum:

| Pair | Prompt | Tokens |
|---|---|---|
| CPU, Q4_K | 1.14 1.10 1.12 1.10 1.19 (1.10 to 1.19) | 0.89 0.89 0.88 0.89 0.90 (0.88 to 0.90) |
| CPU, F16 | 0.94 1.10 0.93 0.91 0.87 (0.87 to 1.10) | 0.95 0.96 0.93 0.99 0.94 (0.93 to 0.99) |
| Vulkan, Q4_K | 0.85 0.83 0.84 0.82 0.85 (0.82 to 0.85) | 0.84 0.82 0.84 0.84 0.83 (0.82 to 0.84) |
| Vulkan, F16 | 0.84 0.82 0.81 0.83 0.83 (0.81 to 0.84) | 0.93 0.92 0.91 0.92 0.91 (0.91 to 0.93) |

The GPU memory is the most that VRAM plus GTT in use (`mem_info_vram_used` and `mem_info_gtt_used`, sampled every 0.1 s) rose above its value before the row. llama.cpp maps the file, so its Vulkan weights are in that column and not in its RSS; ours pass through the process while they load.

- **Speed.** On the CPU Q4_K is about 1.1 times llama.cpp on the prompt (1.10 to 1.19; within the noise of the CPU prompt of the F16 pair, whose ratio moves from 0.87 to 1.10) and 0.89 times on the tokens (every round, 0.88 to 0.90); the F16 CPU rows are 5 percent behind on the tokens (0.93 to 0.99), and their prompt ratio moves from 0.87 to 1.10, so it is noise. On Vulkan ours is at 0.82 to 0.85 of llama.cpp on the prompt and the tokens of Q4_K, and at 0.81 to 0.84 and 0.91 to 0.93 with F16 weights. Against our own F16 weights, Q4_K gives 1.8 times the tokens on the CPU and 2.0 times on Vulkan, at half the memory (0.74 against 1.42 GB, 565 against 1260 MB of GPU) and a third of the load time on the CPU.
- **Memory.** On Vulkan the GPU memory is 62 MB less than llama.cpp's for Q4_K_M (565 against 627 MB) and 345 MB less for F16 (1260 against 1605 MB); the peak host RSS is 0.29 GB more (0.44 against 0.15 GB: the file's bytes pass through the process at load). On the CPU the RSS is the same (0.74 and 0.75 GB).
- **The tokens gap is about 3 ms on every row.** In milliseconds a token, ours is behind by 3.2 ms (CPU Q4_K), 2.7 ms (CPU F16), 3.7 ms (Vulkan Q4_K) and 3.8 ms (Vulkan F16): about the same for every weight type, so it does not come from reading Q4_K. The per-kernel timer of a Vulkan decode step (`GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1`; ours `TENSOR_THREADS=4 WEIGHTS=q4_k agentica tools/profile_qwen.agc Q4_K_M.gguf 0 8 vulkan` with stderr to a file for `python3 -I vk_perf.py FILE`; llama.cpp `llama-bench -m Q4_K_M.gguf -p 0 -n 8 -d 136 -fa 1 -ctk f16 -ctv f16 -t 4 -ngl 99 -r 1`, one run each) gives the same weight kernels at the same speed (`MUL_MAT_VEC q4_K m=3072 k=1024`, 56 calls: 4.36 ms ours, 4.37 ms llama.cpp; the head `q6_K m=151936`: 5.16 against 5.42 ms). The flash attention is 1.87 against 1.68 ms in these runs, a difference within what one run shows. The kernels add up to 22.5 ms for ours and 21.1 ms for llama.cpp, and llama.cpp's sum is more than its 19.7 ms a token without the logger: the logger changes the timing, so the sums cannot be taken from the wall time, and where the 3.7 ms goes is not located. Ours has a few more small kernels where llama.cpp fuses `RMS_NORM`, rope and `SET_ROWS`. Ours is at 137 to 144 rows of cache during those steps (the warm-up of 128 and 8 comes first); the runs behind the numbers here used `-d 128` for llama.cpp, 8 rows less, and `-d 136` is the match. Filed as ml#298.

The cheap gaps: none found in the weight products, so the code of the model is unchanged.

## Layers on the CPU against llama.cpp `-ngl`, and the KV cache in host memory against `-nkvo` (ml#310, ml#312, ml#320, ml#327, ml#85)

The Q4_K_M Qwen3-0.6B of the section above on Vulkan, with the blocks from the `n`th on on the CPU: ours `Qwen3.open(..., layers_on_device: n)` (`AGENTICA_ML_LAYERS_ON_DEVICE`: the first `n` blocks on the card, the table on the CPU, and the head and the final norm on the card for `n` of 1 or more, on the CPU for 0), llama.cpp `llama-bench -ngl N`. llama.cpp's rule (`llama-model.cpp`: `i_gpu_start = n_layer + 1 - ngl`, the output layer being layer `n_layer`): the output layer (the head and the final norm) is on the GPU for `-ngl` 1 or more, the first blocks and the input embedding are on the CPU, and only `-ngl 0` puts the head on the CPU. So ours puts the last blocks on the CPU and llama.cpp the first ones (`-ngl 27` leaves blocks 0 and 1 on the CPU), and `-ngl N` holds N - 1 blocks: the pairs of equal blocks are `n` = 26 with `-ngl 27` (26 blocks and the head on the card), `n` = 13 with `-ngl 14` (13 blocks) and `n` = 0 with `-ngl 0` (none, the head on the CPU). The all-device pair is the control: it shows the default path (no scheduler) next to the section above. The last pair keeps every block on the card and the KV cache in host memory: ours `Generate.open(..., kv: Placement.Kv.Host)` (`AGENTICA_ML_KV=host`) against `llama-bench -ngl 99 -nkvo 1` (`--no-kv-offload`).

- **Machine and tools.** The machine of the sections above, nothing else timing (the other lines frozen), Agentica 0.40.0-beta.1, ours at the code of the PR (ml#345, `49c4317`) over ggml `v1.0.0` (`417a024b4fb6`), the same engines. llama.cpp commit `711f9f4c117042570ef66ef58fba1f15835a331b`, built with `-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_VULKAN=ON` (the CPU build without the Vulkan flag); `llama-bench` ran with 4 threads, flash attention on, F16 KV, generation at depth 128 (the script prints these lines), and ours with `cpu_threads=4`. The GPU clock was at its default (`auto`).
- **Method.** `ROWS=vulkan-q4k THREADS=4 benchmarks/bench_qwen.sh QWEN3_DIR GGUF_DIR BUILD_CPU BUILD_VULKAN 5`: 5 rounds, each pair once per round, the two swapping places from one round to the next, under a lock and a 5 GB memory cap. The load average after the run was 1.6.

Medians of the 5 rounds (128 prompt tokens, then 64 tokens after it):

| Row | Prompt | Tokens | Load | Peak RSS | Peak GPU memory |
|---|---|---|---|---|---|
| ours, all blocks on the card (control) | 603.8 tokens/s | 49.96 tokens/s | 0.52 s | 0.44 GB | 579 MB |
| llama.cpp, `-ngl 99` (control) | 742.4 | 61.47 | | 0.15 GB | 627 MB |
| ours, KV cache in host memory | 442.9 | 6.68 | 0.52 s | 0.45 GB | 512 MB |
| llama.cpp, `-ngl 99 -nkvo 1` | 606.3 | 7.46 | | 0.17 GB | 578 MB |
| ours, `n` = 26 | 397.5 | 27.48 | 0.50 s | 0.54 GB | 453 MB |
| llama.cpp, `-ngl 27` | 617.1 | 58.17 | | 0.15 GB | 604 MB |
| ours, `n` = 13 | 173.2 | 18.35 | 0.51 s | 0.65 GB | 290 MB |
| llama.cpp, `-ngl 14` | 183.3 | 26.49 | | 0.16 GB | 591 MB |
| ours, `n` = 0 | 130.3 | 35.38 | 0.50 s | 0.89 GB | 2 MB |
| llama.cpp, `-ngl 0` | 91.5 | 45.37 | | 0.17 GB | 697 MB |

The ratio of ours to llama.cpp in each round (the rounds in order), then its minimum and maximum:

| Pair | Prompt | Tokens |
|---|---|---|
| all on the card | 0.81 0.81 0.80 0.82 0.81 (0.80 to 0.82) | 0.83 0.81 0.81 0.82 0.80 (0.80 to 0.83) |
| KV cache in host memory and `-nkvo 1` | 0.71 0.78 0.74 0.73 0.73 (0.71 to 0.78) | 0.90 0.89 0.89 0.91 0.90 (0.89 to 0.91) |
| 26 blocks and `-ngl 27` | 0.63 0.66 0.65 0.65 0.61 (0.61 to 0.66) | 0.54 0.48 0.47 0.46 0.54 (0.46 to 0.54) |
| 13 blocks and `-ngl 14` | 0.93 0.96 0.96 0.92 0.96 (0.92 to 0.96) | 0.70 0.69 0.68 0.62 0.67 (0.62 to 0.70) |
| 0 blocks | 1.44 1.42 1.46 1.40 1.45 (1.40 to 1.46) | 0.75 0.73 0.81 0.81 0.78 (0.73 to 0.81) |

The GPU memory is measured as in the section above.

- **The default is unchanged.** The all-device pair takes the plain path (`ggml_run` without a scheduler: `GgmlRaw.mixed_runs` stays at 0, which a test checks) and its ratios (0.80 to 0.82 on the prompt, 0.80 to 0.83 on the tokens) are those of the section above (0.82 to 0.85 and 0.82 to 0.84).
- **Memory is what the knob buys.** The GPU memory falls with `n`: 579, 453, 290 and 2 MB, which is the weights and the KV cache of the blocks that moved; a tied head on the card is a second copy of the table (about 128 MB for this model). llama.cpp's GPU memory does not fall as much (627, 604, 591 and 697 MB: at `-ngl 0` it is more than with every layer); the host RSS rises with `n` on both sides, ours from 0.44 to 0.89 GB.
- **The KV cache in host memory (ml#334, ml#345).** With every block on the card and the cache in RAM a token takes 149.7 ms for ours against 134.0 ms for llama.cpp `-nkvo 1` (0.90, 0.89 to 0.91), and the prompt runs at 0.73 of its speed; with the cache on the card a token takes 20.0 ms (0.81 of llama.cpp's 16.3 ms). Both engines pay for the split of the scheduler, about 2.3 ms each (`GGML_VK_PERF_LOGGER`: the same kernel times), and a layer is 2 splits in each: ours made 4 (113 a step, 255 ms in the profile harness) until the step wrote the keys and the values of a layer after making both (ml#345: 57 splits, 161 ms in the profile harness). The card's peak is 512 MB against 579 MB for the control: the weights stay, and the prompt step still makes the whole cache on the card at the capacity of the prompt before it is copied to RAM (llama.cpp's 578 MB against its 627 MB has no such step).
- **With the head on the card (ml#327).** The first layout (ml#323) put the table, the head and the final norm on the CPU for every `n` below 28, which is not llama.cpp's rule. With 27 blocks on the card a token took 48.7 ms (0.36 of llama.cpp's `-ngl 27`, 17.4 ms): the card was idle during the head on the CPU (5 ms for the product, then the argmax chain over the 151936 logits) and the last blocks, and its governor lowered the clock (`pp_power_profile_mode` `BOOTUP_DEFAULT`, `BUSY_SET_POINT` 70: the clock mean was 301 MHz, 0% of the samples at 800 MHz or more, and the GPU kernels took 3 times as long as in an all-device step, against 1036 MHz and 96% for llama.cpp's `-ngl 27`). With the head on the card, 26 blocks took 31.1 ms a token in the first measure of that layout (0.56 of llama.cpp's 17.3 ms); in the rows above, 36.4 ms against 17.2 ms (0.48, 0.46 to 0.54), the card's clock being what moves it from one run to the next.
- **`n` = 0.** With no block on the card the table, the head and the final norm are on the CPU too (ml#320), as in llama.cpp's `-ngl 0`: 28.3 ms a token against 22.0 ms for llama.cpp (0.78, 0.73 to 0.81), and 1.44 times its speed on the prompt. Before ml#320 this row took 52 ms a step in the profile harness, with the head on the card; the same harness gives 29.8 ms now, 31.4 ms for our CPU-only engine.
- **What is left of the gap, and the clock.** `n` = 26 is 36.4 ms against 17.2 and `n` = 13 is 54.5 ms against 37.7 (the 0.48 and 0.68 of the table; 31.1 and 57.7 ms in the first measure). The card still works in bursts between the CPU splits, and its clock stays low: at `n` = 14 (ml#323) the kernels run at 200 MHz (`MUL_MAT_VEC q4_K m=3072`: 283 us against 65 us in an all-device step, which ramps the clock to 1.0 to 1.2 GHz). It holds with one CPU thread, so it is not the scheduler. The control with the clock forced high (`power_dpm_force_performance_level=high`, ml#323, 300 steps of the profile harness, 3 runs): `n` = 14 took 31.4, 31.4 and 32.2 ms a step instead of 55, and `n` = 0 29.8 and 29.2 ms. That control was not run for this layout, so how much of the remaining gap at `n` = 26 is the clock and how much the cost of a mixed step (the splits and their copies) is not separated; the rows of this section are at the default clock, as the earlier ones.
- **What a first measure of this section found (ml#312).** The same rows before the placement was fixed gave 8.5 and 6.2 tokens per second at `n` = 14 and 0 (0.33 and 0.13 of llama.cpp): the scheduler made 116 splits per step, because the residual stream of the CPU blocks sat on the device and the checks of the graph were copied between the layers. The stream now stays on the CPU, a table of constants goes where its consumers are, and the copies of the checks come after all of them: 3 to 7 splits a graph (`tensor/test/ggml_mixed_test.agc` pins the splits of a stack of blocks).
- **Where `n` pays.** The knob is for a model that does not fit in the memory of the card, not for speed: at `n` = 13 the tokens run at 18 per second instead of 50 and 275 MB of the card's memory are free.

## ResNet-18 and ResNet-50 (ml#83), ours only

`microsoft/resnet-50` and `microsoft/resnet-18` (revisions and hashes in `transformers/DESIGN.md`, `model.safetensors`, `float32`), one batch of 224 by 224 images of constant pixels, with the batch norms folded into the convolutions at load, as `ResNet.open` does. Ours is `transformers/tools/bench_resnet.agc` (Agentica 0.38.0), same machine as above, nothing else running (`--bench` slot): one warm-up run, then the median of 5 timed runs (3 for a training step). "forward" is `ResNet.run` (one graph), whose logits are read back; "forward and backward" is the forward and backward of the folded model, one `Trace.value_and_grad` of a loss (the mean square of the logits) over every weight and the images, all gradients read back, with no optimizer step. The weights are `F32`, Vulkan multiplies in exact `float32` (the tool calls `Tensor.vulkan_fp16(false)` first; the Vulkan rows were measured with that), and there is no peer row: PyTorch and llama.cpp were not run for this model.

| Row | batch 1 | batch 32 |
|---|---|---|
| ResNet-50, ggml CPU, forward | 130 ms (7.6 images/s) | 3949 ms (8.1) |
| ResNet-50, ggml CPU, forward and backward | 1486 ms (0.6) | 34733 ms (0.9) |
| ResNet-50, ggml Vulkan, forward | 56 ms (17.8) | 1148 ms (27.8) |
| ResNet-50, ggml Vulkan, forward and backward | 2827 ms (0.3) | 12617 ms (2.5) |
| ResNet-18, ggml CPU, forward | 55 ms (18.1) | 1601 ms (19.9) |
| ResNet-18, ggml CPU, forward and backward | 767 ms (1.3) | 20155 ms (1.5) |
| ResNet-18, ggml Vulkan, forward | 23 ms (43.4) | 439 ms (72.8) |
| ResNet-18, ggml Vulkan, forward and backward | 344 ms (2.9) | 4368 ms (7.3) |

Loading takes 0.1 to 0.5 s on the CPU and 0.4 to 1.1 s on Vulkan. At batch 32 the backward takes 9 to 13 times the forward on the CPU. ResNet-50 on Vulkan at batch 1 is the odd row: forward and backward take 2.8 s, a fifth of the 12.6 s of 32 images (ResNet-18 takes 0.34 s at batch 1); it was measured three times (2818, 2862 and 2827 ms) and not profiled. Before ml#83 the bias gradient of a convolution (a sum over the batch and the plane) was not lowered to ggml at a batch above 1; `Trace.unbroadcast` now sums the two blocks one after the other.

```sh
agentica --max-steps 1000000000000 transformers/tools/bench_resnet.agc MODEL_DIR cpu|vulkan BATCH REPS forward|backward
```

## ViT-B/16 (ml#83), ours only

`google/vit-base-patch16-224` (revision and hash in `transformers/DESIGN.md`, `model.safetensors`, `float32`), one batch of 224 by 224 images of constant pixels. Ours is `transformers/tools/bench_vit.agc` (Agentica 0.38.0), same machine as above, nothing else running (`--bench` slot): one warm-up run, then the median of 5 timed runs (3 at batch 32). "forward" is `ViT.run` (one graph), whose logits are read back. The weights are `F32` and Vulkan multiplies in exact `float32` (the tool calls `Tensor.vulkan_fp16(false)` first). There is no peer row: PyTorch was not run for this model.

| Row | batch 1 | batch 32 |
|---|---|---|
| ViT-B/16, ggml CPU, forward | 376 ms (2.6 images/s) | 9403 ms (3.4) |
| ViT-B/16, ggml Vulkan, forward | 186 ms (5.3) | 4289 ms (7.4) |

Loading takes 0.25 to 0.27 s on both.

```sh
agentica --max-steps 1000000000000 transformers/tools/bench_vit.agc MODEL_DIR cpu|vulkan BATCH REPS
```

## CLIP ViT-B/32 (ml#83), ours only

`laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (revision and hash in `transformers/DESIGN.md`, `model.safetensors`, `float32`). Ours is `transformers/tools/bench_clip.agc` (Agentica 0.38.0), same machine as above, nothing else running (`--bench` slot): one warm-up run, then the median of 5 timed runs (3 at batch 32). "image embeddings" is `Clip.image_embeds` of a batch of 224 by 224 images of constant pixels; "text embeddings" is `Clip.text_embeds` of rows of 77 ids, the longest a prompt can be (a real prompt of 5 to 15 ids is much cheaper); each is one graph, and the embeddings are read back. The weights are `F32` and Vulkan multiplies in exact `float32` (the tool calls `Tensor.vulkan_fp16(false)` first). There is no peer row: PyTorch was not run for this model.

| Row | batch 1 | batch 32 |
|---|---|---|
| CLIP ViT-B/32, ggml CPU, image embeddings | 97 ms (10.3 images/s) | 2360 ms (13.5) |
| CLIP ViT-B/32, ggml CPU, text embeddings (77 ids) | 70 ms (14.2 texts/s) | 1814 ms (17.6) |
| CLIP ViT-B/32, ggml Vulkan, image embeddings | 75 ms (13.3 images/s) | 1094 ms (29.2) |
| CLIP ViT-B/32, ggml Vulkan, text embeddings (77 ids) | 69 ms (14.4 texts/s) | 806 ms (39.7) |

Loading takes 0.45 to 0.47 s on both.

```sh
agentica --max-steps 1000000000000 transformers/tools/bench_clip.agc MODEL_DIR cpu|vulkan BATCH REPS image|text
```

## Decoding in front of a model, with and without the prefetch task (ml#252), ours only

The same ResNet-50 and CLIP checkpoints, now fed by real files: 32 JPEG files of 640 by 480 (90 KB each, made by the test) go through `Vision.samples` (`Resize`, `CenterCrop(224)`) and `Vision.normalize` to a batch of 32, and the model's image forward runs on it. Ours is `vision/test/prefetch_bench_test.agc` (Agentica 0.38.0), same machine as above, nothing else running (`--bench` slot), Vulkan in exact `float32`. The mean of 3 steps on the CPU and of 6 on Vulkan after one warm-up batch, in three runs of the same batches: "device" decodes one batch once and runs only the model; "sync" decodes each batch in the step, as `fit_batches` does without `prefetch`; "prefetch" has a task decode 2 batches ahead (`Data.ahead`, the stream behind `fit_batches(..., prefetch: 2)`). "idle" is the time a step waits for its batch, which is how long the device does nothing. Times in ms.

| | device | sync step | sync idle | prefetch step | prefetch idle |
|---|---|---|---|---|---|
| ResNet-50, ggml CPU | 4890 | 5311 | 327 | 4513 | 136 |
| CLIP ViT-B/32, ggml CPU | 2373 | 2647 | 289 | 2635 | 173 |
| ResNet-50, ggml Vulkan | 1153 | 1559 | 366 | 1263 | 86 |
| CLIP ViT-B/32, ggml Vulkan | 1110 | 1509 | 355 | 1234 | 100 |

On Vulkan the prefetch hides three quarters of the decoding: a step takes 1263 ms against 1559 (the model alone takes 1153). On the CPU the decoder and ggml's threads share the cores, so the gain is smaller (ResNet-50) or none (CLIP, 2635 against 2647). The "device" column of a run moves with the machine (the same CPU row was 3850 ms in the run before ml#252, and 4890 in this one), so read the columns of one run against each other. Decoding costs about 11 ms an image, resize included.

The `U8` samples of a `U8` image are handed to the backend as bytes now instead of being cast to `F32` by the image library first (the backend widens them to `F32` before it uploads them). Against the code before ml#252 (same bench, "sync" rows, a separate run: 384, 339, 338 and 365 ms of idle in the order above), the idle time is 327, 289, 366 and 355 ms: the cast was a small part of the decode, and the gain is memory: a batch of 32 crops is 4.8 MB of bytes up to the backend, not 19.3 MB.

```sh
PREFETCH_MODEL=resnet|clip PREFETCH_DIR=MODEL_DIR PREFETCH_DEVICE=cpu|vulkan PREFETCH_MODE=device|sync|prefetch PREFETCH_OUT=out.txt \
  agentica test --max-steps 1000000000000 vision/test/prefetch_bench_test.agc
```

## Run it

You need the Hugging Face checkout of `openai/whisper-tiny`, openai/whisper's `whisper/assets/mel_filters.npz`, a whisper.cpp checkout (never modified), `cmake`, a C++ compiler, `/usr/bin/time` (GNU time), `agentica` 0.38.0 on `PATH`, libtorch, and the Vulkan headers and `glslc` for the Vulkan rows. Python needs only the system numpy: the scripts run with `python3 -I`, which ignores virtual environments and `PYTHONPATH`.

```sh
git submodule update --init third_party/ggml
benchmarks/bench_whisper.sh build  WHISPER_TINY_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
benchmarks/bench_whisper.sh run    WHISPER_TINY_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
benchmarks/bench_whisper.sh report WORK_DIR
```

The Qwen3 rows need `qwen3-0.6b-f32.gguf`, `qwen3-0.6b-f16.gguf` and `qwen3-0.6b-q4_k_m.gguf` (`python3 -I convert_hf_to_gguf.py MODEL_DIR --outtype f32|f16 --outfile ...` from llama.cpp's checkout; the Q4_K_M file is `llama-quantize` of the F16 one), the two out-of-tree builds of its `llama-bench` (a file `COMMIT` in each build folder is printed with the rows, and `CMAKE_FLAGS`; `THREADS` sets the threads of both CPU sides, `ROWS` keeps the rows whose name contains one of its comma-separated words), and the engines of `transformers/` (`sh tensor/tools/build-engine.sh vulkan`, its exports in the shell):

```sh
# the builds, from official llama.cpp source at a commit (here the one of the Q4_K_M rows)
git clone https://github.com/ggml-org/llama.cpp && git -C llama.cpp checkout 711f9f4c117042570ef66ef58fba1f15835a331b
cmake -S llama.cpp -B BUILD_CPU -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build BUILD_CPU --target llama-bench llama-quantize -j2
git -C llama.cpp rev-parse HEAD > BUILD_CPU/COMMIT
echo "-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF" > BUILD_CPU/CMAKE_FLAGS
# the same for BUILD_VULKAN with -DGGML_VULKAN=ON added to the flags and to CMAKE_FLAGS
BUILD_CPU/bin/llama-quantize qwen3-0.6b-f16.gguf qwen3-0.6b-q4_k_m.gguf Q4_K_M
benchmarks/bench_qwen.sh QWEN3_DIR GGUF_DIR BUILD_CPU BUILD_VULKAN [ROUNDS]
```

`build` converts the weights to `ggml-tiny-f32.bin` and `ggml-tiny-f16.bin`, builds whisper.cpp for the CPU and Vulkan in `WORK_DIR`, compiles [`whisper_cpp_bench.cpp`](whisper_cpp_bench.cpp) against each, and builds the engines of `asr/`. `run` times every row on both clips, and runs one more pass per row (jfk only) that stops after the load: its `.load.time` file in the raw results holds the memory of the load alone (the tables show the peak of the whole process). `report` prints the tables above (`summarize.py`). `ONLY=wcpp` runs the rows whose name matches. `THREADS` (default: every core) must equal the cores ours uses, which is every online core. Both engines are timed the same way: the harness of whisper.cpp (`whisper_cpp_bench`) and `asr/tools/bench_transcribe.agc` load once, transcribe once to warm up, then time 5 runs.

The machine must be quiet: close builds and other jobs, and run each step under a lock if others share it. Builds and runs take about 15 minutes together on this machine.

whisper.cpp's `whisper-cli` gives the same words (`-l en -nt -bs 1 -bo 1 -nf -t 8`); the harness exists to time a warm process, the way `asr/` is timed.

The training benchmarks of ml#50 need the two venvs and the ggml engines, and take about 40 minutes with the machine quiet:

```sh
sh benchmarks/training/run.sh setup                    # the PyTorch and JAX venvs, once
sh benchmarks/training/run.sh check                    # every row at batch 4 and 8, two steps: they run and agree
sh benchmarks/training/run.sh run results.jsonl
python3 -I benchmarks/training/table.py results.jsonl
python3 -I benchmarks/training/table_test.py           # the loss-curve verdict of table.py
```

The STS-B fine-tune of ml#182 needs the PyTorch venv, `curl` and the ggml engine. The two runs take about 10 minutes together (the LoRA pair about 6); run them one after the other with the machine quiet, and wait about 3 minutes between them: a run leaves the load average above the 1.0 at which `run.sh` refuses to start. The venv is the `venv-torch` of the training benchmarks, so `setup` changes it for both.

```sh
sh benchmarks/stsb/run.sh setup                        # torch, transformers and pyarrow at pinned versions
sh benchmarks/stsb/run.sh fetch                        # the checkpoint and STS-B, at pinned revisions, sha256 checked
sh benchmarks/stsb/run.sh smoke                        # both sides on four pairs agree, LoRA too; without the decay they do not
sh benchmarks/stsb/run.sh ours ours.jsonl              # PAD_TO=8 for the other row
sh benchmarks/stsb/run.sh peer peer.jsonl
python3 -I benchmarks/stsb/summary.py ours.jsonl peer.jsonl 2.4e-7   # the tolerance of the full pair; `run.sh` holds both
sh benchmarks/stsb/run.sh ours-lora ours-lora.jsonl    # run it before peer-lora: it writes the adapter the peer loads
sh benchmarks/stsb/run.sh peer-lora peer-lora.jsonl
python3 -I benchmarks/stsb/summary.py ours-lora.jsonl peer-lora.jsonl 3e-7   # and of the LoRA pair
python3 -I benchmarks/stsb/summary_test.py             # the verdict of summary.py
```

`pyarrow` alone reads the parquet files, not the `datasets` package, which would bring a dozen more dependencies for one `read_table`. Each run writes the JSON lines again from the parquet files, whose hashes it checks first.

### The converter

`convert_whisper_ggml.py` writes whisper.cpp's file format from `model.safetensors` with the standard library and numpy only (no torch, no transformers). It follows whisper.cpp's `models/convert-h5-to-ggml.py`: the header, the mel filters, the vocabulary, the tensor names and which tensors stay F32 in an F16 file. One difference: the original rounds every tensor through float16 first, so its F32 files hold fp16-rounded weights. Ours keeps the checkpoint's float32 values, so F32 against F32 is a fair row; in the F16 file too, the tensors that stay F32 (biases, norms, positions) keep float32 values, where the original rounds them through fp16. whisper.cpp loads both files and transcribes jfk with them.

### Where a Vulkan step goes: the per-kernel timer

`GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1` makes ggml-vulkan print the time of each kernel of each graph to stderr (the GPU timestamps, so the host does not show). On the pick step of whisper-tiny and whisper-medium the largest kernel was not a matmul. The greedy pick (`Asr.picked`) takes two maxima over the 51,865 logits, and a max over one row used to lower to `ggml_pool_1d` with the window as wide as the row, which one thread walks: 13.6 ms a call, 27 ms a step. Over one row of 1,024 floats or more a max is now `ggml_argmax` (a workgroup reduction) and a `ggml_get_rows` of its value (max over many rows keeps the pool), which took the whisper-tiny decoder step on Vulkan from 50 to 21 ms and the medium one from 300 to 268 ms; the CPU takes the same argmax path and did not change. What remains of a medium step on Vulkan is mostly `CONT` (436 copies, 187 ms of 296 before that change).

The inputs of the run above, from the official sources:

| File | sha256 |
|---|---|
| `openai/whisper-tiny` `model.safetensors` | `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395` |
| `config.json` | `ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98` |
| `vocab.json` | `8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9` |
| `added_tokens.json` | `9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1` |
| `tokenizer.json` | `27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566` |
| openai/whisper `mel_filters.npz` | `7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c` |
| `ggml-tiny-f32.bin` (converted) | `ac65b0bb380eac9de3ae6994512a760a4638f53044aee59e769d18c53814a15f` |
| `ggml-tiny-f16.bin` (converted) | `0c501c95606f73ab5a9d19a309ec93e57e595069c7d38465bf7cabcb64ba22d2` |
