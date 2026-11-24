# Benchmarks

Local runs against the Python and C++ stacks, never in CI. This first page is Whisper inference (the inference half of [#50](https://github.com/agentica-lang/ml/issues/50)): whisper-tiny on `asr/` against whisper.cpp, on one machine. Training comes later.

## Whisper-tiny, ours against whisper.cpp

Both engines run the same weights (the Hugging Face `openai/whisper-tiny` checkpoint, converted for whisper.cpp by [`convert_whisper_ggml.py`](convert_whisper_ggml.py)), the same two clips, greedy decoding, no timestamps, a named language and 8 threads. A run is the whole pipeline: log-mel, encoder and decoder. The load is timed on its own, one warm-up transcription runs first, and the number is the median of 5 timed runs.

| | |
|---|---|
| Machine | Ryzen 5 3550H (4 cores, 8 threads, AVX2 and FMA, no AVX512), Radeon Vega iGPU (Mesa RADV 25.0.7), 12 GB, Debian, Linux 6.12 |
| Ours | agentica-lang/ml `0e0bb8b` plus the device pick of ml#81 for the two ggml rows and the medium table (Agentica 0.37.0), `ac1b402` and Agentica 0.36.0 for the libtorch row; ggml `v1.0.0` (the `third_party/ggml` submodule), libtorch 2.4.1 CPU |
| whisper.cpp | `398997ed` (v1.9.4 plus 86 commits), its own ggml 0.24.0, built out of tree with `-DCMAKE_BUILD_TYPE=Release`, and `-DGGML_VULKAN=ON` for the Vulkan one |
| Clips | `jfk.wav` from whisper.cpp (11 s, English) and `whisper/live/fixtures/es-hola.wav` (2.6 s, Spanish) |
| Threads | 8 for ours on ggml CPU (it uses every online core), `-t 8` for whisper.cpp, libtorch's default |

The real-time factor is the run time over the clip length, so lower is better. The load is the time from the file to a model ready to transcribe.

| Row | RTF jfk | RTF es-hola | Load | Peak memory | Tokens against Hugging Face |
|---|---|---|---|---|---|
| ours, ggml CPU | 0.235 | 0.634 | 0.4 s | 0.50 GB | same |
| ours, ggml Vulkan | 0.213 | 0.543 | 0.45 s | 0.35 GB | same |
| ours, libtorch CPU | 0.619 | 1.987 | 0.3 s | 1.5 GB | same |
| whisper.cpp CPU, F32 | 0.080 | 0.287 | 0.10 s | 0.22 GB | one more token on jfk |
| whisper.cpp CPU, F32, no flash attention | 0.119 | 0.429 | 0.11 s | 0.29 GB | same |
| whisper.cpp CPU, F16 (its default) | 0.116 | 0.456 | 0.08 s | 0.15 GB | one more token on jfk |
| whisper.cpp Vulkan, F32 | 0.041 | 0.110 | 0.17 s | 0.16 GB | one more token on jfk |
| whisper.cpp Vulkan, F32, no flash attention | 0.041 | 0.107 | 0.17 s | 0.16 GB | one more token on jfk |
| whisper.cpp Vulkan, F16 (its default) | 0.038 | 0.106 | 0.13 s | 0.12 GB | one more token on jfk |

The two ggml `ours` rows and the medium rows were measured again on ml#107 plus the cross-attention of `asr/` through `Tensor.attention` (ml#109), the same machine, with `Options.timestamps` false, as above (the PR's own table has timestamps on, hence other numbers). Tiny is the median of 5 runs, medium of 2 runs. Real-time factors before (ml#81, `0e0bb8b`) and now:

| Row | jfk before | jfk now | es-hola before | es-hola now |
|---|---|---|---|---|
| tiny, ggml CPU | 0.276 | 0.235 | 0.80 | 0.634 |
| tiny, ggml Vulkan | 0.268 | 0.213 | 0.713 | 0.543 |
| medium, ggml CPU | 8.17 | 7.04 | 22.4 | 18.2 |
| medium, ggml Vulkan | 2.42 | 1.38 | 6.83 | 4.03 |

Before that, the two ggml `ours` rows had been measured again on ml `0e0bb8b` plus the device pick (ml#81), with the same session's runs of `0e0bb8b` alone: 0.299 and 0.393 on jfk, and the medium table below 2.50 and 6.91. They are runs with `Options.timestamps` false. The libtorch row, the whisper.cpp rows and the load and memory columns are from the runs named below, not repeated. The three `ours` rows were measured again after ml#63 (the load and the memory changed, the rest is the same code and the same machine, so the real-time factors moved only by noise). Before it the load was 23.6 s, 23.3 s and 23.6 s, and a pass that stops after the load peaked at 3.05 GB, 2.84 GB and 3.26 GB. The Peak memory column is the whole run, transcriptions included; the load alone now peaks at 0.46 GB, 0.36 GB and 0.49 GB.

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

whisper.cpp turns flash attention on by default; ours has none, so the "no flash attention" row is the like-for-like one on the CPU. On Vulkan it makes no difference. F16 is slower than F32 on this CPU (probably because it has no half-precision arithmetic) and faster on the iGPU only by a little.

### The tokens

The golden for the check is Hugging Face's `generate` on whisper-tiny (`asr/test/fixtures/transcribe_golden.json`). Every `asr/` row, on all three backends and both clips, gives exactly its token ids and text. whisper.cpp gives them on es-hola, and on jfk with flash attention off on the CPU. In the other jfk rows it writes one more token, a comma (id 11) after "for you": the text is the same sentence, `... can do for you, ask what you can do for your country.` It is the same in F32 and F16 and on both devices, and it goes away with flash attention off on the CPU, so it looks like a near tie in the logits and not a converter error.

### Where our time goes

One extra pass over the stages of `Asr.transcribe` (`asr/tools/bench_transcribe.agc`, `phase` lines), on jfk, in milliseconds. The VM steps are `System.Profile.steps()`: the interpreter's own work, which is the same on every run. The stages do not add up to the median because this pass skips the greedy rules and the tokenizer.

| Stage | ours ggml CPU | VM steps | ours ggml Vulkan | whisper.cpp CPU F32, no flash | whisper.cpp Vulkan F32 |
|---|---|---|---|---|---|
| log-mel (one graph, upload included) | 35 | 3.3 M | 95 | 14 | 19 |
| encoder | 994 | 0.27 M | 396 | 1054 | 158 |
| cross-attention cache | 66 | 0.07 M | 37 | | |
| decoder, 24 steps | 1630 | 34.7 M | 3010 | 182 | 261 |

More figures, all on jfk and from the same run (`report` prints the stages; the profile is `AGENTICA_VM_TRACE=profile agentica tools/bench_transcribe.agc ... ggml en -1` in `asr/`):

| Figure | Value |
|---|---|
| whisper.cpp CPU F32 with flash attention, encoder | 662 ms |
| whisper.cpp CPU F32, decoder step | 7 ms (182 ms for the 24 steps) |
| whisper.cpp Vulkan F32, decoder step | 11 ms (261 ms for the 24 steps) |
| ours ggml CPU, decoder step (24 steps in 1630 ms) | 68 ms, 1.45 M VM steps |
| ours ggml Vulkan, decoder step (24 steps in 3010 ms) | 125 ms, 1.45 M VM steps |
| ours, load: VM steps | 2.76 G before ml#63, of which 2.7 G decoded the 37.8 M floats of the checkpoint one by one (`u16`, `u32`, `Float.from_bits32` in `hub/safetensors.agc`) |
| ours, libtorch, encoder | 3171 ms, 117.7 M VM steps |

Reading it:

- **The encoder is on par on the CPU** (994 ms against 1054 ms without flash attention, 662 ms with it). Similar ggml CPU kernels (different ggml versions: v1.0.0 against 0.24.0). On Vulkan ours is 2.5 times slower (396 ms against 158 ms).
- **The decoder is the gap.** A step takes about 68 ms on ggml CPU and 125 ms on Vulkan, against 7 and 11 ms for whisper.cpp. A step is one graph, but it costs 1.45 M VM steps, which is tracing and lowering a graph that does not change (about 6 to 12 ms of the 68, at the 120 to 230 M steps a second the VM runs here). The rest is not split: it is probably native time to feed the graph (the padded self-attention cache and the cross-attention cache go in as inputs every step) and to read the logits back as a list. This is what [#64](https://github.com/agentica-lang/ml/issues/64) is about, and it is 41 percent of a transcription on the CPU and 63 percent on Vulkan.
- **The log-mel costs 35 ms on the CPU and 95 ms on Vulkan.** It was 0.5 s of Agentica code (a radix FFT over lists, 118 M VM steps) plus 62 to 74 ms to upload the result, the same on every backend. It is one graph of tensor ops now (the windowed DFT as a matmul, the mel filters as a second one) that `load` compiled, and its output stays on the device. On the same machine and session, jfk went from 605 + 71 ms to 35 ms on the CPU and from 540 + 74 ms to 95 ms on Vulkan; the median transcription of jfk (11 s) from 3008 to 1949 ms on the CPU (real-time factor 0.273 to 0.177) and from 2675 to 1981 ms on Vulkan (0.243 to 0.180). es-hola (2.6 s) went from 1850 to 1509 ms on the CPU (0.699 to 0.570) and from 1651 to 1414 ms on Vulkan (0.624 to 0.534). The graph computes every one of the 3000 frames, so its cost does not depend on the clip; Vulkan is slower than the CPU here because the matmul alone takes about 20 ms on this iGPU.
- **The load was 23 s and 3 GB, and is 0.3 to 0.45 s, and the load alone peaks under 0.5 GB** (ml#63). 99 percent of it had been `hub/safetensors.agc` decoding the 37.8 M floats one by one (2.7 G VM steps), and the lists they went through held the memory. Now C reads each tensor's byte range into the backend's buffer (`Tensor.from_file`), so no float is an Agentica value. whisper.cpp reads the file in 0.1 s.
- On Vulkan, the encoder gains 2.5 times over the CPU, but the decoder loses: a graph costs about 2 ms to dispatch (`note.md`), and a step has many small ones.
- libtorch runs the encoder op by op: 3.2 s, and 117.7 M VM steps.

### Where a decoder step goes

Whisper-tiny, jfk, ggml CPU and Vulkan. `asr/tools/bench_step.agc` loads the model, runs the log-mel, the encoder and the cross-attention cache once, then runs the prompt and the 23 tokens of jfk through `Asr.step` (24 steps) PASSES times, closing each step's caches as `Asr.transcribe` does. The first pass is a warm-up; the per-step wall time is the mean of the other passes. With 5 passes that is 96 timed steps (4 passes of 24), out of 120 steps run.

Two things are measured, from the same commit:

- **VM steps** (exact) and **where the time goes** (profile on, so the VM runs slower): `AGENTICA_VM_TRACE=profile AGENTICA_VM_TRACE_FILE=REPORT agentica tools/bench_step.agc MODEL_DIR WAV ggml en PASSES`, once with PASSES 0 and once with 5, then `python3 -I benchmarks/step_profile.py BASE FULL 120`. 120 is the steps the 5-pass run has more than the 0-pass run (5 passes of 24).
- **Milliseconds**, with the profile off: `benchmarks/step_timers.sh apply` swaps the `System.Profile.measure` labels for `Time.monotonic_ns` timers (it edits the tree; `revert` undoes it) and `bench_step.agc` then prints `probe NAME NS` totals, again for PASSES 0 and 5; a section's ms per step is the difference over 120 steps. The same script makes `ggml_run` print its allocator and compute times when `TG_TIME` is set. The `ml.*` labels (in `Trace.eval_many`, `Lower.evaluate` and `Asr.step`) also count the encoder's and the cross-attention cache's evaluations, which the 0-pass run holds too, so the subtraction removes them; `ml.compile` of the warm-up pass lands in the steps (it is 0.1 calls a step).

"Before" is ml `b59c68c` (the head of `main` then), "after" is this change on top of it; Agentica 0.37.0; the means of those 120 steps (the wall time is the mean of the 96 timed ones). The ms are rounded one by one, so a column's parts may not add to its sum; the VM steps are rounded to 0.01 M.

| Part of a step | VM steps before | VM steps after | CPU before, ms | CPU after, ms | Vulkan before, ms | Vulkan after, ms |
|---|---|---|---|---|---|---|
| tracing (`traced(m)` and `stepped_graph`: about 1,400 nodes recorded) | 0.19 M | 0.05 M | 2.2 | 0.6 | 3.8 | 1.2 |
| cache key and lowering (`Lower.plan`, `Lower.program`) | 0.23 M | 0.06 M | 7.2 | 1.7 | 11.3 | 2.8 |
| lease lookup (`acquire` by the key string) | 0 | 0 | 0.1 | 0.05 | 0.1 | 0.06 |
| input upload (the id and the position, 8 bytes in 2 calls) | 0 | 0 | 0.1 | 0.1 | 1.1 | 0.8 |
| binding (147 leaves, weights and caches, read in place, no copy) | 0.003 M | 0.005 M | 0.9 | 0.8 | 2.3 | 2.3 |
| native execution (`run`: allocator, compute, finiteness checks, copy-out) | 0 | 0 | 40.7 | 46.3 | 80.5 | 75.6 |
| read-back (the 51,865 logits, 207 KB, as a `List[Float]`, then each as a `Tensor.Values.Float`, then as a `Float`) | 1.04 M | 1.04 M | 18.7 | 19.3 | 31.8 | 31.2 |
| sum of the parts | 1.47 M | 1.15 M | 69.7 | 68.9 | 130.9 | 114.0 |
| wall time of a step | | | 69.9 | 70.9 | 132.0 | 115.2 |

"Before" is `Trace.eval_many` on every step, "after" is `Trace.compile` and `Trace.run`: the decoder step is traced and planned once for each pair of cache rows and capacity (5 graphs for the 24 steps of jfk: 8, 16 and 32 rows), so the tracing and planning rows are those 5 compilations spread over 24 steps, and a step that finds its graph costs nothing there. The encoder blocks share one compiled graph, with the weights as inputs. On Vulkan the step is 17 ms shorter. On the CPU the 8 ms saved in tracing and planning are hidden by the native run, which varied by 6 ms between these runs (40.7, 43.1 and 46.3 ms in three runs of the same graph on this shared machine).

### With the pick on the device

`Asr.step_pick` adds a mask to the logits and takes the first largest id and its log-probability inside the step's graph (`Asr.picked`: `max`, `eq`, `arange`, `sum`, `exp`, `log`), so the 51,865 logits stay on the device and `[id, logprob]` is the only thing that crosses. `Asr.transcribe` used it only when `Options.timestamps` was false, on ggml; the timestamp rules run in the graph too since the section on them below, so the default takes it as well. The `sot` step of a multilingual model reads its logits too, for `no_speech_prob` and for the language detection, whether or not a language is named, so every transcription has one logits step, and the whole-transcription numbers below include it. `bench_step.agc ... - pick` times it. Same session, same code otherwise (the step after `Trace.compile`, with the logits against with the pick):

| Part of a step | VM steps, logits | VM steps, pick | CPU logits, ms | CPU pick, ms | Vulkan logits, ms | Vulkan pick, ms |
|---|---|---|---|---|---|---|
| tracing | 0.05 M | 0.05 M | 0.5 | 0.5 | 0.8 | 0.9 |
| cache key and lowering | 0.06 M | 0.06 M | 1.7 | 1.7 | 2.5 | 2.8 |
| lease lookup | 0 | 0 | 0.05 | 0.04 | 0.07 | 0.05 |
| input upload (8 bytes, 2 calls) | 0 | 0 | 0.1 | 0.1 | 1.1 | 0.6 |
| binding 147 leaves | 0.005 M | 0.005 M | 0.8 | 0.8 | 2.3 | 1.7 |
| native execution (`run`) | 0 | 0 | 40.5 | 39.3 | 78.8 | 62.2 |
| read-back | 1.04 M | 0.0001 M | 19.5 | 0.03 | 33.3 | 0.3 |
| sum | 1.15 M | 0.11 M | 63.2 | 42.4 | 119.0 | 68.5 |
| wall time of a step | | | 61.9 | 38.5 | 120.6 | 68.2 |

The sum of the parts is the mean of 120 steps from one process, the wall time the mean of the 96 timed steps from another: the warm-up pass, which runs each new graph for the first time, takes 1,094 ms against 887 to 974 for the others on the CPU, and the two processes differ by a few percent. That is why the CPU pick column sums to 42.4 ms against a wall time of 38.5.

The read-back is gone, and on Vulkan the native run is 16 ms shorter too: the graph no longer copies 207 KB of logits out and checks them. What is left is the device: 39 ms (CPU) and 62 ms (Vulkan) for a graph of about 1,400 ggml nodes (1,382 to 1,414 across the three cache capacities for the pick graph, 1,377 to 1,393 for the logits graph), and the per-step Agentica work (tracing and planning spread over the 5 compilations, binding, inputs), 3 to 6 ms.

Whole transcriptions, median of 5 runs (medium 2), the code before and after in the same session, no timestamps:

| | jfk before | jfk after | es-hola before | es-hola after |
|---|---|---|---|---|
| tiny ggml CPU | 0.299 | 0.276 | 0.86 to 0.99 | 0.75 to 0.89 |
| tiny ggml Vulkan | 0.393 | 0.268 | 0.972 | 0.713 |
| medium ggml Vulkan | 2.50 | 2.42 | 6.91 | 6.83 |

The CPU es-hola runs vary by 20 percent between runs on this machine (the ranges are the medians of two to three sets). Tokens are those of the Hugging Face goldens everywhere.

The earlier table in this file says 68 ms and 1.45 M VM steps for a CPU step, and this one 69.9 ms and 1.47 M. The earlier table was measured on `ac1b402`; on `b59c68c` the same `bench_transcribe.agc` stage pass gives 1.46 M for the prompt step and 33.8 M for the 23 steps after it (1.47 M each) and 68 ms a step, so the VM steps moved with the code between the two commits and the time is within noise of this run (60 to 78 ms a step across runs of `bench_step.agc` on this machine, which is shared). The parts add up to the wall time within 1 ms in that run (the table above; the table of the pick below has a larger gap, explained under it). In an earlier one `bench_step.agc` did not close the self-attention caches, and about 7 ms a step were unattributed.

What this says:

- **The read-back is the largest VM cost, not the tracing.** 1.04 M of the 1.47 M VM steps, and 19 to 32 ms, are the logits crossing as a list and being converted twice. The planning (key and lowering) is the second, then the tracing; together the VM work that an unchanged graph repeats (tracing, planning, read-back) was 28 ms on the CPU and 47 ms on Vulkan, the rest being native. `Trace.compile` removed the tracing and the planning from it, and the pick on the device removed the read-back (below).
- **Nothing big is uploaded.** The step feeds 8 bytes. The weights and both caches are bound where they are (`bind` makes no copy for an F32 tensor). Each step makes 9 fresh output tensors: the logits (207 KB) and 8 self-attention caches of `capacity x 384` floats (393 KB in all at capacity 32), allocated and written by the graph. Only the logits come back to the host.
- **The device part is what remains**: 41 ms on the CPU and 81 ms on Vulkan for a graph of 1,377 to 1,393 ggml nodes (the logits graph; the pick graph has 1,382 to 1,414). `ggml_run` times show ggml's compute is 39.0 ms on the CPU and 80.3 ms on Vulkan, and its allocator 0.4 and 0.7 ms, so the rest of `run` is under 2 ms. That is 58 us a node on Vulkan, so the node count is the lever there (whisper.cpp's decoder step is 7 and 11 ms).

### With the cache written into the tensors of the step before

A step takes the keys and values of the cache it replaces (`spare`) and writes the new ones into them (`Trace.run`'s `into`), so a decoder step on ggml makes no tensor of its own except the two inputs. The two sets take turns. Same session, `bench_step.agc` (96 timed steps) and `bench_transcribe.agc` (median of 5 runs, medium 2), ml `d688a31` against this branch:

| | before | after |
|---|---|---|
| step, CPU, pick (ms) | 42.9 | 44.0 |
| step, Vulkan, pick (ms) | 68.8 | 68.1 |
| step, CPU, logits (ms) | 64.3 | 64.9 |
| step, Vulkan, logits (ms) | 123.8 | 121.8 |
| tiny CPU, jfk / es-hola (RTF) | 0.252 / 0.783 | 0.244 / 0.749 |
| tiny Vulkan, jfk / es-hola (RTF) | 0.275 / 0.718 | 0.270 / 0.738 |
| medium Vulkan, jfk / es-hola (RTF) | 2.42 / 6.87 | 2.38 / 6.91 |

Every difference is inside the run-to-run noise of this machine (a few percent, 20 percent for es-hola on the CPU). The allocation of 24 small tensors a step was already cheap, as the earlier table said (0.02 to 0.7 ms); the change is kept for what it fixes, not for speed: the tensors a run reads are held until it ends, and the cache never takes a decoder step's graph out from under it. The device is still the cost.

### With the timestamp rules on the device

The default, `Options.timestamps` true, applied Whisper's timestamp rules on the host over the 51,865 logits of every step. They now run in the step's graph (`Asr.scored`, from four numbers per step, `Greedy.state`), so the default path reads an id and a log-probability like the one without timestamps. `Asr.step` and `Greedy.allowed` stay as the reference, and tests compare the graph with them rule by rule on hand-made logits and over ten histories of a tiny model. Whole transcriptions with timestamps, `bench_transcribe.agc ... stamps`, median of 5 runs (medium 2), ml `ba20280`'s parent (`d688a31`, #81) against this branch in one session:

| RTF | jfk before | jfk after | es-hola before | es-hola after |
|---|---|---|---|---|
| tiny ggml CPU | 0.367 | 0.318 | 1.055 | 0.935 |
| tiny ggml Vulkan | 0.429 | 0.328 | 1.122 | 0.878 |
| medium ggml Vulkan | 2.67 | 2.54 | 7.60 | 7.39 |

The machine was slower in that session than in the ones above, so compare the two columns, not the tables. Tokens, text and `avg_logprob` are those of the Hugging Face goldens on ggml CPU and Vulkan for the three cases with timestamps.

## Whisper-medium, ours only

whisper-medium (the Hugging Face `openai/whisper-medium` checkpoint, 3.06 GB of F32, `model.safetensors` sha256 `62f73550fa6db24b0c6f6c5962bd0dae80fa644e93cde9cd9c3792971b47fd28`) could not load before ml#63. Same machine and clips, under a 6 GB memory cap, 2 timed runs after the warm-up (the median). No whisper.cpp row: only the tiny model was converted.

| Row | RTF jfk | RTF es-hola | Load | Peak memory of the load | Tokens |
|---|---|---|---|---|---|
| ours, ggml CPU | 7.04 | 18.2 | 7.7 s | 4.0 GB | same as the medium output of whisper.cpp's server (`whisper/live/fixtures/server-*-medium.json`) |
| ours, ggml Vulkan | 1.38 | 4.03 | 24 s | 0.49 GB | same |

The Vulkan weights sit in the device's own buffers, which on this iGPU are system memory outside the process's resident set, so its row shows the process only. The encoder takes 22 s on the CPU and 5 s on Vulkan, and a decoder step about 2.0 s on the CPU and 0.34 s on Vulkan (one run, jfk), which is the per-step cost of #64 on a bigger model. The Vulkan load is slower than the CPU's (24 s against 7.7 s); where that time goes was not profiled.

## Training loop on Vulkan, ours only

`train/tools/bench_fit.agc` runs `Train.fit` (a relu MLP with 256 inputs, two hidden layers and 10 classes, Adam, cross-entropy) on ggml Vulkan and times the loader alone. It answers one question: does the device wait for the data. Same machine as above (the iGPU is shared with whatever else runs on it, so the runs are noisy: compare only alternating runs). The CPU runs varied from 23 to 54 ms per step on one configuration, so no CPU figure is given.

| Model, batch | ms per step | Loader alone | Time `fit` waits on the loader |
|---|---|---|---|
| 256-512-512-10, 128 rows | 19 to 20 | 4 to 6 ms per batch | 0.08 ms per step |
| 256-128-128-10, 64 rows | 18 | 4 to 5 ms per batch | under 1 ms per step |

- The loader already prefetches: `Data.Loader.prefetch` (2 by default) is a `Task` that builds the next batches on the device into a bounded channel. A batch is one `gather` of the dataset, which sits on the device, so no Agentica list of data is involved. `fit` waits 0.08 ms per step for a batch, so the device does not wait on the loader. Without the task (`prefetch 0`) the step is 3 ms (15%) longer.
- Where a step goes (a profile run, which slows the VM): the device run (`ml.run`, the graph dispatch plus compute) is about 74% of a step of 24 ms and about 70% of one of 19 ms; the host's trace, lowering and binding is the rest.
- Tracing and lowering the step on every batch (about 3 ms of VM time per step) can be removed by compiling it once per batch shape, with the optimizer's per-step numbers as inputs. Measured on Vulkan over 6 alternating runs, the wall time went down by 6% (19.8 to 18.6 ms) on the first model and by 3% (18.4 to 17.9 ms) on the second, with the device share unchanged: the rest of the host time is not the trace. That change is not in `main`.

## Run it

You need the Hugging Face checkout of `openai/whisper-tiny`, openai/whisper's `whisper/assets/mel_filters.npz`, a whisper.cpp checkout (never modified), `cmake`, a C++ compiler, `/usr/bin/time` (GNU time), `agentica` 0.36.0 on `PATH`, libtorch, and the Vulkan headers and `glslc` for the Vulkan rows. Python needs only the system numpy: the scripts run with `python3 -I`, which ignores virtual environments and `PYTHONPATH`.

```sh
git submodule update --init third_party/ggml
benchmarks/bench_whisper.sh build  WHISPER_TINY_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
benchmarks/bench_whisper.sh run    WHISPER_TINY_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
benchmarks/bench_whisper.sh report WORK_DIR
```

`build` converts the weights to `ggml-tiny-f32.bin` and `ggml-tiny-f16.bin`, builds whisper.cpp for the CPU and Vulkan in `WORK_DIR`, compiles [`whisper_cpp_bench.cpp`](whisper_cpp_bench.cpp) against each, and builds the engines of `asr/`. `run` times every row on both clips, and runs one more pass per row (jfk only) that stops after the load: its `.load.time` file in the raw results holds the memory of the load alone (the tables show the peak of the whole process). `report` prints the tables above (`summarize.py`). `ONLY=wcpp` runs the rows whose name matches. `THREADS` (default: every core) must equal the cores ours uses, which is every online core. Both engines are timed the same way: the harness of whisper.cpp (`whisper_cpp_bench`) and `asr/tools/bench_transcribe.agc` load once, transcribe once to warm up, then time 5 runs.

The machine must be quiet: close builds and other jobs, and run each step under a lock if others share it. Builds and runs take about 15 minutes together on this machine.

whisper.cpp's `whisper-cli` gives the same words (`-l en -nt -bs 1 -bo 1 -nf -t 8`); the harness exists to time a warm process, the way `asr/` is timed.

### The converter

`convert_whisper_ggml.py` writes whisper.cpp's file format from `model.safetensors` with the standard library and numpy only (no torch, no transformers). It follows whisper.cpp's `models/convert-h5-to-ggml.py`: the header, the mel filters, the vocabulary, the tensor names and which tensors stay F32 in an F16 file. One difference: the original rounds every tensor through float16 first, so its F32 files hold fp16-rounded weights. Ours keeps the checkpoint's float32 values, so F32 against F32 is a fair row; in the F16 file too, the tensors that stay F32 (biases, norms, positions) keep float32 values, where the original rounds them through fp16. whisper.cpp loads both files and transcribes jfk with them.

### Where a Vulkan step goes: the per-kernel timer

`GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1` makes ggml-vulkan print the time of each kernel of each graph to stderr (the GPU timestamps, so the host does not show). On the pick step of whisper-tiny and whisper-medium the largest kernel was not a matmul. The greedy pick (`Asr.picked`) takes two maxima over the 51,865 logits, and a max over one row lowered to `ggml_pool_1d` with the window as wide as the row, which one thread walks: 13.6 ms a call, 27 ms a step, the same on both models. Over one row of 1,024 floats or more a max is now `ggml_argmax` (a workgroup reduction) and a `ggml_get_rows` of its value; max over many rows keeps the pool. Alternating runs of `bench_step.agc` (pick, 5 passes, medium 3), ml `cbcc6a5` against that change:

| Decoder step | before | after |
|---|---|---|
| tiny, Vulkan | 50.3, 50.4, 50.3 ms | 22.0, 20.8, 22.5 ms |
| medium, Vulkan | 300.6, 299.5 ms | 266.5, 269.5 ms |
| tiny, ggml CPU | 39.6, 35.3, 37.5 ms | 35.7, 35.7, 46.9 ms |
| medium, ggml CPU | 2,380 ms | 2,446 ms |

The CPU rows are within the noise of this shared machine: the CPU takes the same argmax path, and its pool was not slow. The rest of a medium step on Vulkan is mostly `CONT` (436 copies, 187 ms of 296), which the next change looks at.

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
