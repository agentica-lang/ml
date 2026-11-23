# Benchmarks

Local runs against the Python and C++ stacks, never in CI. This first page is Whisper inference (the inference half of [#50](https://github.com/agentica-lang/ml/issues/50)): whisper-tiny on `asr/` against whisper.cpp, on one machine. Training comes later.

## Whisper-tiny, ours against whisper.cpp

Both engines run the same weights (the Hugging Face `openai/whisper-tiny` checkpoint, converted for whisper.cpp by [`convert_whisper_ggml.py`](convert_whisper_ggml.py)), the same two clips, greedy decoding, no timestamps, a named language and 8 threads. A run is the whole pipeline: log-mel, encoder and decoder. The load is timed on its own, one warm-up transcription runs first, and the number is the median of 5 timed runs.

| | |
|---|---|
| Machine | Ryzen 5 3550H (4 cores, 8 threads, AVX2 and FMA, no AVX512), Radeon Vega iGPU (Mesa RADV 25.0.7), 12 GB, Debian, Linux 6.12 |
| Ours | agentica-lang/ml `ac1b402` (`asr/`, `tensor/`), Agentica 0.36.0, ggml `v1.0.0` (the `third_party/ggml` submodule), libtorch 2.4.1 CPU |
| whisper.cpp | `398997ed` (v1.9.4 plus 86 commits), its own ggml 0.24.0, built out of tree with `-DCMAKE_BUILD_TYPE=Release`, and `-DGGML_VULKAN=ON` for the Vulkan one |
| Clips | `jfk.wav` from whisper.cpp (11 s, English) and `whisper/live/fixtures/es-hola.wav` (2.6 s, Spanish) |
| Threads | 8 for ours on ggml CPU (it uses every online core), `-t 8` for whisper.cpp, libtorch's default |

The real-time factor is the run time over the clip length, so lower is better. The load is the time from the file to a model ready to transcribe.

| Row | RTF jfk | RTF es-hola | Load | Peak memory | Tokens against Hugging Face |
|---|---|---|---|---|---|
| ours, ggml CPU | 0.363 | 1.017 | 22.8 s | 3.0 GB | same |
| ours, ggml Vulkan | 0.436 | 1.027 | 22.8 s | 2.8 GB | same |
| ours, libtorch CPU | 0.622 | 1.915 | 23.6 s | 3.6 GB | same |
| whisper.cpp CPU, F32 | 0.080 | 0.287 | 0.10 s | 0.22 GB | one more token on jfk |
| whisper.cpp CPU, F32, no flash attention | 0.119 | 0.429 | 0.11 s | 0.29 GB | same |
| whisper.cpp CPU, F16 (its default) | 0.116 | 0.456 | 0.08 s | 0.15 GB | one more token on jfk |
| whisper.cpp Vulkan, F32 | 0.041 | 0.110 | 0.17 s | 0.16 GB | one more token on jfk |
| whisper.cpp Vulkan, F16 (its default) | 0.038 | 0.106 | 0.13 s | 0.12 GB | one more token on jfk |

Whisper's window is 30 s whatever the clip, so a short clip costs the same encoder time as a long one: that is why es-hola has the larger factor.

whisper.cpp turns flash attention on by default; ours has none, so the "no flash attention" row is the like-for-like one on the CPU. On Vulkan it makes no difference. F16 is slower than F32 on this CPU (probably because it has no half-precision arithmetic) and faster on the iGPU only by a little.

### The tokens

The golden for the check is Hugging Face's `generate` on whisper-tiny (`asr/test/fixtures/transcribe_golden.json`). Every `asr/` row, on all three backends and both clips, gives exactly its token ids and text. whisper.cpp gives them too on es-hola, and on jfk with flash attention off on the CPU. In the other jfk rows it writes one more token, a comma (id 11) after "for you": the text is the same sentence, `... can do for you, ask what you can do for your country.` It is the same in F32 and F16 and on both devices, and it goes away with flash attention off on the CPU, so it looks like a near tie in the logits and not a converter error. I did not look further.

### Where our time goes

One extra pass over the stages of `Asr.transcribe` (`asr/tools/bench_transcribe.agc`, `phase` lines), on jfk, in milliseconds. The VM steps are `System.Profile.steps()`: the interpreter's own work, which is the same on every run. The stages do not add up to the median because this pass skips the greedy rules and the tokenizer.

| Stage | ours ggml CPU | VM steps | ours ggml Vulkan | whisper.cpp CPU F32, no flash | whisper.cpp Vulkan F32 |
|---|---|---|---|---|---|
| log-mel | 508 | 118 M | 663 | 14 | 19 |
| upload of the mel | 62 | 5.5 M | 63 | | |
| encoder | 994 | 0.27 M | 396 | 1054 | 158 |
| cross-attention cache | 66 | 0.07 M | 37 | | |
| decoder, 24 steps | 1630 | 34.7 M | 3010 | 182 | 261 |

Reading it:

- **The encoder is on par on the CPU** (994 ms against 1054 ms without flash attention, 662 ms with it). Same ggml kernels. On Vulkan ours is 2.5 times slower (396 ms against 158 ms).
- **The decoder is the gap.** A step takes about 68 ms on ggml CPU and 125 ms on Vulkan, against 7 and 11 ms for whisper.cpp. A step is one graph, but it costs 1.4 M VM steps each, which is tracing and lowering a graph that does not change (about 6 to 12 ms of the 68, at the 120 to 230 M steps a second the VM runs here). The rest I did not split: it is native time to feed the graph (the padded self-attention cache and the cross-attention cache go in as inputs every step) and to read the logits back as a list. This is what [#64](https://github.com/agentica-lang/ml/issues/64) is about, and it is 40 to 65 percent of a transcription.
- **The log-mel costs 0.5 s** in `mel.agc`, which is Agentica code (a radix FFT over lists): 118 M VM steps, against 14 to 22 ms in whisper.cpp. It runs on the CPU and is the same on every backend; on Vulkan it costs more than the encoder.
- **The load is 23 s**, and 99 percent of it is `hub/safetensors.agc` decoding the 37.8 M floats one by one (`u16`, `u32`, `Float.from_bits32`: 2.7 G VM steps). whisper.cpp reads the file in 0.1 s. The checkpoint is 151 MB and the process peaks at 3 GB, probably because the floats go through lists first (I did not measure that).
- On Vulkan, the encoder gains 2.5 times over the CPU, but the decoder loses: a graph costs about 2 ms to dispatch (`note.md`), and a step has many small ones.
- libtorch runs the encoder op by op: 3.2 s, and 117 M VM steps.

## Run it

You need the Hugging Face checkout of `openai/whisper-tiny`, openai/whisper's `whisper/assets/mel_filters.npz`, a whisper.cpp checkout (never modified), `cmake`, a C++ compiler, libtorch, and the Vulkan headers and `glslc` for the Vulkan rows. Python needs only numpy.

```sh
git submodule update --init third_party/ggml
benchmarks/bench_whisper.sh build  WHISPER_TINY_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
benchmarks/bench_whisper.sh run    WHISPER_TINY_DIR MEL_FILTERS_NPZ WHISPER_CPP_SRC WORK_DIR
benchmarks/bench_whisper.sh report WORK_DIR
```

`build` converts the weights to `ggml-tiny-f32.bin` and `ggml-tiny-f16.bin`, builds whisper.cpp for the CPU and Vulkan in `WORK_DIR`, compiles [`whisper_cpp_bench.cpp`](whisper_cpp_bench.cpp) against each, and builds the engines of `asr/`. `run` times every row twice per clip: the full run, and a pass that stops after the load, whose process peak is the load's memory. `report` prints the tables above (`summarize.py`). `ONLY=wcpp` runs the rows whose name matches. Both engines are timed the same way: the harness of whisper.cpp (`whisper_cpp_bench`) and `asr/tools/bench_transcribe.agc` load once, transcribe once to warm up, then time 5 runs.

The machine must be quiet: close builds and other jobs, and run each step under a lock if others share it. Builds and runs take about 15 minutes together on this machine.

whisper.cpp's `whisper-cli` gives the same words (`-l en -nt -bs 1 -bo 1 -nf -t 8`); the harness exists to time a warm process, the way `asr/` is timed.

### The converter

`convert_whisper_ggml.py` writes whisper.cpp's file format from `model.safetensors` with the standard library and numpy only (no torch, no transformers). It follows whisper.cpp's `models/convert-h5-to-ggml.py`: the header, the mel filters, the vocabulary, the tensor names and which tensors stay F32 in an F16 file. One difference: the original rounds every tensor through float16 first, so its F32 files hold fp16-rounded weights. Ours keeps the checkpoint's float32 values, so F32 against F32 is a fair row. whisper.cpp loads both files and transcribes jfk with them.

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
