# ggml on the iGPU through Vulkan — a measurement-first spike

**This is a spike, not a package.** Nothing here is `tensor/`: there is no
`Backend` arm, no `Handle`, no conformance, no tracing, and no claim that any
of it should be kept. It pins **ggml** as a submodule, builds it twice (CPU
with AVX2, and Vulkan), puts a thin `extern module` over it, proves the graph
really runs on the iGPU instead of falling back to the CPU, and writes down
what the measurements say — including the ones that argue against the iGPU.

It changes nothing outside `spike/ggml-vulkan/` and `third_party/ggml` (the
submodule pointer). `tensor/DESIGN.md` is not edited; the last section says
what these numbers would mean for it, as a proposal.

Read with `tensor/DESIGN.md` open: §2 (types), §3.1 (primitives), §7
(dispatch) and §8 (lifetimes) are the sections this spike measures the cost
of.

## What is real and what is faked

| Thing | State |
|---|---|
| ggml as a pinned submodule (`third_party/ggml`, tag `v1.0.0`) | **real**, no source copied, no binary committed; the pointer is the only third-party artifact in git |
| The CPU (AVX2) and Vulkan builds, flags printed and stored | **real**: one script, two cmake trees, `build/<name>/cmake-flags.txt` |
| The boundary: devices, both backends, `create`, `add`, `relu`, `mul_mat`, readback | **real**, through generated `extern module` C |
| The iGPU really ran the graph | **asserted**: `backend_vulkan` fails unless `ggml_backend_is_vk` and the device type is not CPU, and every program prints the device name and type that ggml reports |
| `mul_mat`, dispatch and readback numbers, both backends plus libtorch CPU | **real measurements**, one machine, commands below |
| fp16 `mul_mat` results that differ from the CPU's fp32 at ~1e-3 | **real, and the default**: ggml's Vulkan backend computes F32 `mul_mat` in fp16 on this device; `GGML_VK_DISABLE_F16=1` turns it off, and both paths are measured |
| Graph batching (many ops per submit) | **measured by a probe, not by the module**: the module does one op per graph on purpose, because that is what an op-by-op `tensor/` backend would do. `tools/probe-dispatch.c` measures the other shape |
| A `tensor/` backend, a `Device` variant, conformance | **absent, on purpose** — that is the section at the end, and it is a proposal |
| CI | **unchanged**: `spike/` is not scanned by the matrix, and no Vulkan runner exists. See "CI" below |

What that adds up to: the **numbers are real**, the **architecture around them
is a stub** sized to produce them, and the **iGPU is a 15 W APU from 2019
whose memory is the CPU's memory** — which is most of what the verdict says.

## The machine

Everything below was measured on one machine, confirmed first:

```console
$ LANG=C lscpu | grep -E '^Model name|^CPU\(s\):|Thread|Core|^L3'
Model name:            AMD Ryzen 5 3550H with Radeon Vega Mobile Gfx
CPU(s):                8
Thread(s) per core:    2
Core(s) per socket:    4
L3 cache:              4 MiB (1 instance)
$ grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx2|fma|avx512f)$'
fma
avx2                      # AVX2 and FMA, no AVX512 (the build pins AVX2, see below)
$ vulkaninfo --summary | sed -n '/GPU0:/,/driverUUID/p'
GPU0:
	apiVersion         = 1.4.305
	driverVersion      = 25.0.7
	vendorID           = 0x1002
	deviceID           = 0x15d8
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = AMD Radeon Graphics (RADV RAVEN)
	driverID           = DRIVER_ID_MESA_RADV
	driverInfo         = Mesa 25.0.7-2+deb13u1
$ ls -l /dev/dri/renderD128; dpkg -l libvulkan1 mesa-vulkan-drivers | awk '/^ii/{print $2, $3}'
crw-rw---- 1 root render 226, 128 oct 30 13:00 /dev/dri/renderD128
libvulkan1:amd64 1.4.309.0-1
mesa-vulkan-drivers:amd64 25.0.7-2+deb13u1
```

Debian 13 (trixie), kernel 6.12.113, gcc 14.2.0, cmake 3.31.6, Agentica
0.35.0, native-gen 0.4.0, libtorch 2.4.1+cpu (found by
`tensor/tools/build-torch.sh`), 12 GiB of RAM. The machine is shared and was
under load 1–2.6 of 8 CPUs while these ran, which is why the numbers are
ranges; the iGPU's numbers were the stable ones (±2 %), and the 8-thread CPU
ones the noisy ones (±50 % on the smallest shape).

What ggml says about the device when it opens it — the `uma: 1` is the
sentence the verdict turns on:

```console
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV RAVEN) (radv) | uma: 1 | rebar: 1 | fp16: 1 | bf16: 0 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: none
```

## Run it

The submodule is pinned at ggml's `v1.0.0`; nothing else has to be fetched:

```sh
git submodule update --init third_party/ggml
```

The Vulkan build needs what ggml's Vulkan backend needs: the Vulkan headers
and loader (`libvulkan-dev`), `glslc`, and `SPIRV-Headers`
(`spirv-headers`). On a machine where they are not system packages (this one
had no root), extract the distribution's `.deb`s into a prefix and point
`VULKAN_SDK` at it — that is exactly what the script looks for:

```sh
for d in libvulkan-dev glslc libshaderc1 spirv-headers; do apt-get download "$d"; done
mkdir -p ~/.local/vulkan-sdk && for d in *.deb; do dpkg-deb -x "$d" ~/.local/vulkan-sdk; done
```

Then, from `spike/ggml-vulkan/`:

```sh
sh tools/build-ggml.sh                    # both builds; prints and stores the cmake flags
eval "$(sh tools/build-ggml.sh | grep '^export ')"   # C_INCLUDE_PATH, LIBRARY_PATH
agentica install                          # builds native/ggmlspike.c against ggml
agentica check && agentica fmt --check && agentica test
agentica demo.agc                         # the boundary, both backends, the assertion
agentica bench.agc vulkan                 # the iGPU numbers (also: cpu1, cpu8)
```

`tools/build-ggml.sh` exits 1 with what it looked for when a Vulkan
dependency is missing, so a run that cannot build Vulkan says so instead of
producing CPU numbers under a Vulkan label. Without the module built, every
program prints `the ggmlspike module is not built or loadable: SKIPPED` and
exits 0.

The build itself, from a clean tree (the flags are the script's, `GGML_NATIVE=OFF`
pins AVX2/FMA and leaves AVX512 off):

```console
$ sh tools/build-ggml.sh
== cmake -S third_party/ggml -B build/cpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_VULKAN=OFF
...
== cmake -S third_party/ggml -B build/vulkan -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_VULKAN=ON -DCMAKE_PREFIX_PATH=/home/sonic182/.local/vulkan-sdk/usr
...
ggml headers: /home/sonic182/apps/agentica-ml-ggml-vulkan/third_party/ggml/include
export C_INCLUDE_PATH=".../third_party/ggml/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export LIBRARY_PATH=".../build/vulkan/src:.../build/vulkan/src/ggml-vulkan:.../build/cpu/src:.../lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
```

| what | first build (empty ccache) | with ccache warm | bytes |
|---|---|---|---|
| CPU tree (`build/cpu`) | 31 s | 1.8 s | 7 481 525 |
| Vulkan tree (`build/vulkan`) | 3 m 32 s | 1 m 33 s | 306 378 247 |
| `libggml.a` / `libggml-base.a` / `libggml-cpu.a` (CPU tree) | | | 59 284 / 1 315 662 / 1 682 034 |
| `libggml-vulkan.a` | | | **43 390 254** (mostly precompiled shaders) |
| our module `libggmlspike.so` | | | **42 MB** (it statically links the Vulkan backend) |
| ggml submodule checkout | | | 35 MB |

The CPU backend archive is the same file in both trees — `GGML_VULKAN=ON` adds
a backend and changes nothing about the CPU one — so the CPU numbers below are
the CPU-only configuration:

```console
$ cmp build/cpu/src/libggml-cpu.a build/vulkan/src/libggml-cpu.a && echo identical
identical
```

## The boundary

`ggml_spike.agc` declares thirteen functions and nothing else: devices
enumeration, the two backends, `describe`, `create`, `add`, `relu`,
`mul_mat`, `empty_graph`, `checksum`, `readback`, `allocated`, `memory`.
`native/ggmlspike.c` is the thin C layer: one ggml call per function, plus the
two checks ggml would otherwise do silently wrong (ggml's `add` broadcasts)
and the readback. The generated header is committed and checked by the same
`--check` every package uses.

The assertion the spike exists for is in `backend_vulkan`:

```c
if (ggml_backend_vk_get_device_count() < 1) ... error
backend = ggml_backend_vk_init(0);
if (!ggml_backend_is_vk(backend)) ... error
device = ggml_backend_get_device(backend);
if (device == NULL || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) ... error
```

and `describe` prints what ggml reports for the backend in use, so a CPU
fallback is visible in the output rather than assumed away. It is not:

```console
$ agentica demo.agc
# ggml devices, as ggml_backend_dev_count/ggml_backend_dev_get report them
  Vulkan0 — AMD Radeon Graphics (RADV RAVEN) (IGPU)
  CPU — AMD Ryzen 5 3550H with Radeon Vega Mobile Gfx (CPU)

# describe: what the backend in use really is, read back from ggml
  cpu:    CPU — AMD Ryzen 5 3550H with Radeon Vega Mobile Gfx (CPU), 8 threads
  vulkan: Vulkan0 — AMD Radeon Graphics (RADV RAVEN) (IGPU), vulkan=1

# device memory (free, total) in bytes
  cpu     free 13546401792, total 13546401792
  vulkan  free 7687839744, total 9994424320

# the same five calls on both backends, [64, 64] F32
  cpu     a [64, 64] F32, 16384 bytes on the backend
  cpu     relu(add) checksum 1259.345452
  cpu     mul_mat checksum  4408.280273 (16384 bytes)
  cpu     first values of mul_mat: [7.342148780822754, -1.804958701133728, -1.7256197929382324, 3.5669422149658203]
  vulkan  a [64, 64] F32, 16384 bytes on the backend
  vulkan  relu(add) checksum 1259.345452
  vulkan  mul_mat checksum  4412.357582 (16384 bytes)
  vulkan  first values of mul_mat: [7.345683574676514, -1.8047490119934082, -1.7258830070495605, 3.568636417388916]

checksum of mul_mat(add, add): cpu 4408.280273, vulkan 4412.357582
gap 4.077308, 0.000925 of the cpu sum (the Vulkan default computes mul_mat in fp16 here)
the two backends agree: the Vulkan run computed the same tensor

# a shape error is a Result error, not a halt, and it names the shapes
  add [2, 3] + [3, 2]: add takes mutually compatible tensors, got [2, 3] and [3, 2]

# the cost of one dispatch round trip with no op in it, 200 rounds
  cpu, 8 threads   median   6635 ns, min   5168 ns, max 111190 ns
  vulkan           median 1999448 ns, min 751784 ns, max 2121463 ns
```

`relu(add)` is bit-identical between the backends: elementwise ops are.
`mul_mat` is not, and the gap is the device's arithmetic, not a bug:

```console
$ GGML_VK_DISABLE_F16=1 agentica demo.agc | grep -A2 'checksum of mul_mat'
checksum of mul_mat(add, add): cpu 4408.280273, vulkan 4408.280233
gap 0.000040, 0.000000 of the cpu sum
```

So on this device ggml's Vulkan backend computes F32 `mul_mat` **in fp16 with
fp32 accumulation** unless `GGML_VK_DISABLE_F16` is set: 9.3e-4 relative error
on a 64×64 product, against 9e-9 in fp32. Both are measured below, because the
choice is worth ~1.6× at 512³ and nothing at 1024³.

### The two helper programs

The libtorch reference columns and the dispatch probe are ordinary C/C++
programs, built with the same environments the module uses:

```sh
# libtorch CPU, the reference point for the mul_mat table
eval "$(sh ../../tensor/tools/build-torch.sh)"
g++ -O2 -std=c++17 tools/bench-torch.cpp -o build/bench-torch \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10 -lpthread
./build/bench-torch 1
./build/bench-torch 8

# one dispatch round trip, an unwaited submit, and N mul_mats in one graph
eval "$(sh tools/build-ggml.sh | grep '^export ')"
cc -O2 tools/probe-dispatch.c -o build/probe-dispatch -I../../third_party/ggml/include \
  build/vulkan/src/libggml.a build/vulkan/src/ggml-vulkan/libggml-vulkan.a \
  build/vulkan/src/libggml-cpu.a build/vulkan/src/libggml-base.a \
  -lvulkan -lstdc++ -lm -lpthread
./build/probe-dispatch
```

### Check, format, header, tests

```console
$ agentica check
check ok: 4 files

$ agentica fmt --check
(no output, exit 0)

$ agentica deps/native-gen/cli.agc ggml_spike.agc --output native/ggmlspike_native.h --check
(no output, exit 0: the committed header is the one the block makes)

$ eval "$(sh tools/build-ggml.sh | grep '^export ')" && agentica test
PASS ... / the module lists ggml devices or is absent
PASS ... / vulkan is an igpu and not a cpu fallback
PASS ... / readback returns the pattern
PASS ... / add and relu agree between the backends
PASS ... / mul mat agrees within the arithmetic the device uses
PASS ... / a shape mismatch is an error and names the shapes
PASS ... / create takes a rank two shape
PASS ... / an empty graph runs on both backends
8 passed, 0 failed, 0 skipped, 0 not executed, 0 file errors
```

The tests assert the device type (not just "it ran"), compare the two backends
on `relu(add)` bit-for-bit and on `mul_mat` within the fp16 path's 2e-3, and
state "no Vulkan device on this machine" when there is none rather than pass
silently. Without the module built, the same run says so:

```console
$ env -u C_INCLUDE_PATH -u LIBRARY_PATH agentica demo.agc
... fatal error: ggml.h: No such file or directory ...
warning: native module ggmlspike not built: the C compiler failed with status 1; the package runs without it
the ggmlspike module is not built or loadable: SKIPPED
  native: cannot open .../native-build/unbuilt/: cannot open shared object file: No such file or directory
  (ggml is a system dependency of this spike: sh tools/build-ggml.sh)
```

This is the `optional = true` failure mode M0 documented: the build error is
not a run failure, and the programs say which half is missing.

## What it costs

Three groups of numbers, each with the command that produces it:

- **fixed costs**: one dispatch round trip with no op in it, one 4 MiB tensor
  made and freed, and reading it back;
- **`mul_mat`**: one module call per op (one graph, one submit) on ggml CPU at
  1 and 8 threads, on the iGPU through Vulkan (fp16 and fp32), and libtorch
  CPU at 1 and 8 threads on the same shapes;
- **what the iGPU is doing inside**: ggml's own per-node GPU timings, and the
  clock ramp of a laptop APU.

Every `mul_mat` row is `a[m, k] · b[n, k]ᵀ` — row-major inputs, ggml's
`mul_mat` convention, and `torch::matmul(a, b.transpose(0, 1))` for libtorch,
so all three multiply the same numbers with the same layouts. Work is
`2 · m · n · k` flops; `GFLOPS` is flop per nanosecond, i.e. 10⁹ flop/s.

### Fixed costs, per call

`agentica bench.agc {cpu1,cpu8,vulkan}`, median / min / max of 200 or 20
calls, milliseconds:

| backend | empty graph | create 1024×1024 (4 MiB) | read back 4 MiB | read back 4096 as `List[Float]` |
|---|---|---|---|---|
| ggml CPU, 1 thread | 0.006 / 0.005 / 0.08 | 2.25 / 2.20 / 4.98 | 1.52 / 1.40 / 1.96 | 0.149 / 0.135 / 0.197 |
| ggml CPU, 8 threads | 0.006 / 0.004 / 0.05 | 2.14 / 2.03 / 4.70 | 1.28 / 1.26 / 1.52 | 0.104 / 0.087 / 0.143 |
| ggml Vulkan (iGPU) | **2.00 / 1.18 / 2.09** | 3.96 / 3.00 / 5.51 | 2.62 / 2.47 / 5.27 | 0.281 / 0.238 / 0.776 |

- **One dispatch round trip on the iGPU costs 2 ms.** An empty graph, computed
  and waited for, is 2.0 ms on Vulkan and 6 µs on the CPU. That is the price
  of `ggml_backend_graph_compute` + synchronize through this driver, and every
  op below pays it before doing any work.
- Making a 4 MiB tensor (allocate + fill + upload) is ~2 ms on the CPU (the
  fill loop dominates) and ~4 ms on the iGPU (the host→device copy is in the
  round trip).
- Reading 4 MiB back costs 2.6 ms on the iGPU — of which 2.0 ms is the round
  trip — and 1.3 ms on the CPU. Reading 4096 values as a `List[Float]` costs
  0.28 ms on the iGPU (68 ns per value, the list decode M0 measured at ~20 ns
  per value plus the round trip), which is why `readback` is not used to
  produce any of the numbers below.

### `mul_mat`, one graph and one synchronize per op

`agentica bench.agc {cpu1,cpu8,vulkan}` and
`GGML_VK_DISABLE_F16=1 agentica bench.agc vulkan`; `tools/bench-torch.cpp` for
the libtorch columns (`./build/bench-torch 1` and `./build/bench-torch 8`,
built by `tensor/tools/build-torch.sh`'s environment). Median (min) of four to
eleven calls after one warm-up; the GFLOPS is the median's:

| shape | ggml CPU 1 thread | ggml CPU 8 threads | ggml Vulkan fp16 | ggml Vulkan fp32 | libtorch CPU 1 thread | libtorch CPU 8 threads |
|---|---|---|---|---|---|---|
| 256³ | 1.89 (1.49) ms · 18 | **0.93** (0.77) · 36 | 2.66 (2.59) · 12.6 | 2.81 (2.76) · 11.9 | 0.69 (0.63) · 49 | **0.27** (0.25) · 124 |
| 512³ | 12.3 (11.1) · 22 | 4.57 (3.92) · 59 | 4.63 (4.57) · 58 | 7.38 (7.27) · 36 | 5.60 (5.20) · 48 | **2.83** (1.84) · 95 |
| 1024³ | 102 (92) · 21 | 33.2 (28.1) · 65 | 18.1 (11.3) · 118 | 18.4 (13.9) · 117 | 48.7 (44.3) · 44 | **17.6** (15.9) · 122 |
| 512×2048, k=512 | 47.9 (44.2) · 22 | 17.8 (14.6) · 60 | **5.8** (5.5) · 185 | 6.8 (6.3) · 158 | 24.6 (22.7) · 44 | 11.8 (9.6) · 91 |

Across runs (5 runs for the ggml CPU columns, 3 for each Vulkan column, 2 for
libtorch), the spread was: Vulkan ±2 % on every shape except 1024³
(17.6–18.4 ms median, 11.3–11.8 ms min); ggml CPU 8 threads 0.88–8.76 ms at
256³ (one run hit the threadpool's wake-up, the rest sat at ~0.9 ms),
4.16–6.04 ms at 512³, 28.6–41.0 ms at 1024³; libtorch 8 threads 0.25–0.29 /
2.77–2.89 / 16.5–18.8 ms.

### What the iGPU is doing inside

`GGML_VK_PERF_LOGGER=1 agentica bench.agc vulkan` prints ggml's own per-node
GPU time from Vulkan timestamp queries. The difference between it and the
wall-clock row is the dispatch, and the sequence across repeats is the clock
ramp:

| shape | GPU time per call, in order | wall clock (median) |
|---|---|---|
| 256³ | 724, 721, 719, … 715 µs (flat) | 2.66 ms |
| 512³ | 2689, 2686, 2637, 2850, 2679, 2696, 2733, 2718 µs | 4.63 ms |
| 1024³ | 17860, 18729, 16204, **11978, 10141 µs** | 18.1 ms |
| 512×2048, k=512 | 6167, 5835, 5499, 5196, 4928, 4747, 4550, **4394 µs** | 5.8 ms |
| 512³, after the table (warm-up block) | 1886, 1330, 1346, 1361, 1398, 1421, 1432, 1483, 1507, 1524 µs | 2.5–2.7 ms |

Three things are in that table:

1. **The kernel is not the cost at small sizes.** 256³ is 0.72 ms of GPU and
   2.66 ms of wall clock: 73 % of the op is the round trip. At 512³ it is
   43 %, and at 1024³ about 10 %.
2. **The iGPU ramps.** The same 1024³ `mul_mat` costs 17.9 ms on its first
   call and 10.1 ms by the fifth — a 1.8× swing with no change in the code.
   The 512×2048 one goes from 6.2 ms to 4.4 ms over eight calls. A laptop APU
   clock-governs itself, so "one op, one synchronize" keeps it slow; the
   warm-up block (`bench.agc`, printed after the table) shows the same 512³ op
   at 1.3–1.5 ms of GPU time once the GPU has been busy for a while.
3. **Sustained, the iGPU does 180–245 GFLOPS** of fp16 `mul_mat` (1.3–1.5 ms
   per 512³, 4.4 ms per 512×2048). That is about a tenth of the Vega 8's
   theoretical fp16 rate, and it is what the "wins" below are made of.

`tools/probe-dispatch.c` isolates the dispatch from the arithmetic, and shows
what a graph with more than one op in it would pay:

```console
$ ./build/probe-dispatch
empty graph, submit + synchronize each:    1.997 ms  (200 rounds, mean)
empty graph, 100 submits then one wait:    1.995 ms total,  0.020 ms each
1 mul_mat 512^3 in one graph:    4.126 ms best,  4.126 ms per op ( 65.1 GFLOPS)
2 mul_mat 512^3 in one graph:    5.796 ms best,  2.898 ms per op ( 92.6 GFLOPS)
4 mul_mat 512^3 in one graph:    8.439 ms best,  2.110 ms per op (127.2 GFLOPS)
8 mul_mat 512^3 in one graph:    8.864 ms best,  1.108 ms per op (242.3 GFLOPS)
```

A submit that nobody waits for costs 20 µs; the wait is the 2 ms. Eight ops in
one graph cost 1.1 ms each instead of 4.1 ms each. **The 2 ms is a per-graph
cost, not a per-op one** — an op-by-op backend pays it per op, a step-shaped
graph pays it once.

### Memory

`agentica bench.agc vulkan` prints the backend buffer ggml allocated for each
tensor of the largest case, and what the device says it has:

```console
# memory, largest case (1024^3)
  a 4 MiB of values, backend buffer 4194304 bytes
  b 4 MiB of values, backend buffer 4194304 bytes
  c 4 MiB of values, backend buffer 4194304 bytes
  device memory: free 7669161984, total 9994424320
```

The buffer is exactly the tensor's bytes (4 MiB each for a 1024×1024 F32) —
ggml's Vulkan allocator does not pad these — and the "device memory" is the
machine's RAM: 10 GB of host memory, `uma: 1` in ggml's own device line. That
is reportable and it is the whole story: **the iGPU has no memory of its own
to be fast in.**

## What the numbers say

1. **Against the iGPU's natural rival, 8-thread AVX2 ggml, the iGPU wins two
   of four sizes, ties one and loses one.** 1024³: 18.1 ms against 33.2 ms (1.8×);
   512×2048: 5.8 ms against 17.8 ms (3.1×); 512³: 4.63 ms against 4.57 ms (a
   tie, and the CPU's min is better); 256³: 2.66 ms against 0.93 ms (the CPU
   wins 2.9×). The iGPU wins once an op carries roughly a gigaflop of work
   (512×2048 is 1.07 GFLOP and wins 3.1×) and ties below that.
2. **Against libtorch CPU at 8 threads it wins one of four sizes.** 512×2048:
   5.8 ms against 11.8 ms (2×); 1024³: 18.1 against 17.6 ms (a tie); 512³:
   4.63 against 2.83 ms (libtorch 1.6× faster); 256³: 2.66 against 0.27 ms
   (libtorch 10× faster). ggml's own CPU `mul_mat` is 2–4× slower than
   libtorch's (which uses a tuned BLAS), so "the iGPU beats ggml on the CPU"
   is a weaker statement than it looks.
3. **The fp16 default is doing real work in those numbers.** With
   `GGML_VK_DISABLE_F16=1` the iGPU loses 512³ outright (7.38 ms against
   4.57 ms for 8-thread AVX2) and gains nothing at 1024³ (18.4 vs 18.1 ms) or
   512×2048 (6.8 vs 5.8). The iGPU's win at 512×2048 is an fp16 win.
4. **There is a warm-up, and it is not a one-time one.** Two separate effects:
   the ~2 ms dispatch round trip, paid on every graph, and a clock ramp that
   makes the first calls ~1.8× slower than the steady state on a laptop APU.
   A per-op, per-synchronize design sees both on every op; a graph-per-step
   design sees the ramp once and the round trip once.
5. **At 256³ the iGPU is dispatch-bound.** 73 % of the wall clock is the
   round trip, and so is everything with less than ~0.7 ms of work in it —
   which is most §3.1 ops on tensors up to a few megabytes.
6. **What the numbers are worth.** This iGPU shares the CPU's memory
   (`uma: 1`, 10 GB of host RAM as "device memory", no heap of its own), so a
   bandwidth-bound op cannot win by moving to it; the wins above are
   arithmetic-bound ones, and they are 1.8–3.1× of wall clock against a
   sustained-kernel gap of 180–245 GFLOPS vs 59–65 GFLOPS (3–4×) — the
   difference between the two ratios is the 2 ms of dispatch and the clock
   ramp. It is a 2019 laptop APU: the honest summary is *a modest,
   size-dependent win over 8-thread AVX2, roughly a tie with a tuned BLAS CPU,
   at 2 ms of dispatch per graph.*

## What this means for the design

Not a proposed change — what the measurements and the boundary imply, for
whoever writes the decision.

### 1. `Device` needs a Vulkan variant, and `Backend` a `Ggml` arm

`tensor/DESIGN.md` §2 has `Device = Cpu | Cuda(Int)` and
`Backend = Reference | Torch(Device) | Xla(Device) | Candle(Device)`. A ggml
backend needs a third device variant — `Vulkan(Int)` is enough, and ggml's own
distinction between `GPU` and `IGPU` (`ggml_backend_dev_type`) argues for
carrying it, because a laptop APU is not a discrete card and this spike's
numbers are what an `IGPU` does. The `case` of §7 is exhaustive over the arm
set, so adding `Backend.Ggml(Device)` touches every dispatch site; that is the
bar the arm has to clear, and on these numbers it clears it only for
graph-shaped work (see 4).

The device type is not cosmetic for correctness either: the same arm computes
in fp16 by default on this device and fp32 elsewhere, so the conformance
suite (§11) would need a per-device tolerance (`2e-3` relative on `mul_mat`
here) or the backend would have to pin `GGML_VK_DISABLE_F16=1` and lose 1.6×
at 512³. The spike measures both; the design has to choose one.

### 2. (b′) fits mechanically: a second `extern module` inside `tensor/`, `optional = true`

The shape works exactly as the `Torch` arm does today: `tensor/` declares

```text
extern module "ggml"
  resource GgmlTensor        # the state owns a ggml_context + backend buffer
  compute def mul_mat(...) -> Result[GgmlTensor]
  ...
end
```

with `[native.ggml] optional = true` in `tensor/agentica.toml`, a
`Handle.Ggml(Ggml.TensorHandle)` arm next to `Handle.Native(...)`, and
`Backend.Ggml(Device.Vulkan(0))` answering `error("backend not available")`
where ggml is not built. The spike's module *is* that layer minus the
Agentica wrapper: 490 lines of C, one ggml call per function.

The cost, measured here:

- **ggml becomes a submodule of the package** (`third_party/ggml`, pinned;
  35 MB checkout) and `tensor/` grows a `tools/build-ggml.sh`-shaped build
  contract. The CI matrix would run it for `tensor/`
  (`tools/build-engine.sh` is the name it looks for), which is 1.8 s with
  ccache and 31 s cold for the CPU build — acceptable; the Vulkan build
  (3 m 32 s, 306 MB of build tree, `glslc` + `SPIRV-Headers` required) is not
  something a CI runner without a GPU should do, so a `tensor/` ggml backend
  would build CPU-only in CI and Vulkan only locally.
- **The module is 42 MB** because `libggml-vulkan.a` is 43 MB of shader
  blobs. The `Torch` arm's 58 KB module next to libtorch's 472 MB
  `libtorch_cpu.so` put the size in the library; here the size is in the
  shaders, and it lands in `tensor/`'s artifact.
- **ggml's C API is graph-first and non-owning.** Every op needs a
  `ggml_context` and a backend buffer, which is what §8's arena-per-call
  already models, but it means the wrapper creates and frees a context per op
  (measured above: `create 1024x1024` costs 2–4 ms on the iGPU, most of it the
  round trip).

### 3. The op closure: ggml is a transformer engine, not a tensor library

Of the 45 §3.1 primitives, ggml v1.0.0 has no constructor for:

- `pow` (no `GGML_OP_POW`), `eq`, `lt`, `where` (no comparison or select ops
  at all);
- `max` as a reduction (only `GGML_OP_POOL_1D` with `MAX` over a window);
- `sum`/`mean` with `axes`/`keep_dims`: `ggml_sum` is all axes,
  `ggml_sum_rows` and `ggml_mean` are dimension-0 only;
- `squeeze`, `unsqueeze`, `stack` (reshape/view covers the first two, nothing
  covers `stack`);
- `slice` (the op exists internally, but the public API only has
  `ggml_view_*`, which is a view, not the primitive);
- a general `gather` (only `ggml_get_rows`/`ggml_set_rows`, rows of dimension
  0 with integer indices);
- `scalar_to_tensor` (a scalar is a one-element tensor made and filled outside
  the graph; there is no op for it);
- `to_device` (not a graph op; a buffer copy the scheduler does).

What it *has* of §3.1 is the transformer core: `add`, `sub`, `mul`, `div`,
`neg`, `exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`, `gelu` (both forms),
`clamp`, `sum`, `mean`, `argmax`, `matmul` (batched through 4-D `mul_mat`),
`transpose`, `reshape`, `concat`, `broadcast_to` (as `repeat`), `conv1d`,
`avg_pool1d`, `to_dtype` (`cast`), `copy`, `contiguous`, `arange`, plus
`softmax`, `rope`, `flash_attn_ext`, `top_k`/`argsort`, `ssm_scan` that §3.1
deliberately leaves out. Two API mismatches matter more than any missing op:
**`ggml_mul_mat(a, b)` is `a · bᵀ`**, not `a · b`, so the wrapper has to
absorb the layout; and **`ggml_add` broadcasts**, so §9's eager rejection of
shape mismatches has to live in the wrapper (the spike's C layer checks by
hand, and its error names both shapes).

A `Ggml` arm would therefore not be a drop-in for §3.1: fourteen of the 45
primitives are missing or narrower than the contract (`sum`/`mean` without
`axes`/`keep_dims`, `gather` only along rows), so they would be
`error("not implemented on the ggml backend")` or emulated in Agentica over
ggml's ops — which is exactly what §3.2 says derived ops are, and the
conformance suite would have to accept them as derived, not as backend.

### 4. The 2 ms round trip is the design-relevant number

A `tensor/` op is one call: validate, dispatch, return. On the CPU backends
that call costs 6 µs of dispatch, so op-by-op is free. On the Vulkan arm it
costs **2 ms per op**, measured three ways (empty graph, `probe-dispatch`, and
the wall-clock minus GPU-time rows above). Consequences:

- an op-by-op `Ggml` arm on Vulkan makes every §3.1 op cost ≥2 ms, including
  `relu` on a 1024-element tensor;
- a step-shaped design (one graph, one submit) pays it once per step and the
  iGPU's throughput becomes the steady-state 180–245 GFLOPS: eight `mul_mat`s
  in one graph cost 1.1 ms each instead of 4.1;
- the submit itself is 20 µs when nobody waits. The 2 ms is the wait.

So the iGPU is not viable behind an op-by-op dispatch, and is interesting
behind a traced graph — which is §10's subject, not §7's. If §7 stays
op-by-op, the honest place for Vulkan is "not yet", and these numbers say why
in one line: 2 ms of dispatch against 0.7 ms of 256³ arithmetic.

### 5. CI

Nothing was added, deliberately. `spike/ggml-vulkan/agentica.toml` sits two
levels below what the workflow scans (`*/agentica.toml`), like `spike/m0`, so
the matrix does not pick it up; and no GitHub-hosted runner has a Vulkan
device or a pinnable driver, so a job that ran the GPU half would be a job
that skips or fails for reasons that are not this repository's code. The CPU
half would pass anywhere, and it is not worth a job: it measures one machine's
iGPU, and the numbers only mean something on the hardware they were taken on.
If a Vulkan runner ever exists, the job is
`git submodule update --init third_party/ggml && sh tools/build-ggml.sh &&
eval … && agentica test` — and it should be non-required until the driver can
be pinned.

## What this spike does not prove

- **No `tensor/` integration.** No `Backend` arm, no `Handle`, no wrapper
  validation, no conformance, no dtypes beyond F32, no autograd. The section
  above is a proposal, not a result.
- **One op per graph on purpose.** The module never batches, so the numbers
  are the op-by-op cost; `probe-dispatch` measures batching but not inside
  Agentica, and not with the FFI on top.
- **fp16 is the default on this device, so Vulkan `mul_mat` does not agree
  with the CPU bit-for-bit** (9.3e-4 relative here). Any conformance claim
  needs the fp32 path and its 1.6× cost, or a tolerance.
- **One machine, one driver, shared and loaded.** The 8-thread CPU numbers
  vary by ~50 % on the smallest shape (threadpool wake-up), the iGPU's by
  ±2 % except at 1024³ (17.6–18.4 ms median). Treat the tables as ranges and
  the verdicts as this machine's.
- **No memory-pressure or multi-op-graph evidence.** Only the buffer sizes of
  one op at a time; nothing about a model's worth of intermediates, and
  nothing about batching several ops' buffers.
- **No quantized types.** ggml's reason to exist is Q4_K and friends; this
  spike only touches F32 (and whatever fp16 arithmetic the driver picks).
- **No `ggml_backend_sched`.** The model loader path (`ggml_backend_sched`,
  which decides per-op backend placement and inserts copies) is not used; the
  module drives one backend directly. A real backend would use the scheduler,
  whose per-op placement costs are not in these numbers.

## Next step

If a Vulkan/ggml backend is wanted:

1. Decide §7's shape for a non-CPU device *first*: op-by-op dispatch cannot
   carry a 2 ms round trip, so the arm needs either graph capture (§10) or a
   scheduler-shaped API (`run(graph)`), neither of which exists in `tensor/`
   today. This spike's contribution is the number that forces the question.
2. If the answer is "graph per step": the spike's module is close to the
   wrapper that would sit behind `Handle.Ggml(...)` — a context per graph, one
   backend buffer for the graph, `ggml_backend_graph_compute` once, one
   readback. `tools/probe-dispatch.c` is the shape of the first benchmark for
   it.
3. If the answer is "op-by-op": keep Vulkan out, and keep the CPU backend
   (libtorch's BLAS beats ggml's CPU `mul_mat` by 2–4× on the same shapes).
4. Whatever the answer: the fourteen missing or narrower §3.1 primitives are a wrapper's
   work, not ggml's, and the fp16 default is a conformance decision that
   cannot be left implicit.
