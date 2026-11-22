# torch

The libtorch (CPU, eager) backend of the tensor stack — the §3.1 primitives
libtorch can express, behind `extern module "torch"` and a thin C++ layer over
libtorch's C++ API (as `tch-rs` is, in spirit).

libtorch is a **system dependency**, never vendored and never downloaded by
this repository: `tools/build-torch.sh` finds it through `TORCH_DIR` (or the
usual `/opt`, `/usr/local` and `$HOME/opt` places), and the module is
`optional = true` in `agentica.toml`, so a checkout without libtorch still
checks, formats and tests.

- `torch.agc` — the public API: `Tensor` (shape, dtype, `Handle.Native`),
  `available()`, `closed()`, and one function per primitive. Every op takes the
  device first, validates shapes, dtypes and devices before the native call
  (`tensor/DESIGN.md` §9), and answers `Result` — `error`, never a halt.
- `src/native.agc` — the FFI boundary and nothing else: `resource
  TensorHandle` and the fourteen `compute def` calls. The generated header
  `native/torch_native.h` comes from this file (native-gen).
- `src/dtypes.agc`, `src/shapes.agc` — the dtype table with the codes that
  cross to C, the two conversions between a storage double and a dtype-tagged
  `Values`, and the shape arithmetic a reduction needs. The backend validates
  its own storage crossing and computes the shape it is about to hand back.
- `native/torch.c` — the thin C++ layer: one libtorch call per function, the
  checks that keep libtorch's implicit behaviour (broadcasting, wrapping) from
  leaking, and the `error` text. libtorch has no C API, so this is C++ in a
  `.c` file: the build table names no compiler, so `CC` selects the driver.
- `tools/build-torch.sh` — finds libtorch and prints the exports the build
  table cannot name (`TORCH_DIR`, `CC` with the rpath, `CPLUS_INCLUDE_PATH`,
  `LIBRARY_PATH`, `LD_LIBRARY_PATH`). Without a libtorch it exits 1 and says
  what it looked for.
- `tools/bench.agc` — the numbers in "What the dependency costs", reproducible.
- `test/` — the conformance suite against `tensor/`'s reference backend.

`torch/` takes `tensor/` **by path** (`tensor = { path = "../tensor" }` in
`agentica.toml`): both are folders of this repository, and `tensor/` has no tag
yet, so a git pin would be a commit id nobody has released. A `path` is a root
manifest's key, which this folder is when `agentica` runs inside it (locally
and in CI); a *fetched* `torch` package will need the tag pin instead, once
`tensor-0.1.0` exists.

## What is implemented, and what is not

`tensor/DESIGN.md` §3.1 names 45 primitives. **18 are implemented here**, and
27 are not. Every implemented one is checked against the reference on the same
inputs, values and shape/dtype, plus its error paths.

| §3.1 group | Implemented | Not implemented |
|---|---|---|
| Creation (7) | `zeros`, `ones`, `empty`, `full`, `like`, `arange`, `scalar_to_tensor` | — |
| Elementwise (17) | `add`, `mul`, `relu`, `exp`, `log` | `sub`, `div`, `neg`, `sqrt`, `pow`, `sigmoid`, `tanh`, `gelu_tanh_approx`, `eq`, `lt`, `where`, `clamp` |
| Reductions (4) | `sum`, `mean` (with `axes` and `keep_dims`) | `max`, `argmax` |
| Linear algebra (3) | `matmul`, `transpose` | `batch_matmul` |
| Shape (8) | `reshape` | `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `broadcast_to`, `gather` |
| Convolution/pooling (2) | — | `conv1d`, `avg_pool1d` |
| Movement (4) | `to_dtype` | `to_device`, `copy`, `contiguous` |

Plus the two host-crossing functions §4 names, `from_list` and `to_list`
(readback), which are not primitives.

Why the gaps, in three groups:

- **`conv1d`/`avg_pool1d`** are missing because `tensor/` refuses them too
  (`error("not implemented")`: no §3.2 derived op needs them and §5 admits an
  op only for a consumer that does). There is nothing to conform against, so a
  torch version could not be checked the way the rest of this package is.
- **`div` is the one that matters.** `softmax`, `layer_norm`,
  `multi_head_attention` (§3.2) all divide, so they cannot be built over the
  M2 primitive set on either backend. That is why the composite checks below
  are an MLP block and a scaled score matrix, not full attention: `mul` by a
  `full` scale tensor is the only division-free way to write the scale, and
  `gelu_tanh_approx` needs `tanh`, which is also missing.
- **Everything else is a native call away** and is deliberately not in this
  slice: M2 is the fourteen native calls of `src/native.agc`, and each op is
  one libtorch line plus its validation. libtorch expresses all 27
  (`x - y`, `x / y`, `-x`, `sqrt`, `pow`, `sigmoid`, `tanh`, `eq`, `lt`,
  `where`, `clamp`, `max`, `argmax`, batched `matmul`, `squeeze`/`unsqueeze`,
  `slice`, `cat`/`stack`, `expand`, `gather`, `conv1d`, `avg_pool1d`,
  `.to(device)`, `clone`, `contiguous`).

Two more honest limits of what crosses the boundary:

- shapes, dtypes and devices cross as `List[Int]`, an `Int` code and nothing
  (CPU only); there is no struct over the boundary, so the wrapper is the one
  that knows both metadata and handles.
- readback crosses as storage doubles: `I64` values past 2^53 lose precision
  on the way out.

## Local setup

libtorch 2.4.1+cpu (the shared-library CPU release) is what this package was
built and tested against:

```sh
# https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cpu.zip -d "$HOME/opt"
export TORCH_DIR="$HOME/opt/libtorch"
eval "$(sh tools/build-torch.sh)"   # in the shell that runs agentica
agentica test                        # builds the module on first use
agentica tools/bench.agc
```

`eval "$(sh tools/build-torch.sh)"` is what makes the build possible at all:
`agentica.toml` can name `.c` sources, `-D` defines and `-l` names, and nothing
else, so the C++ driver (`CC`), libtorch's headers (`CPLUS_INCLUDE_PATH`) and
its libraries (`LIBRARY_PATH`) have to come from the environment. The build
cache key includes those three variables, so a shell without them builds
nothing and warns.

Without libtorch the package still works: `agentica check`, `agentica fmt
--check` and `agentica test` run, the run prints

```text
warning: native module torch not built: the C compiler failed with status 1; the package runs without it
```

and every conformance test takes its `Support.absent()` path, which asserts
that `Torch.available()` is false and that an op answers `Result` error
instead of halting. `agentica test` reports those as passed, so **look for that
warning line to know whether the conformance suite really ran**; nothing in
this package silently pretends to have compared against the reference.

## CI

`.github/workflows/ci.yml` has one generic job per package folder, and this
folder is one of them. It runs `agentica check`, `agentica fmt --check`, the
native-gen `--check` of `native/torch_native.h`, and `agentica test` on the
pinned Agentica. It does **not** install libtorch: a 473 MB `libtorch_cpu.so`
plus 51 MB of headers is not something this repository downloads, and a
download it cannot pin by sha256 would be worse than not running. So:

- what CI proves: the package checks, formats, that the committed header is
  the one native-gen makes from `src/native.agc`, and that the conformance
  suite self-skips by asserting the absent state (with the warning above).
- what was run locally, by hand, with libtorch 2.4.1+cpu at `$HOME/opt/libtorch`:
  all 62 tests (60 conformance comparisons and error paths against the
  reference, 2 module-state checks) and `tools/bench.agc`.

When libtorch becomes pinnable (a `prebuilt` entry with a per-target sha256, or
a CI image that ships it), the fix is a `[native.torch] prebuilt` table and
nothing else: `optional = true` stays for the machines that have neither.

## Tolerances, and why

`test/support.agc` compares against the reference with

```text
|a - b| <= 1e-7                       absolute, for values near zero
|a - b| <= 1e-7 * (|a| + |b|)         relative, above that
```

and requires exact equality for integer and `Bool` values, and exact shape and
dtype equality.

The reason for a tolerance at all: the reference computes in doubles
(`tensor/DESIGN.md` §2 fixes `F32` as the compute dtype but the reference keeps
its storage in Agentica floats), while libtorch stores and computes `F32`.
`F32`'s epsilon is 1.19e-7, so one rounding of a value of magnitude ~1 is about
6e-8 relative — 1e-7 is one float32 ulp of slack, no more. It is deliberately
not looser: a `matmul` that picked the wrong operand, or an `exp` applied
twice, misses it by orders of magnitude.

Where the two backends legitimately disagree and are therefore *not* asserted:

- **saturation**: `exp` overflows `F32` above ~88.7 and gives `+inf`, while the
  reference saturates at 1e308 in doubles. Both are "finite in, non-finite
  out"; the conformance values stay inside `|x| <= 8`.
- **`F64`/`F16` compute**: the reference keeps those values exact; libtorch
  rounds them through the same 24-bit mantissa when it computes in `F32`. The
  tests use `F32` input, so this never hides behind the tolerance.
- **`I64` readback past 2^53**: the boundary carries doubles; the tests use
  values at 2^53 exactly, which round-trip.

## Lifetimes

`tensor/DESIGN.md` §8: one `Native.arena` per call, and one per decode step
when a caller runs several ops.

- Every op in `torch.agc` opens its own `Native.arena` and the tensor it
  returns is the block's value: results leave the arena, the arguments (opened
  by the caller) are not its to close. `test/lifetime_test.agc` checks both
  halves, including a struct field.
- A caller that runs a block — a decode step, an inference call — wraps it in
  one `Native.arena`: the intermediates die at the end of the step, newest
  first, instead of waiting for a GC that sees them as any other resource.
  `test/lifetime_test.agc` runs a three-op step that way, and the composite
  checks run their whole block that way.
- `Torch.closed(t)` answers whether a handle is closed, so a test can assert
  that nothing closed what a value still reaches. A native handle freed under a
  live `Tensor` is `RUNTIME-NATIVE-CLOSED`, never UB.
- M2 needed nothing that arenas cannot express. The one thing a caller cannot
  observe from Agentica is an intermediate whose handle no value reaches (by
  design it is closed at the end of the block), so the suite checks the
  observable half: what the value reaches is alive, and the arguments are.

## What the dependency costs

One machine (Linux x86_64, 12 cores), libtorch 2.4.1+cpu from the plain CPU
release; the build numbers are a single `native/torch.c` compile.

| What | Measured |
|---|---|
| libtorch unpacked | 786 MB (lib/ 736 MB, include/ 51 MB; `libtorch_cpu.so` alone 473 MB) |
| clean build of the module | 18.7 s wall, 1.1 GB peak RSS (one `g++` of `native/torch.c`) |
| the module itself | `libtorch.so`, 87 KB |
| first `Native.load("torch")` | ~400 ms (dlopen of libtorch) |

Per-op cost, 2000 rounds each, `tools/bench.agc`, nanoseconds per op (the
torch column includes the free of the tensor the op made):

| op (shape) | reference | torch |
|---|---|---|
| `add [2, 2]` | 2.8 µs | 17 µs |
| `add [64, 64]` | 480 µs | 21 µs |
| `matmul [4, 3]x[3, 2]` | 3.6 µs | 11.5 µs |
| `matmul [64, 64]x[64, 64]` | 6.6 ms | 33 µs |
| `from_list [64, 64]` (4096 values) | 784 µs | 1.6 ms |
| `exp [64, 64]` | 1.43 ms | 72 µs |
| `sum [64, 64]` | 1.13 ms | 20 µs |
| `linear`+`relu` step, `[4, 3]` | 8.0 µs | 37 µs |

What those two columns say:

- **A small op is dominated by the crossing** (an argument count, ~10 µs of
  arena, resource and term work). `add [2, 2]` at 17 µs is not computing
  anything; a step made of twenty tiny ops pays ~340 µs before libtorch does
  any work. This is the number `tensor/DESIGN.md` §8's arena-per-step exists to
  amortise, and the reason the §3.2 derived ops (`softmax`, `layer_norm`) will
  eventually want to be one native call rather than five.
- **Real work wins outright**: 64×64 mats and transcendentals are 7× to 200×
  faster than the reference, which is the point of the backend.
- **Crossing a big tensor in is the exception**: `from_list [64, 64]` is
  2× slower than the reference because 4096 values cross as VM terms; a
  checkpoint loader (§6) and `create`-from-file will want a bulk path
  (`Bytes` + dims) rather than `List[Float]`.

## Design delta

`tensor/`'s dispatch is not wired to this package, on purpose: wiring it needs
a decision this PR does not take. `tensor/tensor.agc`'s `Backend.Torch` arm
still answers `error("backend not available")`, `tensor/DESIGN.md` is
untouched, and the PR body's "design delta needed (dispatch)" has the finding
and the recommendation in full. Short version: the design's
`Handle = Native(Resource)` needs either a language-level opaque resource type
(in which case the ops table of ml#1's option (a) works as designed) or
`tensor/` must declare the native module itself (option (b)), because a
`resource` is nominal to the `extern module` block that declares it and cannot
cross packages. M2 ships the backend; the move under either option is
mechanical.
