# tensor

Functional tensors over pluggable backends: the public API (`tensor.agc`), a
pure-Agentica **reference** backend that is the oracle, and the **libtorch
(CPU, eager)** backend behind `extern module "torch"` and a thin C++ layer over
libtorch's C++ API (as `tch-rs` is, in spirit). The contract is
[`DESIGN.md`](DESIGN.md); the execution plan is the repository's
[`roadmap.md`](../roadmap.md).

libtorch is a **system dependency**, never vendored and never downloaded by
this repository: `tools/build-torch.sh` finds it through `TORCH_DIR` (or the
usual `/opt`, `/usr/local` and `$HOME/opt` places), and the module is
`optional = true` in `agentica.toml`, so a checkout without libtorch still
checks, formats and tests. `Tensor.available(Tensor.Backend.Torch(Tensor.Device.Cpu))`
answers whether this machine has it, and the dispatch arm answers
`error("backend not available")` when it does not.

- `tensor.agc` — the public API. Every op returns `Result` (`error`, never a
  halt), validates shape/dtype/backend/device before it dispatches
  (`DESIGN.md` §9), and dispatches by value over the four `Backend` arms
  (§7).
- `src/reference.agc` — the reference backend, pure Agentica over
  `List[Float]`; the oracle every other backend is checked against.
- `src/torch.agc` — the libtorch backend: the §3.1 primitives libtorch
  expresses, per-op checks, and one `Native.arena` per call (§8).
- `src/native.agc` — the FFI boundary and nothing else: `resource
  TensorHandle` and the thirty-seven `compute def` calls. The generated header
  `native/torch_native.h` comes from this file (native-gen).
- `src/torch_dtypes.agc`, `src/torch_shapes.agc` — the dtype table with the
  codes that cross to C, the two conversions between a storage double and a
  dtype-tagged `Values`, and the shape arithmetic a reduction needs. The
  backend validates its own storage crossing and computes the shape it is
  about to hand back.
- `native/torch.c` — the thin C++ layer: one libtorch call per function, the
  checks that keep libtorch's implicit behaviour (broadcasting, wrapping) from
  leaking, and the `error` text. libtorch has no C API, so this is C++ in a
  `.c` file: the build table names no compiler, so `CC` selects the driver.
- `tools/build-torch.sh` — finds libtorch and prints the exports the build
  table cannot name (`TORCH_DIR`, `CC` with the rpath, `CPLUS_INCLUDE_PATH`,
  `LIBRARY_PATH`, `LD_LIBRARY_PATH`). Without a libtorch it exits 1 and says
  what it looked for.
- `tools/bench.agc` — the numbers in "What the dependency costs", reproducible.
- `test/` — the unit tests of the reference backend, and the libtorch
  conformance suite (`test/torch_*_test.agc`) that runs the same public call on
  both backends and compares.

The native module lives in this package rather than in a `torch/` of its own
because a `resource` is nominal to the `extern module` block that declares it:
`Tensor.handle` has to name `TensorHandle`, and the `Backend.Torch` arm has to
call the module, so both have to be declared here (`DESIGN.md` §1, §2, §7).
[agentica#1918](https://github.com/agentica-lang/agentica/issues/1918) — an
opaque resource that can live in data and cross modules — is the change that
lets the backend move out into its own package again; until it lands, this one
carries the libtorch layer as a deliberate, documented trade-off.

## What is implemented, and what is not

`DESIGN.md` §3.1 names 45 primitives. Both backends compute them: the
**reference computes all 45**, and the **libtorch backend implements 42** and
answers `error("<op> is not implemented on the torch backend yet")` for the
other three, so the gap is an error naming the op, never a wrong number. Every
implemented one is checked against the reference on the same inputs, values and
shape/dtype, plus its error paths.

| §3.1 group | Implemented on libtorch | Not implemented |
|---|---|---|
| Creation (7) | `zeros`, `ones`, `empty`, `full`, `like`, `arange`, `scalar_to_tensor` | — |
| Elementwise (17) | `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `pow`, `relu`, `sigmoid`, `tanh`, `eq`, `lt`, `where`, `clamp` | `gelu_tanh_approx` |
| Reductions (4) | `sum`, `mean`, `max`, `argmax` (each with `axes` and `keep_dims`) | — |
| Linear algebra (3) | `matmul`, `transpose` | `batch_matmul` |
| Shape (8) | `reshape`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `broadcast_to`, `gather` | — |
| Convolution/pooling (2) | `conv1d` (`stride`, `padding`), `avg_pool1d` (`kernel`, `stride`, `padding`) | — |
| Movement (4) | `to_dtype`, `copy`, `contiguous` | `to_device` |

Plus the two host-crossing functions §4 names, `from_list` and `to_list`
(readback), which are not primitives.

Why those three are still out, in this slice's terms:

- **`gelu_tanh_approx`** is one libtorch call (`at::gelu(..., "tanh")`), but it
  is not part of this change, so no test here pins it against the reference.
  Everything else a transformer block uses is in: the composite conformance
  checks run `softmax`, `layer_norm`, `rms_norm`, `silu`, the masked and
  unmasked `multi_head_attention` and `greedy_step` on **both** backends.
- **`batch_matmul`** is likewise expressible (`at::matmul` already takes a
  batch) and likewise untested here. `multi_head_attention` does not need it:
  it splits heads into rank-2 `matmul` calls (§3.2).
- **`to_device`** would only ever move a tensor to the one device this CPU
  release serves, which is the identity; `Cuda` is already `error("backend not
  available")` at the module level. Adding it would claim a device story this
  build does not have.

The three that *did* land changed what the §3.2 vocabulary can do on libtorch:
`div` (with `sub`, `max`, `argmax`, `sqrt`, `pow` and `broadcast_to`) is what
`softmax`, `layer_norm`, `rms_norm` and attention divide with, and
`conv1d`/`avg_pool1d` now exist on both sides because the reference computes
them too — the oracle comes first, then the backend is checked against it.

`conv1d`/`avg_pool1d` take the window `DESIGN.md` §3.1 admits and no more:
`stride` ≥ 1 and `padding` ≥ 0, no dilation, no groups. `avg_pool1d` also caps
`padding` at half the kernel, because that is the window the libtorch kernel
admits; the cap lives in the public wrapper, so the reference and libtorch
refuse exactly the same windows. `conv1d` takes `[C_in, L]` or
`[N, C_in, L]` against a `[C_out, C_in, K]` weight, and `avg_pool1d` takes
`[C, L]` or `[N, C, L]`; both answer the input's leading axes plus the new
length, and both refuse an empty input, a window wider than the padded input
and a channel-count mismatch.

Two more honest limits of what crosses the boundary:

- shapes, dtypes and devices cross as `List[Int]`, an `Int` code and nothing
  (CPU only); there is no struct over the boundary, so the wrapper is the one
  that knows both metadata and handles. A `List` of handles does not cross
  either, which is why native `concat` joins **two** tensors and
  `src/torch.agc` folds: concatenation along one axis is associative, so the
  fold answers exactly what one call would.
- readback crosses as storage doubles: `I64` values past 2^53 lose precision
  on the way out. Relatedly, `arange` past an int dtype's width diverges by
  side — libtorch's conversion wraps (200 into `I8` reads back `-56`) while the
  reference refuses to read it back at all — and
  `test/torch_creation_test.agc` pins both behaviours rather than hiding them.

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

and every conformance test takes its `Conform.absent()` path, which asserts
that `Tensor.available(...)` is false and that the dispatch arm answers
`Result` error instead of halting. `agentica test` reports those as passed, so
**that warning line is the signal that the conformance comparison did NOT
run** — it is not a note, it is the difference between "the libtorch backend
agrees with the oracle" and "nothing was compared". A green run without it is a
green run with the comparison.

## CI

`.github/workflows/ci.yml` has one generic job per package folder, and this
folder is one of them. It runs `agentica check`, `agentica fmt --check`, the
native-gen `--check` of `native/torch_native.h`, and `agentica test` on the
pinned Agentica. It does **not** install libtorch: a 473 MB `libtorch_cpu.so`
plus 51 MB of headers is not something this repository downloads, and a
download it cannot pin by sha256 would be worse than not running. So:

- what CI proves: the package checks, formats, that the committed header is
  the one native-gen makes from `src/native.agc`, and that the suite runs green
  with the module absent by *asserting* that absence — the `warning:` line
  above is present in the job log, and no test silently skipped.
- what CI does **not** prove: that `native/torch.c` compiles, that it still
  matches libtorch's C++ API, or that the values agree with the reference. With
  `optional = true`, a broken C++ file, a libtorch API drift and a missing
  libtorch are the same one-line warning inside a green job.
- what was run locally, by hand, with libtorch 2.4.1+cpu at `$HOME/opt/libtorch`:
  `agentica check`, `agentica fmt --check`, the header `--check`, `agentica
  test` (318 tests: the reference suite plus the libtorch conformance suite)
  and `agentica tools/bench.agc`.

The exit is a **non-required** CI job that installs libtorch pinned by sha256
and runs `agentica test` with `TORCH_DIR` set, so the build and the comparison
are gated somewhere without making every unrelated PR depend on a 500 MB
download. Until that job exists, the warning line is the contract: green CI
says "the contract still checks", not "the backend still agrees with the
oracle".

## Tolerances, and why

`test/torch_support.agc` compares against the reference with

```text
|a - b| <= 1e-7                       absolute, for values near zero
|a - b| <= 1e-7 * (|a| + |b|)         relative, above that
a == b                                when either side is not finite
```

and requires exact equality for integer and `Bool` values, and exact shape and
dtype equality.

The reason for a tolerance at all: the reference computes in doubles
(`DESIGN.md` §2 fixes `F32` as the compute dtype but the reference keeps its
storage in Agentica floats), while libtorch stores and computes `F32`. `F32`'s
epsilon is 1.19e-7, so one rounding of a value of magnitude ~1 is about 6e-8
relative — 1e-7 is one float32 ulp of slack, no more. It is deliberately not
looser: a `matmul` that picked the wrong operand, or an `exp` applied twice,
misses it by orders of magnitude.

The third line is not decoration: without it the tolerance is vacuous at
infinity (`inf - inf` is `NaN`, and `inf <= inf` is true), so a saturating
libtorch op would "agree" with any large finite value the reference produced.
`test/torch_elementwise_test.agc` pins exactly that case — `exp(100)` is `+inf`
in `F32` and finite in doubles — and asserts that the comparison *fails*.

Where the two backends legitimately disagree and are therefore not asserted:

- **saturation**: `exp` overflows `F32` above ~88.7 and gives `+inf`, while the
  reference saturates at 1e308 in doubles. Both are "finite in, non-finite
  out"; the conformance values stay inside `|x| <= 8`.
- **`F64`/`F16` compute**: the reference keeps those values exact; libtorch
  rounds them through the same 24-bit mantissa when it computes in `F32`. The
  tests use `F32` input, so this never hides behind the tolerance.
- **`I64` readback past 2^53**: the boundary carries doubles; the tests use
  values at 2^53 exactly, which round-trip.

## Lifetimes

`DESIGN.md` §8: one `Native.arena` per call, and one per decode step when a
caller runs several ops.

- Every op in `src/torch.agc` opens its own `Native.arena` and the tensor it
  returns is the block's value: results leave the arena, the arguments (opened
  by the caller) are not its to close. `test/torch_lifetime_test.agc` checks
  both halves, including a struct field.
- A caller that runs a block — a decode step, an inference call — wraps it in
  one `Native.arena`: the intermediates die at the end of the step, newest
  first, instead of waiting for a GC that sees them as any other resource.
  `test/torch_lifetime_test.agc` runs a three-op step that way, and the
  composite conformance checks run their whole block that way.
- `Handle.Native` and `Handle.Plain` coexist in one process: a `Reference`
  tensor and a `Torch` tensor go through the same `Tensor.*` calls, dispatch by
  value, and mixing the two backends is an `error`. The same test asserts the
  arena rules from Agentica's side: a value that leaves an arena is still
  readable, an argument is not closed by an op's arena, and an explicitly
  `Native.close`d handle reads as closed.
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
  any work. This is the number `DESIGN.md` §8's arena-per-step exists to
  amortise, and the reason the §3.2 derived ops (`softmax`, `layer_norm`) will
  eventually want to be one native call rather than five.
- **Real work wins outright**: 64×64 mats and transcendentals are 7× to 200×
  faster than the reference, which is the point of the backend.
- **Crossing a big tensor in is the exception**: `from_list [64, 64]` is
  2× slower than the reference because 4096 values cross as VM terms; a
  checkpoint loader (§6) and `create`-from-file will want a bulk path
  (`Bytes` + dims) rather than `List[Float]`.
