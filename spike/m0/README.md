# M0 spike — a tensor, a hand-written gradient, and libtorch behind the FFI

**This is a spike, not a package.** Nothing here is `tensor/`: there is no
`tensor` API, no backend interface, no tracing, no autograd, and no claim that
any of it should be kept. It is the time-boxed experiment `roadmap.md` calls
Fase 0 / `tensor/DESIGN.md` calls M0: build the smallest thing that runs
`f(w) = sum(relu(matmul(x, w)))` with an SGD step on a pure-Agentica tensor and
on libtorch, over one fixed input, and write down what the FFI boundary could
not express. The gap list below is the deliverable; the code exists to produce
it.

Read with `tensor/DESIGN.md` open: §2 (types), §3.1 (primitives), §4 (API
shape), §7 (dispatch), §8 (lifetimes/arenas), §10 (tracing and grad) are the
sections this spike tests.

## What is real and what is faked

| Thing | State |
|---|---|
| A dense tensor as `struct` (`shape: List[Int]`, `dtype: Dtype`, `data: List[Float]`, row-major) and the reference ops | **real**, pure Agentica, no native code |
| `f(w) = sum(relu(matmul(x, w)))`, the SGD step, the 20-step loss curve | **real**, deterministic: two runs print the same curve |
| The backward | **hand-written, per function.** Five calls written out by hand in `model.agc`, checked against a central difference in the tests. There is no traced graph and no `grad`: `tensor/DESIGN.md` §10 asks for a graph traced from the primitives, and this spike stands in for it with a straight-line function. Only `dL/dw` exists; `dL/dx` is not computed. |
| The same loss curve on libtorch CPU (`create`, `add`, `matmul`, `relu`, `sum`) | **real**, through a generated `extern module`, matched against the reference in the tests and in the demo |
| The gradient on libtorch | **absent, on purpose**: the demo steps the parameters with the reference backward, because a backend is forward-only (§10). So the two curves step identical parameters and differ only in the forward arithmetic. |
| Shape/dtype/backend dispatch | **faked**: the demo names the two backends explicitly; there is no `Backend` enum and no `case` dispatch of §7, because that is `tensor/`'s job, not the spike's |
| dtypes | **faked**: the reference says `F64` and computes in Agentica `Float` (IEEE 754 double); the native side takes doubles and stores `float32`. Nothing on either side converts or checks, and nothing crosses that says "F32". |
| Model sizes | **tiny**: 4×3 by 3×2. The per-op numbers below are dominated by per-call overhead, which is exactly what the spike wanted to measure, and say nothing about matrix work. |

What that adds up to: the **numbers on this page are real measurements**; the
**architecture around them is a stub** that exists to make the boundary do
something representative.

## Run it

The reference half needs nothing but Agentica:

```sh
cd spike/m0
agentica check
agentica test
agentica demo.agc     # prints the reference curve and says the native side was skipped
agentica bench.agc    # prints the reference numbers, native column says "skipped"
```

The generated header is committed, so nothing has to be fetched to run any of
that. `agentica install` fetches native-gen and is needed only to remake
`native/torchspike_native.h` after a change to the `extern module` block (the
command is in the header of `torch_spike.agc`).

The native half needs a CPU libtorch release unpacked somewhere, and then:

```sh
cd spike/m0
eval "$(sh tools/build-torch.sh)"   # prints TORCH_DIR, CC, CPLUS_INCLUDE_PATH, LIBRARY_PATH
agentica install                    # builds the module (once; ~19 s)
agentica demo.agc                   # both curves and the gap between them
agentica bench.agc                  # both columns
```

`tools/build-torch.sh` finds libtorch through `TORCH_DIR`, `/opt/libtorch`,
`/usr/local/libtorch`, `/usr/lib/libtorch`, `~/opt/libtorch` or `~/libtorch`,
and exits 1 with what it looked for when there is none. A CPU release is one
zip: <https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip>.

**libtorch was not installed on this machine.** The check that says so:

```console
$ echo "TORCH_DIR=$TORCH_DIR"; ls /usr/lib/libtorch* /usr/local/lib/libtorch* /opt/libtorch 2>/dev/null; find / -maxdepth 4 -name 'libtorch*' -not -path '/proc/*' 2>/dev/null | head
TORCH_DIR=
(nothing)
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

To run the native half once, this spike unpacked the official CPU release
above into `~/opt/libtorch` — **outside the repository**; nothing from libtorch
is committed, and the module is `optional = true` in `agentica.toml` so the
spike works without it. Every number below that involves libtorch comes from
that unpacked release, and the "without libtorch" output in the evidence is
what this machine prints with no libtorch and no environment set.

## The reference side

`tensor_ref.agc` — a tensor as data, nothing else:

- `struct Tensor { shape: List[Int], dtype: Dtype, data: List[Float] }` plus
  `enum Dtype { F32, F64 }`. Immutable, row-major, `shape == []` is a rank-0
  scalar.
- Ops: `zeros`, `ones`, `from_list`, `add`, `mul`, `sum`, `matmul`, `relu`, and
  the two the backward needs, `transpose` and `relu_backward`. `ones` and
  `transpose` are in §3.1 already; `relu_backward` is the backward rule §10
  says lives in `tensor/`, not in a backend. `mul` exists because the backward
  multiplies the upstream gradient by the relu mask, which is the only reason
  the spike needed it.
- Immutability is the language's: `w.data.set(...)`, `xs.push(...)` return new
  values, so "a training step returns new parameters" (§1) costs nothing to
  express.

`model.agc` — `f` and its backward, written out:

```text
y  = matmul(x, w)            [4, 2]
r  = relu(y)
L  = sum(r)                  rank-0 tensor
dr = ones(r.shape)           dL/dr of a sum
dy = relu_backward(dr, y)    = dr * 1[y > 0]
dw = matmul(transpose(x), dy)
w' = w + (-lr) * dw
```

`dataset.agc` holds the one input (x, w, 20 steps, lr 0.02, tolerance) so the
demo, the benchmark and the tests cannot disagree about it.
`test/m0_test.agc` holds 16 tests: the ops against hand-computed values, the
error paths, the backward against a central difference, the curve twice, and
the two native comparisons.

## The native side

`torch_spike.agc` declares the module and one wrapper:

```text
extern module "torchspike"
  resource TorchTensor
  compute def create(shape: List[Int], data: List[Float]) -> Result[TorchTensor]
  compute def add(a: TorchTensor, b: TorchTensor) -> Result[TorchTensor]
  compute def matmul(a: TorchTensor, b: TorchTensor) -> Result[TorchTensor]
  compute def relu(a: TorchTensor) -> Result[TorchTensor]
  compute def sum(a: TorchTensor) -> Result[Float]
end
```

Five functions and no more: creation, `add`, `matmul`, `relu`, `sum`.
`native/torchspike_native.h` is generated by native-gen 0.4.0 from that block
and is committed; `native/torchspike.c` is the thin C++ layer — 149 lines, of
which the libtorch calls are `torch::from_blob(...).clone()`, `left + right`,
`torch::matmul`, `relu()`, and `sum().item<double>()`. The resource state is a
`torch::Tensor *` that the module's free function deletes.

Two things in that setup are worth knowing before reading the gaps:

1. **The source is C++ in a `.c` file, and `CC` is `g++`.** The build table
   takes `.c` sources and names no compiler; `g++` compiles `.c` as C++ (and
   warns about the `-std=c11` the build always passes).
2. **The build table names no include or library path.** libtorch's headers
   arrive in `CPLUS_INCLUDE_PATH`, its `.so` names in `libs` (`-l`), its
   directory in `LIBRARY_PATH`, and the `-Wl,-rpath` it needs at run time is
   smuggled in as a word of `CC` (`CC="g++ -Wl,-rpath,$TORCH_DIR/lib"`).
   `tools/build-torch.sh` prints exactly those exports.

## Evidence

All of it from one machine: AMD Ryzen 5 3550H, 12.6 GiB, Debian 13, g++
14.2.0, Agentica 0.35.0, native-gen 0.4.0, libtorch 2.4.1+cpu. The machine is
shared and was under load (~2.5), which is why the numbers below are ranges.

### Check, format, header, tests

```console
$ agentica check
check ok: 7 files

$ agentica fmt --check
(no output, exit 0)

$ agentica deps/native-gen/cli.agc torch_spike.agc --output native/torchspike_native.h --check
(no output, exit 0: the committed header is the one the block makes)

$ eval "$(sh tools/build-torch.sh)" && agentica test
PASS ... / from list builds a row major tensor
PASS ... / zeros fills the shape
PASS ... / from list rejects a count that does not fit the shape
PASS ... / a negative dimension is an error
PASS ... / add is elementwise
PASS ... / add rejects two shapes
PASS ... / mul is elementwise
PASS ... / matmul multiplies two matrices
PASS ... / matmul rejects a mismatched inner dimension
PASS ... / sum reduces to a rank zero tensor
PASS ... / relu clamps at zero
PASS ... / transpose swaps rows and columns
PASS ... / the backward matches a central difference
PASS ... / the curve is the same twice
PASS ... / native loss matches the reference or libtorch is absent
PASS ... / native matmul sums like the reference or libtorch is absent
16 passed, 0 failed, 0 skipped, 0 not executed, 0 file errors
```

The same `agentica test` with libtorch's environment removed still ends
`16 passed, 0 failed`: the two native tests then assert that the module is
unbuilt, so "no libtorch" is a stated result and not a silent pass.

### The loss curve, both backends

```console
$ eval "$(sh tools/build-torch.sh)" && /usr/bin/time -f 'wall %e s, max rss %M KiB' agentica demo.agc
# reference: f(w) = sum(relu(matmul(x, w))), 20 steps, lr 0.02, F64
step  1  loss 3.125000000
step  2  loss 2.883750000
step  3  loss 2.645000000
step  4  loss 2.408750000
step  5  loss 2.167500000
step  6  loss 1.938750000
step  7  loss 1.726250000
step  8  loss 1.517500000
step  9  loss 1.308750000
step 10  loss 1.096250000
step 11  loss 0.883750000
step 12  loss 0.671250000
step 13  loss 0.495000000
step 14  loss 0.456250000
step 15  loss 0.430000000
step 16  loss 0.403750000
step 17  loss 0.377500000
step 18  loss 0.357500000
step 19  loss 0.332500000
step 20  loss 0.306250000

# torchspike: the same function on libtorch CPU, float32
step  1  loss 3.125000000
step  2  loss 2.883749962
step  3  loss 2.644999981
step  4  loss 2.408750057
step  5  loss 2.167500019
step  6  loss 1.938750029
step  7  loss 1.726249933
step  8  loss 1.517499924
step  9  loss 1.308750153
step 10  loss 1.096250057
step 11  loss 0.883750081
step 12  loss 0.671250045
step 13  loss 0.494999975
step 14  loss 0.456250012
step 15  loss 0.430000007
step 16  loss 0.403750002
step 17  loss 0.377499998
step 18  loss 0.357499987
step 19  loss 0.332500011
step 20  loss 0.306250006

max |reference - torch| = 0.000000153, tolerance 0.000001
loss curves: MATCH
wall 0.52 s, max rss 116660 KiB
```

The tolerance is 1e-6 in `dataset.agc`; the measured gap is **1.53e-7**, and it
is float32 rounding, not a difference in the function: the reference is double
and libtorch is `float32`, the parameters are identical step by step (the
gradient is the reference's), and 20 steps of float32 accumulate to ~1e-7 on a
loss of order 1. It is empirical: the spike does not derive it, and a longer
run would need a bigger number.

The same command with libtorch's environment removed:

```console
$ env -u TORCH_DIR -u CC -u CPLUS_INCLUDE_PATH -u LIBRARY_PATH -u LD_LIBRARY_PATH agentica demo.agc
... the 20 reference steps ...
native side: SKIPPED, the torchspike module is not built or loadable
  native: cannot open /home/sonic182/.cache/agentica/pkg/native-build/unbuilt/: cannot open shared object file: No such file or directory
  (libtorch is a system dependency: see README, "The native side")
wall 0.15 s
```

(stderr, on the first run of that configuration: the `g++` error naming
`torch/torch.h`, then `warning: native module torchspike not built: the C
compiler failed with status 1; the package runs without it`.)

### Per-op cost

`agentica bench.agc`, three runs, ns per op; each native number includes the
free of the tensor the op made (`Native.close`):

| op (shape) | reference | torchspike (`compute def`) |
|---|---|---|
| `add` [2, 2] | 453 – 1982 | 8167 – 15198 |
| `matmul` [4, 3]×[3, 2] | 1541 – 1970 | 11782 – 14163 |
| `create` [64, 64] (4096 values, `List[Float]`) | 207 – 260 | 90490 – 99082 |
| `f`, demo shapes | 2586 – 3050 | 56768 – 97232 |
| `f` + backward, demo shapes | 6948 – 7457 | n/a |
| library load, first call | – | 391 – 504 ms |

The reference `create` number is not a copy: a reference tensor shares its
`List[Float]`, so it measures almost nothing. The native `create` is a copy of
4096 doubles across the boundary, which is the interesting part.

`f` is the same function on both sides (`matmul`, `relu`, `sum`).
`f` + backward has no native column on purpose: the backward is Agentica code
and identical on both backends, so a "native step" would be the libtorch
forward *plus* the reference backward — the crossing plus a duplicated step,
which measures neither. What putting libtorch's forward into a step costs is
the difference between the columns of the `f` row (about 54–94 µs here).

Two follow-up experiments, both run on copies of this folder in `/tmp` (they
change the committed surface, so they are not committed):

**A. The same module with `def` instead of `compute def`** — isolates the
compute-pool dispatch from the rest of the call:

```sh
cp -r spike/m0 /tmp/spike-short && cd /tmp/spike-short
sed -i 's/^  compute def /  def /' torch_spike.agc
agentica deps/native-gen/cli.agc torch_spike.agc --output native/torchspike_native.h
eval "$(sh tools/build-torch.sh)" && agentica bench.agc
```

| op (shape) | reference | torchspike (`def`, short) |
|---|---|---|
| `add` [2, 2] | 504 – 677 | 2979 – 6482 |
| `matmul` [4, 3]×[3, 2] | 1584 – 1848 | 4359 – 6103 |
| `create` [64, 64] (`List[Float]`) | 258 – 392 | 70646 – 72653 |
| `f`, demo shapes | 3018 – 6855 | 23180 – 25542 |
| `f` + backward, demo shapes | 8135 – 20830 | n/a |

So the mode of the call is worth roughly **4–9 µs per op** on these shapes:
`compute def` costs about twice a short `def`, and the reference is
~0.5–3 µs. Both native columns are still an order of magnitude above the
reference for a 2×2 add, which is the fixed cost of a call plus a resource
allocate/free.

**B. `create` taking bulk data as `Bytes` (raw little-endian float32) instead
of `List[Float]`** — the same 4096 values, one `Bytes` built once outside the
loop:

```sh
cp -r spike/m0 /tmp/spike-bytes && cd /tmp/spike-bytes
# create(shape: List[Int], data: Bytes), C side reads ng_buf as a float32 array
eval "$(sh tools/build-torch.sh)" && agentica bench.agc
```

| op (shape) | reference | torchspike (`Bytes`, short) |
|---|---|---|
| `create` [64, 64] (4096 values) | 168 – 748 | **3747 – 4228** |

`List[Float]` costs ~63 500–99 700 ns for 4096 values and `Bytes` costs
~3 700–4 200 ns: the cursor decode of a `List` is about **15–24 ns per value**,
and a `Bytes` argument is one borrowed buffer (`ng_buf`) with no per-element
work. Extrapolated — arithmetic, not measured — a 250 M-value F32 tensor
(1 GB, a small model) would cost ~5 s per crossing as `List[Float]`, against a
memcpy as `Bytes`. Note that turning a `List[Float]` into `Bytes` in
Agentica is not free either (`Binary.write_f32_le` per value, outside the
timed loop here), which is why the recommendation is that **data starts as
bytes**, which is exactly what a safetensors file is (§6).

### Build and artifact size

```console
$ rm -rf ~/.cache/agentica/pkg/native-build && eval "$(sh tools/build-torch.sh)" && time agentica install
cc1plus: warning: command-line option ‘-std=c11’ is valid for C/ObjC but not for C++
built native torchspike: /home/sonic182/.cache/agentica/pkg/native-build/4bd2e6.../libtorchspike.so
in the cache native-gen@native-gen-0.4.0
real	0m19,274s
```

| artifact | bytes |
|---|---|
| `libtorchspike.so` (our module, g++ 14, `-O2`) | 58 296 |
| `libtorch_cpu.so` (the dependency that matters) | 495 079 065 |
| `libtorch.so`, `libc10.so` | 7 584 · 1 132 560 |
| libtorch release, unpacked (`du -sb`) | 798 963 200 (786 MiB) |
| libtorch release, the zip | 189 456 424 |
| libtorch headers (`du -sh include`) | 51 MiB |

A cold build (headers, template instantiation, link) is ~19 s for a 58 KB
module that calls five libtorch functions. The module's own size is
irrelevant next to the 472 MiB of `libtorch_cpu.so`, which is what a user
actually downloads.

## What the FFI boundary could not express

Ordered by how much they hurt the design.

### 1. A tensor value cannot cross; only its handle can

`tensor/DESIGN.md` §2 wants `Tensor = { shape, dtype, backend, handle }`. The
language does allow a resource in a struct field, but such a struct cannot be a
parameter of a native module:

```console
$ cat probe.agc
struct Boxed
  handle: Stream
  name: String
end

extern module "probe"
  resource Stream
  def use(b: Boxed) -> Result[Float]
end

$ agentica check probe.agc
probe.agc:8:14 error[CHECK-NATIVE-TYPE]
the resource Stream does not cross to a native module in Boxed

8 |   def use(b: Boxed) -> Result[Float]
  |              ^^^^^
  = help: a resource of the module block crosses as a whole parameter type, and in a result
    outside the fields of a struct or the payloads of an enum
```

So every op signature must be `add(a: Handle, b: Handle)`: the handle and the
metadata travel on different paths, and the split is at every call. What the
wrapper must then do is either (a) keep `shape`/`dtype` in Agentica and pass
only the handle, or (b) ask the backend. The design has to pick one; see
"Decisions" below.

A plain data struct *does* cross, so the metadata side is fine on its own:

```agentica
struct Values
  shape: List[Int]
  dtype: Dtype
  data: List[Float]
end

extern module "probe"
  resource Stream
  def create(v: Values) -> Result[Stream]
end
```

`agentica check` accepts that. The spike did not use it (five functions, as the
task asked) and passes `shape` and `data` as two arguments instead, which is
where the dtype got lost.

### 2. Shape and dtype do not travel with the tensor

`TorchTensor` is opaque: the Agentica side cannot ask its shape, its dtype or
its device, and the signature does not say them. Consequences measured here:

- §9 asks the Agentica wrapper to reject a shape mismatch **before** dispatch.
  With only a handle, the wrapper cannot: it has nothing to compare. libtorch
  will not say no either — it *broadcasts*, so `add` of `[2, 2]` and `[2]`
  returns a tensor instead of an error. The native layer in
  `native/torchspike.c` checks `sizes()` by hand to keep the reference's error
  behaviour; in a real `torch/` that check belongs to the wrapper, and the
  wrapper cannot make it.
- dtype: `create` takes `List[Float]` (doubles) and stores `float32`. Nothing
  in the signature or the value says which dtype the tensor holds, so §2's
  "numeric compute is F32" rule is invisible at the boundary, and the F64/F32
  gap in the loss curves above is a property of the spike's plumbing rather
  than of the tensor that came back.
- `sum` had to return a bare `Float`. `sum` over all axes is rank-0 per §3.1
  (`axes: []`, `keep_dims: false`), but a rank-0 **tensor** is a resource, and
  reading one element is a call of its own (§4 has `Tensor.to_list` for
  exactly that). With five functions to spend, the readback got fused into the
  reduction: the spike's `sum` is `sum` **plus** `item`. That is a real API
  decision, not a shortcut: a reduction that returns a `Tensor` needs the
  readback to be a primitive.

### 3. Two failure channels, and no error type on the boundary

A module function can fail two ways, and the design has to say which is which:

- **returned**: `Result[T]` with an error made by `an_error(env, text, len)` —
  a message and, since native API 1.4, a kind and details. This is what the
  spike's ops return.
- **halting**: `an_raise`/`an_badarg`, and any bad term or bad decode in the
  generated cursor code, fail the whole call with `RUNTIME-NATIVE-ERROR`
  (`RUNTIME-NATIVE-CLOSED` for a closed resource). §4 promises "every fallible
  op returns `Result`; shape/dtype mismatches are `error`, never a halt", and
  §8 accepts `RUNTIME-NATIVE-CLOSED` as a halt — so the split exists in the
  design already, but nothing in `tensor/` says which failures belong where,
  and the spike had to invent it per op.

The message text is also backend-specific, because the native side has less
information: the reference says `shape [4, 3] holds 6 values, got 5`, the
native side can only say `a shape of 6 values takes 5` (it has no `[4, 3]`
term to print — see gap 2). §11 wants conformance on "error paths" across
backends; matching **text** would mean rebuilding shape strings in C for every
op. Matching a **kind** would not: `an_error_kind` is in the API 1.4 that this
Agentica carries, and native-gen generates no helper for it, so the wrapper
layer in `tensor/` would have to call it directly.

### 4. Bulk tensor data: `List[Float]` is the wrong shape for it

Measured above: ~22 ns per value through a `List[Float]` cursor
(~95 µs for 4096 values) against ~4 µs for the whole `Bytes` buffer, which is
the call overhead plus one memcpy of 16 KB. Both
directions matter: writing a tensor out has the symmetric problem (a `List`
builder), and §4's `to_list` is the API that would pay it. This is also the
safetensors path of §6: the file *is* a contiguous little-endian buffer, so
`Bytes` in both directions makes loading one `mmap` plus one `memcpy`, while
`List[Float]` makes it a per-value decode on both sides of the call. The spike
used `List[Float]` because that is the API in §4 (`from_list(data: List[Float],
...)`), and that API is the thing this measurement argues against as the *only*
way in.

### 5. Resource lifetime and arenas: what worked, and what only a spike this small can say

What worked, exactly as §8 proposes:

- A module resource is a type, owned by the task that made it, freed once by
  the module's free function: at `Native.close`, at the end of the task, or at
  the end of the run.
- `Native.arena do ... end` runs the block, returns its value, and closes
  everything the block opened that the value does not hold. The demo does one
  arena per `f` call — the granularity §8 wants for an inference call — and the
  benchmark does one per measurement; both run with the four intermediate
  tensors of a step dying at a known point.
- `try` inside the block releases what the error does not hold, which is how
  the wrappers are written.
- `Native.close` and the arena agree: closing a resource twice is a no-op, and
  a closed one passed to a call is `RUNTIME-NATIVE-CLOSED`, not UB.

The gaps in lifetime handling that the spike hit:

- **The native state is zeroed bytes, so a C++ backend must hold a pointer and
  construct in its own `create`.** `torch::Tensor` has a constructor, so
  `native/torchspike.c` stores `torch::Tensor *` and `new`s it; the free
  function `delete`s it. Every C++ (and Rust) backend hits this, and it is the
  first thing a C++ layer has to do. Worth one line in the design so four
  backends do not invent it four times.
- **The state holds C data only, never terms**, so a backend cannot cache
  Agentica values in it — a shape it wants to report must be re-encoded in C.
  Combined with gap 2 (no shape on the boundary) this is why the native error
  messages cannot print a shape.
- **No views, so no `keep` rule was exercised.** §8's subtle part (a kept view
  keeping its buffer) is untested here: nothing in the spike returns a view.
- **`Native.arena` + `try` has a writing rule that §8 does not mention** (see
  decisions 5).
- **Nothing about the GC-pressure argument.** The benchmark creates and frees
  ~14 000 native tensors (0.99 s, max RSS 144 MB, which is libtorch's mapped
  image); a leak of 14 000 small tensors would not show. §8's real question —
  thousands of tens of MB of intermediates per step — is not answered by a
  4×2 tensor.

### 6. Build and link friction (all measured, all worked around)

- **C++ is not a first-class source kind.** `build.sources` are `.c` files, so
  the C++ layer lives in `native/torchspike.c` and `CC` must be `g++`
  (`CC="g++ -Wl,-rpath,$TORCH_DIR/lib"`). The build passes `-std=c11`
  unconditionally, so every build prints
  `cc1plus: warning: command-line option '-std=c11' is valid for C/ObjC but not
  for C++`. A libtorch layer without that warning needs a `CC` wrapper script;
  with `-Werror` it would need one anyway.
- **No include/library/flag keys in the manifest.** The build table has
  `sources`, `defines` and `libs` (`-l` names, no paths), and "the manifest
  names no compiler and no other flag". libtorch needs `-I` (two of them),
  `-L` and `-Wl,-rpath`; all four therefore live in the environment
  (`CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, and `CC` for the rpath), which
  `tools/build-torch.sh` prints. A reader of `agentica.toml` cannot see how
  the module is built. `whisper/` does the same thing (`tools/build-engine.sh`
  prints the same two variables), so this is a repository-wide pattern, not a
  one-off: it is the manifest that is missing the key.
- **The rpath is not optional and cannot be expressed.** Without it, the built
  module loads but cannot find `libtorch_cpu.so` unless the user also exports
  `LD_LIBRARY_PATH`. Smuggling `-Wl,-rpath` into `CC` works and participates in
  the build cache key (the key covers the words of `CC`), so it is
  reproducible — but it is a trick, and the manifest does not say it.
- **`optional = true` is the only honest way to be runnable without the
  dependency**, and its failure mode is a compiler error plus
  `warning: native module torchspike not built: ...; the package runs without
  it`. `Native.load("torchspike")` then fails with
  `native: cannot open .../native-build/unbuilt/: cannot open shared object
  file`, which is what the demo prints. A program cannot distinguish "libtorch
  is not installed" from "our C++ does not compile" through that message; a
  real package wants `Native.load` to answer with the reason the build failed.
- **A prebuilt `.so` does not remove the dependency.** §7 plans "prebuilt per
  platform (CPU first; sizes documented)". Our prebuilt is 58 KB; the 472 MiB
  is libtorch itself, which the user still installs. The size table for the
  design should be the *engine's*, not the wrapper's.

## Decisions the design needs adjusted

Recommendations, in the order I would settle them; each is backed by something
above.

1. **§2/§9: say where `shape`/`dtype` live, because they cannot live on the
   handle.** Recommend: `Tensor` is an Agentica struct `{ shape, dtype,
   backend, handle }` where `handle` is the resource, and the *wrapper* holds
   the authoritative shape/dtype; ops take `handle` only, and the wrapper
   validates before dispatch (§9) using its own copy. Cost: two sources of
   truth, so the backend must still check (libtorch broadcasts — measured), and
   a `to_dtype`/`to_device` that changes the tensor must keep the shadow in
   step. Alternative to consider: `handle` is itself a resource whose C state
   holds shape/dtype, and a native `info(handle) -> Values` call answers (one
   call per validation, or a per-op check inside C). The spike's evidence
   favours the shadow: validation then costs zero FFI, which is what §9 and §4
   ("introspection is free") ask for.
2. **§3.1: `sum`/`mean`/`max`/`argmax` need a scalar readback in the
   primitive set.** Either `to_list`/`item` is a primitive (not the convenience
   it looks like in §4), or a reduction that consumes every axis returns a
   scalar value (`Float`/`Int`). The spike spent one of its five functions on
   the fusion and should not have had to.
3. **§7/§3.1: bulk data crosses as `Bytes`, not `List[Float]`.** Keep
   `from_list`/`to_list` as the convenient path for tests and small tensors,
   and add `from_bytes`/`to_bytes` (raw, dtype-checked, C-order, little-endian)
   as the path a checkpoint and a training step use. 19× measured difference at
   4096 values; the gap grows linearly.
4. **§11: conformance must compare error *kinds*, not messages.** Define a
   small kind taxonomy (`tensor.shape`, `tensor.dtype`, `tensor.closed`, ...)
   and make the native layer emit it with `an_error_kind`; then the shared
   suite can assert equality across backends. Matching text would force every
   backend to rebuild the reference's strings, including shapes it does not
   have.
5. **§8: the per-call arena has a shape constraint in the language today.**
   `Native.arena` works with module resources and frees them at the end of the
   block (the demo and the benchmark rely on it), but `try` inside the block
   only checks when the arena call *is* the value of a `Result`-returning
   expression:

   ```console
   $ cat arena.agc
   def work() -> Result[Int]
     measured = Native.arena do
       seed = try parse("3")
       ok(seed + 1)
     end

     measured
   end

   $ agentica check arena.agc
   arena.agc:3:12 error[CHECK-TRY-CONTEXT]
   try requires a function returning Result

   3 |     seed = try parse("3")
     |            ^^^^^^^^^^^^^^
     = help: return Result[T] from this function, or handle the Result explicitly
   ```
   Writing the same body as `Native.arena do ... end` with the arena call as
   the returned expression checks (`ok`, no diagnostic); assigning it to a
   variable first does not, unless every function the block calls is declared
   above it. Either the design states that rule for `tensor/` wrappers, or the
   checker is asked to infer the block's result from the `try`. This is the one
   place where the spike's code is shaped by the checker rather than by the
   design.
6. **§10: the spike confirms the "backward rules live in Agentica" shape, and
   shows the cost.** The rules are ordinary functions over tensors
   (`relu_backward` is `mul(dy, mask)`; `matmul`'s rule is
   `matmul(transpose(x), dy)`), so they are backend-agnostic and only need
   `transpose` and elementwise `mul` — both already primitives. What the spike
   does not show is any of the *machinery*: no trace, no graph, no `vmap`, no
   `jit`, and `dL/dx` was never needed. A real `grad` also has to keep the
   forward values (`y` here) alive for the backward, which is a lifetime
   question §8 does not answer for tracked intermediates.
7. **§7: the design has no rule for `def` vs `compute def` per op, and the
   choice is measurable.** On a 2×2 `add`, the same module with `def` costs
   3.0–6.5 µs and with `compute def` 8.2–15.2 µs (measured above); the dirty
   call buys the ability to block on a real kernel and costs ~4–9 µs of
   dispatch.
   A backend needs a rule ("dirty above N elements", or "dirty for ops that
   may call into a thread pool"), which the contract should state once.
8. **§3.1's "adding an op later would be breaking" is right, and `mul` is the
   evidence.** The one op the spike added beyond the demo's list earned its
   place only through a backward rule; `ones` and `transpose` were already in
   §3.1. Nothing in this spike needed an op §3.1 lacks.

## What this spike does not prove

- **No tracing, no autograd, no `grad`/`value_and_grad`.** The backward is one
  hand-written function for one function. Nothing here says the traced version
  is expressible; it says the *rules* are.
- **No `tensor/` API and no dispatch.** No `Backend` enum, no `case`, no
  conformance suite, no `nn/`, no `train/`.
- **The parity number is about arithmetic, not about backends.** The two
  curves share the parameters, so 1.53e-7 measures F64-vs-F32 forward
  rounding, not "the torch backend agrees with the reference".
- **Only CPU, only the five ops, only tiny shapes.** `matmul` on 4×3 by 3×2 is
  a fixed-cost measurement. Nothing here about real matrix sizes, threading,
  BLAS, or device placement.
- **No memory-pressure evidence.** Arenas free their resources (the benchmark
  creates and frees ~14 000 native tensors), but nothing in the spike allocates
  megabytes per tensor, so §8's actual problem — GC vs. thousands of large intermediates — is
  untested. RSS for the demo run is 116 MB, essentially libtorch's mapped
  image.
- **No concurrency.** All native calls here are `compute def` (dirty) from one
  task; short-call thread safety is untested.
- **No checkpoint path, no `mmap`, no prebuilt distribution, no CI.** The spike
  is not a package: `spike/m0/agentica.toml` sits one level below what the CI
  matrix scans (`*/agentica.toml`), so no workflow changed and none is needed
  for a spike. Everything on this page was run by hand with the commands
  quoted.

## Next step

The M1 work this spike argues for, in order:

1. `tensor/`: the `Tensor` struct and `Backend` enum of §2 with the wrapper
   owning shape/dtype (decision 1), dispatch by `case` (§7), validation in the
   wrapper (§9).
2. `tensor/`: primitives for scalar readback and bulk `Bytes` in/out
   (decisions 2, 3), with `from_list`/`to_list` on top as the slow path.
3. The conformance suite with error kinds (decision 4), reference first.
4. `torch/`: the real ~45-op contract with resources per op, `optional` build
   for the CPU release, `tools/build-torch.sh` as the documented env contract,
   and the `Bytes` data path.
5. Language asks, smallest first: a `[native.NAME] build` key for include and
   library paths (and C++ sources); `try` inference inside an assigned
   `Native.arena` block; a `Native.load` that reports why a build failed.
