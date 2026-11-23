## Shared helpers of the libtorch backend's conformance tests (not a test file
## itself).
##
## The public API is what is under test: every check builds the same inputs
## twice, once on `Reference` (the oracle, `tensor/DESIGN.md` §11) and once on
## `Torch`, calls the same `Tensor.*` function, and compares shape, dtype and
## values. One call therefore exercises the dispatch in `tensor.agc`, the
## backend in `src/torch.agc` and the native layer in `native/torch.c`.
##
## The reference computes in doubles and libtorch computes in `F32`
## (`tensor/DESIGN.md` §2), so values are compared within the tolerance `F32`
## justifies and shape/dtype exactly (`../README.md`, "Tolerances and why").

import Support from "support"
import Tensor from "../tensor"

## The backend the native libtorch module runs on: the one device its CPU
## release serves.
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
## The backend the oracle runs on.
REF = Tensor.Backend.Reference

## The relative tolerance: `F32` carries 24 bits of mantissa, so a value that
## went through a float32 rounding and back differs by about 2^-24 of itself.
REL = 0.0000001

## The absolute floor, for values near zero where a relative gap says nothing.
ABS = 0.0000001

## True when the native module is loaded: the conformance comparison runs only
## then.
def available() -> Bool = Tensor.available(TORCH)

## What a test proves where libtorch is not installed: the backend is
## unavailable through the public API, and the dispatch arm answers
## `error("backend not available")` naming the reason. Spelled out, never a
## silent pass (`../README.md`, "CI").
def absent() -> Result[Void]
  try Test.assert(!available(), "the native torch module is loaded after all")
  case Tensor.zeros([2], Tensor.Dtype.F32, TORCH)
    ok(_) -> error("the Torch arm computed a tensor without the native module")
    error(e) -> Test.assert(e.message.include?("backend not available"), "the Torch arm said \"#{e.message}\"")
  end
end

## True for ordinary values: false for `NaN` and both infinities.
def finite?(x: Float) -> Bool = x - x == 0.0

## Two floats agree: exactly below `ABS`, within `REL` relatively above, and
## exactly (same sign, and only then) when either side is non-finite. Without
## that first line the tolerance is vacuous at infinity: `gap` is `inf` and
## `REL * (|a| + |b|)` is `inf`, so `inf <= inf` would let a saturating op
## agree with any finite value the reference produced.
def close?(a: Float, b: Float) -> Bool
  return a == b if !finite?(a) or !finite?(b)
  gap = (a - b).abs
  return true if gap <= ABS
  gap <= REL * (a.abs + b.abs)
end

## `data` as an `F32` reference tensor of `shape`: the oracle side.
def on_ref(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor] = Support.f32(data, shape)

## `data` as an `I64` reference tensor of `shape`.
def on_ref_i64(data: List[Int], shape: List[Int]) -> Result[Tensor.Tensor] = Support.i64(data, shape)

## `data` as an `F32` torch tensor of `shape`.
def on_torch(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.floats(data, shape, TORCH)
end

## `data` as an `I64` torch tensor of `shape`.
def on_torch_i64(data: List[Int], shape: List[Int]) -> Result[Tensor.Tensor]
  ints = data.map do |x: Int| Tensor.Values.Int(x) end
  Tensor.from_list(ints, shape, Tensor.Dtype.I64, TORCH)
end

## The row-major floats of `t`, whatever backend holds them.
def floats_of(t: Tensor.Tensor) -> Result[List[Float]] = Support.floats_of(t)

## The row-major integers of `t`, whatever backend holds them.
def ints_of(t: Tensor.Tensor) -> Result[List[Int]] = Support.ints_of(t)

## True once `t`'s native handle is closed; a `Plain` tensor has no handle to
## close and answers `false`. The lifetime checks use it to see, from
## Agentica, that a value still reaches a live resource.
def native_closed(t: Tensor.Tensor) -> Bool
  case t.handle
    Tensor.Handle.Native(h) -> Native.closed?(h)
    Tensor.Handle.Plain(_) -> false
    Tensor.Handle.Traced(_) -> false
    Tensor.Handle.Ggml(h) -> Native.closed?(h)
  end
end

## One value of each side agree: the same tag, floats within the tolerance,
## integers and bools exactly.
def values_agree(ours: Tensor.Values, theirs: Tensor.Values) -> Result[Bool]
  case ours
    Tensor.Values.Float(x) ->
      case theirs
        Tensor.Values.Float(y) -> ok(close?(x, y))
        _ -> error("the torch tensor holds a Float where the reference holds #{theirs}")
      end

    Tensor.Values.Int(i) ->
      case theirs
        Tensor.Values.Int(j) -> ok(i == j)
        _ -> error("the torch tensor holds an Int where the reference holds #{theirs}")
      end

    Tensor.Values.Bool(flag) ->
      case theirs
        Tensor.Values.Bool(other) -> ok(flag == other)
        _ -> error("the torch tensor holds a Bool where the reference holds #{theirs}")
      end
  end
end

## `got` and `want` agree on backend-independent shape, dtype and values: the
## conformance check `tensor/DESIGN.md` §11 asks for.
def agrees(got: Tensor.Tensor, want: Tensor.Tensor, what: String) -> Result[Void]
  try Test.eq(Tensor.shape(got), Tensor.shape(want))
  try Test.eq(Tensor.dtype(got), Tensor.dtype(want))
  ours = try Tensor.to_list(got)
  theirs = try Tensor.to_list(want)
  try Test.eq(ours.size, theirs.size)
  for i in 0...theirs.size
    same = try values_agree(ours[i], theirs[i])
    return error("#{what}: value #{i} is #{ours[i]}, the reference says #{theirs[i]}") if !same
  end

  ok()
end

## `t` holds `expected`, element by element, within the tolerance.
def expect_floats(t: Tensor.Tensor, expected: List[Float], what: String) -> Result[Void]
  got = try floats_of(t)
  try Test.eq(got.size, expected.size)
  for i in 0...got.size
    return error("#{what}: value #{i} is #{got[i]}, expected #{expected[i]}") if !close?(got[i], expected[i])
  end

  ok()
end
