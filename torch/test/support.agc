## Shared helpers of the `torch/` conformance tests (not a test file itself).
##
## `tensor/`'s reference backend is the oracle (`tensor/DESIGN.md` §11): every
## check builds the same inputs on both backends and compares shapes, dtypes
## and values. The reference computes in doubles, libtorch computes in `F32`
## (`tensor/DESIGN.md` §2), so the value tolerance is the one `F32` justifies
## and no tighter (README, "Tolerances").

import Ref from "pkg/tensor/tensor"
import Torch from "../torch"

## The one device libtorch's CPU release serves.
CPU = Torch.Device.Cpu

## The relative tolerance: `F32` carries 24 bits of mantissa, so a value that
## went through a float32 rounding and back differs by about 2^-24 of itself.
REL = 0.0000001

## The absolute floor, for values near zero where a relative gap says nothing.
ABS = 0.0000001

## True when the native module is loaded: the conformance comparison runs only
## then.
def available() -> Bool = Torch.available()

## What a test proves where libtorch is not installed: the module is absent,
## so every op answers a `Result` error instead of halting. Spelled out, never
## a silent pass (README, "CI").
def absent() -> Result[Void]
  try Test.assert(!Torch.available(), "the native torch module is loaded after all")
  Test.error?(Torch.zeros(CPU, [2], Torch.Dtype.F32))
end

## The two floats agree: exactly below `ABS`, within `REL` relatively above.
def close?(a: Float, b: Float) -> Bool
  gap = (a - b).abs
  return true if gap <= ABS
  gap <= REL * (a.abs + b.abs)
end

## `data` as an `F32` reference tensor of `shape`: the oracle side.
def ref_f32(data: List[Float], shape: List[Int]) -> Result[Ref.Tensor]
  floats = data.map do |x: Float| Ref.Values.Float(x) end
  Ref.from_list(floats, shape, Ref.Dtype.F32, Ref.Backend.Reference)
end

## `data` as an `I64` reference tensor of `shape`.
def ref_i64(data: List[Int], shape: List[Int]) -> Result[Ref.Tensor]
  ints = data.map do |x: Int| Ref.Values.Int(x) end
  Ref.from_list(ints, shape, Ref.Dtype.I64, Ref.Backend.Reference)
end

## `data` as an `F32` torch tensor of `shape`.
def torch_f32(data: List[Float], shape: List[Int]) -> Result[Torch.Tensor]
  floats = data.map do |x: Float| Torch.Values.Float(x) end
  Torch.from_list(CPU, floats, shape, Torch.Dtype.F32)
end

## `data` as an `I64` torch tensor of `shape`.
def torch_i64(data: List[Int], shape: List[Int]) -> Result[Torch.Tensor]
  ints = data.map do |x: Int| Torch.Values.Int(x) end
  Torch.from_list(CPU, ints, shape, Torch.Dtype.I64)
end

## The row-major floats of a torch tensor (an `F32` or `F64` one).
def floats_of(t: Torch.Tensor) -> Result[List[Float]]
  raw = try Torch.to_list(t)
  out: List[Float] = []
  for v in raw
    case v
      Torch.Values.Float(x) -> out = out.push(x)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

## The row-major integers of a torch tensor.
def ints_of(t: Torch.Tensor) -> Result[List[Int]]
  raw = try Torch.to_list(t)
  out: List[Int] = []
  for v in raw
    case v
      Torch.Values.Int(i) -> out = out.push(i)
      _ -> return error("an integer tensor holds Int values")
    end
  end

  ok(out)
end

## One value of the torch tensor and one of the reference tensor agree: the
## same tag, and floats within the tolerance, integers and bools exactly.
def values_agree(ours: Ref.Values, theirs: Ref.Values) -> Result[Bool]
  case ours
    Ref.Values.Float(x) ->
      case theirs
        Ref.Values.Float(y) -> ok(close?(x, y))
        _ -> error("the torch tensor holds a Float where the reference holds #{theirs}")
      end

    Ref.Values.Int(i) ->
      case theirs
        Ref.Values.Int(j) -> ok(i == j)
        _ -> error("the torch tensor holds an Int where the reference holds #{theirs}")
      end

    Ref.Values.Bool(flag) ->
      case theirs
        Ref.Values.Bool(other) -> ok(flag == other)
        _ -> error("the torch tensor holds a Bool where the reference holds #{theirs}")
      end
  end
end

## `got` and `want` agree on shape, dtype and every value: the conformance
## check `tensor/DESIGN.md` §11 asks for.
def agrees(got: Torch.Tensor, want: Ref.Tensor, what: String) -> Result[Void]
  try Test.eq(Torch.shape(got), Ref.shape(want))
  try Test.eq(Torch.dtype(got), Ref.dtype(want))
  ours = try Torch.to_list(got)
  theirs = try Ref.to_list(want)
  try Test.eq(ours.size, theirs.size)
  for i in 0...theirs.size
    same = try values_agree(ours[i], theirs[i])
    return error("#{what}: value #{i} is #{ours[i]}, the reference says #{theirs[i]}") if !same
  end

  ok()
end

## `t` holds `expected`, element by element, within the tolerance.
def expect_floats(t: Torch.Tensor, expected: List[Float], what: String) -> Result[Void]
  got = try floats_of(t)
  try Test.eq(got.size, expected.size)
  for i in 0...got.size
    return error("#{what}: value #{i} is #{got[i]}, expected #{expected[i]}") if !close?(got[i], expected[i])
  end

  ok()
end
