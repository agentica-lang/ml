## The ops of the StableHLO emitter (`src/stablehlo.agc`) that a transformer step needs beyond the MLP's, each compiled
## by XLA (`Backend.Xla`) against the reference backend on the same inputs.
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Support from "xla_support"
import Tensor from "../tensor"
import Trace from "../trace"

private

def test_elementwise_ops_and_dtype_conversions_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) { ok([try wide(b), try positive(b)]) },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.sigmoid(xs[0]),
        try Tensor.pow(xs[1], xs[0]),
        try Tensor.clamp(xs[0], -1.5, 2.0),
        try Tensor.to_dtype(xs[0], Tensor.Dtype.I32),
        try Tensor.to_dtype(try Tensor.to_dtype(xs[0], Tensor.Dtype.I32), Tensor.Dtype.F32),
        try Tensor.to_dtype(xs[0], Tensor.Dtype.Bool),
        try Tensor.to_dtype(try Tensor.to_dtype(xs[0], Tensor.Dtype.Bool), Tensor.Dtype.F32),
      ])
    },
  )
end

def test_shape_ops_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([try cube(2, b), try cube(3, b), try Support.floats(Support.uniform(6, 5, 1.0), [1, 3, 1, 2], b)])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.permute(xs[0], [2, 0, 1]),
        try Tensor.concat([xs[0], xs[1], xs[0]], 0),
        try Tensor.concat([xs[0], xs[0]], 1),
        try Tensor.stack([xs[0], xs[0]], 2),
        try Tensor.squeeze(xs[2]),
        try Tensor.squeeze(xs[2], some(0)),
        try Tensor.unsqueeze(xs[0], 3),
        try Tensor.unsqueeze(xs[0], 0),
      ])
    },
  )
end

## The batch is the leading axes of both operands.
def test_a_batched_product_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.floats(Support.uniform(2 * 3 * 4 * 5, 1, 1.0), [2, 3, 4, 5], b),
        try Support.floats(Support.uniform(2 * 3 * 5 * 2, 2, 1.0), [2, 3, 5, 2], b),
        try Support.floats(Support.uniform(3 * 4, 3, 1.0), [3, 4], b),
        try Support.floats(Support.uniform(4 * 2, 4, 1.0), [4, 2], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.batch_matmul(xs[0], xs[1]),
        try Tensor.batch_matmul(try Tensor.unsqueeze(xs[2], 0), try Tensor.unsqueeze(xs[3], 0)),
      ])
    },
  )
end

## The index inside the reduced block, numbered by the ascending axes, and a tie keeps the lowest.
def test_argmax_matches_the_reference_on_one_axis_many_axes_and_ties() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) { ok([try cube(3, b), try ties(b)]) },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.argmax(xs[0]),
        try Tensor.argmax(xs[0], some([1])),
        try Tensor.argmax(xs[0], some([2]), true),
        try Tensor.argmax(xs[0], some([0, 2])),
        try Tensor.argmax(xs[0], some([2, 0]), true),
        try Tensor.argmax(xs[1], some([1])),
        try Tensor.argmax(xs[1]),
      ])
    },
  )
end

def test_an_op_the_emitter_still_lacks_is_an_error_naming_it() -> Result[Void]
  return ok() if !try Support.present()
  xs = [try wide(Support.XLA)]
  r = Trace.compile(xs.map { |t| Trace.spec_of(t) }, ->(ys: List[Tensor.Tensor]) { ok([try Tensor.gelu_erf(ys[0])]) })
  try Test.assert(!r.ok?, "gelu_erf compiled")
  Test.assert(r.message.include?("gelu_erf"), r.message)
end

## `make` the inputs on a backend, `f` over them compiled by XLA, and `f` over the same inputs on the reference: every
## output within `Support.TOL`, with the same shape.
def agrees(
  make: Fn[Tensor.Backend, Result[List[Tensor.Tensor]]],
  f: Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[Void]
  inputs = try make.call(Support.XLA)
  compiled = try Trace.compile(inputs.map { |t| Trace.spec_of(t) }, f)
  got = try Trace.run(compiled, inputs)
  want = try f.call(try make.call(Support.REF))
  Support.close(got, want)
end

## Values of both signs, past the clamp and past the rounding of a conversion, with zeros.
def wide(b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats([-3.7, -1.5, -0.4, 0.0, 0.3, 1.0, 2.5, 9.9, -0.0, 4.2, -2.2, 0.7], [3, 4], b)
end

def positive(b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats([0.5, 1.5, 2.0, 3.0, 0.25, 7.0, 1.1, 4.0, 9.0, 0.1, 6.5, 2.2], [3, 4], b)
end

## `[n, 2, 3]` of distinct values.
def cube(n: Int, b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats(Support.uniform(n * 6, n, 1.0), [n, 2, 3], b)
end

## `[2, 4]` where each row's largest value appears twice.
def ties(b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats([1.0, 5.0, 2.0, 5.0, 3.0, 3.0, 0.5, 3.0], [2, 4], b)
end
