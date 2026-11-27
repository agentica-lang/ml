## The StableHLO emitter (`src/stablehlo.agc`): the text of a graph, the order of its arguments, and the ops it
## refuses. The numbers are checked by the goldens of `train/` (IREE's outputs).

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

private

def test_a_graph_is_one_function_of_its_inputs_then_its_constants() -> Result[Void]
  bias = try Support.f32([0.5, 1.5], [2])
  m = try Trace.stablehlo([spec([3, 4], Tensor.Dtype.F32)], ->(xs: List[Tensor.Tensor]) { shifted(xs[0], bias) })
  try Test.eq(m.constants.size, 1)
  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<3x4xf32>, %arg1: tensor<2xf32>) -> (tensor<6x2xf32>) {
          %0 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<6x2xf32>
          %1 = stablehlo.reshape %arg0 : (tensor<3x4xf32>) -> tensor<6x2xf32>
          %2 = stablehlo.add %1, %0 : tensor<6x2xf32>
          func.return %2 : tensor<6x2xf32>
        }
      }
    MLIR
  )
end

## A maximum starts from `-inf` and keeps its axis as a 1; a comparison of integers is `SIGNED`.
def test_the_reductions_and_the_comparisons_take_their_neutral_elements_and_types() -> Result[Void]
  specs = [spec([2, 3], Tensor.Dtype.F32), spec([3], Tensor.Dtype.I32)]
  m = try Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { reduced(xs[0], xs[1]) })
  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3xi32>) -> (tensor<2x1xf32>, tensor<3xi1>) {
          %0 = stablehlo.iota dim = 0 : tensor<3xi32>
          %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
          %2 = stablehlo.reduce(%arg0 init: %1) applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
          %3 = stablehlo.reshape %2 : (tensor<2xf32>) -> tensor<2x1xf32>
          %4 = stablehlo.compare EQ, %arg1, %0, SIGNED : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
          func.return %3, %4 : tensor<2x1xf32>, tensor<3xi1>
        }
      }
    MLIR
  )
end

## An `arange` that starts at 1 in steps of 2, a `mean` that keeps its axis, and a fill whose float has an exponent.
def test_an_arange_a_kept_mean_and_a_small_fill() -> Result[Void]
  m = try Trace.stablehlo([spec([2, 3], Tensor.Dtype.F32)], ->(xs: List[Tensor.Tensor]) { counted(xs[0]) })
  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x3xf32>) -> (tensor<3xi32>, tensor<2x1xf32>) {
          %0 = stablehlo.constant dense<1.0e-8> : tensor<2x1xf32>
          %1 = stablehlo.iota dim = 0 : tensor<3xi32>
          %2 = stablehlo.constant dense<2> : tensor<3xi32>
          %3 = stablehlo.multiply %1, %2 : tensor<3xi32>
          %4 = stablehlo.constant dense<1> : tensor<3xi32>
          %5 = stablehlo.add %3, %4 : tensor<3xi32>
          %6 = stablehlo.constant dense<0.0> : tensor<f32>
          %7 = stablehlo.reduce(%arg0 init: %6) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
          %8 = stablehlo.constant dense<3.0> : tensor<2xf32>
          %9 = stablehlo.divide %7, %8 : tensor<2xf32>
          %10 = stablehlo.reshape %9 : (tensor<2xf32>) -> tensor<2x1xf32>
          %11 = stablehlo.add %10, %0 : tensor<2x1xf32>
          func.return %5, %11 : tensor<3xi32>, tensor<2x1xf32>
        }
      }
    MLIR
  )
end

def test_an_op_without_an_emitter_is_an_error_naming_it() -> Result[Void]
  r = Trace.stablehlo(
    [spec([2, 2], Tensor.Dtype.F32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.avg_pool1d(xs[0], 2)]) },
  )

  case r
    ok(_) -> error("avg_pool1d was emitted")
    error(e) -> Test.eq(e.message, "StableHLO has no emitter for avg_pool1d yet")
  end
end

## A slice is `stablehlo.slice` with the start and end of each dimension.
def test_a_slice_names_its_extents() -> Result[Void]
  m = try Trace.stablehlo(
    [spec([2, 7], Tensor.Dtype.F32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.slice(xs[0], [1, 2], [2, 5])]) },
  )

  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x7xf32>) -> (tensor<1x3xf32>) {
          %0 = stablehlo.slice %arg0 [1:2, 2:5] : (tensor<2x7xf32>) -> tensor<1x3xf32>
          func.return %0 : tensor<1x3xf32>
        }
      }
    MLIR
  )
end

## An `F16` weight stays half as an argument and is widened inside the program, where an `F32` result reads it: a product
## and the rows of a table. A product of two other mixed dtypes is an error naming it.
def test_an_f16_weight_is_widened_where_an_f32_result_reads_it() -> Result[Void]
  m = try Trace.stablehlo(
    [spec([2, 3], Tensor.Dtype.F32), spec([3, 2], Tensor.Dtype.F16)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.matmul(xs[0], xs[1])]) },
  )

  try Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf16>) -> (tensor<2x2xf32>) {
          %0 = stablehlo.convert %arg1 : (tensor<3x2xf16>) -> tensor<3x2xf32>
          %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
          func.return %1 : tensor<2x2xf32>
        }
      }
    MLIR
  )

  rows = try Trace.stablehlo(
    [spec([5, 3], Tensor.Dtype.F16), spec([2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.to_dtype(try Tensor.gather(xs[0], xs[1]), Tensor.Dtype.F32)]) },
  )

  try Test.assert(rows.text.include?("(tensor<2x3xf16>) -> tensor<2x3xf32>"), rows.text)
  r = Trace.stablehlo(
    [spec([2, 2], Tensor.Dtype.F32), spec([2, 2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.matmul(xs[0], xs[1])]) },
  )

  try Test.assert(!r.ok?, "a matmul of F32 by I32 was emitted")
  ok()
end

def spec(shape: List[Int], dtype: Tensor.Dtype) -> Trace.Spec
  Trace.Spec(shape: shape, dtype: dtype, backend: Tensor.Backend.Reference)
end

## `x` as `[6, 2]` plus the `bias` constant widened.
def shifted(x: Tensor.Tensor, bias: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  wide = try Tensor.broadcast_to(Trace.constant(bias), [6, 2])
  ok([try Tensor.add(try Tensor.reshape(x, [6, 2]), wide)])
end

def reduced(x: Tensor.Tensor, labels: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  classes = try Tensor.arange(0, 3, 1, Tensor.Dtype.I32, Tensor.Backend.Trace)
  ok([try Tensor.max(x, some([1]), true), try Tensor.eq(labels, classes)])
end

def counted(x: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  tiny = try Tensor.full([2, 1], Tensor.Values.Float(0.00000001), Tensor.Dtype.F32, Tensor.Backend.Trace)
  ok([
    try Tensor.arange(1, 7, 2, Tensor.Dtype.I32, Tensor.Backend.Trace),
    try Tensor.add(try Tensor.mean(x, some([1]), true), tiny),
  ])
end

## The ops of a transformer step are written without a plugin (`xla_ops_test.agc` runs them): the text names each one.
def test_the_ops_of_a_transformer_step_are_written() -> Result[Void]
  specs = [
    spec([2, 4, 3, 8], Tensor.Dtype.F32),
    spec([2, 2, 5, 8], Tensor.Dtype.F32),
    spec([3, 5], Tensor.Dtype.F32),
    spec([3], Tensor.Dtype.I32),
    spec([4], Tensor.Dtype.I32),
    spec([4, 3], Tensor.Dtype.F32),
  ]

  m = try Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { transformer_ops(xs) })
  wanted = [
    "stablehlo.logistic",
    "stablehlo.power",
    "stablehlo.clamp",
    "stablehlo.compare NE",
    "stablehlo.transpose",
    "stablehlo.concatenate",
    "batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3]",
    "contracting_dims = [1] x [0]",
    "applies stablehlo.minimum",
    "\"chlo.erf\"",
    "stablehlo.cosine",
    "stablehlo.sine",
    "stablehlo.negate",
    "stablehlo.exponential",
    "dims = [0, 1, 3, 4]",
    "index_vector_dim = 1",
    "\"stablehlo.sort\"",
    "#1 [0:",
    "\"stablehlo.scatter\"",
    "stablehlo.add %acc, %upd",
  ]

  for w in wanted
    try Test.assert(m.text.include?(w), "the program has no #{w}")
  end

  ok()
end

def transformer_ops(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  q = xs[0]
  small = try Tensor.reshape(q, [24, 8])
  flat = try Tensor.reshape(q, [2, 4, 24])
  ok([
    try Tensor.pow(try Tensor.sigmoid(q), try Tensor.clamp(q, -1.0, 1.0)),
    try Tensor.to_dtype(try Tensor.to_dtype(q, Tensor.Dtype.I32), Tensor.Dtype.Bool),
    try Tensor.permute(try Tensor.stack([q, q], 1), [1, 0, 2, 3, 4]),
    try Tensor.concat([try Tensor.squeeze(try Tensor.unsqueeze(small, 0), some(0)), small], 1),
    try Tensor.batch_matmul(flat, try Tensor.permute(flat, [0, 2, 1])),
    try Tensor.batch_matmul(small, try Tensor.permute(small, [1, 0])),
    try Tensor.argmax(q, some([1, 3]), true),
    try Tensor.layer_norm(try Tensor.gelu_erf(small), 0.00001),
    try Tensor.rms_norm(small, 0.000001),
    try Tensor.rope(q, xs[3]),
    try Tensor.rope(q, xs[3], 10000.0, some(4), false, true),
    try Tensor.attention(q, xs[1], xs[1], mask: some(xs[2])),
    try Tensor.scatter_add(xs[4], xs[5], 5),
    try Tensor.top_ids(try Tensor.reshape(q, [6, 32]), 3),
  ])
end
