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
          %0 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = array<i64: 1>} : (tensor<2xf32>) -> (tensor<6x2xf32>)
          %1 = "stablehlo.reshape"(%arg0) : (tensor<3x4xf32>) -> (tensor<6x2xf32>)
          %2 = "stablehlo.add"(%1, %0) : (tensor<6x2xf32>, tensor<6x2xf32>) -> (tensor<6x2xf32>)
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
          %0 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<3xi32>)
          %1 = "stablehlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> (tensor<f32>)
          %2 = "stablehlo.reduce"(%arg0, %1) ({
          ^bb0(%ir2_0r0_a0: tensor<f32>, %ir2_0r0_a1: tensor<f32>):
            %ir2_0r0_0_0 = "stablehlo.maximum"(%ir2_0r0_a0, %ir2_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
            "stablehlo.return"(%ir2_0r0_0_0) : (tensor<f32>) -> ()
          }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<f32>) -> (tensor<2xf32>)
          %3 = "stablehlo.reshape"(%2) : (tensor<2xf32>) -> (tensor<2x1xf32>)
          %4 = "stablehlo.compare"(%arg1, %0) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type SIGNED>} : (tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi1>)
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
          %0 = "stablehlo.constant"() {value = dense<1.0e-8> : tensor<2x1xf32>} : () -> (tensor<2x1xf32>)
          %1 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<3xi32>)
          %2 = "stablehlo.constant"() {value = dense<2> : tensor<3xi32>} : () -> (tensor<3xi32>)
          %3 = "stablehlo.multiply"(%1, %2) : (tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>)
          %4 = "stablehlo.constant"() {value = dense<1> : tensor<3xi32>} : () -> (tensor<3xi32>)
          %5 = "stablehlo.add"(%3, %4) : (tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>)
          %6 = "stablehlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
          %7 = "stablehlo.reduce"(%arg0, %6) ({
          ^bb0(%ir7_0r0_a0: tensor<f32>, %ir7_0r0_a1: tensor<f32>):
            %ir7_0r0_0_0 = "stablehlo.add"(%ir7_0r0_a0, %ir7_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
            "stablehlo.return"(%ir7_0r0_0_0) : (tensor<f32>) -> ()
          }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<f32>) -> (tensor<2xf32>)
          %8 = "stablehlo.constant"() {value = dense<3.0> : tensor<2xf32>} : () -> (tensor<2xf32>)
          %9 = "stablehlo.divide"(%7, %8) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
          %10 = "stablehlo.reshape"(%9) : (tensor<2xf32>) -> (tensor<2x1xf32>)
          %11 = "stablehlo.add"(%10, %0) : (tensor<2x1xf32>, tensor<2x1xf32>) -> (tensor<2x1xf32>)
          func.return %5, %11 : tensor<3xi32>, tensor<2x1xf32>
        }
      }
    MLIR
  )
end

def test_xla_rejects_a_device_other_than_cpu() -> Result[Void]
  r = Trace.stablehlo(
    [spec([2, 2], Tensor.Dtype.F32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.to_device(xs[0], Tensor.Device.Vulkan(0))]) },
  )

  case r
    ok(_) -> error("XLA accepted a Vulkan to_device target")
    error(e) -> Test.assert(e.message.include?("XLA runs on the CPU only"), e.message)
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
          %0 = "stablehlo.slice"(%arg0) {start_indices = array<i64: 1, 2>, limit_indices = array<i64: 2, 5>, strides = array<i64: 1, 1>} : (tensor<2x7xf32>) -> (tensor<1x3xf32>)
          func.return %0 : tensor<1x3xf32>
        }
      }
    MLIR
  )
end

## An `argmax` takes the first index of the maximum: the maximum reduced, the positions where it is reached, and the
## smallest of them. Without an axis it reduces every axis.
def test_argmax_text_reduces_the_maximum_then_the_first_index() -> Result[Void]
  m = try Trace.stablehlo(
    [spec([2, 3], Tensor.Dtype.F32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.argmax(xs[0], some([1])), try Tensor.argmax(xs[0])]) },
  )

  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x3xf32>) -> (tensor<2xi64>, tensor<i64>) {
          %0 = "stablehlo.reshape"(%arg0) : (tensor<2x3xf32>) -> (tensor<2x3xf32>)
          %1 = "stablehlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> (tensor<f32>)
          %2 = "stablehlo.reduce"(%0, %1) ({
          ^bb0(%ir2_0r0_a0: tensor<f32>, %ir2_0r0_a1: tensor<f32>):
            %ir2_0r0_0_0 = "stablehlo.maximum"(%ir2_0r0_a0, %ir2_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
            "stablehlo.return"(%ir2_0r0_0_0) : (tensor<f32>) -> ()
          }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<f32>) -> (tensor<2xf32>)
          %3 = "stablehlo.broadcast_in_dim"(%2) {broadcast_dimensions = array<i64: 0>} : (tensor<2xf32>) -> (tensor<2x3xf32>)
          %4 = "stablehlo.compare"(%0, %3) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2x3xi1>)
          %5 = "stablehlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<2x3xi64>)
          %6 = "stablehlo.constant"() {value = dense<3> : tensor<2x3xi64>} : () -> (tensor<2x3xi64>)
          %7 = "stablehlo.select"(%4, %5, %6) : (tensor<2x3xi1>, tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>)
          %8 = "stablehlo.constant"() {value = dense<3> : tensor<i64>} : () -> (tensor<i64>)
          %9 = "stablehlo.reduce"(%7, %8) ({
          ^bb0(%ir9_0r0_a0: tensor<i64>, %ir9_0r0_a1: tensor<i64>):
            %ir9_0r0_0_0 = "stablehlo.minimum"(%ir9_0r0_a0, %ir9_0r0_a1) : (tensor<i64>, tensor<i64>) -> (tensor<i64>)
            "stablehlo.return"(%ir9_0r0_0_0) : (tensor<i64>) -> ()
          }) {dimensions = array<i64: 1>} : (tensor<2x3xi64>, tensor<i64>) -> (tensor<2xi64>)
          %10 = "stablehlo.reshape"(%arg0) : (tensor<2x3xf32>) -> (tensor<6xf32>)
          %11 = "stablehlo.constant"() {value = dense<0xFF800000> : tensor<f32>} : () -> (tensor<f32>)
          %12 = "stablehlo.reduce"(%10, %11) ({
          ^bb0(%ir12_0r0_a0: tensor<f32>, %ir12_0r0_a1: tensor<f32>):
            %ir12_0r0_0_0 = "stablehlo.maximum"(%ir12_0r0_a0, %ir12_0r0_a1) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
            "stablehlo.return"(%ir12_0r0_0_0) : (tensor<f32>) -> ()
          }) {dimensions = array<i64: 0>} : (tensor<6xf32>, tensor<f32>) -> (tensor<f32>)
          %13 = "stablehlo.broadcast_in_dim"(%12) {broadcast_dimensions = array<i64>} : (tensor<f32>) -> (tensor<6xf32>)
          %14 = "stablehlo.compare"(%10, %13) {comparison_direction = #stablehlo<comparison_direction EQ>, compare_type = #stablehlo<comparison_type FLOAT>} : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xi1>)
          %15 = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> (tensor<6xi64>)
          %16 = "stablehlo.constant"() {value = dense<6> : tensor<6xi64>} : () -> (tensor<6xi64>)
          %17 = "stablehlo.select"(%14, %15, %16) : (tensor<6xi1>, tensor<6xi64>, tensor<6xi64>) -> (tensor<6xi64>)
          %18 = "stablehlo.constant"() {value = dense<6> : tensor<i64>} : () -> (tensor<i64>)
          %19 = "stablehlo.reduce"(%17, %18) ({
          ^bb0(%ir19_0r0_a0: tensor<i64>, %ir19_0r0_a1: tensor<i64>):
            %ir19_0r0_0_0 = "stablehlo.minimum"(%ir19_0r0_a0, %ir19_0r0_a1) : (tensor<i64>, tensor<i64>) -> (tensor<i64>)
            "stablehlo.return"(%ir19_0r0_0_0) : (tensor<i64>) -> ()
          }) {dimensions = array<i64: 0>} : (tensor<6xi64>, tensor<i64>) -> (tensor<i64>)
          func.return %9, %19 : tensor<2xi64>, tensor<i64>
        }
      }
    MLIR
  )
end

## A product and a batched product are one `dot_general` each, and the rows of a table are one `gather`.
def test_products_and_gather_text_name_their_dimensions() -> Result[Void]
  specs = [spec([2, 3], Tensor.Dtype.F32), spec([3, 4], Tensor.Dtype.F32), spec([2], Tensor.Dtype.I32)]
  m = try Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { products(xs) })
  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2xi32>) -> (tensor<2x4xf32>, tensor<1x2x4xf32>, tensor<2x4xf32>) {
          %0 = "stablehlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x3xf32>, tensor<3x4xf32>) -> (tensor<2x4xf32>)
          %1 = "stablehlo.reshape"(%arg0) : (tensor<2x3xf32>) -> (tensor<1x2x3xf32>)
          %2 = "stablehlo.reshape"(%arg1) : (tensor<3x4xf32>) -> (tensor<1x3x4xf32>)
          %3 = "stablehlo.dot_general"(%1, %2) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<1x2x3xf32>, tensor<1x3x4xf32>) -> (tensor<1x2x4xf32>)
          %4 = "stablehlo.gather"(%arg1, %arg2) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>, indices_are_sorted = false} : (tensor<3x4xf32>, tensor<2xi32>) -> (tensor<2x4xf32>)
          func.return %0, %3, %4 : tensor<2x4xf32>, tensor<1x2x4xf32>, tensor<2x4xf32>
        }
      }
    MLIR
  )
end

def products(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ok([
    try Tensor.matmul(xs[0], xs[1]),
    try Tensor.batch_matmul(try Tensor.reshape(xs[0], [1, 2, 3]), try Tensor.reshape(xs[1], [1, 3, 4])),
    try Tensor.gather(xs[1], xs[2]),
  ])
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
          %0 = "stablehlo.convert"(%arg1) : (tensor<3x2xf16>) -> (tensor<3x2xf32>)
          %1 = "stablehlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<2x3xf32>, tensor<3x2xf32>) -> (tensor<2x2xf32>)
          func.return %1 : tensor<2x2xf32>
        }
      }
    MLIR
  )

  rows = try Trace.stablehlo(
    [spec([5, 3], Tensor.Dtype.F16), spec([2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.to_dtype(try Tensor.gather(xs[0], xs[1]), Tensor.Dtype.F32)]) },
  )

  try Test.assert(rows.text.include?("(tensor<2x3xf16>) -> (tensor<2x3xf32>)"), rows.text)
  r = Trace.stablehlo(
    [spec([2, 2], Tensor.Dtype.F32), spec([2, 2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.matmul(xs[0], xs[1])]) },
  )

  try Test.assert(!r.ok?, "a matmul of F32 by I32 was emitted")
  ok()
end

## The text of `set_rows` on a rank-2 cache: the positions as a column, the rows moved to the scatter layout, and one
## scatter that replaces the rows. This golden is the one the emitter must keep when it builds the scatter with `agc-ir`.
def test_set_rows_text_of_rank_two_is_a_scatter_of_the_rows() -> Result[Void]
  m = try Trace.stablehlo(
    [spec([4, 3], Tensor.Dtype.F32), spec([2, 3], Tensor.Dtype.F32), spec([2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<4x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2xi32>) -> (tensor<4x3xf32>) {
          %0 = "stablehlo.reshape"(%arg2) : (tensor<2xi32>) -> (tensor<2x1xi32>)
          %1 = "stablehlo.transpose"(%arg1) {permutation = array<i64: 0, 1>} : (tensor<2x3xf32>) -> (tensor<2x3xf32>)
          %2 = "stablehlo.scatter"(%arg0, %0, %1) ({
          ^bb0(%set_rows0_r0_a0: tensor<f32>, %set_rows0_r0_a1: tensor<f32>):
            "stablehlo.return"(%set_rows0_r0_a1) : (tensor<f32>) -> ()
          }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<4x3xf32>, tensor<2x1xi32>, tensor<2x3xf32>) -> (tensor<4x3xf32>)
          func.return %2 : tensor<4x3xf32>
        }
      }
    MLIR
  )
end

## The text of `set_rows` on a rank-3 cache: the rows axis is the one before the width, so the updates move it to the front.
def test_set_rows_text_of_rank_three_moves_the_rows_axis() -> Result[Void]
  m = try Trace.stablehlo(
    [spec([2, 4, 3], Tensor.Dtype.F32), spec([2, 2, 3], Tensor.Dtype.F32), spec([2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<2x4x3xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2xi32>) -> (tensor<2x4x3xf32>) {
          %0 = "stablehlo.reshape"(%arg2) : (tensor<2xi32>) -> (tensor<2x1xi32>)
          %1 = "stablehlo.transpose"(%arg1) {permutation = array<i64: 1, 0, 2>} : (tensor<2x2x3xf32>) -> (tensor<2x2x3xf32>)
          %2 = "stablehlo.scatter"(%arg0, %0, %1) ({
          ^bb0(%set_rows0_r0_a0: tensor<f32>, %set_rows0_r0_a1: tensor<f32>):
            "stablehlo.return"(%set_rows0_r0_a1) : (tensor<f32>) -> ()
          }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (tensor<2x4x3xf32>, tensor<2x1xi32>, tensor<2x2x3xf32>) -> (tensor<2x4x3xf32>)
          func.return %2 : tensor<2x4x3xf32>
        }
      }
    MLIR
  )
end

## The text of `expert_matmul`: the selected expert matrices gathered, the input broadcast over them, then one batched
## product per row. The golden is the emitter's text before it is built with `agc-ir`.
def test_expert_matmul_text_gathers_then_contracts_each_row() -> Result[Void]
  m = try Trace.stablehlo(
    [spec([3, 2, 4], Tensor.Dtype.F32), spec([2, 1, 4], Tensor.Dtype.F32), spec([2, 2], Tensor.Dtype.I32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.expert_matmul(xs[0], xs[1], xs[2])]) },
  )

  Test.eq(
    m.text,
    <<~MLIR,
      module {
        func.func public @main(%arg0: tensor<3x2x4xf32>, %arg1: tensor<2x1x4xf32>, %arg2: tensor<2x2xi32>) -> (tensor<2x2x2xf32>) {
          %0 = "stablehlo.gather"(%arg0, %arg2) {dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2, 4>, indices_are_sorted = false} : (tensor<3x2x4xf32>, tensor<2x2xi32>) -> (tensor<2x2x2x4xf32>)
          %1 = "stablehlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = array<i64: 0, 1, 2>} : (tensor<2x1x4xf32>) -> (tensor<2x2x4xf32>)
          %2 = "stablehlo.dot_general"(%1, %0) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [3]>} : (tensor<2x2x4xf32>, tensor<2x2x2x4xf32>) -> (tensor<2x2x2xf32>)
          func.return %2 : tensor<2x2x2xf32>
        }
      }
    MLIR
  )
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
    "comparison_direction NE",
    "stablehlo.transpose",
    "stablehlo.concatenate",
    "lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [3]",
    "lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]",
    "\"stablehlo.minimum\"",
    "\"chlo.erf\"",
    "stablehlo.cosine",
    "stablehlo.sine",
    "stablehlo.negate",
    "stablehlo.exponential",
    "broadcast_dimensions = array<i64: 0, 1, 3, 4>",
    "index_vector_dim = 1",
    "\"stablehlo.sort\"",
    "#1) {start_indices",
    "\"stablehlo.scatter\"",
    "inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]",
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
