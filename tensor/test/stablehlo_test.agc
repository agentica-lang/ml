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

def test_an_op_without_an_emitter_is_an_error_naming_it() -> Result[Void]
  r = Trace.stablehlo(
    [spec([2, 2], Tensor.Dtype.F32)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.permute(xs[0], [1, 0])]) },
  )

  case r
    ok(_) -> error("permute was emitted")
    error(e) -> Test.eq(e.message, "StableHLO has no emitter for permute yet")
  end
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
