import IR from "pkg/agc-ir/src/ir"

COMBINERS = ["add", "maximum", "minimum", "or"]

def reduce_op(b: IR.Block, x: IR.Value, init: IR.Value, axes: List[Int], combine: String) -> Result[IR.Added]
  return error("reduce IR has no combiner #{combine}") if !COMBINERS.include?(combine)
  scalar = IR.TensorType(shape: [], element: x.type.element)
  return error("reduce IR needs a scalar init of the input's element type") if init.type != scalar
  rank = x.type.shape.size
  for i in 0...axes.size
    ax = axes[i]
    return error("reduce IR axis #{ax} is outside rank #{rank}") if ax < 0 or ax >= rank
    return error("reduce IR repeats axis #{ax}") if axes.take(i).include?(ax)
  end

  kept = (0...rank).to_a.select { |i| !axes.include?(i) }.map { |i| x.type.shape[i] }
  region = IR.block([scalar, scalar])
  combined = try IR.append(region, "stablehlo.#{combine}", region.args, [scalar])
  yielded = try IR.append(combined.block, "stablehlo.return", combined.values, [])
  IR.append(
    b,
    "stablehlo.reduce",
    [x, init],
    [IR.TensorType(shape: kept, element: x.type.element)],
    [IR.Attribute(key: "dimensions", value: IR.Attr.I64Array(axes))],
    [yielded.block],
  )
end
