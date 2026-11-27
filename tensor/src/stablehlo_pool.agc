import IR from "pkg/agc-ir/src/ir"

## An average pool over the last axis of a [C, L] or [N, C, L] input. The
## divisor includes padded zeros, as `count_include_pad` requires.
def avg_pool1d(b: IR.Block, input: IR.Value, kernel: Int, stride: Int, padding: Int) -> Result[IR.Added]
  shape = input.type.shape
  return error("avg_pool1d IR needs rank 2 or 3") if shape.size != 2 and shape.size != 3
  return error("avg_pool1d IR needs a floating-point input") if !float?(input.type.element)
  return error("avg_pool1d IR needs a positive kernel and stride, and valid padding") if kernel < 1 or stride < 1 or padding < 0 or padding > kernel / 2
  for d in shape
    return error("avg_pool1d IR needs nonempty input dimensions") if d < 1
  end

  span = shape[shape.size - 1] + 2 * padding - kernel
  return error("avg_pool1d kernel exceeds the padded input") if span < 0
  out_shape = shape.take(shape.size - 1) + [span / stride + 1]
  window: List[Int] = []
  strides: List[Int] = []
  pad: List[List[Int]] = []
  for i in 0...shape.size
    if i == shape.size - 1
      window = window.push(kernel)
      strides = strides.push(stride)
      pad = pad.push([padding, padding])
    else
      window = window.push(1)
      strides = strides.push(1)
      pad = pad.push([0, 0])
    end
  end

  reduce_window(b, input, out_shape, window, strides, pad, kernel.to_f)
end

## An average pool over H and W of an NCHW input. The divisor is the full
## square window even at padded boundaries.
def avg_pool2d(b: IR.Block, input: IR.Value, kernel: Int, stride: Int, padding: Int) -> Result[IR.Added]
  shape = input.type.shape
  return error("avg_pool2d IR needs rank 4") if shape.size != 4
  return error("avg_pool2d IR needs a floating-point input") if !float?(input.type.element)
  return error("avg_pool2d IR needs a positive kernel and stride, and valid padding") if kernel < 1 or stride < 1 or padding < 0 or padding > kernel / 2
  for d in shape
    return error("avg_pool2d IR needs nonempty input dimensions") if d < 1
  end

  h = shape[2] + 2 * padding - kernel
  w = shape[3] + 2 * padding - kernel
  return error("avg_pool2d kernel exceeds the padded input") if h < 0 or w < 0
  out_shape = [shape[0], shape[1], h / stride + 1, w / stride + 1]
  reduce_window(
    b,
    input,
    out_shape,
    [1, 1, kernel, kernel],
    [1, 1, stride, stride],
    [[0, 0], [0, 0], [padding, padding], [padding, padding]],
    kernel.to_f * kernel.to_f,
  )
end

private

def reduce_window(
  b: IR.Block,
  input: IR.Value,
  shape: List[Int],
  windows: List[Int],
  strides: List[Int],
  padding: List[List[Int]],
  divisor: Float,
) -> Result[IR.Added]
  element = input.type.element
  scalar = IR.TensorType(shape: [], element: element)
  output = IR.TensorType(shape: shape, element: element)
  pad_type = IR.TensorType(shape: [padding.size, 2], element: IR.Element.I64)
  zero = try IR.append(b, "stablehlo.constant", [], [scalar], [attr("value", IR.Attr.DenseF32(0.0, scalar))])
  region = IR.block([scalar, scalar], some(zero.block))
  total = try IR.append(region, "stablehlo.add", [region.args[0], region.args[1]], [scalar])
  yielded = try IR.append(total.block, "stablehlo.return", total.values, [])
  attrs = [
    attr("window_dimensions", IR.Attr.I64Array(windows)),
    attr("window_strides", IR.Attr.I64Array(strides)),
    attr("padding", IR.Attr.DenseIntRows(padding, pad_type)),
  ]
  reduced = try IR.append(
    zero.block,
    "stablehlo.reduce_window",
    [input, zero.values[0]],
    [output],
    attrs,
    [yielded.block],
  )
  scale = try IR.append(
    reduced.block,
    "stablehlo.constant",
    [],
    [output],
    [attr("value", IR.Attr.DenseF32(1.0 / divisor, output))],
  )
  IR.append(scale.block, "stablehlo.multiply", [reduced.values[0], scale.values[0]], [output])
end

def float?(element: IR.Element) -> Bool
  element == IR.Element.F16 or element == IR.Element.F32 or element == IR.Element.F64
end

def attr(key: String, value: IR.Attr) -> IR.Attribute = IR.Attribute(key: key, value: value)
