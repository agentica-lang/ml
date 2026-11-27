import IR from "pkg/agc-ir/src/ir"

## NCHW input and output, OIHW filters, symmetric padding and equal strides on both spatial axes.
def conv2d(b: IR.Block, input: IR.Value, weight: IR.Value, stride: Int, padding: Int, groups: Int) -> Result[IR.Added]
  x = input.type.shape
  w = weight.type.shape
  element = input.type.element
  if element != weight.type.element or (element != IR.Element.F16 and element != IR.Element.F32 and element != IR.Element.F64)
    return error("conv2d needs matching floating-point element types")
  end

  return error("conv2d needs rank-4 input and weight") if x.size != 4 or w.size != 4
  for d in x + w
    return error("conv2d needs nonempty input and weight") if d < 1
  end

  return error("conv2d needs positive stride and groups, and nonnegative padding") if stride < 1 or groups < 1 or padding < 0
  if x[1] % groups != 0 or w[0] % groups != 0 or w[1] != x[1] / groups
    return error("conv2d channels do not match the feature groups")
  end

  h = x[2] + 2 * padding - w[2]
  width = x[3] + 2 * padding - w[3]
  return error("conv2d kernel exceeds the padded input") if h < 0 or width < 0
  out = IR.TensorType(shape: [x[0], w[0], h / stride + 1, width / stride + 1], element: element)
  dims = IR.Attr.DialectAttr("stablehlo", "conv", "[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]")
  pad = IR.TensorType(shape: [2, 2], element: IR.Element.I64)
  attrs = [
    attr("dimension_numbers", dims),
    attr("window_strides", IR.Attr.I64Array([stride, stride])),
    attr("padding", IR.Attr.DenseInt(padding, pad)),
    attr("batch_group_count", IR.Attr.I64(1)),
    attr("feature_group_count", IR.Attr.I64(groups)),
  ]
  IR.append(b, "stablehlo.convolution", [input, weight], [out], attrs)
end

## Conv1d over NCL input and OIK weights. A rank-2 [C, L] input is represented
## internally as [1, C, L], then reshaped back to [O, L_out].
def conv1d(b: IR.Block, input: IR.Value, weight: IR.Value, stride: Int, padding: Int) -> Result[IR.Added]
  x = input.type.shape
  w = weight.type.shape
  return error("conv1d needs a rank-2 or rank-3 input and rank-3 weight") if (x.size != 2 and x.size != 3) or w.size != 3
  return error("conv1d needs matching floating-point element types") if input.type.element != weight.type.element
  element = input.type.element
  if element != IR.Element.F16 and element != IR.Element.F32 and element != IR.Element.F64
    return error("conv1d needs matching floating-point element types")
  end

  for d in x + w
    return error("conv1d needs nonempty input and weight") if d < 1
  end

  channels = x[x.size - 2]
  if stride < 1 or padding < 0 or w[1] != channels
    return error("conv1d needs positive stride, nonnegative padding, and matching input channels")
  end

  length = x[x.size - 1] + 2 * padding - w[2]
  return error("conv1d kernel exceeds the padded input") if length < 0
  out_length = length / stride + 1
  current = b
  source = input
  if x.size == 2
    batched = IR.TensorType(shape: [1] + x, element: element)
    reshaped = try IR.append(current, "stablehlo.reshape", [source], [batched])
    current = reshaped.block
    source = reshaped.values[0]
  end

  output = IR.TensorType(shape: [source.type.shape[0], w[0], out_length], element: element)
  dims = IR.Attr.DialectAttr("stablehlo", "conv", "[b, f, 0]x[o, i, 0]->[b, f, 0]")
  pad = IR.TensorType(shape: [1, 2], element: IR.Element.I64)
  attrs = [
    attr("dimension_numbers", dims),
    attr("window_strides", IR.Attr.I64Array([stride])),
    attr("padding", IR.Attr.DenseInt(padding, pad)),
    attr("batch_group_count", IR.Attr.I64(1)),
    attr("feature_group_count", IR.Attr.I64(1)),
  ]
  convolution = try IR.append(current, "stablehlo.convolution", [source, weight], [output], attrs)
  current = convolution.block
  answer = convolution.values[0]
  if x.size == 2
    flat = IR.TensorType(shape: [w[0], out_length], element: element)
    squeezed = try IR.append(current, "stablehlo.reshape", [answer], [flat])
    current = squeezed.block
    answer = squeezed.values[0]
  end

  ok(IR.Added(block: current, values: [answer]))
end

private

def attr(key: String, value: IR.Attr) -> IR.Attribute = IR.Attribute(key: key, value: value)
