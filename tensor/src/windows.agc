## The backward rules of `conv2d`, `max_pool2d` and `avg_pool2d` (`tensor/DESIGN.md` §10.3), written over the ops of
## §3.1 so that every backend runs them (a ggml tensor has at most four axes, so no step goes through a rank above
## four). A stride is undone by spreading a tensor with zeros (`spread`) and read back by taking every `stride`-th
## position (`pick`), which are each other's adjoint.
##
## - The gradient of the input of a convolution is a convolution of the spread `ct`, padded by `kernel - 1 - padding`,
##   with the weight flipped on both kernel axes and its two channel axes swapped.
## - The gradient of the weight is a convolution of the input, its batch and channel axes swapped, with the spread `ct`
##   as the weight; the first `KH` by `KW` positions of its answer are the gradient.
## - A max pool sends each `ct` to the first position of its window that holds the maximum (as libtorch does), by
##   looking at every one of the `kernel * kernel` offsets of the window in turn.
## - An average pool is a convolution of every plane with a kernel of `1 / kernel^2`.

import Graph from "graph"
import Tensor from "../tensor"
import Types from "types"

NEGATIVE_INFINITY = -1.0e308 * 10.0

# The floats of the work space of the convolution of ggml on the CPU (16 MiB).
WORK_FLOATS = 4194304

## The gradient of argument `i` of `conv2d(x, w, stride, padding, groups)` given the gradient `ct` of its answer.
def conv2d_rule(
  x: Types.Tensor,
  w: Types.Tensor,
  ct: Types.Tensor,
  stride: Int,
  padding: Int,
  groups: Int,
  i: Int,
) -> Result[Types.Tensor]
  if padding >= w.shape[2] or padding >= w.shape[3]
    return error(
      "conv2d has no backward rule for a padding of #{padding} with a kernel of #{w.shape[2]} by #{w.shape[3]}: the padding must be smaller than the kernel",
    )
  end

  return input_grad(x.shape[2], x.shape[3], w, ct, stride, padding, groups) if i == 0
  weight_grad(x, w, ct, stride, padding, groups)
end

## The gradient of `avg_pool2d(x, kernel, stride, padding)` given the gradient `ct` of its answer.
def avg_pool2d_rule(x: Types.Tensor, ct: Types.Tensor, kernel: Int, stride: Int, padding: Int) -> Result[Types.Tensor]
  planes = x.shape[0] * x.shape[1]
  weight = try Graph.filled([1, 1, kernel, kernel], 1.0 / (kernel * kernel).to_f, x.dtype)
  flat = try Tensor.reshape(ct, [planes, 1, ct.shape[2], ct.shape[3]])
  grad = try input_grad(x.shape[2], x.shape[3], weight, flat, stride, padding, 1)
  Tensor.reshape(grad, x.shape)
end

## The gradient of `max_pool2d(x, kernel, stride, padding)`, whose answer is `out`, given the gradient `ct` of the
## answer. The input is padded with `-inf`, so a padded position never holds the maximum.
def max_pool2d_rule(
  x: Types.Tensor,
  out: Types.Tensor,
  ct: Types.Tensor,
  kernel: Int,
  stride: Int,
  padding: Int,
) -> Result[Types.Tensor]
  rows = x.shape[2]
  cols = x.shape[3]
  reach_rows = kernel - 1 + out.shape[2] * stride
  reach_cols = kernel - 1 + out.shape[3] * stride
  padded_rows = rows + 2 * padding
  if reach_rows > padded_rows
    padded_rows = reach_rows
  end

  padded_cols = cols + 2 * padding
  if reach_cols > padded_cols
    padded_cols = reach_cols
  end

  padded = try pad(x, padding, padded_rows - rows - padding, padding, padded_cols - cols - padding, NEGATIVE_INFINITY)
  seen = try Graph.filled(out.shape, 0.0, x.dtype)
  acc = try Graph.filled([x.shape[0], x.shape[1], padded_rows, padded_cols], 0.0, x.dtype)
  for dy in 0...kernel
    for dx in 0...kernel
      window = try patch(padded, dy, dx, out.shape[2], out.shape[3], stride)
      hit = try Tensor.to_dtype(try Tensor.eq(window, out), x.dtype)
      take = try Tensor.mul(hit, try Tensor.sub(try Graph.filled(out.shape, 1.0, x.dtype), seen))
      seen = try Tensor.add(seen, take)
      back = try unpatch(try Tensor.mul(ct, take), dy, dx, padded_rows, padded_cols, stride)
      acc = try Tensor.add(acc, back)
    end
  end

  Tensor.slice(acc, [0, 0, padding, padding], [x.shape[0], x.shape[1], padding + rows, padding + cols])
end

private

# The gradient of the `[N, C, rows, cols]` input of a convolution by `w`, given the gradient `ct` of its answer.
def input_grad(
  rows: Int,
  cols: Int,
  w: Types.Tensor,
  ct: Types.Tensor,
  stride: Int,
  padding: Int,
  groups: Int,
) -> Result[Types.Tensor]
  kh = w.shape[2]
  kw = w.shape[3]
  extra_rows = (rows + 2 * padding - kh) % stride
  extra_cols = (cols + 2 * padding - kw) % stride
  spread_ct = try spread(ct, stride)
  full = try pad(
    spread_ct,
    kh - 1 - padding,
    kh - 1 - padding + extra_rows,
    kw - 1 - padding,
    kw - 1 - padding + extra_cols,
    0.0,
  )

  return Tensor.conv2d(full, try flipped(w), 1, 0, 1) if groups == 1
  per_group = w.shape[0] / groups
  parts: List[Types.Tensor] = []
  for g in 0...groups
    channels = try cut(full, 1, g * per_group, (g + 1) * per_group)
    kernel = try flipped(try cut(w, 0, g * per_group, (g + 1) * per_group))
    parts = parts.push(try Tensor.conv2d(channels, kernel, 1, 0, 1))
  end

  Tensor.concat(parts, 1)
end

# The gradient of the weight of a convolution of `x`, given the gradient `ct` of its answer. The batch is the channel
# axis of the convolution, and ggml's CPU kernel works in 16 MiB, so a big batch is cut in chunks whose answers add up.
def weight_grad(
  x: Types.Tensor,
  w: Types.Tensor,
  ct: Types.Tensor,
  stride: Int,
  padding: Int,
  groups: Int,
) -> Result[Types.Tensor]
  out_per_group = w.shape[0] / groups
  in_per_group = w.shape[1]
  spread_ct = try spread(ct, stride)
  chunk = (WORK_FLOATS - out_per_group) / (spread_ct.shape[2] * spread_ct.shape[3])
  if chunk < 1
    chunk = 1
  end

  parts: List[Types.Tensor] = []
  for g in 0...groups
    input = try cut(x, 1, g * in_per_group, (g + 1) * in_per_group)
    window = try cut(spread_ct, 1, g * out_per_group, (g + 1) * out_per_group)
    grown: Optional[Types.Tensor] = none
    first = 0
    while first < x.shape[0]
      last = first + chunk
      if last > x.shape[0]
        last = x.shape[0]
      end

      part = try Tensor.conv2d(
        try Tensor.permute(try cut(input, 0, first, last), [1, 0, 2, 3]),
        try Tensor.permute(try cut(window, 0, first, last), [1, 0, 2, 3]),
        1,
        padding,
        1,
      )

      case grown
        some(so_far) -> grown = some(try Tensor.add(so_far, part))
        none -> grown = some(part)
      end

      first = last
    end

    case grown
      some(sum) ->
        kernel = try Tensor.slice(sum, [0, 0, 0, 0], [in_per_group, out_per_group, w.shape[2], w.shape[3]])
        parts = parts.push(try Tensor.permute(kernel, [1, 0, 2, 3]))
      none -> return error("conv2d weight gradient of an empty batch")
    end
  end

  return ok(parts[0]) if groups == 1
  Tensor.concat(parts, 0)
end

# `w` read backwards along both kernel axes, with its first two axes swapped.
def flipped(w: Types.Tensor) -> Result[Types.Tensor]
  Tensor.permute(try reversed(try reversed(w, 2), 3), [1, 0, 2, 3])
end

# `t` read backwards along `axis`.
def reversed(t: Types.Tensor, axis: Int) -> Result[Types.Tensor]
  parts: List[Types.Tensor] = []
  for k in 0...t.shape[axis]
    parts = parts.push(try cut(t, axis, t.shape[axis] - 1 - k, t.shape[axis] - k))
  end

  return ok(t) if parts.size == 1
  Tensor.concat(parts, axis)
end

# The part of `t` from `from` to `to` along `axis`, or `t` itself when that is all of it.
def cut(t: Types.Tensor, axis: Int, from: Int, to: Int) -> Result[Types.Tensor]
  return ok(t) if from == 0 and to == t.shape[axis]
  Tensor.slice(t, t.shape.map { |_| 0 }.set(axis, from), t.shape.set(axis, to))
end

# `t` with `value` around its two last axes.
def pad(t: Types.Tensor, top: Int, bottom: Int, left: Int, right: Int, value: Float) -> Result[Types.Tensor]
  rows = try pad_axis(t, 2, top, bottom, value)
  pad_axis(rows, 3, left, right, value)
end

def pad_axis(t: Types.Tensor, axis: Int, before: Int, after: Int, value: Float) -> Result[Types.Tensor]
  parts = [t]
  if before > 0
    parts = [try Graph.filled(t.shape.set(axis, before), value, t.dtype)] + parts
  end

  if after > 0
    parts = parts.push(try Graph.filled(t.shape.set(axis, after), value, t.dtype))
  end

  return ok(t) if parts.size == 1
  Tensor.concat(parts, axis)
end

# `t` (`[N, C, H, W]`) with `stride - 1` zeros after each position of its two last axes, up to the last position:
# `[N, C, (H - 1) * stride + 1, (W - 1) * stride + 1]`.
def spread(t: Types.Tensor, stride: Int) -> Result[Types.Tensor]
  return ok(t) if stride == 1
  n = t.shape[0]
  c = t.shape[1]
  h = t.shape[2]
  w = t.shape[3]
  across = try Tensor.reshape(t, [n * c * h, w, 1])
  across = try Tensor.concat([across, try Graph.filled([n * c * h, w, stride - 1], 0.0, t.dtype)], 2)
  down = try Tensor.reshape(across, [n * c, h, 1, w * stride])
  down = try Tensor.concat([down, try Graph.filled([n * c, h, stride - 1, w * stride], 0.0, t.dtype)], 2)
  wide = try Tensor.reshape(down, [n, c, h * stride, w * stride])
  Tensor.slice(wide, [0, 0, 0, 0], [n, c, (h - 1) * stride + 1, (w - 1) * stride + 1])
end

# Every `stride`-th position of the two last axes of `t`, from the first: `[N, C, H / stride, W / stride]`.
def pick(t: Types.Tensor, stride: Int) -> Result[Types.Tensor]
  return ok(t) if stride == 1
  n = t.shape[0]
  c = t.shape[1]
  h = t.shape[2] / stride
  w = t.shape[3] / stride
  down = try Tensor.reshape(t, [n * c, h, stride, w * stride])
  down = try Tensor.slice(down, [0, 0, 0, 0], [n * c, h, 1, w * stride])
  across = try Tensor.reshape(down, [n * c * h, w, stride])
  across = try Tensor.slice(across, [0, 0, 0], [n * c * h, w, 1])
  Tensor.reshape(across, [n, c, h, w])
end

# The `[N, C, rows, cols]` positions `(dy + stride * y, dx + stride * x)` of `padded`: the elements the offset `(dy,
# dx)` of every window of a pool reads.
def patch(padded: Types.Tensor, dy: Int, dx: Int, rows: Int, cols: Int, stride: Int) -> Result[Types.Tensor]
  n = padded.shape[0]
  c = padded.shape[1]
  block = try Tensor.slice(padded, [0, 0, dy, dx], [n, c, dy + rows * stride, dx + cols * stride])
  pick(block, stride)
end

# The adjoint of `patch`: `g` (`[N, C, rows, cols]`) put back at the positions of the offset `(dy, dx)` in zeros of
# `[N, C, padded_rows, padded_cols]`.
def unpatch(g: Types.Tensor, dy: Int, dx: Int, padded_rows: Int, padded_cols: Int, stride: Int) -> Result[Types.Tensor]
  wide = try spread(g, stride)
  after_rows = padded_rows - dy - wide.shape[2]
  after_cols = padded_cols - dx - wide.shape[3]
  pad(wide, dy, after_rows, dx, after_cols, 0.0)
end
