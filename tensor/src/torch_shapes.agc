## Shape arithmetic of the libtorch backend: the element count, the axis
## checks and the shape a reduction leaves (`tensor/DESIGN.md` §9 asks the
## wrapper to validate eagerly, before dispatch; the native side re-checks as a
## backstop and is never the only check).
##
## These mirror the reference (`src/types.agc`, `src/reference.agc`) without
## importing it: the backend computes the shape of the tensor it is about to
## hand back, so a metadata mismatch cannot hide behind a valid native buffer.

## The number of elements `shape` names: `1` for `[]`, the product otherwise.
def volume(shape: List[Int]) -> Result[Int]
  total = 1
  for d in shape
    return error("a shape holds no negative dimension, got #{shape}") if d < 0
    total = total * d
  end

  ok(total)
end

## The axes `axes` will reduce: every axis for `none`, the named ones for
## `some`, after checking each names a distinct axis of a rank-`rank` tensor.
## The wrapper (`tensor.agc`) has already sorted them ascending, which is the
## order this backend and the reference both number a reduced block by.
def normalize_axes(axes: Optional[List[Int]], rank: Int, op: String) -> Result[List[Int]]
  case axes
    some(given) ->
      _ = try check_axes(given, rank, op)
      ok(given)
    none -> ok((0...rank).to_a)
  end
end

## Every entry of `axes` names a distinct axis of a rank-`rank` tensor.
def check_axes(axes: List[Int], rank: Int, op: String) -> Result[Bool]
  for ax in axes
    return error("#{op} needs axes in 0...#{rank}, got #{ax}") if ax < 0 or ax >= rank
  end

  return error("#{op} names an axis twice, got #{axes}") if axes.uniq.size != axes.size
  ok(true)
end

## The shape left after reducing `axes`: dropped, or `1` with `keep_dims`.
def reduced_shape(shape: List[Int], axes: List[Int], keep_dims: Bool) -> List[Int]
  out: List[Int] = []
  for i in 0...shape.size
    next if !keep_dims and axes.include?(i)
    if axes.include?(i)
      out = out.push(1)
    else
      out = out.push(shape[i])
    end
  end

  out
end

## The product of the reduced dimensions (the count `mean` divides by).
def reduced_count(shape: List[Int], axes: List[Int]) -> Int
  total = 1
  for ax in axes
    total = total * shape[ax]
  end

  total
end

## `shape` without the axes `axes` names, or without every size-1 axis when
## `axes` is `[]`: the shape a `squeeze` leaves.
def squeezed_shape(shape: List[Int], axes: List[Int]) -> List[Int]
  kept: List[Int] = []
  for i in 0...shape.size
    next if axes.empty? and shape[i] == 1
    next if axes.include?(i)
    kept = kept.push(shape[i])
  end

  kept
end

## `shape` with a `1` at `axis`: the shape an `unsqueeze` leaves.
def unsqueezed_shape(shape: List[Int], axis: Int) -> List[Int]
  out: List[Int] = []
  for i in 0...shape.size
    if i == axis
      out = out.push(1)
    end

    out = out.push(shape[i])
  end

  if axis == shape.size
    out = out.push(1)
  end

  out
end

## `a` with `axis` grown by the same dimension of `b`: the shape a `concat`
## along `axis` leaves.
def concatenated_shape(a: List[Int], b: List[Int], axis: Int) -> List[Int]
  out: List[Int] = []
  for i in 0...a.size
    if i == axis
      out = out.push(a[i] + b[i])
    else
      out = out.push(a[i])
    end
  end

  out
end

## The length a window of `kernel` positions leaves in `length`, padded by
## `padding` on each side and moved by `stride`: the `L_out` of `conv1d` and
## `avg_pool1d`.
def windowed_length(length: Int, kernel: Int, stride: Int, padding: Int) -> Int
  (length + 2 * padding - kernel) / stride + 1
end

## The `[...leading, out_channels, L_out]` a `conv1d` leaves.
def conv1d_output_shape(shape: List[Int], out_channels: Int, kernel: Int, stride: Int, padding: Int) -> List[Int]
  out: List[Int] = []
  for i in 0...(shape.size - 2)
    out = out.push(shape[i])
  end

  out.push(out_channels).push(windowed_length(shape[shape.size - 1], kernel, stride, padding))
end

## The `[...leading, L_out]` an `avg_pool1d` leaves.
def avg_pool1d_output_shape(shape: List[Int], kernel: Int, stride: Int, padding: Int) -> List[Int]
  out: List[Int] = []
  for i in 0...(shape.size - 1)
    out = out.push(shape[i])
  end

  out.push(windowed_length(shape[shape.size - 1], kernel, stride, padding))
end

## `indices`'s shape followed by the tail of `source`'s: the shape a `gather`
## leaves.
def gathered_shape(source: List[Int], indices: List[Int]) -> List[Int]
  out = indices
  for i in 1...source.size
    out = out.push(source[i])
  end

  out
end
