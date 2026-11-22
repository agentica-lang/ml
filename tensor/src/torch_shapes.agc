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

## `none` is every axis; entries name distinct axes of a rank-`rank` tensor.
def check_axes_option(axes: Optional[List[Int]], rank: Int, op: String) -> Result[Bool]
  case axes
    some(given) -> check_axes(given, rank, op)
    none -> ok(true)
  end
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
