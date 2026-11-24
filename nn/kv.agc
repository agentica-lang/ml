## The padded cache of a decoder, shared by the models that step one: the rows
## a cache has (8, 16, 32, ... so a compiled step changes shape only when the
## capacity grows), the mask that hides the later positions and the padding,
## and the write of new rows into a cache. Each model keeps its own forward.

import Nn from "nn"
import Tensor from "pkg/tensor/tensor"

## Where a step writes into a padded cache: `positions` (`I32`, `[n]`) are the
## new tokens' rows, `capacity` the rows after the step.
struct Slots
  positions: Tensor.Tensor
  capacity: Int
end

## The rows a padded cache has for `needed` tokens: 8, 16, 32 and so on, at
## most `limit`.
def capacity(needed: Int, limit: Int) -> Int
  rows = 8
  while rows < needed
    rows *= 2
  end

  [rows, limit].min.unwrap!
end

## `[n, capacity]`: `MASKED` where the key's row is after the query's position
## (later tokens and the padding), 0 elsewhere.
def padded_mask(s: Slots, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(s.positions)[0]
  backend = Tensor.backend(s.positions)
  grid = [n, s.capacity]
  queries = try Tensor.broadcast_to(try Tensor.reshape(s.positions, [n, 1]), grid)
  keys = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, s.capacity, 1, Tensor.Dtype.I32, backend), [1, s.capacity]),
    grid,
  )

  blocked = try Tensor.full(grid, Tensor.Values.Float(Nn.MASKED), dtype, backend)
  Tensor.where(try Tensor.lt(queries, keys), blocked, try Tensor.zeros(grid, dtype, backend))
end

## `[capacity, n]`: 1 at row `positions[j]` of column `j`, 0 elsewhere. A
## position of `capacity` (the padding of a prompt) names no row.
def spread_of(s: Slots, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(s.positions)[0]
  grid = [s.capacity, n]
  rows = try Tensor.broadcast_to(
    try Tensor.reshape(
      try Tensor.arange(0, s.capacity, 1, Tensor.Dtype.I32, Tensor.backend(s.positions)),
      [s.capacity, 1],
    ),
    grid,
  )

  at = try Tensor.broadcast_to(try Tensor.reshape(s.positions, [1, n]), grid)
  Tensor.to_dtype(try Tensor.eq(rows, at), dtype)
end

## `fresh` written into `old` (zero past the cache's length; none when empty)
## at the rows `spread` (`[capacity, n]`) names, padded with zeros to `capacity`
## rows. The rows are axis `axis` of both (`1` of `[1, rows, d]`, `2` of
## `[1, heads, rows, width]`). A product with a 0/1 matrix and a sum with
## zeros, so exact in F32.
def written(
  old: Optional[Tensor.Tensor],
  fresh: Tensor.Tensor,
  spread: Tensor.Tensor,
  axis: Int,
) -> Result[Tensor.Tensor]
  shape = Tensor.shape(fresh)
  if axis < 0 or axis >= shape.size
    return error("written: the rows axis #{axis} is not an axis of a tensor of shape #{shape}")
  end

  room = Tensor.shape(spread)[0]
  n = shape[axis]
  rest = (0...shape.size).select { |i| i != axis }
  order = [axis] + rest
  inverse = (0...shape.size).map { |i| order.index(i).unwrap! }
  kept = rest.map { |i| shape[i] }
  width = 1
  for k in kept
    width *= k
  end

  # With only 1s before the rows axis the rows are already first: no permute, as the cache of asr (`[1, rows, d]`).
  placed = fresh
  if ((0...axis).all? { |i| shape[i] == 1 })
    here = (0...shape.size).map { |i| axis_rows(shape, i, axis, room) }
    placed = try Tensor.reshape(try Tensor.matmul(spread, try Tensor.reshape(fresh, [n, width])), here)
  else
    flat = try Tensor.reshape(try Tensor.permute(fresh, order), [n, width])
    moved = try Tensor.reshape(try Tensor.matmul(spread, flat), [room] + kept)
    placed = try Tensor.permute(moved, inverse)
  end

  case old
    none -> ok(placed)
    some(o) ->
      rows = Tensor.shape(o)[axis]
      if rows < room
        have = Tensor.shape(o)
        gap_shape = (0...have.size).map { |i| axis_rows(have, i, axis, room - rows) }
        gap = try Tensor.zeros(gap_shape, Tensor.dtype(o), Tensor.backend(o))
        o = try Tensor.concat([o, gap], axis)
      end

      Tensor.add(o, placed)
  end
end

private

def axis_rows(shape: List[Int], i: Int, axis: Int, rows: Int) -> Int
  return rows if i == axis
  shape[i]
end
