## Which attention nodes the lowering lets flash attention take (the second int of the node): no mask, the causal
## flag, a mask of `where` over fills of 0 and of -1e9 (the padded mask of a decoder), also with one query; not a fill
## above -65504, a mask of arbitrary values, `-inf`, a fill beyond what F32 holds, nor the causal flag added to a mask.
## The results of the ggml tests agree on the fallback too, so only the flag tells which path a mask takes. It needs no
## ggml module.

import Graph from "../src/graph"
import Lower from "../src/lower"
import Tensor from "../tensor"
import Types from "../src/types"

private

def flash_flag(shapes: List[List[Int]], f: Fn[List[Types.Tensor], Result[Types.Tensor]]) -> Result[Int]
  inputs = shapes.map { |s| Graph.record(Types.Op.Input, [], s, Types.Dtype.F32) }
  out = try f.call(inputs)
  p = try Lower.program([out], inputs, Tensor.Device.Cpu)
  for l in p.nodes
    return ok(l.ints[1]) if l.op == Lower.ATTENTION
  end

  error("the program has no attention node")
end

## `[sq, 4]`: `fill` where the key is after the query, 0 elsewhere.
def filled_mask(like: Types.Tensor, sq: Int, fill: Float) -> Result[Types.Tensor]
  backend = Tensor.backend(like)
  grid = [sq, 4]
  rows = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, sq, 1, Tensor.Dtype.I32, backend), [sq, 1]),
    grid,
  )

  cols = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, 4, 1, Tensor.Dtype.I32, backend), [1, 4]),
    grid,
  )

  blocked = try Tensor.full(grid, Tensor.Values.Float(fill), Tensor.Dtype.F32, backend)
  Tensor.where(try Tensor.lt(rows, cols), blocked, try Tensor.zeros(grid, Tensor.Dtype.F32, backend))
end

def masked(sq: Int, fill: Float) -> Result[Int]
  flash_flag([[1, 2, sq, 8], [1, 2, 4, 8], [1, 2, 4, 8]]) do |x|
    Tensor.attention(x[0], x[1], x[2], some(try filled_mask(x[0], sq, fill)), none, false)
  end
end

def test_no_mask_and_the_causal_flag_take_flash_attention() -> Result[Void]
  shapes = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]]
  try Test.eq(try flash_flag(shapes) do |x| Tensor.attention(x[0], x[1], x[2], none, none, false) end, 1)
  Test.eq(try flash_flag(shapes) do |x| Tensor.attention(x[0], x[1], x[2], none, none, true) end, 1)
end

def test_a_mask_of_fills_takes_flash_attention_also_with_one_query() -> Result[Void]
  try Test.eq(try masked(4, -1000000000.0), 1)
  try Test.eq(try masked(4, -65504.0), 1)
  try Test.eq(try masked(1, -1000000000.0), 1)
  Test.eq(
    try flash_flag([[1, 2, 1, 8], [1, 2, 4, 8], [1, 2, 4, 8]]) do |x|
      Tensor.attention(x[0], x[1], x[2], none, none, false)
    end,
    1,
  )
end

def test_other_masks_keep_the_products() -> Result[Void]
  try Test.eq(try masked(4, -1.0), 0)
  try Test.eq(try masked(4, -1.0e308 * 10.0), 0)
  try Test.eq(try masked(4, -1.0e39), 0)
  try Test.eq(try masked(4, -3.5e38), 0)
  try Test.eq(try masked(4, 1.0), 0)
  shapes = [[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [4, 4]]
  try Test.eq(try flash_flag(shapes) do |x| Tensor.attention(x[0], x[1], x[2], some(x[3]), none, false) end, 0)
  Test.eq(try flash_flag(shapes) do |x| Tensor.attention(x[0], x[1], x[2], some(x[3]), none, true) end, 0)
end
