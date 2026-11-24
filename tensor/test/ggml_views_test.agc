## A permute that keeps the last axis (a head's width) is a view on ggml, not a copy: every class of op that takes
## such a view gives what the reference gives, node by node with `Trace.diff`, on ggml CPU (and Vulkan when it is
## there). The classes: a binary op (the view first or second, and as the source of a broadcast), a reduction over a
## trailing and a leading block, a product, a gather, `concat`, `rms_norm`, `rope` (which permutes the view back, also
## after a norm that ggml-vulkan fuses with it), `where` (a mask that is a view) and `attention`, also with one key head
## over a batch (a view whose head axis is 1 and whose batch stride is not packed: ggml-vulkan copies such a product
## operand through F16). CI has no Vulkan: the Vulkan cases run on a machine that has it, so the one-head case there
## cannot fail in CI if `operand` goes; ggml's CPU backend takes both paths.

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
REF = Tensor.Backend.Reference

private

def engines() -> Result[List[Tensor.Backend]]
  return ok([]) if !try Support.ggml_present()
  return ok([CPU, VULKAN]) if Tensor.available(VULKAN)
  ok([CPU])
end

def agree(
  what: String,
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Void]
  d = try Trace.diff(args, REF, backend, f: f)
  case d.first
    some(n) -> error("#{what} on #{backend}: node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
    none -> Test.assert(d.nodes >= 2, "#{what}: only #{d.nodes} nodes compared")
  end
end

## The result alone, as one compiled graph (`diff` makes every node an output, which ggml-vulkan does not fuse).
def agree_end(
  what: String,
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Void]
  moved: List[Tensor.Tensor] = []
  for t in args
    moved = moved.push(try Tensor.from_list(try Tensor.to_list(t), t.shape, t.dtype, backend))
  end

  want = try Support.floats_of(try Trace.eval(args, f))
  got = try Support.floats_of(try Trace.eval(moved, f))
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    if (got[i] - want[i]).abs > 0.00001 + 0.0001 * want[i].abs
      return error("#{what} on #{backend}: value #{i} is #{got[i]}, expected #{want[i]}")
    end
  end

  ok()
end

def f32(seed: Int, shape: List[Int]) -> Result[Tensor.Tensor]
  Support.f32(Support.ramp(seed, shape.reduce(1) do |acc, n| acc * n end), shape)
end

## `[b, h, s, d]` of a `[b, s, h, d]` tensor: the view that `split_heads` makes.
def heads(x: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.permute(x, [0, 2, 1, 3])

def test_binary_ops_take_a_view_first_second_and_as_a_broadcast_source() -> Result[Void]
  a = try f32(1, [2, 3, 4, 5])
  b = try f32(2, [2, 4, 3, 5])
  c = try f32(3, [2, 3, 1, 5])
  for backend in try engines()
    try agree("add of a view and a tensor", backend, [a, b]) do |x| Tensor.add(try heads(x[0]), x[1]) end

    try agree("mul of a tensor and a view", backend, [a, b]) do |x| Tensor.mul(x[1], try heads(x[0])) end

    try agree("sub of two views", backend, [a, b]) do |x|
      Tensor.sub(try heads(x[0]), try heads(try Tensor.add(x[0], x[0])))
    end

    try agree("where with a mask that is a view", backend, [a, b]) do |x|
      mask = try heads(try Tensor.lt(x[0], try Tensor.add(x[0], x[0])))
      Tensor.where(mask, try heads(x[0]), try heads(try Tensor.add(x[0], x[0])))
    end

    try agree("a view broadcast to a bigger shape", backend, [b, c]) do |x|
      Tensor.add(x[0], try Tensor.broadcast_to(try heads(x[1]), [2, 4, 3, 5]))
    end
  end

  ok()
end

def test_reductions_products_and_gathers_take_a_view() -> Result[Void]
  a = try f32(4, [2, 3, 4, 5])
  w = try f32(5, [2, 4, 5, 2])
  table = try f32(6, [6, 3, 5])
  ids = try Tensor.from_list([1, 0, 1].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, REF)
  rows = try Tensor.from_list([5, 0, 3].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, REF)
  for backend in try engines()
    try agree("sum over the last axis of a view", backend, [a]) do |x| Tensor.sum(try heads(x[0]), some([3])) end

    try agree("max over the last axis of a view", backend, [a]) do |x| Tensor.max(try heads(x[0]), some([3])) end

    try agree("sum over the leading axes of a view", backend, [a]) do |x| Tensor.sum(try heads(x[0]), some([0, 1])) end

    try agree("a batched product of a view", backend, [a, w]) do |x| Tensor.batch_matmul(try heads(x[0]), x[1]) end

    try agree("a product of a reshaped view", backend, [a, w]) do |x|
      Tensor.matmul(try Tensor.reshape(try heads(x[0]), [24, 5]), try Tensor.reshape(x[1], [5, 16]))
    end

    try agree("gather of the rows of a view", backend, [a, ids]) do |x| Tensor.gather(try heads(x[0]), x[1]) end

    try agree("gather from a table that was added", backend, [table, rows]) do |x|
      Tensor.gather(try Tensor.add(x[0], x[0]), x[1])
    end
  end

  ok()
end

def test_concat_norm_rope_and_attention_take_a_view() -> Result[Void]
  a = try f32(7, [1, 3, 2, 8])
  b = try f32(8, [1, 3, 2, 8])
  k = try f32(9, [1, 5, 1, 8])
  v = try f32(15, [1, 5, 1, 8])
  wide = try f32(13, [2, 3, 2, 4])
  scale = try f32(14, [4])
  at = try Tensor.from_list([0, 1, 2].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, REF)
  for backend in try engines()
    try agree("concat of two views", backend, [a, b]) do |x| Tensor.concat([try heads(x[0]), try heads(x[1])], 2) end

    try agree("an rms norm of a view", backend, [a]) do |x| Tensor.rms_norm(try heads(x[0]), 0.000001) end

    try agree("rope of a view", backend, [a, at]) do |x| Tensor.rope(try heads(x[0]), x[1], dims: some(4)) end

    try agree_end("a norm, its weight and a rope (ggml-vulkan fuses the three)", backend, [wide, scale, at]) do |x|
      normed = try Tensor.mul(try Tensor.rms_norm(x[0], 0.000001), try Tensor.broadcast_to(x[1], [2, 3, 2, 4]))
      Tensor.rope(try heads(normed), x[2], dims: some(4))
    end

    try agree("rope of a view, inverse", backend, [a, at]) do |x|
      Tensor.rope(try heads(x[0]), x[1], dims: some(4), inverse: true)
    end

    try agree("rope of a rope", backend, [a, at]) do |x|
      Tensor.rope(try Tensor.rope(try heads(x[0]), x[1], dims: some(4)), x[1], dims: some(4), inverse: true)
    end

    try agree("causal attention over views", backend, [a, k, v]) do |x|
      Tensor.attention(try heads(x[0]), try heads(x[1]), try heads(x[2]), none, none, true)
    end
  end

  ok()
end

def test_a_product_takes_a_view_of_one_head_over_a_batch() -> Result[Void]
  q = try f32(10, [2, 3, 2, 4])
  k = try f32(11, [2, 3, 1, 4])
  v = try f32(12, [2, 3, 1, 4])
  at = try Tensor.from_list([0, 1, 2].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, REF)
  for backend in try engines()
    try agree("attention over the rope of one key head", backend, [q, k, v, at]) do |x|
      query = try Tensor.rope(try heads(x[0]), x[3], dims: some(4))
      key = try Tensor.rope(try heads(x[1]), x[3], dims: some(4))
      Tensor.attention(query, key, try heads(x[2]), none, none, true)
    end
  end

  ok()
end

def test_slices_norms_unary_ops_and_a_cache_write_take_a_view() -> Result[Void]
  a = try f32(30, [2, 3, 4, 5])
  x = try f32(31, [1, 3, 2, 4])
  cache = try f32(32, [1, 2, 8, 4])
  at = try Tensor.from_list([5, 0, 3].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, REF)
  for backend in try engines()
    try agree("a slice of a view", backend, [a]) do |y| Tensor.slice(try heads(y[0]), [0, 1, 1, 0], [2, 3, 3, 4]) end

    try agree("a layer norm of a view", backend, [a]) do |y| Tensor.layer_norm(try heads(y[0]), 0.000001) end

    try agree("a unary op of a view", backend, [a]) do |y| Tensor.exp(try heads(y[0])) end

    try agree_run("rows of a cache write that are a view", backend, [cache, x, at]) do |y|
      Tensor.set_rows(y[0], try heads(y[1]), y[2])
    end
  end

  ok()
end

## A compiled function, run on `backend` and on the reference: the cache write runs in place on ggml.
def agree_run(
  what: String,
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Void]
  want = try run_once(REF, args, f)
  got = try run_once(backend, args, f)
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    if (got[i] - want[i]).abs > 0.000001
      return error("#{what} on #{backend}: value #{i} is #{got[i]}, expected #{want[i]}")
    end
  end

  ok()
end

def run_once(
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[List[Float]]
  moved: List[Tensor.Tensor] = []
  for t in args
    moved = moved.push(try Tensor.from_list(try Tensor.to_list(t), t.shape, t.dtype, backend))
  end

  compiled = try Trace.compile(moved.map { |t| Trace.spec_of(t) }) do |y| ok([try f.call(y)]) end
  out = try Trace.run(compiled, moved)
  Trace.close(compiled)
  Support.floats_of(out[0])
end

## Flash attention takes the heads as views and a mask of fills (0 and -1e9, as the padded mask of a decoder is):
## grouped heads, a batch, a width of 8 (ggml-vulkan has a kernel from there), a causal mask, an explicit one and one query
## (the only shape Vulkan takes with fp16 products, its default). `lower_flash_test.agc` tells which masks take this path.
def test_attention_reads_the_heads_as_views_and_a_mask_of_fills() -> Result[Void]
  q = try f32(20, [2, 5, 4, 8])
  k = try f32(21, [2, 9, 2, 8])
  v = try f32(22, [2, 9, 2, 8])
  long = try f32(24, [1, 70, 2, 8])
  one = try f32(23, [2, 1, 4, 8])
  last = try Tensor.from_list([Tensor.Values.Int(6)], [1], Tensor.Dtype.I32, REF)
  positions = try Tensor.from_list([4, 5, 6, 7, 8].map { |i| Tensor.Values.Int(i) }, [5], Tensor.Dtype.I32, REF)
  for backend in try engines()
    try agree_end("causal attention", backend, [q, k, v]) do |x|
      Tensor.attention(try heads(x[0]), try heads(x[1]), try heads(x[2]), none, none, true)
    end

    try agree_end("attention with a padded mask", backend, [q, k, v, positions]) do |x|
      Tensor.attention(try heads(x[0]), try heads(x[1]), try heads(x[2]), some(try fills_mask(x[3])), none, false)
    end

    try agree_end("attention of one query (a decode step)", backend, [one, k, v, last]) do |x|
      Tensor.attention(try heads(x[0]), try heads(x[1]), try heads(x[2]), some(try fills_mask(x[3])), none, false)
    end

    try agree_end("causal attention of 70 queries (the tiled kernel of the CPU)", backend, [long]) do |x|
      Tensor.attention(try heads(x[0]), try heads(x[0]), try heads(x[0]), none, none, true)
    end
  end

  ok()
end

## `[n, 9]` for `n` positions: -1e9 where the key is after the query's position, 0 elsewhere.
def fills_mask(positions: Tensor.Tensor) -> Result[Tensor.Tensor]
  backend = Tensor.backend(positions)
  n = positions.shape[0]
  grid = [n, 9]
  queries = try Tensor.broadcast_to(try Tensor.reshape(positions, [n, 1]), grid)
  keys = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, 9, 1, Tensor.Dtype.I32, backend), [1, 9]),
    grid,
  )

  blocked = try Tensor.full(grid, Tensor.Values.Float(-1000000000.0), Tensor.Dtype.F32, backend)
  Tensor.where(try Tensor.lt(queries, keys), blocked, try Tensor.zeros(grid, Tensor.Dtype.F32, backend))
end
