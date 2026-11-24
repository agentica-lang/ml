## The recording side of tracing (`tensor/DESIGN.md` §10.2): the `Trace` arms
## of `../tensor.agc` call these after the wrapper's checks, so a traced op is
## checked exactly as an eager one and only records a node.
##
## A node's `id` is a `Ref`: unique within a run and ordered by creation, so
## an argument always has a smaller id than its user: sorting by id is a
## topological order, and two uses of one value are one node.

import Shapes from "torch_shapes"
import Types from "types"

## The id no node has, for a tensor that is not traced.
UNTRACED: Ref = Ref.new()

## A traced tensor of `shape` and `dtype` whose node computes `op` over `args`.
def record(op: Types.Op, args: List[Types.Tensor], shape: List[Int], dtype: Types.Dtype) -> Types.Tensor
  node = Types.Node(id: Ref.new(), op: op, args: args)
  Types.Tensor(shape: shape, dtype: dtype, backend: Types.Backend.Trace, handle: Types.Handle.Traced(node))
end

## The id of a traced tensor's node; `UNTRACED` for a tensor that is not traced.
def id_of(t: Types.Tensor) -> Ref
  case t.handle
    Types.Handle.Traced(n) -> n.id
    _ -> UNTRACED
  end
end

## The node behind a traced tensor.
def node_of(t: Types.Tensor) -> Result[Types.Node]
  case t.handle
    Types.Handle.Traced(n) -> ok(n)
    _ -> error("a #{Types.backend_name(t.backend)} tensor is not traced: lift it with Trace.constant")
  end
end

## A traced fill of `shape` with `value`, which must fit `dtype`.
def fill(shape: List[Int], value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try Types.value_to_float(value, dtype)
  ok(record(Types.Op.Full(value), [], shape, dtype))
end

## A traced fill of `shape` with the storage float `x` (`0.0`, `1.0`) tagged
## for `dtype`.
def filled(shape: List[Int], x: Float, dtype: Types.Dtype) -> Result[Types.Tensor]
  fill(shape, try Types.float_to_value(x, dtype), dtype)
end

## A traced `sum` or `mean` over the ascending `axes` (every axis for `none`).
def reduction(sum: Bool, t: Types.Tensor, axes: Optional[List[Int]], keep_dims: Bool) -> Types.Tensor
  axs = axes.unwrap_or((0...t.shape.size).to_a)
  op = Types.Op.Mean(axs, keep_dims)
  if sum
    op = Types.Op.Sum(axs, keep_dims)
  end

  record(op, [t], Types.reduced_shape(t.shape, axs, keep_dims), t.dtype)
end

## A traced `reshape`: the volumes must match, since no backend sees the
## values to check it.
def reshape(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  have = try Types.volume(t.shape)
  return error("reshape needs #{have} elements for #{shape}, got #{n}") if n != have
  ok(record(Types.Op.Reshape, [t], shape, t.dtype))
end

## A traced op over `args` whose result has the shape and dtype of `like`.
def same(op: Types.Op, args: List[Types.Tensor], like: Types.Tensor) -> Result[Types.Tensor]
  ok(record(op, args, like.shape, like.dtype))
end

## A traced `max` or `argmax` (`index`) over the ascending `axes`.
def reduced(index: Bool, t: Types.Tensor, axes: Optional[List[Int]], keep_dims: Bool) -> Types.Tensor
  axs = axes.unwrap_or((0...t.shape.size).to_a)
  shape = Types.reduced_shape(t.shape, axs, keep_dims)
  return record(Types.Op.Argmax(axs, keep_dims), [t], shape, Types.Dtype.I64) if index
  record(Types.Op.Max(axs, keep_dims), [t], shape, t.dtype)
end

## A traced `arange`: the count of `start, start + step, ...` before `stop`.
def arange(start: Int, stop: Int, step: Int, dtype: Types.Dtype) -> Types.Tensor
  span = stop - start
  if step < 0
    span = start - stop
  end

  count = 0
  if span > 0
    count = (span + step.abs - 1) / step.abs
  end

  record(Types.Op.Arange(start, stop, step), [], [count], dtype)
end

## A traced `squeeze`: without every size-1 axis, or without `axis`.
def squeeze(t: Types.Tensor, axis: Optional[Int]) -> Types.Tensor
  axes: List[Int] = []
  case axis
    some(ax) -> axes = [ax]
    none -> nil
  end

  record(Types.Op.Squeeze(axis), [t], Shapes.squeezed_shape(t.shape, axes), t.dtype)
end

## A traced `unsqueeze`: a `1` at `axis`.
def unsqueeze(t: Types.Tensor, axis: Int) -> Types.Tensor
  record(Types.Op.Unsqueeze(axis), [t], Shapes.unsqueezed_shape(t.shape, axis), t.dtype)
end

## A traced `slice`: `ends[i] - starts[i]` along each axis.
def slice(t: Types.Tensor, starts: List[Int], ends: List[Int]) -> Types.Tensor
  shape = (0...starts.size).to_a.map { |i| ends[i] - starts[i] }
  record(Types.Op.Slice(starts, ends), [t], shape, t.dtype)
end

## A traced `concat` along `axis`.
def concat(tensors: List[Types.Tensor], axis: Int) -> Types.Tensor
  shape = tensors[0].shape
  for t in tensors.drop(1)
    shape = Shapes.concatenated_shape(shape, t.shape, axis)
  end

  record(Types.Op.Concat(axis), tensors, shape, tensors[0].dtype)
end

## A traced `stack` along a new `axis`.
def stack(tensors: List[Types.Tensor], axis: Int) -> Types.Tensor
  first = tensors[0]
  shape = first.shape.take(axis) + [tensors.size] + first.shape.drop(axis)
  record(Types.Op.Stack(axis), tensors, shape, first.dtype)
end

## A traced `batch_matmul`: `[B..., m, k] . [B..., k, n]` is `[B..., m, n]`.
def batch_matmul(a: Types.Tensor, b: Types.Tensor) -> Types.Tensor
  rank = a.shape.size
  record(Types.Op.BatchMatmul, [a, b], a.shape.take(rank - 1).push(b.shape[rank - 1]), a.dtype)
end

## The dtype of the rows a traced `gather` reads: a half table (`F16`, held
## only by ggml) answers `F32` rows, as ggml widens them.
def rows_dtype(table: Types.Dtype) -> Types.Dtype
  return Types.Dtype.F32 if table == Types.Dtype.F16
  table
end

## A traced `gather` of the `indices` rows of `source`.
def gather(source: Types.Tensor, indices: Types.Tensor) -> Types.Tensor
  shape = Shapes.gathered_shape(source.shape, indices.shape)
  record(Types.Op.Gather, [source, indices], shape, rows_dtype(source.dtype))
end

## A traced `scatter_add` of the `updates` rows into `rows` rows, at `indices`.
def scatter_add(indices: Types.Tensor, updates: Types.Tensor, rows: Int) -> Types.Tensor
  record(
    Types.Op.ScatterAdd(rows),
    [indices, updates],
    Shapes.scattered_shape(rows, indices.shape, updates.shape),
    updates.dtype,
  )
end

## A traced `conv1d` of `input` by `weight` (`[C_out, C_in, K]`).
def conv1d(input: Types.Tensor, weight: Types.Tensor, stride: Int, padding: Int) -> Types.Tensor
  shape = Shapes.conv1d_output_shape(input.shape, weight.shape[0], weight.shape[2], stride, padding)
  record(Types.Op.Conv1d(stride, padding), [input, weight], shape, input.dtype)
end

## A traced `avg_pool1d`.
def avg_pool1d(input: Types.Tensor, kernel: Int, stride: Int, padding: Int) -> Types.Tensor
  shape = Shapes.avg_pool1d_output_shape(input.shape, kernel, stride, padding)
  record(Types.Op.AvgPool1d(kernel, stride, padding), [input], shape, input.dtype)
end

## A traced `permute`: the axes of `t` in the order `axes` names.
def permute(t: Types.Tensor, axes: List[Int]) -> Types.Tensor
  record(Types.Op.Permute(axes), [t], Types.permuted_shape(t.shape, axes), t.dtype)
end

## A traced `attention` over `[batch, heads, q_len, value_dim]`; the additive
## `mask`, when there is one, is the fourth argument.
def attention(
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  mask: Optional[Types.Tensor],
  scale: Float,
) -> Types.Tensor
  args = [q, k, v]
  case mask
    some(m) -> args = args.push(m)
    none -> nil
  end

  record(Types.Op.Attention(scale), args, q.shape.take(3).push(v.shape[3]), q.dtype)
end

## A traced `rope` of `x`, whose shape it keeps.
def rope(x: Types.Tensor, positions: Types.Tensor, base: Float, dims: Int, neox: Bool, inverse: Bool) -> Types.Tensor
  record(Types.Op.Rope(base, dims, neox, inverse), [x, positions], x.shape, x.dtype)
end
