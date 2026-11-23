## The recording side of tracing (`tensor/DESIGN.md` §10.2): the `Trace` arms
## of `../tensor.agc` call these after the wrapper's checks, so a traced op is
## checked exactly as an eager one and only records a node.
##
## A node's `id` is unique within a run and grows with creation, so an
## argument always has a smaller id than its user: sorting by id is a
## topological order, and two uses of one value are one node.

import Types from "types"

## The source of node ids: the one counter every trace of a run shares.
IDS: ATS[String, Int] = ATS.new()

## A traced tensor of `shape` and `dtype` whose node computes `op` over `args`.
def record(op: Types.Op, args: List[Types.Tensor], shape: List[Int], dtype: Types.Dtype) -> Types.Tensor
  node = Types.Node(id: IDS.add("next", 1), op: op, args: args)
  Types.Tensor(shape: shape, dtype: dtype, backend: Types.Backend.Trace, handle: Types.Handle.Traced(node))
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

## The answer of an op the trace cannot record yet.
def untraceable(op: String) -> Result[Types.Tensor] = error("#{op} cannot be traced yet")
