## `Trace.eval` against the same `Tensor.*` calls run eagerly: for every op of
## §3.1 and two derived ones, over good and bad inputs, both answer an error or
## both answer the same shape, dtype and values. The checks a traced call
## shares with an eager one (§9) and the ones only the backend can make
## (element domains, index bounds) both show up here.

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

## One call: its name, its arguments and the function run both ways.
struct Case
  name: String
  args: List[Tensor.Tensor]
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
end

private

def test_traced_and_eager_calls_agree() -> Result[Void]
  cases = try all_cases()
  for c in cases
    try agree(c)
  end

  Test.assert(cases.size >= 40, "only #{cases.size} cases ran")
end

## Both runs fail, or both give one shape, dtype and the same values, and the
## shape and dtype the trace records on lifted inputs are the eager ones.
def agree(c: Case) -> Result[Void]
  eager = c.f.call(c.args)
  traced = Trace.eval(c.args, c.f)
  case eager
    error(_) -> Test.assert(!traced.ok?, "#{c.name}: eager failed, traced did not")
    ok(want) ->
      case traced
        error(e) -> error("#{c.name}: eager ran, traced failed with \"#{e.message}\"")
        ok(got) ->
          try Test.assert(Tensor.shape(got) == Tensor.shape(want), "#{c.name}: shape #{Tensor.shape(got)}")
          try Test.assert(Tensor.dtype(got) == Tensor.dtype(want), "#{c.name}: dtype")
          recorded = try c.f.call(c.args.map { |a| Trace.constant(a) })
          try Test.assert(Tensor.shape(recorded) == Tensor.shape(want), "#{c.name}: recorded shape")
          try Test.assert(Tensor.dtype(recorded) == Tensor.dtype(want), "#{c.name}: recorded dtype")
          Test.assert(Tensor.to_list(got) == Tensor.to_list(want), "#{c.name}: values differ")
      end
  end
end

def all_cases() -> Result[List[Case]]
  v = try Support.f32([4.0, 9.0, 0.25, 1.0, 2.0, 0.5], [2, 3])
  w = try Support.f32([1.0, -2.0, 0.5, 3.0, -1.0, 2.0], [2, 3])
  neg = try Support.f32([-1.0, 4.0], [2])
  m = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
  row = try Support.f32([1.0, 2.0, 3.0], [1, 3])
  b3 = try Support.f32([1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0], [2, 3, 2])
  b2 = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  sig = try Support.f32([1.0, 2.0, 0.0, -1.0, 3.0, 1.0, 2.0, 0.5], [2, 4])
  kernel = try Support.f32([1.0, -1.0, 0.5, 0.5, 2.0, 1.0, 0.0, 1.0], [2, 2, 2])
  ids = try Support.i64([2, 0], [2])
  far = try Support.i64([5], [1])
  ok([
    Case(name: "add", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.add(x[0], x[1]) }),
    Case(name: "add of two shapes", args: [v, m], f: ->(x: List[Tensor.Tensor]) { Tensor.add(x[0], x[1]) }),
    Case(name: "sub", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.sub(x[0], x[1]) }),
    Case(name: "mul", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.mul(x[0], x[1]) }),
    Case(name: "div", args: [w, v], f: ->(x: List[Tensor.Tensor]) { Tensor.div(x[0], x[1]) }),
    Case(name: "neg", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.neg(x[0]) }),
    Case(name: "exp", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.exp(x[0]) }),
    Case(name: "log", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.log(x[0]) }),
    Case(name: "log of a negative", args: [neg], f: ->(x: List[Tensor.Tensor]) { Tensor.log(x[0]) }),
    Case(name: "sqrt", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.sqrt(x[0]) }),
    Case(name: "sqrt of a negative", args: [neg], f: ->(x: List[Tensor.Tensor]) { Tensor.sqrt(x[0]) }),
    Case(name: "pow", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.pow(x[0], x[1]) }),
    Case(name: "relu", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.relu(x[0]) }),
    Case(name: "sigmoid", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.sigmoid(x[0]) }),
    Case(name: "tanh", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.tanh(x[0]) }),
    Case(name: "gelu", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.gelu_tanh_approx(x[0]) }),
    Case(name: "gelu_erf", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.gelu_erf(x[0]) }),
    Case(name: "eq", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.eq(x[0], x[1]) }),
    Case(name: "lt", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.lt(x[0], x[1]) }),
    Case(
      name: "where",
      args: [v, w],
      f: ->(x: List[Tensor.Tensor]) { Tensor.where(try Tensor.lt(x[0], x[1]), x[0], x[1]) },
    ),
    Case(name: "clamp", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.clamp(x[0], -1.0, 1.0) }),
    Case(name: "clamp of an empty range", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.clamp(x[0], 1.0, -1.0) }),
    Case(name: "sum", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.sum(x[0], some([1]), true) }),
    Case(name: "mean", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.mean(x[0], some([0])) }),
    Case(name: "max", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.max(x[0], some([1])) }),
    Case(name: "argmax", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.argmax(x[0], some([1]), true) }),
    Case(name: "sum over a missing axis", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.sum(x[0], some([2])) }),
    Case(name: "matmul", args: [v, m], f: ->(x: List[Tensor.Tensor]) { Tensor.matmul(x[0], x[1]) }),
    Case(
      name: "matmul of a bad inner size",
      args: [v, w],
      f: ->(x: List[Tensor.Tensor]) { Tensor.matmul(x[0], x[1]) },
    ),
    Case(name: "batch_matmul", args: [b3, b2], f: ->(x: List[Tensor.Tensor]) { Tensor.batch_matmul(x[0], x[0]) }),
    Case(name: "batch_matmul 2", args: [b2, b2], f: ->(x: List[Tensor.Tensor]) { Tensor.batch_matmul(x[0], x[1]) }),
    Case(name: "transpose", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.transpose(x[0]) }),
    Case(name: "reshape", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.reshape(x[0], [3, 2]) }),
    Case(name: "reshape to another volume", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.reshape(x[0], [4]) }),
    Case(name: "squeeze", args: [row], f: ->(x: List[Tensor.Tensor]) { Tensor.squeeze(x[0], some(0)) }),
    Case(name: "squeeze of a wide axis", args: [row], f: ->(x: List[Tensor.Tensor]) { Tensor.squeeze(x[0], some(1)) }),
    Case(name: "unsqueeze", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.unsqueeze(x[0], 2) }),
    Case(name: "slice", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.slice(x[0], [0, 1], [2, 3]) }),
    Case(name: "slice past the end", args: [v], f: ->(x: List[Tensor.Tensor]) { Tensor.slice(x[0], [0, 1], [2, 4]) }),
    Case(name: "concat", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.concat([x[0], x[1], x[0]], 1) }),
    Case(name: "stack", args: [v, w], f: ->(x: List[Tensor.Tensor]) { Tensor.stack([x[0], x[1]], 1) }),
    Case(name: "broadcast_to", args: [row], f: ->(x: List[Tensor.Tensor]) { Tensor.broadcast_to(x[0], [2, 2, 3]) }),
    Case(
      name: "broadcast_to a narrower shape",
      args: [v],
      f: ->(x: List[Tensor.Tensor]) { Tensor.broadcast_to(x[0], [3]) },
    ),
    Case(name: "gather", args: [m, ids], f: ->(x: List[Tensor.Tensor]) { Tensor.gather(x[0], x[1]) }),
    Case(name: "gather out of range", args: [m, far], f: ->(x: List[Tensor.Tensor]) { Tensor.gather(x[0], x[1]) }),
    Case(name: "scatter_add", args: [ids, w], f: ->(x: List[Tensor.Tensor]) { Tensor.scatter_add(x[0], x[1], 3) }),
    Case(
      name: "scatter_add out of range",
      args: [far, row],
      f: ->(x: List[Tensor.Tensor]) { Tensor.scatter_add(x[0], x[1], 3) },
    ),
    Case(name: "to_dtype", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.to_dtype(x[0], Tensor.Dtype.I32) }),
    Case(name: "to_device", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.to_device(x[0], Tensor.Device.Cpu) }),
    Case(name: "copy", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.copy(x[0]) }),
    Case(name: "contiguous", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.contiguous(x[0]) }),
    Case(name: "conv1d", args: [sig, kernel], f: ->(x: List[Tensor.Tensor]) { Tensor.conv1d(x[0], x[1], 2, 1) }),
    Case(name: "avg_pool1d", args: [sig], f: ->(x: List[Tensor.Tensor]) { Tensor.avg_pool1d(x[0], 2, 1, 1) }),
    Case(
      name: "arange",
      args: [w],
      f: ->(x: List[Tensor.Tensor]) { Tensor.arange(7, 0, -2, Tensor.Dtype.I64, Tensor.backend(x[0])) },
    ),
    Case(
      name: "full",
      args: [w],
      f: ->(x: List[Tensor.Tensor]) {
        Tensor.full([2], Tensor.Values.Float(1.5), Tensor.Dtype.F32, Tensor.backend(x[0]))
      },
    ),
    Case(name: "stop_grad", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.stop_grad(x[0]) }),
    Case(name: "softmax", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.softmax(x[0], 1) }),
    Case(name: "layer_norm", args: [w], f: ->(x: List[Tensor.Tensor]) { Tensor.layer_norm(x[0], 0.00001) }),
  ])
end
