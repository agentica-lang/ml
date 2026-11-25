## `conv2d`, `max_pool2d` and `avg_pool2d` against `conv2d_peer.py`, which writes nine cases with plain loops in doubles:
## the answer, and the gradient of `sum(out * ct)` for the input and the weight. The reference (in doubles), libtorch and
## ggml (CPU, and Vulkan when it is there) each run the forward pass as a trace, and the reference and libtorch also
## run it eagerly; every backend's gradient comes from the backward rules of `trace.agc`. Run from `tensor/`.

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

## One backend under test, the dtype it computes in and the gap it may have from the peer's doubles.
struct Run
  backend: Tensor.Backend
  dtype: Tensor.Dtype
  tolerance: Float
  eager: Bool
end

## One case of the peer: the call, its inputs (`w` is empty for a pool) and what the peer computed.
struct Case
  name: String
  kind: String
  stride: Int
  padding: Int
  groups: Int
  kernel: Int
  x_shape: List[Int]
  x: List[Float]
  w_shape: List[Int]
  w: List[Float]
  ct: List[Float]
  out_shape: List[Int]
  out: List[Float]
  dx: List[Float]
  dw: List[Float]
end

private

def test_every_backend_matches_the_peer_forward_and_backward() -> Result[Void]
  with_peer do |dir|
    cases = try read_cases(dir)
    runs = try backends()
    for r in runs
      for c in cases
        try check(r, c)
      end
    end

    Test.assert(cases.size == 9 and !runs.empty?, "ran #{cases.size} cases on #{runs.size} backends")
  end
end

def check(r: Run, c: Case) -> Result[Void]
  what = "#{c.name} on #{r.backend}"
  args = try inputs(r, c)
  if r.eager
    try close(try Support.floats_of(try apply(c, args)), c.out, r.tolerance, "#{what}, eager")
  end

  try close(
    try Support.floats_of(try Trace.eval(args, ->(a: List[Tensor.Tensor]) { apply(c, a) })),
    c.out,
    r.tolerance,
    "#{what}, traced",
  )

  ct = try tensor(r, c.out_shape, c.ct)
  step = try Trace.value_and_grad(args + [ct], ->(a: List[Tensor.Tensor]) { loss(c, a) })
  try close(try Support.floats_of(step.grads[0]), c.dx, r.tolerance, "#{what}, input gradient")
  if c.kind == "conv2d"
    try close(try Support.floats_of(step.grads[1]), c.dw, r.tolerance, "#{what}, weight gradient")
  end

  ok()
end

## The call of `c` on `a` (`[x, w]` for a convolution, `[x]` for a pool).
def apply(c: Case, a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  return Tensor.conv2d(a[0], a[1], c.stride, c.padding, c.groups) if c.kind == "conv2d"
  return Tensor.max_pool2d(a[0], c.kernel, c.stride, c.padding) if c.kind == "max_pool2d"
  Tensor.avg_pool2d(a[0], c.kernel, c.stride, c.padding)
end

## `sum(apply(c, a) * ct)`, with `ct` the last of `a`.
def loss(c: Case, a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  Tensor.sum(try Tensor.mul(try apply(c, a), a[a.size - 1]), none)
end

def inputs(r: Run, c: Case) -> Result[List[Tensor.Tensor]]
  x = try tensor(r, c.x_shape, c.x)
  return ok([x]) if c.kind != "conv2d"
  ok([x, try tensor(r, c.w_shape, c.w)])
end

def tensor(r: Run, shape: List[Int], values: List[Float]) -> Result[Tensor.Tensor]
  Tensor.from_list(values.map { |v: Float| Tensor.Values.Float(v) }, shape, r.dtype, r.backend)
end

def close(got: List[Float], want: List[Float], tolerance: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    return error("#{what}: value #{i} is #{got[i]}, the peer says #{want[i]}") if (got[i] - want[i]).abs > tolerance
  end

  ok()
end

## The backends this machine has: the reference, libtorch and ggml when they are built.
def backends() -> Result[List[Run]]
  runs = [Run(backend: Tensor.Backend.Reference, dtype: Tensor.Dtype.F64, tolerance: 0.000000001, eager: true)]
  torch = Tensor.Backend.Torch(Tensor.Device.Cpu)
  if Tensor.available(torch)
    runs = runs.push(Run(backend: torch, dtype: Tensor.Dtype.F32, tolerance: 0.0001, eager: true))
  end

  if try Support.ggml_present()
    for device in [Tensor.Device.Cpu, Tensor.Device.Vulkan(0)]
      b = Tensor.Backend.Ggml(device)
      if Tensor.available(b)
        runs = runs.push(Run(backend: b, dtype: Tensor.Dtype.F32, tolerance: 0.0001, eager: false))
      end
    end
  end

  ok(runs)
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", "test/conv2d_peer.py"])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("conv2d_peer.py exited with #{code}: #{out.stderr}") if code != 0
  dir = out.stdout.strip.split("\n").last
  return error("conv2d_peer.py printed no folder: #{dir}") if !dir.start_with?("/")
  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

def read_cases(dir: String) -> Result[List[Case]]
  out: List[Case] = []
  for item in try JSON.list(try JSON.get(try JSON.parse(try File.read_text("#{dir}/cases.json")), "cases"))
    out = out.push(try read_case(item))
  end

  ok(out)
end

def read_case(item: JSON.Value) -> Result[Case]
  kind = try JSON.string(try JSON.get(item, "kind"))
  conv = kind == "conv2d"
  w_shape: List[Int] = []
  w: List[Float] = []
  dw: List[Float] = []
  stride = try JSON.int(try JSON.get(item, "stride"))
  padding = try JSON.int(try JSON.get(item, "padding"))
  groups = 1
  kernel = 0
  if conv
    w_shape = try ints(try JSON.get(item, "w"))
    w = try floats(try JSON.get(item, "wv"))
    dw = try floats(try JSON.get(item, "dw"))
    groups = try JSON.int(try JSON.get(item, "groups"))
  else
    kernel = try JSON.int(try JSON.get(item, "kernel"))
  end

  ok(
    Case(
      name: try JSON.string(try JSON.get(item, "name")),
      kind: kind,
      stride: stride,
      padding: padding,
      groups: groups,
      kernel: kernel,
      x_shape: try ints(try JSON.get(item, "x")),
      x: try floats(try JSON.get(item, "xv")),
      w_shape: w_shape,
      w: w,
      ct: try floats(try JSON.get(item, "ct")),
      out_shape: try ints(try JSON.get(item, "out_shape")),
      out: try floats(try JSON.get(item, "out")),
      dx: try floats(try JSON.get(item, "dx")),
      dw: dw,
    ),
  )
end

def ints(v: JSON.Value) -> Result[List[Int]]
  out: List[Int] = []
  for item in try JSON.list(v)
    out = out.push(try JSON.int(item))
  end

  ok(out)
end

def floats(v: JSON.Value) -> Result[List[Float]]
  out: List[Float] = []
  for item in try JSON.list(v)
    out = out.push(try JSON.float(item))
  end

  ok(out)
end
