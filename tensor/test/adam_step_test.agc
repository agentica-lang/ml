## `Tensor.adam_step`, the update kernel of Adam: ggml runs it as one `ggml_opt_step_adamw` node, every other backend
## composes it. The numbers come from the formula in doubles (Python), at step 3 with a weight decay.

import Tensor from "../tensor"
import Trace from "../trace"
import S from "support"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

WEIGHT = [0.5, -0.25, 1.0, 2.0, 0.0, -1.5]
GRAD = [0.1, -0.2, 0.3, 0.0, 0.05, -0.4]
FIRST = [0.01, 0.02, -0.03, 0.0, 0.04, 0.05]
SECOND = [0.001, 0.002, 0.003, 0.0, 0.004, 0.005]

## lr 0.01, beta1 0.9, beta2 0.999, eps 1e-8, decay 0.1, then 1 / (1 - beta1^3) and 1 / (1 - beta2^3).
HYPER = [0.01, 0.9, 0.999, 0.00000001, 0.1, 3.690036900369005, 333.66688900003714]

WANT_P = [
  0.4982916794450591, -0.24966050432383302, 0.99889092453107, 1.998, -0.0013098134804852852, -1.498640679177104,
]
WANT_M = [0.019, -0.002, 0.003, 0.0, 0.041, 0.005]
WANT_V = [0.001009, 0.002038, 0.003087, 0.0, 0.0039985, 0.005155]

private

## The ggml backends of this machine: none without the module, the CPU one, and Vulkan when a device is there (without
## one the Vulkan cases below do not run and pass; with `TENSOR_REQUIRE_VULKAN` set a missing device is an error).
def ggml_backends() -> Result[List[Tensor.Backend]]
  return ok([]) if !try S.ggml_present()
  return ok([CPU, VULKAN]) if Tensor.available(VULKAN)
  flag = System.env("TENSOR_REQUIRE_VULKAN")
  return error("a Vulkan device is required (TENSOR_REQUIRE_VULKAN is set)") if flag.ok? and try flag != ""
  ok([CPU])
end

def tensors(backend: Tensor.Backend, dtype: Tensor.Dtype) -> Result[List[Tensor.Tensor]]
  rows = [WEIGHT, GRAD, FIRST, SECOND]
  out: List[Tensor.Tensor] = []
  for r in rows
    out = out.push(try Tensor.from_list(r.map { |x: Float| Tensor.Values.Float(x) }, [2, 3], dtype, backend))
  end

  ok(out.push(try Tensor.from_list(HYPER.map { |x: Float| Tensor.Values.Float(x) }, [7], dtype, backend)))
end

def step_on(backend: Tensor.Backend, dtype: Tensor.Dtype) -> Result[List[List[Float]]]
  args = try tensors(backend, dtype)
  outs = try Trace.eval_many(args) do |a| Tensor.adam_step(a[0], a[1], a[2], a[3], a[4]) end
  out: List[List[Float]] = []
  for t in outs
    out = out.push(try S.floats_of(t))
  end

  ok(out)
end

def close(got: List[Float], want: List[Float], tol: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if (got[i] - want[i]).abs > tol
  end

  ok()
end

def follows(backend: Tensor.Backend, dtype: Tensor.Dtype, tol: Float) -> Result[Void]
  got = try step_on(backend, dtype)
  try close(got[0], WANT_P, tol, "p on #{backend}")
  try close(got[1], WANT_M, tol, "m on #{backend}")
  close(got[2], WANT_V, tol, "v on #{backend}")
end

def test_the_composition_follows_the_formula() -> Result[Void]
  try follows(Tensor.Backend.Reference, Tensor.Dtype.F64, 0.0000000001)
  follows(Tensor.Backend.Reference, Tensor.Dtype.F32, 0.000001)
end

def test_ggml_runs_one_node_and_follows_the_formula() -> Result[Void]
  for backend in try ggml_backends()
    try follows(backend, Tensor.Dtype.F32, 0.000001)
  end

  ok()
end

## The kernel works on copies: the weight, the gradient and the moments it was given are what they were, so a state
## fed back (or a tensor another node reads) is not changed under its holder.
def test_the_arguments_are_not_changed() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    _ = try Trace.eval_many(args) do |a| Tensor.adam_step(a[0], a[1], a[2], a[3], a[4]) end
    for i in 0...4
      want = [WEIGHT, GRAD, FIRST, SECOND][i]
      try close(try S.floats_of(args[i]), want, 0.0000001, "argument #{i} on #{backend}")
    end
  end

  ok()
end

def test_a_bad_hyper_is_an_error() -> Result[Void]
  args = try tensors(Tensor.Backend.Reference, Tensor.Dtype.F32)
  short = try Tensor.from_list([Tensor.Values.Float(1.0)], [1], Tensor.Dtype.F32, Tensor.Backend.Reference)
  Test.assert(!Tensor.adam_step(args[0], args[1], args[2], args[3], short).ok?, "7 numbers are needed")
end

## A value that is not finite in the gradient is an error of the run, not a silent step: the three outputs of the step
## share one check.
def test_a_gradient_that_is_not_finite_is_an_error() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    got = Trace.eval_many(args) do |a| overflowing(a) end
    try Test.assert(!got.ok?, "an infinite gradient must fail on #{backend}")
  end

  ok()
end

## The step of a gradient that overflows `F32` in the trace: `exp(1000 g)`.
def overflowing(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  big = try Tensor.full(Tensor.shape(a[1]), Tensor.Values.Float(1000.0), Tensor.Dtype.F32, Tensor.backend(a[1]))
  Tensor.adam_step(a[0], try Tensor.exp(try Tensor.mul(a[1], big)), a[2], a[3], a[4])
end

## Rank 4 is past what the node takes (the stack needs one more axis): the composition runs there and agrees.
def test_a_rank_4_weight_composes() -> Result[Void]
  for backend in try ggml_backends()
    args: List[Tensor.Tensor] = []
    for r in [WEIGHT, GRAD, FIRST, SECOND]
      args = args.push(
        try Tensor.from_list(r.map { |x: Float| Tensor.Values.Float(x) }, [1, 2, 1, 3], Tensor.Dtype.F32, backend),
      )
    end

    hyper = try Tensor.from_list(HYPER.map { |x: Float| Tensor.Values.Float(x) }, [7], Tensor.Dtype.F32, backend)
    outs = try Trace.eval_many(args + [hyper]) do |a| Tensor.adam_step(a[0], a[1], a[2], a[3], a[4]) end
    try close(try S.floats_of(outs[0]), WANT_P, 0.000001, "p on #{backend}")
  end

  ok()
end

## Weights of 3 and 10 values: the moments are not at an aligned offset of anything (ggml-Vulkan binds each of them as a
## buffer, and aborts on a misaligned one), so each is a tensor of its own.
def test_small_odd_sizes_run_on_every_backend() -> Result[Void]
  for backend in try ggml_backends()
    for n in [3, 10]
      try odd_size(backend, n)
    end
  end

  ok()
end

def odd_size(backend: Tensor.Backend, n: Int) -> Result[Void]
  args: List[Tensor.Tensor] = []
  for r in [WEIGHT, GRAD, FIRST, SECOND]
    values = r.take(n).map { |x: Float| Tensor.Values.Float(x) }
    if n > r.size
      values = (0...n).map { |i| Tensor.Values.Float(r[i % r.size]) }
    end

    args = args.push(try Tensor.from_list(values, [n], Tensor.Dtype.F32, backend))
  end

  hyper = try Tensor.from_list(HYPER.map { |x: Float| Tensor.Values.Float(x) }, [7], Tensor.Dtype.F32, backend)
  got = try Trace.eval_many(args + [hyper]) do |a| Tensor.adam_step(a[0], a[1], a[2], a[3], a[4]) end
  want = try Trace.eval_many(try on_reference(args + [hyper])) do |a|
    Tensor.adam_step(a[0], a[1], a[2], a[3], a[4])
  end

  for i in 0...3
    try close(try S.floats_of(got[i]), try S.floats_of(want[i]), 0.000001, "part #{i} of #{n} on #{backend}")
  end

  ok()
end

## `tensors` as reference tensors with the same values.
def on_reference(tensors: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for t in tensors
    values = try Tensor.to_list(t)
    out = out.push(try Tensor.from_list(values, Tensor.shape(t), Tensor.Dtype.F32, Tensor.Backend.Reference))
  end

  ok(out)
end

## An argument that is the result of another node and is read by another output: the step works on a copy of it.
def test_a_weight_that_is_read_elsewhere_is_not_changed() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    got = try Trace.eval_many(args) do |a| doubled_and_stepped(a) end
    want = try Trace.eval_many(try on_reference(args)) do |a| doubled_and_stepped(a) end
    for i in 0...4
      try close(try S.floats_of(got[i]), try S.floats_of(want[i]), 0.000001, "output #{i} on #{backend}")
    end
  end

  ok()
end

## The new `[p, m, v]` of a weight that is twice the input, and that weight too.
def doubled_and_stepped(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  w = try Tensor.add(a[0], a[0])
  stepped = try Tensor.adam_step(w, a[1], a[2], a[3], a[4])
  ok(stepped + [w])
end

## In a compiled function the moments are updated where they lie: their results are the very inputs, which hold the new
## values after the run, and the weight is never written. A second run from the results follows the reference.
def test_a_compiled_step_updates_its_moments_in_place() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    specs = args.map { |t| Trace.spec_of(t) }
    program = try Trace.compile(specs) do |a| Tensor.adam_step(a[0], a[1], a[2], a[3], a[4]) end
    try Test.eq(Trace.donated(program), [false, true, true])
    outs = try Trace.run(program, args)
    try close(try S.floats_of(outs[0]), WANT_P, 0.000001, "p on #{backend}")
    try close(try S.floats_of(args[0]), WEIGHT, 0.0000001, "the weight on #{backend}")
    try close(try S.floats_of(args[2]), WANT_M, 0.000001, "the first moment, an input, on #{backend}")
    try close(try S.floats_of(args[3]), WANT_V, 0.000001, "the second moment, an input, on #{backend}")
    try close(try S.floats_of(outs[1]), WANT_M, 0.000001, "the first moment on #{backend}")
    Trace.close(program)
  end

  ok()
end

## When a moment is also read by another node, or its result is not an output, the step copies.
def test_a_moment_that_is_read_elsewhere_is_copied() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    specs = args.map { |t| Trace.spec_of(t) }
    program = try Trace.compile(specs) do |a| stepped_and_first(a) end
    try Test.eq(Trace.donated(program), [false, false, false, false])
    outs = try Trace.run(program, args)
    try close(try S.floats_of(args[2]), FIRST, 0.0000001, "the first moment on #{backend}")
    try close(try S.floats_of(outs[3]), FIRST, 0.0000001, "the first moment read elsewhere on #{backend}")
    Trace.close(program)
  end

  ok()
end

def stepped_and_first(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  stepped = try Tensor.adam_step(a[0], a[1], a[2], a[3], a[4])
  ok(stepped + [a[2]])
end

## One compiled graph run twice, into two sets of tensors: the outputs are the new weight (a view of the copy the step
## updates) and the moments, written where the caller holds them. Each set keeps its run, and the second run is the
## fresh result.
def test_a_graph_run_twice_into_two_targets_keeps_each_run() -> Result[Void]
  for backend in try ggml_backends()
    one = try tensors(backend, Tensor.Dtype.F32)
    two = try with_grad(one, [0.3, 0.1, -0.2, 0.4, -0.05, 0.2], backend)
    program = try Trace.compile(one.map { |t| Trace.spec_of(t) }) do |a| stepped_and_first(a) end
    a_set = try targets(one, backend)
    b_set = try targets(one, backend)
    _ = try Trace.run(program, one, a_set.map { |t| some(t) })
    _ = try Trace.run(program, two, b_set.map { |t| some(t) })
    want_one = try Trace.eval_many(try on_reference(one)) do |a| stepped_and_first(a) end
    want_two = try Trace.eval_many(try on_reference(two)) do |a| stepped_and_first(a) end
    fresh = try Trace.run(program, two)
    for i in 0...4
      try close(try S.floats_of(a_set[i]), try S.floats_of(want_one[i]), 0.000001, "run 1, output #{i} on #{backend}")
      try close(try S.floats_of(b_set[i]), try S.floats_of(want_two[i]), 0.000001, "run 2, output #{i} on #{backend}")
      try close(try S.floats_of(fresh[i]), try S.floats_of(b_set[i]), 0.0, "a fresh run, output #{i} on #{backend}")
    end

    Trace.close(program)
  end

  ok()
end

## `args` with another gradient.
def with_grad(args: List[Tensor.Tensor], grad: List[Float], backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  g = try Tensor.from_list(
    grad.map { |x: Float| Tensor.Values.Float(x) },
    Tensor.shape(args[1]),
    Tensor.Dtype.F32,
    backend,
  )

  ok([args[0], g, args[2], args[3], args[4]])
end

## Four zero tensors of the shape of the first four arguments.
def targets(args: List[Tensor.Tensor], backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for i in 0...4
    out = out.push(try Tensor.zeros(Tensor.shape(args[i]), Tensor.Dtype.F32, backend))
  end

  ok(out)
end

## The gradient of a parameter the loss does not read is a fill of one value, a view at an offset of the fill store
## (a second fill goes first here): the kernel binds its gradient as a buffer, which Vulkan aborts on when it is not
## aligned, so the lowering copies it.
def test_a_constant_gradient_of_one_value_runs_on_every_backend() -> Result[Void]
  for backend in try ggml_backends()
    args: List[Tensor.Tensor] = []
    for x in [0.5, 0.0, 0.0]
      args = args.push(try Tensor.from_list([Tensor.Values.Float(x)], [1], Tensor.Dtype.F32, backend))
    end

    hyper = try Tensor.from_list(HYPER.map { |x: Float| Tensor.Values.Float(x) }, [7], Tensor.Dtype.F32, backend)
    got = try Trace.eval_many(args + [hyper]) do |a| after_a_fill(a) end
    want = try Trace.eval_many(try on_reference(args + [hyper])) do |a| after_a_fill(a) end
    for i in 0...4
      try close(try S.floats_of(got[i]), try S.floats_of(want[i]), 0.000001, "output #{i} on #{backend}")
    end
  end

  ok()
end

## A fill of 3, then the step with a gradient that is a fill of 0.25.
def after_a_fill(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  three = try Tensor.full([1], Tensor.Values.Float(3.0), Tensor.Dtype.F32, Tensor.backend(a[0]))
  grad = try Tensor.full([1], Tensor.Values.Float(0.25), Tensor.Dtype.F32, Tensor.backend(a[0]))
  stepped = try Tensor.adam_step(a[0], grad, a[1], a[2], a[3])
  ok([three] + stepped)
end

## A gradient of 1e20 squares past `F32`: the second moment is infinite and the weight stays finite (its step is 0). The
## second moment is checked where it is written in place, so the run is an error and the weight does not freeze unseen.
def test_an_infinite_second_moment_is_an_error() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    big = try Tensor.from_list(
      [
        Tensor.Values.Float(1e20),
        Tensor.Values.Float(1e20),
        Tensor.Values.Float(1e20),
        Tensor.Values.Float(1e20),
        Tensor.Values.Float(1e20),
        Tensor.Values.Float(1e20),
      ],
      [2, 3],
      Tensor.Dtype.F32,
      backend,
    )

    args = [args[0], big, args[2], args[3], args[4]]
    program = try Trace.compile(args.map { |t| Trace.spec_of(t) }) do |a|
      Tensor.adam_step(a[0], a[1], a[2], a[3], a[4])
    end

    try Test.eq(Trace.donated(program), [false, true, true])
    try Test.assert(!Trace.run(program, args).ok?, "an infinite second moment must fail on #{backend}")
    Trace.close(program)
  end

  ok()
end

## A moment that is also the gradient is read twice, so the step copies it.
def test_a_moment_read_twice_is_copied() -> Result[Void]
  for backend in try ggml_backends()
    args = try tensors(backend, Tensor.Dtype.F32)
    program = try Trace.compile(args.map { |t| Trace.spec_of(t) }) do |a|
      Tensor.adam_step(a[0], a[2], a[2], a[3], a[4])
    end

    try Test.eq(Trace.donated(program), [false, false, false])
    _ = try Trace.run(program, args)
    try close(try S.floats_of(args[2]), FIRST, 0.0000001, "the moment on #{backend}")
    Trace.close(program)
  end

  ok()
end

## On a backend that computes, `adam_step` makes the ops of `adam_composed` one at a time (`adam_staged`): the same
## numbers to the last bit, at step 3 with a weight decay and at a later step of a schedule with a lower rate. The
## libtorch kernel is one pass that rounds in the last place differently (`torch_adam_test.agc`).
def test_the_eager_step_is_the_composition_bit_for_bit() -> Result[Void]
  later = [0.0025, 0.9, 0.999, 0.00000001, 0.05, 1.0000265613, 1.1123]
  backend = Tensor.Backend.Reference
  for hyper in [HYPER, later]
    args = try tensors(backend, Tensor.Dtype.F32)
    h = try Tensor.floats(hyper, [7], backend)
    eager = try Tensor.adam_step(args[0], args[1], args[2], args[3], h)
    composed = try Tensor.adam_composed(args[0], args[1], args[2], args[3], h)
    for i in 0...3
      try Test.eq(try S.floats_of(eager[i]), try S.floats_of(composed[i]))
    end
  end

  ok()
end
