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

## The ggml backends of this machine: none without the module, the CPU one, and Vulkan when a device is there.
def ggml_backends() -> Result[List[Tensor.Backend]]
  return ok([]) if !try S.ggml_present()
  return ok([CPU, VULKAN]) if Tensor.available(VULKAN)
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
