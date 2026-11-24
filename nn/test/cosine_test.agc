## `Nn.cosine_similarity` against PyTorch (`cosine_golden.py`, torch 2.6.0+cpu in `float64`, whose printed values are
## the constants below): the cosines of the rows of `ROWS_A` and `ROWS_B`, and the gradients of the mean of
## `(cosine - score)^2` for the scores 0.2, 0.9 and 0.4, on the reference, against a central difference, and on
## libtorch and ggml in `F32`.

import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

ROWS_A = [0.3, -1.2, 2.5, 0.7, -0.4, 1.1, 0.2, -0.9, 1.5, 0.1, -0.6, 0.8]
ROWS_B = [1.0, 0.5, -0.3, 0.2, -0.2, 0.9, 0.4, -1.3, -0.7, 0.6, 1.2, 0.1]
SCORES = [0.2, 0.9, 0.4]
COSINES = [-0.26936991816748385, 0.9476099573653523, -0.5952712206103672]
LOSS = 0.4043798768987104
GRAD_A = [
  -0.09568339845836529, -0.034082240607375115, 0.002307243366782539, -0.025659682297567005, 0.0028264450812710306,
  -0.003235228460793705, 0.0024760593273944554, -0.0046601305265584324, -0.012115671049715782, -0.1575037236463056,
  -0.21808207889488473, -0.12115671049715815,
]
GRAD_B = [
  -0.08886701526564691, 0.08061124269942746, -0.213240564576254, -0.077053877284715, -0.0029577748527240266,
  0.004234995811854858, -0.001863014030611888, 0.002813727376130325, -0.24326160394603322, -0.12726722285266703,
  -0.06068370890988101, -0.21102338358765899,
]
TOL = 0.000000000001
F32_TOL = 0.000001
EPS = 0.000001
CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

def tensor(data: List[Float], shape: List[Int], dtype: Tensor.Dtype, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, dtype, backend)
end

## The mean of `(cosine - score)^2` over `ps`: `a`, `b` and the scores.
def loss(ps: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  gap = try Tensor.sub(try Nn.cosine_similarity(ps[0], ps[1]), ps[2])
  Tensor.mean(try Tensor.mul(gap, gap))
end

def args(dtype: Tensor.Dtype, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try tensor(ROWS_A, [3, 4], dtype, backend),
    try tensor(ROWS_B, [3, 4], dtype, backend),
    try tensor(SCORES, [3], dtype, backend),
  ])
end

def close(got: Tensor.Tensor, want: List[Float], tol: Float, what: String) -> Result[Void]
  values = try S.floats_of(got)
  try Test.eq(values.size, want.size)
  for i in 0...want.size
    try Test.assert((values[i] - want[i]).abs <= tol, "#{what}[#{i}]: #{values[i]}, expected #{want[i]}")
  end

  ok()
end

def test_the_cosine_and_its_gradients_are_pytorchs() -> Result[Void]
  xs = try args(Tensor.Dtype.F64, S.REF)
  try close(try Nn.cosine_similarity(xs[0], xs[1]), COSINES, TOL, "cosine")
  both = try Trace.value_and_grad(xs, &loss)
  try close(both.value, [LOSS], TOL, "loss")
  try close(both.grads[0], GRAD_A, TOL, "gradient of a")
  close(both.grads[1], GRAD_B, TOL, "gradient of b")
end

def test_the_gradient_matches_a_central_difference() -> Result[Void]
  xs = try args(Tensor.Dtype.F64, S.REF)
  analytic = try S.floats_of((try Trace.grad(xs, &loss))[0])
  for j in 0...ROWS_A.size
    up = try at(ROWS_A.set(j, ROWS_A[j] + EPS), xs)
    down = try at(ROWS_A.set(j, ROWS_A[j] - EPS), xs)
    slope = (up - down) / (2.0 * EPS)
    try Test.assert((slope - analytic[j]).abs <= 0.00001, "[#{j}]: slope #{slope}, gradient #{analytic[j]}")
  end

  ok()
end

def at(a: List[Float], xs: List[Tensor.Tensor]) -> Result[Float]
  moved = [try tensor(a, [3, 4], Tensor.Dtype.F64, S.REF), xs[1], xs[2]]
  ok((try S.floats_of(try Trace.eval(moved, &loss)))[0])
end

## A row of zeros has a cosine of 0 and a finite gradient; other shapes are errors.
def test_a_zero_row_has_cosine_zero_and_bad_shapes_are_errors() -> Result[Void]
  zero = try tensor([0.0, 0.0, 3.0, 4.0], [2, 2], Tensor.Dtype.F64, S.REF)
  other = try tensor([1.0, 2.0, 3.0, 4.0], [2, 2], Tensor.Dtype.F64, S.REF)
  try close(try Nn.cosine_similarity(zero, other), [0.0, 1.0], TOL, "cosine")
  scores = try tensor([0.5, 0.5], [2], Tensor.Dtype.F64, S.REF)
  grads = try Trace.grad([zero, other, scores], &loss)
  for g in try S.floats_of(grads[0])
    try Test.assert(g == g and g.abs < 1.0e12, "the gradient #{g} is finite")
  end

  try Test.error?(Nn.cosine_similarity(zero, try tensor([1.0, 2.0], [1, 2], Tensor.Dtype.F64, S.REF)))
  one = try tensor([1.0, 2.0], [2], Tensor.Dtype.F64, S.REF)
  Test.error?(Nn.cosine_similarity(one, one))
end

## The native backends of this machine; a missing ggml is an error when `TENSOR_REQUIRE_GGML` is set (CI sets it).
def native() -> Result[List[Tensor.Backend]]
  try Tensor.vulkan_fp16(false)
  if !Tensor.available(CPU)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
  end

  ok([S.TORCH, CPU, VULKAN].select { |b: Tensor.Backend| Tensor.available(b) })
end

def test_the_traced_cosine_runs_on_libtorch_and_ggml() -> Result[Void]
  for backend in try native()
    both = try Trace.value_and_grad(try args(Tensor.Dtype.F32, backend), &loss)
    try close(both.value, [LOSS], F32_TOL, "loss on #{backend}")
    try close(both.grads[0], GRAD_A, F32_TOL, "gradient of a on #{backend}")
    try close(both.grads[1], GRAD_B, F32_TOL, "gradient of b on #{backend}")
  end

  ok()
end
