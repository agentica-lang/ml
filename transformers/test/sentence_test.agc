## The sentence-embedding pieces (`sentence.agc`): the mean over the kept tokens and the unit length by hand, and the
## cosine loss with its gradients against PyTorch (`nn/test/cosine_golden.py`, torch 2.6.0+cpu in `float64`, whose
## printed values are the constants below), on the reference and, traced, on ggml.

import S from "support"
import Sentence from "../sentence"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

ROWS_A = [0.3, -1.2, 2.5, 0.7, -0.4, 1.1, 0.2, -0.9, 1.5, 0.1, -0.6, 0.8]
ROWS_B = [1.0, 0.5, -0.3, 0.2, -0.2, 0.9, 0.4, -1.3, -0.7, 0.6, 1.2, 0.1]
SCORES = [0.2, 0.9, 0.4]
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

private

def f64(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F64, S.REF)
end

def loss(ps: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = Sentence.cosine_similarity_loss(ps[0], ps[1], ps[2])

def test_the_cosine_loss_and_its_gradients_are_pytorchs() -> Result[Void]
  args = [try f64(ROWS_A, [3, 4]), try f64(ROWS_B, [3, 4]), try f64(SCORES, [3])]
  both = try Trace.value_and_grad(args, &loss)
  try Test.eq(Tensor.shape(both.value), [])
  try S.expect_close(try S.floats_of(both.value), [LOSS], TOL, 0.0, "loss")
  try S.expect_close(try S.floats_of(both.grads[0]), GRAD_A, TOL, 0.0, "gradient of a")
  try S.expect_close(try S.floats_of(both.grads[1]), GRAD_B, TOL, 0.0, "gradient of b")
  eager = try Sentence.cosine_similarity_loss(args[0], args[1], args[2])
  try S.expect_close(try S.floats_of(eager), [LOSS], TOL, 0.0, "eager loss")
  Test.error?(Sentence.cosine_similarity_loss(args[0], args[1], try f64([0.2, 0.9], [2])))
end

def test_the_traced_cosine_loss_runs_on_ggml() -> Result[Void]
  for backend in try S.ggml_backends()
    args = [try S.f32(ROWS_A, [3, 4], backend), try S.f32(ROWS_B, [3, 4], backend), try S.f32(SCORES, [3], backend)]

    both = try Trace.value_and_grad(args, &loss)
    try S.expect_close(try S.floats_of(both.value), [LOSS], F32_TOL, 0.0, "loss on #{backend}")
    try S.expect_close(try S.floats_of(both.grads[0]), GRAD_A, F32_TOL, 0.0, "gradient of a on #{backend}")
    try S.expect_close(try S.floats_of(both.grads[1]), GRAD_B, F32_TOL, 0.0, "gradient of b on #{backend}")
  end

  ok()
end

## Two sentences of three positions and width 2; the second has one position of padding, which the mean leaves out.
def test_the_mean_leaves_the_padding_out() -> Result[Void]
  hidden = try f64([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, -1.0, 3.0, 5.0, 100.0, 100.0], [2, 3, 2])
  mask = try f64([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [2, 3])
  mean = try Sentence.mean_pooled(hidden, mask)
  try Test.eq(Tensor.shape(mean), [2, 2])
  try S.expect_close(try S.floats_of(mean), [3.0, 4.0, 2.0, 2.0], TOL, 0.0, "mean")
  Test.error?(Sentence.mean_pooled(hidden, try f64([1.0, 1.0], [2, 1])))
end

def test_normalized_rows_have_length_one_and_zeros_stay_zeros() -> Result[Void]
  unit = try Sentence.normalized(try f64([3.0, 4.0, 0.0, 0.0, -1.0, 0.0], [3, 2]))
  try S.expect_close(try S.floats_of(unit), [0.6, 0.8, 0.0, 0.0, -1.0, 0.0], TOL, 0.0, "unit")
  Test.error?(Sentence.normalized(try f64([3.0, 4.0], [2])))
end
