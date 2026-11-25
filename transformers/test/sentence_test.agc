## The sentence-embedding pieces (`sentence.agc`): the mean over the kept tokens and the unit length by hand, and the
## cosine loss with a gradient against PyTorch (`nn/test/cosine_golden.py`, torch 2.6.0+cpu in `float64`, whose
## printed values are the constants below; `nn/test/cosine_test.agc` covers the cosine on ggml).

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
TOL = 0.000000000001

private

def f64(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F64, S.REF)
end

def loss(ps: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = Sentence.cosine_similarity_loss(ps[0], ps[1], ps[2])

def test_the_cosine_loss_and_a_gradient_are_pytorchs() -> Result[Void]
  args = [try f64(ROWS_A, [3, 4]), try f64(ROWS_B, [3, 4]), try f64(SCORES, [3])]
  both = try Trace.value_and_grad(args, &loss)
  try Test.eq(Tensor.shape(both.value), [])
  try S.expect_close(try S.floats_of(both.value), [LOSS], TOL, 0.0, "loss")
  try S.expect_close(try S.floats_of(both.grads[0]), GRAD_A, TOL, 0.0, "gradient of a")
  Test.error?(Sentence.cosine_similarity_loss(args[0], args[1], try f64([0.2, 0.9], [2])))
end

## Two sentences of three positions and width 2; the second has one position of padding, which the mean leaves out.
def test_the_mean_leaves_the_padding_out() -> Result[Void]
  hidden = try f64([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, -1.0, 3.0, 5.0, 100.0, 100.0], [2, 3, 2])
  mask = try f64([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [2, 3])
  mean = try Sentence.mean_pooled(hidden, mask)
  try Test.eq(Tensor.shape(mean), [2, 2])
  try S.expect_close(try S.floats_of(mean), [3.0, 4.0, 2.0, 2.0], TOL, 0.0, "mean")
  empty = try Sentence.mean_pooled(hidden, try f64([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [2, 3]))
  try S.expect_close(try S.floats_of(empty), [3.0, 4.0, 0.0, 0.0], TOL, 0.0, "a row with no token")
  Test.error?(Sentence.mean_pooled(hidden, try f64([1.0, 1.0], [2, 1])))
end

def test_normalized_rows_have_length_one_and_zeros_stay_zeros() -> Result[Void]
  unit = try Sentence.normalized(try f64([3.0, 4.0, 0.0, 0.0, -1.0, 0.0], [3, 2]))
  try S.expect_close(try S.floats_of(unit), [0.6, 0.8, 0.0, 0.0, -1.0, 0.0], TOL, 0.0, "unit")
  Test.error?(Sentence.normalized(try f64([3.0, 4.0], [2])))
end

def unit_sum(ps: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = Tensor.sum(try Sentence.normalized(ps[0]))

## The gradient of the sum of the unit row of `[3, 4]`: `(1 - x_i 7 / 25) / 5`.
def test_normalized_has_a_gradient_under_a_trace() -> Result[Void]
  both = try Trace.value_and_grad([try f64([3.0, 4.0], [1, 2])], &unit_sum)
  try S.expect_close(try S.floats_of(both.value), [1.4], TOL, 0.0, "sum of the unit row")
  S.expect_close(try S.floats_of(both.grads[0]), [0.032, -0.024], TOL, 0.0, "gradient")
end
