## One or two optimizer steps against values worked out by hand.

import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"

TOL = 0.000000000001

private

def test_sgd_moves_against_the_gradient() -> Result[Void]
  got = try run(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1)), [[0.5, -1.0]])
  S.expect_close(got, [0.95, -0.9], TOL, 0.0, "sgd")
end

## `v1 = g1`, `v2 = 0.9 v1 + g2`: `p2 = p0 - 0.1 (g1 + 0.9 g1 + g2)`.
def test_sgd_momentum_accumulates_a_velocity() -> Result[Void]
  got = try run(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1, momentum: 0.9)), [[0.5, -1.0], [1.0, 0.0]])
  S.expect_close(got, [1.0 - 0.1 * (0.5 + 0.45 + 1.0), -1.0 - 0.1 * (-1.0 - 0.9 - 0.0)], TOL, 0.0, "momentum")
end

## With the same gradient twice, the bias-corrected moments are `g` and `g^2`
## on both steps, so each step is `lr g / (|g| + eps)`: with `eps = 0.5`,
## `0.05` for `g = 0.5` and `-0.1 / 3` for `g = -1`.
def test_adam_corrects_the_bias_of_its_moments() -> Result[Void]
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.1, eps: 0.5))
  once = try run(adam, [[0.5, -1.0]])
  try S.expect_close(once, [0.95, -1.0 + 0.1 / 1.5], TOL, 0.0, "adam step 1")
  twice = try run(adam, [[0.5, -1.0], [0.5, -1.0]])
  S.expect_close(twice, [0.9, -1.0 + 0.2 / 1.5], TOL, 0.0, "adam step 2")
end

## The same holds at any step, which pins `beta^t` at `t = 3` and `t = 5`.
def test_adam_keeps_the_correction_on_later_steps() -> Result[Void]
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.1, eps: 0.5))
  for n in [3, 5]
    got = try run(adam, (0...n).to_a.map { |_| [0.5, -1.0] })
    try S.expect_close(got, [1.0 - 0.05 * n.to_f, -1.0 + (0.1 / 1.5) * n.to_f], TOL, 0.0, "adam step #{n}")
  end

  ok()
end

## AdamW shrinks the parameter by `lr weight_decay` first, then takes Adam's step: `[1, -1] (1 - 0.05)`, then the
## `0.05` and `-0.1 / 1.5` of the test above. A parameter with a false flag does not decay, which is Adam; the
## inputs of the step are two rows of 7 numbers, the second without decay.
def test_adamw_decays_the_parameter_then_takes_adams_step() -> Result[Void]
  o = Optim.AdamW(lr: 0.1, eps: 0.5, weight_decay: 0.5)
  once = try run(Optim.Optimizer.AdamW(o), [[0.5, -1.0]])
  try S.expect_close(once, [0.95 - 0.05, -0.95 + 0.1 / 1.5], TOL, 0.0, "adamw step 1")
  kept = try run(Optim.Optimizer.AdamW(o.with(decay: [false])), [[0.5, -1.0]])
  try S.expect_close(kept, [0.95, -1.0 + 0.1 / 1.5], TOL, 0.0, "adamw without decay")
  p = try S.tensor([1.0, -1.0], [2])
  g = try S.tensor([0.5, -1.0], [2])
  state = try Optim.init(Optim.Optimizer.AdamW(o), [p])
  held = Optim.step_inputs(state)
  try Test.eq(held.size, 2)
  try S.expect_close([held[0][4], held[1][4]], [0.5, 0.0], TOL, 0.0, "the decay of each row")
  rows = try tensors(held)
  given = try Optim.step(state, [p], [g], rows)
  try S.expect_close(try S.floats_of(given.params[0]), once, TOL, 0.0, "adamw with rows")
  try Test.error?(Optim.step(state, [p], [g], rows.take(1)))
  try Test.error?(Optim.init(Optim.Optimizer.AdamW(o.with(decay: [true, false])), [p]))
  flagged = try Optim.init(Optim.Optimizer.AdamW(o.with(decay: [true])), [p])
  Test.error?(Optim.step(flagged, [p, p], [g, g]))
end

def test_a_gradient_per_parameter_is_required() -> Result[Void]
  p = try S.tensor([1.0, -1.0], [2])
  state = try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1)), [p])
  Test.assert(!Optim.step(state, [p], []).ok?, "a missing gradient is an error")
end

## The inputs of a step carry what it would hold as constants, so the step is the same; a wrong number or shape is an
## error. SGD takes its learning rate as a rank-0 tensor, Adam one row of 7 numbers.
def test_the_inputs_of_a_step_change_nothing() -> Result[Void]
  p = try S.tensor([1.0, -1.0], [2])
  g = try S.tensor([0.5, -1.0], [2])
  state = try Optim.init(Optim.Optimizer.Adam(Optim.Adam(lr: 0.1, eps: 0.5)), [p])
  held = Optim.step_inputs(state)
  try Test.eq(held.size, 1)
  try Test.eq(held[0].size, 7)
  rows = try tensors(held)
  given = try Optim.step(state, [p], [g], rows)
  baked = try Optim.step(state, [p], [g])
  try S.expect_close(try S.floats_of(given.params[0]), try S.floats_of(baked.params[0]), TOL, 0.0, "adam")
  try Test.assert(!Optim.step(state, [p], [g], [try S.tensor([0.1], [])]).ok?, "an Adam input has shape [7]")
  try Test.assert(!Optim.step(state, [p], [g], rows + rows).ok?, "an Adam step takes one input")
  sgd = try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1, momentum: 0.9)), [p])
  rate = Optim.step_inputs(sgd)
  try Test.eq(rate, [[0.1]])
  given = try Optim.step(sgd, [p], [g], try tensors(rate))
  baked = try Optim.step(sgd, [p], [g])
  try S.expect_close(try S.floats_of(given.params[0]), try S.floats_of(baked.params[0]), TOL, 0.0, "sgd")
  try Test.assert(!Optim.step(sgd, [p], [g], [try S.tensor([0.1], [1])]).ok?, "an SGD rate is rank 0")
  twice = try tensors(rate + rate)
  Test.assert(!Optim.step(sgd, [p], [g], twice).ok?, "an SGD step takes one input")
end

## The tensors of `step_inputs`: a rank-0 one for one number, else a rank-1 one.
def tensors(rows: List[List[Float]]) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for numbers in rows
    out = out.push(try S.tensor(numbers, Optim.input_shape(numbers)))
  end

  ok(out)
end

## The parameter `[1, -1]` after one step per gradient in `grads`.
def run(optimizer: Optim.Optimizer, grads: List[List[Float]]) -> Result[List[Float]]
  params = [try S.tensor([1.0, -1.0], [2])]
  state = try Optim.init(optimizer, params)
  for g in grads
    done = try Optim.step(state, params, [try S.tensor(g, [2])])
    state = done.state
    params = done.params
  end

  S.floats_of(params[0])
end
