## One or two optimizer steps against values worked out by hand.

import Optim from "../optim"
import S from "support"

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

def test_a_gradient_per_parameter_is_required() -> Result[Void]
  p = try S.tensor([1.0, -1.0], [2])
  state = try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1)), [p])
  Test.assert(!Optim.step(state, [p], []).ok?, "a missing gradient is an error")
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
