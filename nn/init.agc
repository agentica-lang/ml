## Deterministic initializers: the same seed gives the same `F32` values on
## every backend, so a model made on `Reference` and on `Torch` from one seed
## is the same model (`nn/DESIGN.md`).
##
## The generator is Park-Miller's minimal standard (`x = 48271 x mod 2^31-1`),
## run in Agentica and handed to the backend through `from_list`. It is for
## reproducible starting weights, not for statistics or secrets.

import Tensor from "pkg/tensor/tensor"

MODULUS = 2147483647
MULTIPLIER = 48271

## `count` floats in `-limit...limit` from `seed`.
def uniform_values(count: Int, limit: Float, seed: Int) -> List[Float]
  state = start(seed)
  out: List[Float] = []
  for _ in 0...count
    state = (state * MULTIPLIER) % MODULUS
    out = out.push(((2.0 * state.to_f) / MODULUS.to_f - 1.0) * limit)
  end

  out
end

## An `F32` tensor of `shape` drawn uniformly from `-limit...limit`.
def uniform(shape: List[Int], limit: Float, seed: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  return error("uniform needs limit >= 0, got #{limit}") if limit < 0.0
  count = shape.reduce(1) do |acc, d| acc * d end
  values = uniform_values(count, limit, seed).map do |x: Float| Tensor.Values.Float(x) end
  Tensor.from_list(values, shape, Tensor.Dtype.F32, backend)
end

## Glorot (Xavier) uniform for a `[fan_in, fan_out]` weight: the limit is
## `sqrt(6 / (fan_in + fan_out))`.
def glorot_uniform(shape: List[Int], seed: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  return error("glorot_uniform needs a rank-2 shape, got #{shape}") if shape.size != 2
  return error("glorot_uniform needs positive fans, got #{shape}") if shape[0] < 1 or shape[1] < 1
  uniform(shape, Math.sqrt(6.0 / (shape[0] + shape[1]).to_f), seed, backend)
end

## The seed of the `k`-th part of a layer made from `seed`: parts of one layer,
## and the same part of two layers with neighbouring seeds, draw apart.
def derive(seed: Int, k: Int) -> Int = (start(seed) * 131 + 7919 * (k + 1)) % (MODULUS - 1)

private

## A state in `1...MODULUS` for any seed. `seed.abs` makes `s` and `-s` the
## same seed, and neighbouring seeds start states `MULTIPLIER` apart, so their
## sequences step together linearly: fine for starting weights, nothing more.
def start(seed: Int) -> Int
  s = seed.abs % (MODULUS - 1) + 1
  (s * MULTIPLIER) % MODULUS
end
