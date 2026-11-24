## `nn/` composes with `Trace`: the gradient of a scalar loss through `dense`
## (on a rank-3 input, so its fold and unfold are traced), a relu and a silu
## `mlp`, `embedding` with `positional`, and a gelu encoder block matches a
## central difference of the same loss.

import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

EPS = 0.000001
TOL = 0.00001
## `[2, 2, 3]`; every pre-activation of the relu `mlp` below stays 0.12 or
## more away from the kink, where a central difference says nothing.
X_DATA = [0.5, -1.0, 2.0, 0.3, 1.2, -0.7, -0.4, 0.9, 0.1, 1.5, -0.2, 0.6]

private

def test_the_gradient_through_dense_matches_a_central_difference() -> Result[Void]
  x = try f64(X_DATA, [2, 2, 3])
  args = [try f64([0.2, -0.4, 0.7, 0.1, -0.3, 0.6], [3, 2]), try f64([0.05, -0.15], [2])]
  check(args, ->(ps: List[Tensor.Tensor]) { dense_loss(ps, x) })
end

def test_the_gradient_through_a_relu_mlp_matches_a_central_difference() -> Result[Void]
  x = try f64(X_DATA, [2, 2, 3])
  args = [
    try f64([0.2, -0.4, 0.7, 0.1, -0.3, 0.6, 0.5, -0.2, 0.4, 0.3, -0.6, 0.8], [3, 4]),
    try f64([0.05, -0.15, 0.1, 0.02], [4]),
    try f64([0.3, -0.5, 0.2, 0.6, 0.1, -0.4, -0.2, 0.7, 0.5, 0.4, -0.3, 0.1], [4, 3]),
    try f64([0.1, 0.0, -0.1], [3]),
  ]

  check(args, ->(ps: List[Tensor.Tensor]) { mlp_loss(ps, x) })
end

def test_the_gradient_through_embedding_and_positional_matches_a_central_difference() -> Result[Void]
  ids = try Tensor.from_list([2, 0, 2].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I64, S.REF)
  args = [
    try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7, 0.4, 0.9], [4, 2]),
    try f64([0.1, -0.2, 0.3, 0.6, -0.4, 0.8, 0.2, 0.5], [4, 2]),
  ]

  check(
    args,
    ->(ps: List[Tensor.Tensor]) {
      rows = try Nn.embedding(Nn.Embedding(table: ps[0]), Trace.constant(ids))
      squared(try Nn.positional(Nn.Embedding(table: ps[1]), rows))
    },
  )
end

def test_the_gradient_through_a_gelu_encoder_block_matches_a_central_difference() -> Result[Void]
  args = [try filler([1, 2, 4], 0), try filler([4], 1)]
  for k in 2...8
    args = args.push(try filler([4, 4], k))
  end

  args = args.push(try filler([4], 9))
  args = args.push(try filler([4], 10))
  check(args, ->(ps: List[Tensor.Tensor]) { block_loss(ps) })
end

## The input of a decoder block and its encoder output (keys and values of the
## cross-attention), differentiated together. The block's `F32` weights enter
## as `F64` constants.
def test_the_gradient_of_a_decoder_block_matches_a_central_difference() -> Result[Void]
  block = try S.lift_decoder_block(try Nn.init_decoder_block(2, 2, 3, Nn.Activation.Silu, 29, S.REF), Tensor.Dtype.F64)
  check(
    [try filler([1, 2, 2], 10), try filler([1, 3, 2], 9)],
    ->(ps: List[Tensor.Tensor]) { squared(try Nn.decoder_block(block, ps[0], ps[1])) },
  )
end

## `sum(encoder_block(x)^2)` with a gelu `mlp` and bias-free weights, for
## `ps = [x, norm1 gamma, query, key, value, output, up, down, beta, norm2
## gamma]`: every parameter of the block is differentiated.
def block_loss(ps: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  attention = Nn.Attention(
    query: Nn.Dense(w: ps[2], b: none),
    key: Nn.Dense(w: ps[3], b: none),
    value: Nn.Dense(w: ps[4], b: none),
    output: Nn.Dense(w: ps[5], b: none),
    heads: 2,
  )

  block = Nn.EncoderBlock(
    norm1: Nn.LayerNorm(gamma: ps[1], beta: ps[8], eps: 0.00001),
    attention: attention,
    norm2: Nn.LayerNorm(gamma: ps[9], beta: ps[8], eps: 0.00001),
    mlp: Nn.Mlp(up: Nn.Dense(w: ps[6], b: none), down: Nn.Dense(w: ps[7], b: none), activation: Nn.Activation.Gelu),
  )

  squared(try Nn.encoder_block(block, ps[0]))
end

## An `F64` tensor of `shape` with small values that differ by `seed`.
def filler(shape: List[Int], seed: Int) -> Result[Tensor.Tensor]
  n = shape.reduce(1) do |acc, d| acc * d end
  f64((0...n).to_a.map { |j| 0.1 * ((j * 7 + seed * 3) % 11 - 5).to_f + 0.05 * seed.to_f }, shape)
end

def test_the_gradient_through_a_silu_mlp_matches_a_central_difference() -> Result[Void]
  x = try f64(X_DATA, [2, 2, 3])
  args = [
    try f64([0.2, -0.4, 0.7, 0.1, -0.3, 0.6, 0.5, -0.2, 0.4, 0.3, -0.6, 0.8], [3, 4]),
    try f64([0.3, -0.5, 0.2, 0.6, 0.1, -0.4, -0.2, 0.7, 0.5, 0.4, -0.3, 0.1], [4, 3]),
  ]

  check(
    args,
    ->(ps: List[Tensor.Tensor]) {
      m = Nn.Mlp(up: Nn.Dense(w: ps[0], b: none), down: Nn.Dense(w: ps[1], b: none), activation: Nn.Activation.Silu)
      squared(try Nn.mlp(m, Trace.constant(x)))
    },
  )
end

## `sum(dense(x)^2)` for the weight and bias in `ps`.
def dense_loss(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  squared(try Nn.dense(Nn.Dense(w: ps[0], b: some(ps[1])), Trace.constant(x)))
end

def mlp_loss(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  m = Nn.Mlp(
    up: Nn.Dense(w: ps[0], b: some(ps[1])),
    down: Nn.Dense(w: ps[2], b: some(ps[3])),
    activation: Nn.Activation.Relu,
  )

  squared(try Nn.mlp(m, Trace.constant(x)))
end

def squared(y: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.sum(try Tensor.mul(y, y))

## The gradient of `loss` at `args` against its central difference, value by
## value.
def check(args: List[Tensor.Tensor], loss: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]) -> Result[Void]
  got = try Trace.value_and_grad(args, loss)
  for i in 0...args.size
    analytic = try S.floats_of(got.grads[i])
    base = try S.floats_of(args[i])
    for j in 0...base.size
      up = try at(args, i, base.set(j, base[j] + EPS), loss)
      down = try at(args, i, base.set(j, base[j] - EPS), loss)
      slope = (up - down) / (2.0 * EPS)
      try Test.assert(
        (slope - analytic[j]).abs <= TOL * (1.0 + slope.abs),
        "arg #{i}[#{j}]: slope #{slope}, gradient #{analytic[j]}",
      )
    end
  end

  ok()
end

def f64(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F64, S.REF)
end

## The loss with argument `i` replaced by `values`.
def at(
  args: List[Tensor.Tensor],
  i: Int,
  values: List[Float],
  loss: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Float]
  moved = args.set(i, try f64(values, Tensor.shape(args[i])))
  out = try S.floats_of(try Trace.eval(moved, loss))
  ok(out[0])
end

## Plain gradient descent on a Qwen-like block (grouped-query attention, rope,
## query and key norms, RMS norms, a gated MLP, causal) towards a fixed
## target: the loss falls on every step.
def test_a_training_step_through_the_configured_block_lowers_the_loss() -> Result[Void]
  options = Nn.BlockOptions(
    kv_heads: some(1),
    rope: some(Nn.Rope()),
    qk_norm: true,
    bias: false,
    rms: true,
    gated: true,
  )

  start = try Nn.init_encoder_block(8, 4, 12, Nn.Activation.Silu, 61, S.REF, options)
  params: List[Tensor.Tensor] = []
  for t in S.qwen_params(start)
    params = params.push(try as_f64(t))
  end

  x = try filler([1, 3, 8], 3)
  target = try filler([1, 3, 8], 7)
  loss = ->(ps: List[Tensor.Tensor]) { fit_loss(start, ps, x, target) }

  losses: List[Float] = []
  for _ in 0...6
    step = try Trace.value_and_grad(params, loss)
    losses = losses.push((try S.floats_of(step.value))[0])
    moved: List[Tensor.Tensor] = []
    for i in 0...params.size
      rate = try Tensor.full(Tensor.shape(params[i]), Tensor.Values.Float(0.002), Tensor.Dtype.F64, S.REF)
      moved = moved.push(try Tensor.sub(params[i], try Tensor.mul(step.grads[i], rate)))
    end

    params = moved
  end

  for i in 1...losses.size
    try Test.assert(losses[i] < losses[i - 1], "the loss rose at step #{i}: #{losses}")
  end

  ok()
end

def fit_loss(
  start: Nn.EncoderBlock,
  ps: List[Tensor.Tensor],
  x: Tensor.Tensor,
  target: Tensor.Tensor,
) -> Result[Tensor.Tensor]
  y = try Nn.encoder_block(S.qwen_block(start, ps), Trace.constant(x), none, true)
  squared(try Tensor.sub(y, Trace.constant(target)))
end

def as_f64(t: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.to_dtype(t, Tensor.Dtype.F64)
