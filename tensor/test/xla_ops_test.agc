## The ops of the StableHLO emitter (`src/stablehlo.agc`) that a transformer step needs beyond the MLP's, each compiled
## by XLA (`Backend.Xla`) against the reference backend on the same inputs.
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Support from "xla_support"
import Tensor from "../tensor"
import Trace from "../trace"

private

def test_padded_max_pool_does_not_choose_zero_over_negative_pixels() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      xs = Support.uniform(60, 1, 1.0).map { |x| -1.0 - x.abs }
      ok([try Support.floats(xs, [1, 2, 5, 6], b)])
    },
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.max_pool2d(xs[0], 3, 2, 1)]) },
  )
end

def test_expert_matmul_matches_the_reference_with_shared_and_per_expert_inputs() -> Result[Void]
  return ok() if !try Support.present()
  for uses in [1, 2]
    try agrees(
      ->(b: Tensor.Backend) {
        ok([
          try Support.floats(Support.uniform(24, 1, 0.5), [3, 2, 4], b),
          try Support.floats(Support.uniform(8 * uses, 2, 0.5), [2, uses, 4], b),
          try Tensor.from_list([2, 0, 1, 1].map { |i| Tensor.Values.Int(i) }, [2, 2], Tensor.Dtype.I32, b),
        ])
      },
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.expert_matmul(xs[0], xs[1], xs[2])]) },
    )
  end

  ok()
end

def test_expert_matmul_rejects_invalid_expert_ids() -> Result[Void]
  return ok() if !try Support.present()
  weights = try Support.floats(Support.uniform(12, 1, 0.5), [3, 2, 2], Support.XLA)
  x = try Support.floats([1.0, 2.0], [1, 1, 2], Support.XLA)
  for bad in [-1, 3]
    ids = try Tensor.from_list([Tensor.Values.Int(bad)], [1, 1], Tensor.Dtype.I32, Support.XLA)
    r = Trace.eval([weights, x, ids], ->(xs: List[Tensor.Tensor]) { Tensor.expert_matmul(xs[0], xs[1], xs[2]) })
    try Test.error?(r)
    try Test.assert(r.message.include?("outside the rows"), r.message)
  end

  ok()
end

def test_gated_delta_output_and_state_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  for tokens in [1, 5]
    try agrees(
      ->(b: Tensor.Backend) {
        shape = [tokens, 2, 3]
        count = tokens * 6
        ok([
          try Support.floats(Support.uniform(count, 1, 0.5), shape, b),
          try Support.floats(Support.uniform(count, 2, 0.5), shape, b),
          try Support.floats(Support.uniform(count, 3, 0.5), shape, b),
          try Support.floats(Support.uniform(tokens * 2, 4, 0.5).map { |x| -x.abs }, [tokens, 2], b),
          try Support.floats(Support.uniform(tokens * 2, 5, 0.5).map { |x| x + 0.5 }, [tokens, 2], b),
          try Support.floats(Support.uniform(18, 6, 0.5), [2, 3, 3], b),
        ])
      },
      ->(xs: List[Tensor.Tensor]) {
        r = try Tensor.gated_delta_rule(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5])
        return ok([r.out, r.state]) if tokens == 1
        left = try delta_rows(xs, 0, 2)
        right = try delta_rows(xs, 2, tokens)
        prefix = try Tensor.gated_delta_rule(left[0], left[1], left[2], left[3], left[4], xs[5])
        suffix = try Tensor.gated_delta_rule(right[0], right[1], right[2], right[3], right[4], prefix.state)
        ok([r.out, r.state, try Tensor.concat([prefix.out, suffix.out], 0), suffix.state])
      },
    )
  end

  ok()
end

def delta_rows(xs: List[Tensor.Tensor], start: Int, stop: Int) -> Result[List[Tensor.Tensor]]
  rows: List[Tensor.Tensor] = []
  for x in xs.take(5)
    tail = Tensor.shape(x).drop(1)
    rows = rows.push(try Tensor.slice(x, [start] + tail.map { |_| 0 }, [stop] + tail))
  end

  ok(rows)
end

def test_grouped_conv2d_with_stride_and_padding_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.floats(Support.uniform(2 * 4 * 5 * 7, 1, 1.0), [2, 4, 5, 7], b),
        try Support.floats(Support.uniform(6 * 2 * 2 * 3, 2, 1.0), [6, 2, 2, 3], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.conv2d(xs[0], xs[1], 2, 1, 2)]) },
  )
end

def test_conv1d_matches_the_reference_for_batched_and_unbatched_inputs() -> Result[Void]
  return ok() if !try Support.present()
  for unbatched in [false, true]
    try agrees(
      ->(b: Tensor.Backend) {
        shape = [2, 3, 12]
        count = 2 * 3 * 12
        if unbatched
          shape = [3, 12]
          count = 3 * 12
        end

        ok([
          try Support.floats(Support.uniform(count, 81, 1.0), shape, b),
          try Support.floats(Support.uniform(5 * 3 * 3, 82, 1.0), [5, 3, 3], b),
        ])
      },
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.conv1d(xs[0], xs[1], 2, 1)]) },
    )
  end

  ok()
end

def test_avg_pool1d_includes_zero_padding_in_the_divisor() -> Result[Void]
  return ok() if !try Support.present()
  for unbatched in [false, true]
    try agrees(
      ->(b: Tensor.Backend) {
        shape = [2, 3, 8]
        count = 2 * 3 * 8
        if unbatched
          shape = [3, 8]
          count = 3 * 8
        end

        xs = Support.uniform(count, 83, 1.0).map { |x| -x.abs - 0.1 }
        ok([try Support.floats(xs, shape, b)])
      },
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.avg_pool1d(xs[0], 3, 2, 1)]) },
    )
  end

  ok()
end

def test_avg_pool2d_includes_zero_padding_in_the_divisor() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      xs = Support.uniform(2 * 3 * 5 * 6, 84, 1.0).map { |x| -x.abs - 0.1 }
      ok([try Support.floats(xs, [2, 3, 5, 6], b)])
    },
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.avg_pool2d(xs[0], 3, 2, 1)]) },
  )
end

def test_elementwise_ops_and_dtype_conversions_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) { ok([try wide(b), try positive(b)]) },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.sigmoid(xs[0]),
        try Tensor.pow(xs[1], xs[0]),
        try Tensor.clamp(xs[0], -1.5, 2.0),
        try Tensor.to_dtype(xs[0], Tensor.Dtype.I32),
        try Tensor.to_dtype(try Tensor.to_dtype(xs[0], Tensor.Dtype.I32), Tensor.Dtype.F32),
        try Tensor.to_dtype(xs[0], Tensor.Dtype.Bool),
        try Tensor.to_dtype(try Tensor.to_dtype(xs[0], Tensor.Dtype.Bool), Tensor.Dtype.F32),
      ])
    },
  )
end

def test_shape_ops_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try cube(2, b),
        try cube(3, b),
        try Support.floats(Support.uniform(6, 5, 1.0), [1, 3, 1, 2], b),
        try Support.floats([2.5], [1], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.permute(xs[0], [2, 0, 1]),
        try Tensor.concat([xs[0], xs[1], xs[0]], 0),
        try Tensor.concat([xs[0], xs[0]], 1),
        try Tensor.concat([xs[0], xs[0]], 2),
        try Tensor.stack([xs[0], xs[0]], 2),
        try Tensor.stack([xs[0], xs[0]], 0),
        try Tensor.stack([xs[0], xs[0]], 3),
        try Tensor.squeeze(xs[2]),
        try Tensor.squeeze(xs[2], some(2)),
        try Tensor.squeeze(xs[2], some(0)),
        try Tensor.unsqueeze(xs[0], 3),
        try Tensor.unsqueeze(xs[0], 0),
        try Tensor.unsqueeze(try Tensor.squeeze(xs[3]), 0),
      ])
    },
  )
end

## The batch is the leading axes of both operands.
def test_a_batched_product_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.floats(Support.uniform(2 * 3 * 4 * 5, 1, 1.0), [2, 3, 4, 5], b),
        try Support.floats(Support.uniform(2 * 3 * 5 * 2, 2, 1.0), [2, 3, 5, 2], b),
        try Support.floats(Support.uniform(3 * 4, 3, 1.0), [3, 4], b),
        try Support.floats(Support.uniform(4 * 2, 4, 1.0), [4, 2], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.batch_matmul(xs[0], xs[1]),
        try Tensor.batch_matmul(try Tensor.unsqueeze(xs[2], 0), try Tensor.unsqueeze(xs[3], 0)),
        try Tensor.batch_matmul(xs[2], xs[3]),
      ])
    },
  )
end

## The index inside the reduced block, numbered by the ascending axes, and a tie keeps the lowest.
def test_argmax_matches_the_reference_on_one_axis_many_axes_and_ties() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) { ok([try cube(3, b), try ties(b)]) },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.argmax(xs[0]),
        try Tensor.argmax(xs[0], some([1])),
        try Tensor.argmax(xs[0], some([2]), true),
        try Tensor.argmax(xs[0], some([0, 2])),
        try Tensor.argmax(xs[0], some([2, 0]), true),
        try Tensor.argmax(xs[1], some([1])),
        try Tensor.argmax(xs[1]),
      ])
    },
  )
end

def test_norms_gelu_and_rope_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.floats(Support.uniform(2 * 3 * 8, 21, 2.0), [2, 3, 8], b),
        try Support.floats(Support.uniform(2 * 2 * 3 * 8, 22, 1.0), [2, 2, 3, 8], b),
        try Tensor.from_list(
          [Tensor.Values.Int(0), Tensor.Values.Int(5), Tensor.Values.Int(17)],
          [3],
          Tensor.Dtype.I32,
          b,
        ),
        try Support.floats(Support.uniform(8, 23, 2.0), [8], b),
        try Support.floats(Support.uniform(3 * 8, 24, 1.0), [3, 8], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.layer_norm(xs[0], 0.00001),
        try Tensor.rms_norm(xs[0], 0.000001),
        try Tensor.layer_norm(xs[3], 0.00001),
        try Tensor.rms_norm(xs[3], 0.000001),
        try Tensor.gelu_erf(xs[0]),
        try Tensor.rope(xs[1], xs[2]),
        try Tensor.rope(xs[4], xs[2]),
        try Tensor.rope(xs[1], xs[2], 500.0, some(4), false),
        try Tensor.rope(xs[1], xs[2], 10000.0, some(8), false, true),
        try Tensor.rope(xs[1], xs[2], 1000000.0, some(6), true, true),
      ])
    },
  )
end

## Heads that share keys (a group of 2), a mask of each shape, the causal mask, a scale, and one head per key.
def test_attention_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.floats(Support.uniform(2 * 4 * 3 * 8, 41, 1.0), [2, 4, 3, 8], b),
        try Support.floats(Support.uniform(2 * 2 * 5 * 8, 42, 1.0), [2, 2, 5, 8], b),
        try Support.floats(Support.uniform(2 * 2 * 5 * 6, 43, 1.0), [2, 2, 5, 6], b),
        try Support.floats(Support.uniform(3 * 5, 44, 2.0), [3, 5], b),
        try Support.floats(Support.uniform(2 * 4 * 3 * 5, 45, 2.0), [2, 4, 3, 5], b),
        try Support.floats(Support.uniform(2 * 4 * 5 * 8, 46, 1.0), [2, 4, 5, 8], b),
        try Support.floats(Support.uniform(1 * 1 * 3 * 5, 47, 2.0), [1, 1, 3, 5], b),
        try Support.floats(Support.uniform(2 * 1 * 3 * 5, 48, 2.0), [2, 1, 3, 5], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.attention(xs[0], xs[1], xs[2]),
        try Tensor.attention(xs[0], xs[1], xs[2], mask: some(xs[3])),
        try Tensor.attention(xs[0], xs[1], xs[2], mask: some(xs[4]), scale: some(0.3)),
        try Tensor.attention(xs[0], xs[1], xs[2], causal: true),
        try Tensor.attention(xs[0], xs[5], xs[5], mask: some(xs[6]), causal: true),
        try Tensor.attention(xs[0], xs[5], xs[5], mask: some(xs[7])),
      ])
    },
  )
end

## Rows added at their indices, duplicates summing, for indices of rank 1 and 2 and updates of rank 1 and 2; an index
## outside the rows is an error, as on the reference.
def test_scatter_add_matches_the_reference_and_checks_its_indices() -> Result[Void]
  return ok() if !try Support.present()
  try agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.labels([1, 3, 1, 0], b),
        try Support.floats(Support.uniform(4 * 3, 51, 1.0), [4, 3], b),
        try Support.floats(Support.uniform(4, 52, 1.0), [4], b),
        try Tensor.from_list(
          [Tensor.Values.Int(4), Tensor.Values.Int(4), Tensor.Values.Int(0), Tensor.Values.Int(2)],
          [2, 2],
          Tensor.Dtype.I64,
          b,
        ),
        try Support.floats(Support.uniform(2 * 2 * 3, 53, 1.0), [2, 2, 3], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.scatter_add(xs[0], xs[1], 5),
        try Tensor.scatter_add(xs[0], xs[2], 5),
        try Tensor.scatter_add(xs[3], xs[4], 5),
      ])
    },
  )

  updates = try Support.floats(Support.uniform(6, 54, 1.0), [2, 3], Support.XLA)
  for bad in [[1, 5], [-1, 0]]
    r = Trace.eval(
      [try Support.labels(bad, Support.XLA), updates],
      ->(xs: List[Tensor.Tensor]) { Tensor.scatter_add(xs[0], xs[1], 5) },
    )

    try Test.assert(!r.ok?, "scatter_add at #{bad} of 5 rows ran on XLA")
    try Test.assert(r.message.include?("outside the rows"), r.message)
  end

  ok()
end

## The gradient through attention (its backward is written with `batch_matmul`) and through the embedding gather (whose
## backward is `scatter_add`), compiled by XLA.
def test_value_and_grad_through_attention_and_an_embedding_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  got = try embedding_gradients(Support.XLA)
  want = try embedding_gradients(Support.REF)
  Support.close(got, want)
end

def embedding_gradients(b: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  args = [
    try Support.floats(Support.uniform(6 * 8, 61, 1.0), [6, 8], b),
    try Support.floats(Support.uniform(2 * 4 * 3 * 4, 62, 1.0), [2, 4, 3, 4], b),
    try Support.floats(Support.uniform(2 * 2 * 3 * 4, 63, 1.0), [2, 2, 3, 4], b),
    try Support.floats(Support.uniform(2 * 2 * 3 * 4, 64, 1.0), [2, 2, 3, 4], b),
  ]

  ids = try Support.labels([2, 5, 2, 0, 3, 3], b)
  loss = ->(xs: List[Tensor.Tensor]) { embedding_loss(xs, ids) }
  got = try Trace.value_and_grad(args, loss)
  ok([got.value] + got.grads)
end

## Rows of a table gathered into `[2, 3, 8]`, split into heads as the queries of an attention over keys and values, and a
## weighted sum.
def embedding_loss(xs: List[Tensor.Tensor], ids: Tensor.Tensor) -> Result[Tensor.Tensor]
  words = try Tensor.reshape(try Tensor.gather(xs[0], Trace.constant(ids)), [2, 3, 2, 4])
  q = try Tensor.concat([try Tensor.permute(words, [0, 2, 1, 3]), xs[1]], 1)
  mask = try Tensor.full([3, 3], Tensor.Values.Float(0.0), Tensor.Dtype.F32, Tensor.backend(q))
  out = try Tensor.attention(q, xs[2], xs[3], mask: some(mask), causal: true)
  Tensor.sum(try Tensor.mul(out, out))
end

## The gradient of a loss through the ops of a transformer's block, by the rules of `trace.agc`, compiled by XLA.
def test_value_and_grad_through_the_new_ops_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  got = try gradients(Support.XLA)
  want = try gradients(Support.REF)
  Support.close(got, want)
end

## The ids of the largest values, the largest first, of a vector and of each row of a matrix; among equal values the
## lowest index first.
def test_top_ids_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  agrees(
    ->(b: Tensor.Backend) {
      ok([
        try Support.floats(Support.uniform(40, 71, 3.0), [40], b),
        try Support.floats(Support.uniform(3 * 12, 72, 3.0), [3, 12], b),
        try Support.floats([1.0, 3.0, 3.0, 0.0, 3.0], [5], b),
      ])
    },
    ->(xs: List[Tensor.Tensor]) {
      ok([
        try Tensor.top_ids(xs[0], 1),
        try Tensor.top_ids(xs[0], 7),
        try Tensor.top_ids(xs[1], 4),
        try Tensor.top_ids(xs[1], 12),
        try Tensor.top_ids(xs[2], 3),
      ])
    },
  )
end

def test_an_op_the_emitter_still_lacks_is_an_error_naming_it() -> Result[Void]
  return ok() if !try Support.present()
  xs = [
    try Support.floats(Support.uniform(6, 85, 1.0), [3, 2], Support.XLA),
    try Support.floats(Support.uniform(2, 86, 1.0), [1, 2], Support.XLA),
    try Support.labels([1], Support.XLA),
  ]
  r = Trace.compile(
    xs.map { |t| Trace.spec_of(t) },
    ->(ys: List[Tensor.Tensor]) { ok([try Tensor.set_rows(ys[0], ys[1], ys[2])]) },
  )

  try Test.assert(!r.ok?, "set_rows compiled")
  Test.assert(r.message.include?("set_rows"), r.message)
end

## A sigmoid, a permute, a concat, a squeeze, the norms, a gelu and a rope, then a weighted sum.
def block_loss(xs: List[Tensor.Tensor], positions: Tensor.Tensor) -> Result[Tensor.Tensor]
  moved = try Tensor.permute(try Tensor.sigmoid(xs[0]), [0, 2, 1, 3])
  s = try Tensor.squeeze(try Tensor.concat([moved, moved], 2), some(0))
  n = try Tensor.rms_norm(try Tensor.gelu_erf(try Tensor.layer_norm(s, 0.00001)), 0.000001)
  r = try Tensor.rope(n, Trace.constant(positions))
  Tensor.sum(try Tensor.mul(r, xs[1]))
end

def gradients(b: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  args = [
    try Support.floats(Support.uniform(1 * 2 * 4 * 8, 31, 1.0), [1, 2, 4, 8], b),
    try Support.floats(Support.uniform(4 * 4 * 8, 32, 1.0), [4, 4, 8], b),
  ]

  positions = try Tensor.from_list(
    [Tensor.Values.Int(0), Tensor.Values.Int(1), Tensor.Values.Int(2), Tensor.Values.Int(3)],
    [4],
    Tensor.Dtype.I32,
    b,
  )

  loss = ->(xs: List[Tensor.Tensor]) { block_loss(xs, positions) }
  got = try Trace.value_and_grad(args, loss)
  ok([got.value] + got.grads)
end

## `make` the inputs on a backend, `f` over them compiled by XLA, and `f` over the same inputs on the reference: every
## output within `Support.TOL`, with the same shape.
def agrees(
  make: Fn[Tensor.Backend, Result[List[Tensor.Tensor]]],
  f: Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[Void]
  inputs = try make.call(Support.XLA)
  compiled = try Trace.compile(inputs.map { |t| Trace.spec_of(t) }, f)
  got = try Trace.run(compiled, inputs)
  want = try f.call(try make.call(Support.REF))
  Support.close(got, want)
end

## Values of both signs, past the clamp and past the rounding of a conversion, with zeros.
def wide(b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats([-3.7, -1.5, -0.4, 0.0, 0.3, 1.0, 2.5, 9.9, -0.0, 4.2, -2.2, 0.7], [3, 4], b)
end

def positive(b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats([0.5, 1.5, 2.0, 3.0, 0.25, 7.0, 1.1, 4.0, 9.0, 0.1, 6.5, 2.2], [3, 4], b)
end

## `[n, 2, 3]` of distinct values.
def cube(n: Int, b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats(Support.uniform(n * 6, n, 1.0), [n, 2, 3], b)
end

## `[2, 4]` where each row's largest value appears twice.
def ties(b: Tensor.Backend) -> Result[Tensor.Tensor]
  Support.floats([1.0, 5.0, 2.0, 5.0, 3.0, 3.0, 0.5, 3.0], [2, 4], b)
end
