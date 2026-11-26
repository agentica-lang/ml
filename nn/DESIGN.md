# nn — design

Functional layers over `tensor/`. Depends only on `tensor/` (a member of the
same workspace, read from `../tensor`). The reference and libtorch run a layer
eagerly; ggml and XLA run it traced only: the layer is called on `Trace`
tensors and the graph goes to `Trace.eval` or `Trace.compile`
(`tensor/DESIGN.md` §10).

## API

```text
struct Dense        { w: [in, out] ([out, in] with out_in), b: Optional [out], out_in: Bool = false }
struct LayerNorm    { gamma: [width], beta: [width], eps: Float, rms: Bool = false }
struct Embedding    { table: [rows, width] }
struct Conv1d       { weight: [out, in, kernel], bias: Optional [out], stride, padding: Int }
struct Conv2d       { weight: [out, in / groups, kh, kw], bias: Optional [out], stride = 1, padding = 0, groups = 1 }
enum   Activation   = Relu | Gelu | GeluErf | Silu
struct Rope         { base: Float = 10000.0, neox: Bool = true, dims: Optional[Int] = none, zero_tail: Optional[Int] = none, sections: Optional[List[Int]] = none }
struct Attention    { query, key, value, output: Dense, heads: Int, kv_heads: Optional[Int] = none,
                      rope: Optional[Rope] = none, query_norm, key_norm: Optional[LayerNorm] = none,
                      gate: Bool = false, scale: Optional[Float] = none, head_gate: Optional[Dense] = none }
struct Mlp          { up, down: Dense, activation: Activation, gate: Optional[Dense] = none }
struct EncoderBlock { norm1: LayerNorm, attention: Attention, norm2: LayerNorm, mlp: Mlp,
                      norm_first: Bool = true }
struct DecoderBlock { norm1, self_attention: Attention, norm2, cross_attention: Attention,
                      norm3: LayerNorm, mlp: Mlp, norm_first: Bool = true }
struct EncoderStack { blocks: List[EncoderBlock], final_norm: Optional[LayerNorm] = none }
struct DecoderStack { blocks: List[DecoderBlock], final_norm: Optional[LayerNorm] = none }
struct BlockOptions { kv_heads, rope, qk_norm, bias, rms, gated, norm_first, eps }   (for init_*)

Nn.dense(p, x)                     x [..., in] -> [..., out]
Nn.cosine_similarity(a, b)         [batch, width] twice -> [batch], the cosine of each pair of rows
Nn.layer_norm(p, x)                last axis, then * gamma + beta
Nn.embedding(p, ids)               ids rank-1 integer -> [ids, width]
Nn.conv1d(p, x)                    x [in, L] or [batch, in, L] -> Tensor.conv1d + bias per channel
Nn.conv2d(p, x)                    x [batch, in, H, W] -> Tensor.conv2d + bias per channel
Nn.fold_batch_norm(conv, gamma, beta, mean, var, eps)  Conv2d and the batch norm after it -> one Conv2d (inference)
Nn.global_avg_pool2d(x)            [batch, channels, H, W] -> [batch, channels], the mean over H and W
Nn.positional(p, x)                x [seq, width] or [batch, seq, width] + table rows 0...seq
Nn.activate(a, x)
Nn.attention(p, queries, memory, mask?, causal?, positions?, memory_positions?)
                                   [batch, q, width] over [batch, k, width]: project, then attend
Nn.project(p, queries, memory, positions?, memory_positions?) -> Heads { query, key, value, gate? }
                                   split by head, q/k norms, rope at `positions` (default 0...seq); with `gate`, the
                                   query projection is [heads * 2 * head]: per head the query, then its gate
Nn.attend(p, query, key, value, mask?, causal?, gate?)   Tensor.attention over heads or a cache, merged (times
                                   sigmoid(gate) when given), output dense
Nn.self_attention(p, x, mask?, causal?)
Nn.mlp(p, x)                       down(act(up x)), or down(act(gate x) * up x) with a gate
Nn.encoder_block(p, x, mask?, causal?)
                                   pre-norm: h = x + attn(norm1(x)); h + mlp(norm2(h))
                                   post-norm: h = norm1(x + attn(x)); norm2(h + mlp(h))
Nn.causal_mask(size, dtype, backend, window?)   [size, size]: 0.0 on and below the diagonal, MASKED above; with a
                                   window, 0.0 only where i - window < key <= i
Nn.window_bias(queries, keys, window, dtype)   that window mask over grids of I32 positions (a padded cache names its own)
Nn.padding_bias(mask)              [batch, seq] of 1.0/0.0 -> [batch, 1, seq, seq]: MASKED on padding keys
Nn.decoder_block(p, x, memory)     x [batch, t, width] over memory [batch, s, width]:
                                   h = x + self_attn(norm1(x), causal);
                                   h = h + attn(cross, norm2(h), memory); h + mlp(norm3(h))
Nn.encoder_stack(p, x, mask?, causal?) / decoder_stack(p, x, memory)
                                   the blocks in order, then final_norm

Moe (`pkg/nn/moe`, the mixture of experts of Granite MoE):
struct Moe          { router: Dense ([experts, width], out_in), gate_up: [experts, 2 * inner, width],
                      down: [experts, width, inner], top_k: Int, activation: Activation, normalize: Bool = true }
Moe.moe(p, x)                      x [..., width] -> the same shape: the gated MLPs of the picked experts, added by their gates

ShortConv (`pkg/nn/short_conv`, the gated short convolution of LFM3):
struct ShortConv    { input: Dense ([3 * width, width], out_in), conv: [width, taps], output: Dense }
ShortConv.short_conv(p, x, state, padding?)  x [tokens, width] -> ShortOut { out, state }: output(c * conv(b * x)), with
                                   b, c and x the three parts of input(x); state [taps - 1, width] as Delta's conv state
ShortConv.short_start(p, backend)  the zero state of a layer

Delta (`pkg/nn/delta`, the linear attention of Qwen3.5):
Delta.gated_delta_net(p, x, state, padding?)  x [tokens, width] after a DeltaState -> DeltaOut { out, state }
Delta.causal_conv1d(x, weight, state)  depthwise over time, the past only -> Convolved { out, state }
Delta.gated_rms_norm(x, z, weight, eps)  weight * rms_norm(x) * silu(z)
Delta.delta_start(p, backend)      the zero DeltaState of a layer
Delta.padding(real, tokens, taps, backend)  the Padding of a call whose last rows are padding
```

The layer is for inference: a state is not a parameter and `gated_delta_rule` has no gradient, so nothing flows back
through it. It takes one head width for keys and values (the rule's), and a value head count that is a multiple of
the key head count. `DeltaState` is not Hugging Face's cache: `conv` is `[taps - 1, channels]`, the last `taps - 1`
rows of the convolution's input in time order (Hugging Face keeps `[channels, taps]`, the last `taps` columns), and
`recurrent` is `[value heads, value, key]` (Hugging Face's `[value heads, key, value]`, transposed). A loader reads the
checkpoint's convolution `[channels, 1, taps]` as `[channels, taps]`; `test/delta_test.agc` checks both states against
the cache of Hugging Face.

`ShortConv` is the convolution layer of LFM3 (LFM2 has the same layer, untested here): `in_proj` makes `b`, `c` and `x`, the depthwise convolution
(`Delta.causal_conv1d`, so the same state, the same `Padding` and the same checks) runs over `b * x`, and `c` gates its
output before `out_proj`; there is no bias (`conv_bias` is false in every checkpoint, a loader refuses it). The attention
of LFM3 gates each head by one number, `sigmoid(gate_proj(x))` with `gate_proj` `[heads, width]`, which is
`Attention.head_gate`: `project` repeats it over the head width as a `Heads.gate`, so `attend` is the one of `gate`. A
sliding-window layer is an attention under `causal_mask` with a `window` (the window counts the key itself: a window of 3 sees the
position and the two before it, as Hugging Face's `sliding_window`); a padded cache gives `padded_mask` the window.
The full-attention layers of LFM3 rotate only part of the head, in a layout that `dims` does not give: Hugging Face
keeps the frequencies `base^(-2i / head)` of the whole head and zeroes those of the last pairs, so the pairs that turn are
`(i, i + head / 2)` for `i < zero_tail / 2` and the others stay. `Rope.zero_tail` (the rotated columns, `head *
partial_rotary_factor`) brings those columns together (four slices and a `concat` of the head) and runs `Tensor.rope` over
them with `dims: zero_tail` and the base `base^(zero_tail / head)`, which gives pair `i` the same angle. The head comes out
reordered, the same for the queries and the keys, so `q k^T` is the one of Hugging Face; the values are not reordered, and
nothing reads the reordered heads but the scores (a cache keeps the keys as they come out). A `zero_tail` of the whole
head is an error (it is a plain rotation: leave it out). Doing it in the graph, not in the weights at load, works for `F16`
and quantized weights, which a host permutation of rows could not touch.
`test/lfm3_test.agc` checks the three layers and the rotary against the official code's tiny model (`test/lfm3_golden.py`).

Qwen3.5 turns each pair of the rotated columns by one of three positions (the interleaved M-RoPE, `Rope.sections` =
`[time, height, width]`): the pair `i` takes the height row when `i % 3 == 1` and `i < 3 * height`, the width row when
`i % 3 == 2` and `i < 3 * width`, the time row otherwise. `Tensor.rope` takes one position vector, so with positions of
`[3, seq]` the head is rotated three times, once by each row, and a 0/1 mask over the columns (a cycle of three, tiled,
cut at the limit, built from fills so that it is a graph constant) picks each pair from its axis. Positions of `[seq]`
are the plain rope, which is what the three rows are for text. `test/layers_test.agc` checks it against the plain rope of
each row with the pairs picked by hand.

A compiled decode step has one shape for the prompts of a capacity, so a prompt is padded after its last real token.
`Padding` makes the padding leave the state as the real tokens left it: `valid` (1.0 real, 0.0 padding) multiplies
`beta` and `g`, so a padding row neither writes into the recurrent state nor decays it, and `keep` (`[taps - 1, taps - 1
+ tokens]`, one-hot) picks the convolution's state rows that end at the last real token, a product with a 0/1 matrix
(exact). `test/delta_test.agc` checks the answers and both states against the same prompt unpadded.

`Moe.moe` is the sparse block of Granite MoE (and Qwen3 MoE): the router `Dense` scores the experts, `Tensor.top_ids` picks
`top_k` of them per token, and each pick is a gated MLP whose two projections are one stacked weight a checkpoint stores
(`gate_up` holds the gate and then the value of each expert; Granite's `input_linear` is that, and Qwen3 MoE's separate
projections are concatenated when loaded). The gates are the softmax over the picked scores (`normalize`, Granite), or
the picked entries of the softmax over all the experts. They are found with a mask of the ids against the expert
numbers, not a gather of integers, because ggml has no integer sum. The ids come back in no order on ggml, so
the routing is not compared node by node; `test/moe_test.agc` compares the answers with Hugging Face's blocks (one token,
40 tokens, and a tie for the last place between two experts that are the same expert: whichever is picked, the answer is
the same). `normalize = false` (the gates of Qwen3 MoE) has no model in this repo yet: Hugging Face's block checks it.

```text
Nn.init_dense / init_layer_norm / init_embedding / init_attention /
  init_mlp / init_encoder_block / init_decoder_block / init_encoder_stack /
  init_decoder_stack(..., seed, backend, options?) -> Result[layer]
Init.uniform(shape, limit, seed, backend), Init.glorot_uniform(shape, seed, backend)

Padded (kv.agc)   the padded KV cache that the decoding models share: capacity, reach, reached,
                  padded_mask (with a `window` for a sliding-window layer), spread_of, written
```

Every function returns `Result`; the checks are `tensor/`'s, plus the few
that are the layer's own (rank of `positional`, table length, heads).

## Decisions

- **A layer is a struct of parameters and a function `layer(params, x)`.**
  No classes, no state, no model object: a model is a struct of layer
  structs, and a training step returns new structs. Structs over maps
  because the fields are fixed and the checker then knows them.
- **No math of its own.** Every layer composes `tensor/`'s public calls,
  mostly its derived ops (`linear`, `layer_norm`, `rms_norm`, `embedding`,
  `gelu`, `silu`) and its attention primitives (`attention`, `permute`,
  `rope`). nn/ adds what is model-shaped: the
  parameters, the affine part of the norm, the attention projections, the
  residual block, and folding leading axes so `dense` takes any rank.
- **Differentiable for free.** Since a layer is `Tensor.*` calls, it records a
  graph on `Trace` tensors and `Trace.value_and_grad` differentiates it
  (`tensor/DESIGN.md` §10). The tests check `dense`, a relu and a silu
  `mlp`, `embedding` with `positional`, a gelu `encoder_block` (the
  gradient of its input and of one gamma) and a silu `decoder_block` (its
  input and its encoder output) against a central difference.
- **The cosine has no clamp.** PyTorch divides by each norm clamped to
  `1e-8`; `clamp` has no backward rule, so `cosine_similarity` divides by
  `sqrt(|a|^2 |b|^2 + 1e-16)`: the same values to `F32` rounding for a row
  that is not zero, and a cosine of 0 with a finite gradient for one that is.
  In `F16` the term would round to 0; training in `F16` is not a goal.
- **Options are fields with defaults.** A model changes a detail of a block,
  so the detail is a field: `kv_heads`, `rope` and the query and key norms of
  `Attention`, `gate` of `Mlp`, `rms` of `LayerNorm`, `norm_first` of the
  blocks. Each default is the old behaviour, so a struct built field by field
  (as `hub/` does) keeps working. `BlockOptions` carries the same choices to
  `init_*`; it is not a layer. The cross-attention of a decoder block takes
  only its `bias`. A loader that rebuilds a layer (`hub/`) must use
  `.with(...)` on the struct it has, not a constructor call, or it drops the
  options added later. The head width is the width of `query` over `heads`,
  so loaded weights may have a head width other than `width / heads`
  (Qwen3); `init_attention` makes only `width / heads`.
- **`project` and `attend` split attention for a cache.** The layers hold no
  cache (`Padded` has the helpers the models share), but a decoder that keeps one must rotate the new token's key at its own
  position and read the cached keys as they are: `project` takes the integer
  `positions` (queries) and `memory_positions` (keys; the queries' when only
  those are given: only `positions` means self-attention, pass
  `memory_positions` for cross-attention), and `attend` takes the heads or a
  cache of them.
  `attention` is their composition.
- **Attention is `Tensor.attention`.** `attention` splits the projections into
  heads with `reshape` and `permute`, applies the query and key norms, then
  `rope` (positions `0...seq` of each input, so a cross-attention gets rope
  only if its `Attention` has one), runs `Tensor.attention` and merges the
  heads. Grouped-query attention is `kv_heads` fewer than `heads` (it must
  divide them); the mask is `Tensor.attention`'s (`[q, k]` or
  `[batch|1, heads|1, q, k]`).
- **Pre-norm and post-norm.** `norm_first: true` is the pre-norm of Whisper,
  Qwen and Llama; `false` is BERT's post-norm. A pre-norm stack ends in
  `final_norm`; a post-norm one has none. `causal: true` on `encoder_block`
  makes it a decoder-only block (Qwen). `RMSNorm` is `LayerNorm` with `rms`:
  `beta` is kept and ignored.
- **The decoder is Whisper's, pre-norm by default.** Self-attention is causal
  (`causal: true` in `Tensor.attention`, which adds `-1e9` where a query must
  not see a key: finite, and far enough down that its softmax weight is `0`;
  `causal_mask` builds the same bias for a caller that wants it) and
  cross-attention reuses `attention(p, queries, memory)`: queries come from
  the decoder and keys and values from the encoder output, so `t` and `s`
  differ freely. With fewer queries than keys the mask ends on the last key
  (the end-aligned mask of a decode step); more queries than keys is an
  error, and on `Reference` `q * k` of the mask is at most 1,048,576
  (`Tensor.full`). It assumes an `F32` or `F64` compute dtype. `decoder_block`
  has no KV cache and no decode loop: each call recomputes every position.
- **Weights are `[in, out]`** (`x w`, as `Tensor.linear` takes them), or
  `[out, in]` with `Dense.out_in`. Hugging Face and PyTorch checkpoints store
  `[out, in]`, and `hub/` keeps that layout: `dense` multiplies by its
  transpose, which ggml reads with no copy (a weight `[in, out]` is copied
  transposed on every run: copies were 187 of the 296 ms of kernels of a
  whisper-medium step on Vulkan, `benchmarks/README.md`). `Tensor.linear(..., out_in)` reads it as it is on ggml and
  libtorch.
- **Attention costs what `tensor/` costs.** `Tensor.attention` is one op on
  every backend (a fused kernel on ggml and libtorch), so a layer is a few
  nodes. `Tensor.multi_head_attention` remains in `tensor/` as a wrapper over
  it for callers with `[batch, seq, width]` and no `nn/` layer (`asr/` until it
  moves); `nn/` does not call it.
- **Seeded initializers in Agentica.** `tensor/` has no random op at
  inference (`tensor/DESIGN.md` §3.1), so `init.agc` draws from a
  Park-Miller generator and crosses with `from_list`: one seed gives the same
  weights on every backend, which is what lets the tests compare a model on
  the reference with the same model on libtorch. Sub-layers draw from
  `Init.derive(seed, k)`. Only uniform and Glorot uniform: real weights come
  from checkpoints (`hub/`).
- **`F32` only** for `init_*`, the compute dtype of `tensor/DESIGN.md` §2.
  Layers themselves take any float dtype (the gradient test uses `F64`).
  The gradient tests need `agentica test --max-steps 10000000`: a bare
  `agentica test` runs out of its 1M steps (per file, times its tests + 1).
- **Two GELUs**: `Gelu` is the tanh approximation (`Tensor.gelu`);
  `GeluErf` is the exact `erf` form Whisper and BERT use
  (`Tensor.gelu_erf`). `Trace` differentiates both. The libtorch comparisons
  use `relu` and `silu`.
