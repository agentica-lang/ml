## `nn/` layers loaded by name: a dense layer, a layer norm and a
## Whisper-style decoder block give the forward pass the Python peer computed,
## on the reference and on libtorch, and a Whisper-style encoder block loads
## and runs. Where libtorch is not installed, the test asserts that absence
## instead (`tensor/README.md`, "CI").

import Hub from "../hub"
import Load from "../load"
import Nn from "pkg/nn/nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The reference computes in doubles, as the peer does.
ABS = 0.000000001
REL = 0.000000001

## libtorch computes in `F32`.
TORCH_ABS = 0.00001
TORCH_REL = 0.0001

private

def test_dense_and_layer_norm_match_the_peer() -> Result[Void]
  S.with_fixtures("layer") do |dir|
    c = try Hub.open("#{dir}/layer.safetensors")
    want = try S.json(dir, "layer.json")
    dense = try Load.dense(c, "enc.fc", S.REF)
    try Test.eq(Tensor.shape(dense.w), [3, 4])
    try Test.assert(dense.out_in, "a loaded dense layer keeps the checkpoint's [out, in]")
    norm = try Load.layer_norm(c, "enc.ln", try JSON.float(try JSON.get(want, "eps")), S.REF)
    try Test.eq([Tensor.dtype(norm.gamma), Tensor.dtype(norm.beta)], [Tensor.Dtype.F32, Tensor.Dtype.F32])
    got = try forward(dense, norm, try S.rows(try JSON.get(want, "x")), S.REF)
    S.expect_close(got, try S.rows(try JSON.get(want, "y")), ABS, REL, "dense + layer_norm")
  end
end

def test_dense_and_layer_norm_match_the_peer_on_libtorch() -> Result[Void]
  S.with_fixtures("layer") do |dir|
    c = try Hub.open("#{dir}/layer.safetensors")
    return expect_absent(Load.dense(c, "enc.fc", S.TORCH)) if not Tensor.available(S.TORCH)

    want = try S.json(dir, "layer.json")
    eps = try JSON.float(try JSON.get(want, "eps"))
    x = try S.rows(try JSON.get(want, "x"))
    got: Result[List[Float]] = Native.arena do
      dense = try Load.dense(c, "enc.fc", S.TORCH)
      forward(dense, try Load.layer_norm(c, "enc.ln", eps, S.TORCH), x, S.TORCH)
    end

    S.expect_close(try got, try S.rows(try JSON.get(want, "y")), TORCH_ABS, TORCH_REL, "torch dense + layer_norm")
  end
end

def test_a_weight_of_the_wrong_kind_is_refused() -> Result[Void]
  S.with_fixtures("layer") do |dir|
    c = try Hub.open("#{dir}/layer.safetensors")
    try Test.error?(Load.embedding(c, "enc.ids", S.REF))
    try Test.error?(Load.dense(c, "enc.fc.bias", S.REF))
    Test.error?(Load.dense(c, "enc.missing", S.REF))
  end
end

def test_a_whisper_style_encoder_block_loads_layer_by_layer() -> Result[Void]
  S.with_fixtures("block") do |dir|
    c = try Hub.open("#{dir}/block.safetensors")
    want = try S.json(dir, "block.json")
    p = "encoder.layers.0."
    names = Load.BlockNames(
      norm1: "#{p}self_attn_layer_norm",
      query: "#{p}self_attn.q_proj",
      key: "#{p}self_attn.k_proj",
      value: "#{p}self_attn.v_proj",
      output: "#{p}self_attn.out_proj",
      norm2: "#{p}final_layer_norm",
      up: "#{p}fc1",
      down: "#{p}fc2",
    )

    block = try Load.encoder_block(c, names, 2, Nn.Activation.Relu, 0.00001, S.REF)
    a = block.attention
    prefixes = [names.query, names.key, names.value, names.output, names.up, names.down]
    denses = [a.query, a.key, a.value, a.output, block.mlp.up, block.mlp.down]
    for i in 0...denses.size
      try same_dense(denses[i], try JSON.get(try JSON.get(want, "dense"), prefixes[i]))
    end

    try same_norm(block.norm1, try JSON.get(try JSON.get(want, "norm"), names.norm1))
    try same_norm(block.norm2, try JSON.get(try JSON.get(want, "norm"), names.norm2))
    try Test.eq(Tensor.shape(block.mlp.up.w), [8, 4])
    x = try S.f32((0...12).map { |i| i.to_f / 10.0 }, [1, 3, 4], S.REF)
    Test.eq(Tensor.shape(try Nn.encoder_block(block, x)), [1, 3, 4])
  end
end

def test_an_embedding_table_loads_as_stored() -> Result[Void]
  S.with_fixtures("block") do |dir|
    c = try Hub.open("#{dir}/block.safetensors")
    e = try Load.embedding(c, "encoder.embed_positions.weight", S.REF)
    try Test.eq(Tensor.shape(e.table), [6, 4])
    try Test.eq(Tensor.dtype(e.table), Tensor.Dtype.F32)
    Test.eq(try S.floats_of(e.table), try S.floats(try JSON.get(try S.json(dir, "block.json"), "embed")))
  end
end

def test_a_whisper_style_decoder_block_loads_and_matches_the_peer() -> Result[Void]
  S.with_fixtures("decoder") do |dir|
    c = try Hub.open("#{dir}/decoder.safetensors")
    want = try S.json(dir, "decoder.json")
    names = decoder_names()
    eps = try JSON.float(try JSON.get(want, "eps"))
    block = try Load.decoder_block(c, names, 2, Nn.Activation.Relu, eps, S.REF)
    a = block.self_attention
    x = block.cross_attention
    prefixes = [
      names.query, names.key, names.value, names.output, names.cross_query, names.cross_key, names.cross_value,
      names.cross_output, names.up, names.down,
    ]

    denses = [a.query, a.key, a.value, a.output, x.query, x.key, x.value, x.output, block.mlp.up, block.mlp.down]
    for i in 0...denses.size
      try same_dense(denses[i], try JSON.get(try JSON.get(want, "dense"), prefixes[i]))
    end

    try same_norm(block.norm1, try JSON.get(try JSON.get(want, "norm"), names.norm1))
    try same_norm(block.norm2, try JSON.get(try JSON.get(want, "norm"), names.norm2))
    try same_norm(block.norm3, try JSON.get(try JSON.get(want, "norm"), names.norm3))

    got = try decode(block, want, S.REF)
    S.expect_close(got, try S.rows(try JSON.get(want, "y")), ABS, REL, "decoder_block")
  end
end

def test_a_decoder_block_matches_the_peer_on_libtorch() -> Result[Void]
  S.with_fixtures("decoder") do |dir|
    c = try Hub.open("#{dir}/decoder.safetensors")
    return expect_absent(Load.dense(c, "decoder.layers.0.fc1", S.TORCH)) if not Tensor.available(S.TORCH)

    want = try S.json(dir, "decoder.json")
    eps = try JSON.float(try JSON.get(want, "eps"))
    got: Result[List[Float]] = Native.arena do
      decode(try Load.decoder_block(c, decoder_names(), 2, Nn.Activation.Relu, eps, S.TORCH), want, S.TORCH)
    end

    S.expect_close(try got, try S.rows(try JSON.get(want, "y")), TORCH_ABS, TORCH_REL, "torch decoder_block")
  end
end

## Whisper's names for the layers of decoder block 0.
def decoder_names() -> Load.DecoderBlockNames
  p = "decoder.layers.0."
  Load.DecoderBlockNames(
    norm1: "#{p}self_attn_layer_norm",
    query: "#{p}self_attn.q_proj",
    key: "#{p}self_attn.k_proj",
    value: "#{p}self_attn.v_proj",
    output: "#{p}self_attn.out_proj",
    norm2: "#{p}encoder_attn_layer_norm",
    cross_query: "#{p}encoder_attn.q_proj",
    cross_key: "#{p}encoder_attn.k_proj",
    cross_value: "#{p}encoder_attn.v_proj",
    cross_output: "#{p}encoder_attn.out_proj",
    norm3: "#{p}final_layer_norm",
    up: "#{p}fc1",
    down: "#{p}fc2",
  )
end

## The block over the peer's `x` (3 positions) and `memory` (5 positions).
def decode(block: Nn.DecoderBlock, want: JSON.Value, backend: Tensor.Backend) -> Result[List[Float]]
  x = try S.f32(try S.rows(try JSON.get(want, "x")), [1, 3, 4], backend)
  memory = try S.f32(try S.rows(try JSON.get(want, "memory")), [1, 5, 4], backend)
  S.floats_of(try Nn.decoder_block(block, x, memory))
end

def forward(dense: Nn.Dense, norm: Nn.LayerNorm, x: List[Float], backend: Tensor.Backend) -> Result[List[Float]]
  h = try Nn.dense(dense, try S.f32(x, [2, 4], backend))
  S.floats_of(try Nn.layer_norm(norm, h))
end

## Without the native module, loading on libtorch is an error, not a halt.
def expect_absent(r: Result[Nn.Dense]) -> Result[Void]
  case r
    ok(_) -> error("a torch layer loaded without the native module")
    error(_) -> ok()
  end
end

## `d.w` transposed is the peer's `[in, out]` weight, and `d.b` its bias or `none`.
def same_dense(d: Nn.Dense, want: JSON.Value) -> Result[Void]
  try Test.eq(try S.floats_of(try Tensor.transpose(d.w)), try S.floats(try JSON.get(want, "w")))
  b = try JSON.get(want, "b")
  case d.b
    some(t) -> Test.eq(try S.floats_of(t), try S.floats(b))
    none -> Test.assert(JSON.null?(b), "a bias the peer wrote did not load")
  end
end

def same_norm(n: Nn.LayerNorm, want: JSON.Value) -> Result[Void]
  try Test.eq(try S.floats_of(n.gamma), try S.floats(try JSON.get(want, "gamma")))
  Test.eq(try S.floats_of(n.beta), try S.floats(try JSON.get(want, "beta")))
end
