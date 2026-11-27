## The GGUF reader against the Python peer: every type of key-value and of tensor reads as the peer wrote it, the names
## are Hugging Face's, malformed files are refused with a reason, and a header longer than the first read is read whole.

import Gguf from "../gguf"
import Hub from "../hub"
import Load from "../load"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

private

def test_every_tensor_reads_as_the_peer_wrote_it() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    c = try Hub.open("#{dir}/model.gguf")
    try S.same_as_peer(c, try S.json(dir, "model.json"))
    # GGUF's first dimension is the last of the shape
    try Test.eq(try Hub.shape(c, "model.embed_tokens.weight"), [3, 2])
    ok()
  end
end

def test_the_key_values_read_as_the_peer_wrote_them() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    h = (try Hub.open("#{dir}/model.gguf")).gguf.unwrap!
    try Test.eq(try Gguf.string(h, "general.architecture"), "qwen3")
    try Test.eq(try Gguf.string(h, "t.str"), "héllo")
    try Test.eq(try Gguf.int(h, "t.u8"), 200)
    try Test.eq(try Gguf.int(h, "t.i8"), -5)
    try Test.eq(try Gguf.int(h, "t.u16"), 65000)
    try Test.eq(try Gguf.int(h, "t.i16"), -30000)
    try Test.eq(try Gguf.int(h, "t.u32"), 4000000000)
    try Test.eq(try Gguf.int(h, "t.i32"), -2000000000)
    try Test.eq(try Gguf.int(h, "t.u64"), 1099511627776)
    try Test.eq(try Gguf.int(h, "t.i64"), -1099511627776)
    try Test.eq(try Gguf.float(h, "t.f32"), 0.5)
    try Test.eq(try Gguf.float(h, "t.f64"), 0.1)
    try Test.eq(try Gguf.float(h, "qwen3.rope.freq_base"), 1000000.0)
    try S.expect_error(Gguf.int(h, "t.str"), "not an integer")
    try S.expect_error(Gguf.string(h, "t.u8"), "not a string")
    try S.expect_error(Gguf.float(h, "t.u8"), "not a float")
    S.expect_error(Gguf.int(h, "t.missing"), "no key t.missing")
  end
end

def test_an_array_reads_from_the_file_when_asked() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    h = (try Hub.open("#{dir}/model.gguf")).gguf.unwrap!
    try Test.eq(try Gguf.strings(h, "t.arr_str"), ["a", "bc", ""])
    try Test.eq(try Gguf.ints(h, "t.arr_u32"), [1, 2, 3])
    try Test.eq(try Gguf.ints(h, "t.arr_i16"), [-2, 7])
    try Test.eq(try Gguf.ints(h, "t.arr_empty"), [])
    try S.expect_error(Gguf.strings(h, "t.arr_u32"), "not an array of strings")
    try S.expect_error(Gguf.ints(h, "t.arr_str"), "not an array of integers")
    try S.expect_error(Gguf.strings(h, "t.str"), "not an array")
    try S.expect_error(Gguf.ints(h, "t.missing"), "no key t.missing")
    big = (try Hub.open("#{dir}/big.gguf")).gguf.unwrap!
    try S.expect_error(Gguf.ints(big, "big"), "more than 1000000")
    ok()
  end
end

def test_a_header_longer_than_the_first_read_is_read_whole() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    c = try Hub.open("#{dir}/big.gguf")
    try S.same_as_peer(c, try S.json(dir, "big.json"))
    ok()
  end
end

def test_the_native_backends_read_floats_from_a_gguf_as_the_reference_does() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    c = try Hub.open("#{dir}/model.gguf")
    for backend in try S.native()
      for name in ["model.embed_tokens.weight", "lm_head.weight", "model.layers.0.self_attn.q_proj.weight"]
        want = try Hub.tensor(c, name, S.REF)
        got = try Hub.tensor(c, name, backend)
        try Test.eq(Tensor.dtype(got), Tensor.Dtype.F32)
        try Test.eq(Tensor.shape(got), Tensor.shape(want))
        try Test.eq(S.bits(try Tensor.to_list(got)), S.bits(try Tensor.to_list(want)))
      end
    end

    ok()
  end
end

def test_an_f16_gguf_keeps_half_weights_on_ggml() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    small = try Hub.open("#{dir}/model.gguf")
    big = try Hub.open("#{dir}/half.gguf")
    for backend in try S.ggml_backends()
      # the F16 tensor of the file at its offset, as a half weight
      half = try Hub.tensor(small, "lm_head.weight", backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(half), Tensor.Dtype.F16)
      ids = try Tensor.from_list([Tensor.Values.Int(0), Tensor.Values.Int(1)], [2], Tensor.Dtype.I32, backend)
      widened = try Trace.eval_many([ids, half]) do |a|
        ok([try Tensor.to_dtype(try Tensor.embedding(a[0], a[1]), Tensor.Dtype.F32)])
      end

      try Test.eq(try S.floats_of(widened[0]), [1.0, -2.0, 0.5, 65504.0])
      # Load keeps the big ones half: an F16 table, a BF16 matrix, and not an F32 one
      table = try Load.embedding(big, "model.embed_tokens.weight", backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(table.table), Tensor.Dtype.F16)
      try Test.eq(Tensor.shape(table.table), [400, 300])
      head = try Load.dense(big, "lm_head", backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(head.w), Tensor.Dtype.F16)
      q = try Load.dense(big, "model.layers.0.self_attn.q_proj", backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(q.w), Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype((try Load.dense(big, "model.layers.0.self_attn.q_proj", backend)).w), Tensor.Dtype.F32)
    end

    ok()
  end
end

## The values the peer wrote for the tensor `name` (floats as text).
def peer_floats(want: JSON.Value, name: String) -> Result[List[Float]]
  out: List[Float] = []
  for v in try JSON.list(try JSON.get(try JSON.get(want, name), "values"))
    out = out.push(try S.float(try JSON.string(v)))
  end

  ok(out)
end

## Q8_0 tensors read as the peer dequantizes them (a half scale times the byte), on every backend that reads a file.
def test_a_q8_0_tensor_reads_as_the_peer_dequantizes_it() -> Result[Void]
  S.with_fixtures("gguf_q8") do |dir|
    c = try Hub.open("#{dir}/q8.gguf")
    want = try S.json(dir, "q8.json")
    try S.same_as_peer(c, want)
    try Test.eq(try Hub.shape(c, "model.layers.0.mlp.up_proj.weight"), [1, 32])
    try Test.eq(try Hub.dtype(c, "model.layers.0.mlp.up_proj.weight"), "Q8_0")
    for backend in try S.native()
      for name in Hub.names(c)
        t = try Hub.tensor(c, name, backend)
        expected = try peer_floats(want, name)
        try Test.eq(try S.floats_of(t), expected)
      end
    end

    ok()
  end
end

## `held_bytes` of a Q8_0 matrix under `weights: Q8_0` is its blocks (34 bytes for 32 values); under F32 it is the values
## it is read as, and a norm is F32 in both.
def test_held_bytes_of_q8_0_blocks() -> Result[Void]
  S.with_fixtures("gguf_q8") do |dir|
    c = try Hub.open("#{dir}/q8.gguf")
    q = "model.layers.0.self_attn.q_proj.weight"
    try Test.eq(try Load.held_bytes(c, q, Tensor.Dtype.Q8_0), ((3 * 96) / 32) * 34)
    try Test.eq(try Load.held_bytes(c, q, Tensor.Dtype.F32), 3 * 96 * 4)
    try Test.eq(try Load.held_bytes(c, "model.layers.0.input_layernorm.weight", Tensor.Dtype.Q8_0), 64 * 4)
    large = try Hub.open("#{dir}/q8-large.gguf")
    bf16 = Tensor.Dtype.BF16
    try Test.eq(try Load.held_bytes(large, q, bf16), 320 * 320 * 4)
    for backend in try S.ggml_backends()
      dense = try Load.dense(large, "model.layers.0.self_attn.q_proj", backend, bf16)
      try Test.eq(Tensor.dtype(dense.w), Tensor.Dtype.F32)
    end

    ok()
  end
end

## With `weights: Q8_0` the Q8_0 matrices of a file stay as blocks on ggml (they hold the same values), and nothing else
## changes: a norm stays F32, a Q8_0 matrix read as F32 or F16 is F32, and another backend or another dtype is refused.
def test_weights_q8_0_keeps_the_blocks_of_a_q8_0_file() -> Result[Void]
  S.with_fixtures("gguf_q8") do |dir|
    c = try Hub.open("#{dir}/q8.gguf")
    want = try S.json(dir, "q8.json")
    q = "model.layers.0.self_attn.q_proj"
    for backend in try S.ggml_backends()
      kept = try Load.dense(c, q, backend, Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.dtype(kept.w), Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.shape(kept.w), [3, 96])
      try Test.eq(try S.floats_of(kept.w), try peer_floats(want, "#{q}.weight"))
      table = try Load.embedding(c, "model.embed_tokens.weight", backend, Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.dtype(table.table), Tensor.Dtype.Q8_0)
      norm = try Load.weight(c, "model.layers.0.input_layernorm.weight", 1, backend)
      try Test.eq(Tensor.dtype(norm), Tensor.Dtype.F32)
      try Test.eq(Tensor.dtype((try Load.dense(c, q, backend)).w), Tensor.Dtype.F32)
      try Test.eq(Tensor.dtype((try Load.dense(c, q, backend, Tensor.Dtype.F16)).w), Tensor.Dtype.F32)
      try S.expect_error(
        Hub.tensor(c, "model.layers.0.input_layernorm.weight", backend, Tensor.Dtype.Q8_0),
        "only a Q8_0",
      )
    end

    try S.expect_error(Load.dense(c, q, S.REF, Tensor.Dtype.Q8_0), "needs the ggml backend")
    S.expect_error(Hub.tensor(c, "model.embed_tokens.weight", S.REF, Tensor.Dtype.Q8_0), "ggml")
  end
end

## Q4_K and Q6_K tensors of random blocks, written by the official `gguf` package (`kquants_golden.py`), read as it
## dequantizes them (float32 values; the bound is a few ulps of values under 16), on every backend that reads a file.
def test_a_k_quant_tensor_reads_as_the_gguf_package_dequantizes_it() -> Result[Void]
  c = try Hub.open("test/fixtures/kquants.gguf")
  want = try JSON.parse(try File.read_text("test/fixtures/kquants.json"))
  names = {
    "model.layers.0.self_attn.q_proj.weight": ["blk.0.attn_q.weight", "Q4_K"],
    "model.layers.0.mlp.up_proj.weight": ["blk.0.ffn_up.weight", "Q6_K"],
  }

  for name in names.keys
    peer = try JSON.get(want, names[name][0])
    expected = (try JSON.decode(try JSON.get(peer, "bits"), List[Int])).map { |b: Int|
      Float32.from_bits(b.to_uint32!).to_f
    }

    try Test.eq(try Hub.dtype(c, name), names[name][1])
    try Test.eq(try Hub.shape(c, name), try JSON.decode(try JSON.get(peer, "shape"), List[Int]))
    for backend in [S.REF] + try S.native()
      t = try Hub.tensor(c, name, backend)
      try Test.eq(Tensor.dtype(t), Tensor.Dtype.F32)
      try S.expect_close(try S.floats_of(t), expected, 0.000004, 0.0, "#{name} on #{backend}")
    end
  end

  ok()
end

## With `weights: Q4_K` each rank-2 weight of a file keeps the bit width it is stored in (Q4_K and Q6_K) as blocks on
## ggml, and nothing else changes: `weights: Q8_0` keeps none of them, and Q6_K is no request of its own.
def test_weights_q4_k_keeps_each_tensor_at_its_own_bit_width() -> Result[Void]
  c = try Hub.open("test/fixtures/kquants.gguf")
  four = "model.layers.0.self_attn.q_proj"
  six = "model.layers.0.mlp.up_proj"
  for backend in try S.ggml_backends()
    try Test.eq(Tensor.dtype((try Load.dense(c, four, backend, Tensor.Dtype.Q4_K)).w), Tensor.Dtype.Q4_K)
    try Test.eq(Tensor.dtype((try Load.dense(c, six, backend, Tensor.Dtype.Q4_K)).w), Tensor.Dtype.Q6_K)
    table = try Load.embedding(c, "#{six}.weight", backend, Tensor.Dtype.Q4_K)
    try Test.eq(Tensor.dtype(table.table), Tensor.Dtype.Q6_K)
    try Test.eq(Tensor.shape(table.table), [2, 512])
    try Test.eq(Tensor.dtype((try Load.dense(c, four, backend)).w), Tensor.Dtype.F32)
    try Test.eq(Tensor.dtype((try Load.dense(c, four, backend, Tensor.Dtype.Q8_0)).w), Tensor.Dtype.F32)
    try Test.eq(Tensor.dtype(try Hub.tensor(c, "#{six}.weight", backend, Tensor.Dtype.Q6_K)), Tensor.Dtype.Q6_K)
    try S.expect_error(Hub.tensor(c, "#{six}.weight", backend, Tensor.Dtype.Q4_K), "only a Q4_K")
    try S.expect_error(Hub.tensor(c, "#{four}.weight", backend, Tensor.Dtype.Q8_0), "only a Q8_0")
    try S.expect_error(Load.dense(c, four, backend, Tensor.Dtype.Q6_K), "weights is")
  end

  S.expect_error(Load.dense(c, four, S.REF, Tensor.Dtype.Q4_K), "needs the ggml backend")
end

## A quantized tensor that is all of the data: sized in blocks, not in values, so the file opens (a type that is not read
## is refused when it is asked for).
def test_a_quantized_tensor_that_fills_the_data_opens() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    c = try Hub.open("#{dir}/q5k.gguf")
    try Test.eq(try Hub.shape(c, "model.layers.0.mlp.up_proj.weight"), [1, 256])
    S.expect_error(Hub.tensor(c, "model.layers.0.mlp.up_proj.weight", S.REF), "Q5_K")
  end
end

def test_malformed_files_are_refused() -> Result[Void]
  S.with_fixtures("gguf_bad") do |dir|
    cases = try JSON.decode(try S.json(dir, "cases.json"), Map[String, String])
    try Test.eq(cases.size, 27)
    for file in cases.keys
      try S.expect_error(Hub.open("#{dir}/#{file}"), cases[file])
    end

    ok()
  end
end
