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
    for backend in S.native()
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

def test_a_quantized_tensor_is_refused_alone() -> Result[Void]
  S.with_fixtures("gguf") do |dir|
    c = try Hub.open("#{dir}/q8.gguf")
    try Test.eq(try Hub.shape(c, "model.layers.0.mlp.up_proj.weight"), [1, 32])
    try S.expect_error(Hub.tensor(c, "model.layers.0.mlp.up_proj.weight", S.REF), "Q8_0")
    Test.eq(Tensor.shape(try Hub.tensor(c, "model.embed_tokens.weight", S.REF)), [2])
  end
end

def test_malformed_files_are_refused() -> Result[Void]
  S.with_fixtures("gguf_bad") do |dir|
    cases = try JSON.decode(try S.json(dir, "cases.json"), Map[String, String])
    for file in cases.keys
      try S.expect_error(Hub.open("#{dir}/#{file}"), cases[file])
    end

    ok()
  end
end
