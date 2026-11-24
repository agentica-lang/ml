## The reader against the Python peer: every dtype it writes reads back as
## the same values, malformed files are refused with a reason, and a sharded
## index finds each tensor in its shard.

import Hub from "../hub"
import S from "support"
import Tensor from "pkg/tensor/tensor"

private

def test_every_dtype_reads_as_the_peer_wrote_it() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/values.safetensors")
    try S.same_as_peer(c, try S.json(dir, "values.json"))
    try Test.eq(Hub.metadata(c)["values.safetensors"], {"format": "pt"})
    try Test.eq(Tensor.shape(try Hub.tensor(c, "empty", S.REF)), [0, 3])
    Test.eq(Tensor.shape(try Hub.tensor(c, "empty_last", S.REF)), [3, 0])
  end
end

def test_the_native_backends_read_floats_from_the_file_as_the_reference_does() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/values.safetensors")
    for backend in S.native()
      for name in ["f32", "f16", "bf16", "scalar", "empty"]
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

def test_a_shape_that_overflows_an_int_is_no_crash() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/hostile.safetensors")
    for backend in [S.REF].concat(S.native())
      _ = Hub.tensor(c, "h", backend)
    end

    ok()
  end
end

def test_a_dtype_tensor_lacks_is_refused_alone() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/u16.safetensors")
    try S.expect_error(Hub.tensor(c, "w", S.REF), "U16")
    try Test.eq(try Tensor.to_list(try Hub.tensor(c, "ok", S.REF)), [Tensor.Values.Float(1.0)])
    c2 = try Hub.open("#{dir}/bad_bool.safetensors")
    S.expect_error(Hub.tensor(c2, "b", S.REF), "not 0 or 1")
  end
end

def test_a_tensor_past_one_chunk_of_the_file_reads_from_its_offset() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/chunks.safetensors")
    Test.eq(try Tensor.to_list(try Hub.tensor(c, "small", S.REF)), [Tensor.Values.Int(7), Tensor.Values.Int(-9)])
  end
end

def test_malformed_files_are_refused() -> Result[Void]
  S.with_fixtures("malformed") do |dir|
    cases = try JSON.decode(try S.json(dir, "malformed.json"), Map[String, String])
    try Test.eq(cases.size, 21)
    for file in cases.keys
      try S.expect_error(Hub.open("#{dir}/#{file}"), cases[file])
    end

    ok()
  end
end

def test_a_sharded_index_reads_each_shard() -> Result[Void]
  S.with_fixtures("sharded") do |dir|
    c = try Hub.open("#{dir}/model.safetensors.index.json")
    try Test.eq(c.shards.size, 2)
    try Test.eq(Hub.names(c), ["enc.a", "enc.b", "enc.c"])
    try S.same_as_peer(c, try S.json(dir, "sharded.json"))
    try S.expect_error(Hub.tensor(c, "enc.z", S.REF), "no tensor enc.z")
    bad = try JSON.decode(try S.json(dir, "bad_index.json"), Map[String, String])
    try Test.eq(bad.size, 6)
    for file in bad.keys
      try S.expect_error(Hub.open("#{dir}/#{file}"), bad[file])
    end

    ok()
  end
end

## Every tensor of `expected` (name -> dtype, shape, values) reads back the
## same from `c`.
def test_an_f64_tensor_comes_as_f32_on_ggml() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/values.safetensors")
    for backend in try S.ggml_backends()
      t = try Hub.tensor(c, "f64", backend)
      try Test.eq(Tensor.dtype(t), Tensor.Dtype.F32)
      # the narrowing is silent: 0.1 rounds to F32 and -1e300 is -inf
      inf = Float.from_bits(0x7FF0000000000000)
      try Test.eq(try Tensor.to_list(t), [Tensor.Values.Float(0.10000000149011612), Tensor.Values.Float(-inf)])
    end

    ok()
  end
end
