## The reader against the Python peer: every dtype it writes reads back as
## the same values, malformed files are refused with a reason, and a sharded
## index finds each tensor in its shard.

import Hub from "../hub"
import S from "support"
import Safetensors from "../safetensors"
import Tensor from "pkg/tensor/tensor"

private

def test_every_dtype_reads_as_the_peer_wrote_it() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/values.safetensors")
    try same_as_peer(c, try S.json(dir, "values.json"))
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
        try Test.eq(bits(try Tensor.to_list(got)), bits(try Tensor.to_list(want)))
      end
    end

    ok()
  end
end

def test_a_dtype_tensor_lacks_is_refused_alone() -> Result[Void]
  S.with_fixtures("values") do |dir|
    c = try Hub.open("#{dir}/u16.safetensors")
    try expect_error(Hub.tensor(c, "w", S.REF), "U16")
    try Test.eq(try Tensor.to_list(try Hub.tensor(c, "ok", S.REF)), [Tensor.Values.Float(1.0)])
    c2 = try Hub.open("#{dir}/bad_bool.safetensors")
    expect_error(Hub.tensor(c2, "b", S.REF), "not 0 or 1")
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
      try expect_error(Hub.open("#{dir}/#{file}"), cases[file])
    end

    ok()
  end
end

def test_a_sharded_index_reads_each_shard() -> Result[Void]
  S.with_fixtures("sharded") do |dir|
    c = try Hub.open("#{dir}/model.safetensors.index.json")
    try Test.eq(c.shards.size, 2)
    try Test.eq(Hub.names(c), ["enc.a", "enc.b", "enc.c"])
    try same_as_peer(c, try S.json(dir, "sharded.json"))
    try expect_error(Hub.tensor(c, "enc.z", S.REF), "no tensor enc.z")
    bad = try JSON.decode(try S.json(dir, "bad_index.json"), Map[String, String])
    try Test.eq(bad.size, 6)
    for file in bad.keys
      try expect_error(Hub.open("#{dir}/#{file}"), bad[file])
    end

    ok()
  end
end

## Every tensor of `expected` (name -> dtype, shape, values) reads back the
## same from `c`.
def same_as_peer(c: Hub.Checkpoint, expected: JSON.Value) -> Result[Void]
  fields: Map[String, JSON.Value] = {}
  case JSON.node(expected)
    JSON.Node.Object(m) -> fields = m
    _ -> return error("the expected values are not a JSON object")
  end

  try Test.eq(Hub.names(c), fields.keys.sort)
  for name in fields.keys
    want = fields[name]
    t = try Hub.tensor(c, name, S.REF)
    try Test.eq(Tensor.dtype(t), loaded_dtype(try JSON.string(try JSON.get(want, "dtype"))))
    try Test.eq(Tensor.shape(t), try JSON.decode(try JSON.get(want, "shape"), List[Int]))
    try Test.eq(bits(try Tensor.to_list(t)), bits(try values(try JSON.get(want, "values"), Tensor.dtype(t))))
  end

  ok()
end

## F32, F16 and BF16 load as `F32`; the other dtypes as the file has them.
def loaded_dtype(file_dtype: String) -> Tensor.Dtype
  return Tensor.Dtype.F32 if ["F32", "F16", "BF16"].include?(file_dtype)
  Safetensors.DTYPES[file_dtype]
end

## The peer's values: floats come as text, so `inf` and `nan` fit in JSON.
def values(json: JSON.Value, dtype: Tensor.Dtype) -> Result[List[Tensor.Values]]
  out: List[Tensor.Values] = []
  for v in try JSON.list(json)
    case dtype
      Tensor.Dtype.F32 -> out = out.push(Tensor.Values.Float(try float(try JSON.string(v))))
      Tensor.Dtype.F64 -> out = out.push(Tensor.Values.Float(try float(try JSON.string(v))))
      Tensor.Dtype.Bool -> out = out.push(Tensor.Values.Bool(try JSON.bool(v)))
      _ -> out = out.push(Tensor.Values.Int(try JSON.int(v)))
    end
  end

  ok(out)
end

def float(text: String) -> Result[Float]
  inf = Float.from_bits(0x7FF0000000000000)
  case text
    "inf" -> ok(inf)
    "-inf" -> ok(-inf)
    "nan" -> ok(inf - inf)
    _ -> text.to_f
  end
end

## Floats as their bits, so `-0.0` differs from `0.0` and NaN equals NaN
## (`to_bits` gives every NaN one pattern: payloads are not compared).
def bits(vs: List[Tensor.Values]) -> List[String]
  vs.map do |v|
    case v
      Tensor.Values.Float(f) -> "f#{Float.to_bits(f)}"
      Tensor.Values.Int(i) -> "i#{i}"
      Tensor.Values.Bool(b) -> "b#{b}"
    end
  end
end

def expect_error[T](r: Result[T], phrase: String) -> Result[Void]
  case r
    ok(_) -> error("expected an error with \"#{phrase}\", got a value")
    error(e) -> Test.assert(e.message.include?(phrase), "\"#{e.message}\" does not hold \"#{phrase}\"")
  end
end
