## The GGUF writer (`gguf_write.agc`) against a Python peer: `peer.py gguf_write` writes a safetensors file of every
## float format with values that make the rounding matter, and `gguf_verify.py` reads the file the writer made, with the
## standard library alone, and compares its header, key-values, names, dimensions, alignment and every byte of every
## tensor with the source rounded to the target format. The file is also read back with the reader of `gguf.agc`.

import Gguf from "../gguf"
import GgufWrite from "../gguf_write"
import Hub from "../hub"
import S from "support"
import Shards from "../shard"
import Tensor from "pkg/tensor/tensor"

## One key-value of `spec.json`: its key and the value, under the name of its kind.
struct Kv
  key: String
  u32: Optional[Int]
  f32: Optional[Float]
  bool: Optional[Bool]
  text: Optional[String]
  texts: Optional[List[String]]
  ints: Optional[List[Int]]
end

struct Item
  out: String
  src: String
  format: String
end

struct Spec
  kvs: List[Kv]
  tensors: List[Item]
  refused: Item
end

private

def spec_of(dir: String) -> Result[Spec] = JSON.decode(try File.read_text("#{dir}/spec.json"), Spec)

def format_of(name: String) -> Result[Tensor.FloatFormat]
  case name
    "F32" -> ok(Tensor.FloatFormat.F32)
    "F16" -> ok(Tensor.FloatFormat.F16)
    "BF16" -> ok(Tensor.FloatFormat.BF16)
    _ -> error("no format #{name}")
  end
end

def field_of(kv: Kv) -> Result[GgufWrite.Field]
  value = GgufWrite.Meta.Bool(false)
  if kv.u32.some?
    value = GgufWrite.Meta.U32(kv.u32.unwrap!)
  elsif kv.f32.some?
    value = GgufWrite.Meta.F32(kv.f32.unwrap!)
  elsif kv.bool.some?
    value = GgufWrite.Meta.Bool(kv.bool.unwrap!)
  elsif kv.text.some?
    value = GgufWrite.Meta.Text(kv.text.unwrap!)
  elsif kv.texts.some?
    value = GgufWrite.Meta.Texts(kv.texts.unwrap!)
  elsif kv.ints.some?
    value = GgufWrite.Meta.Ints(kv.ints.unwrap!)
  else
    return error("the key-value #{kv.key} has no value")
  end

  ok(GgufWrite.Field(key: kv.key, value: value))
end

def out_of(c: Hub.Checkpoint, item: Item) -> Result[GgufWrite.Out]
  ok(GgufWrite.Out(name: item.out, source: try Hub.source(c, item.src), format: try format_of(item.format)))
end

def fields_of(spec: Spec) -> Result[List[GgufWrite.Field]]
  out: List[GgufWrite.Field] = []
  for kv in spec.kvs
    out = out.push(try field_of(kv))
  end

  ok(out)
end

def outs_of(c: Hub.Checkpoint, spec: Spec) -> Result[List[GgufWrite.Out]]
  out: List[GgufWrite.Out] = []
  for item in spec.tensors
    out = out.push(try out_of(c, item))
  end

  ok(out)
end

## `t` with the shape `shape` in its header, the place of its values the same.
def reshaped(t: GgufWrite.Out, shape: List[Int]) -> GgufWrite.Out
  s = t.source
  GgufWrite.Out(
    name: t.name,
    source: Shards.Source(path: s.path, offset: s.offset, format: s.format, shape: shape),
    format: t.format,
  )
end

def test_the_file_is_what_an_independent_reader_finds() -> Result[Void]
  S.with_fixtures("gguf_write") do |dir|
    c = try Hub.open("#{dir}/src.safetensors")
    spec = try spec_of(dir)
    try GgufWrite.write("#{dir}/out.gguf", try fields_of(spec), try outs_of(c, spec))
    try Test.eq(Path.real("#{dir}/out.gguf.part").ok?, false)
    p = try Process.spawn_captured("python3", ["-I", "test/gguf_verify.py", dir])
    out = try Process.output(p, 30000)
    code = try Process.wait(p, 30000)
    try Test.assert(code == 0, "gguf_verify.py: #{out.stdout}#{out.stderr}")
    Test.eq(out.stdout.strip, "ok #{spec.tensors.size}")
  end
end

## What the reader of `gguf.agc` finds: the key-values, the arrays, the shapes, and every F32 tensor equal to its source
## (the rounded ones are compared with their source, byte by byte, by `gguf_verify.py`).
def test_the_reader_finds_what_was_written() -> Result[Void]
  S.with_fixtures("gguf_write") do |dir|
    c = try Hub.open("#{dir}/src.safetensors")
    spec = try spec_of(dir)
    try GgufWrite.write("#{dir}/out.gguf", try fields_of(spec), try outs_of(c, spec))
    back = try Hub.open("#{dir}/out.gguf")
    h = back.gguf.unwrap!
    try Test.eq(try Gguf.int(h, "t.u32"), 4000000000)
    try Test.eq(try Gguf.float(h, "t.f32"), 0.5)
    try Test.eq(try Gguf.string(h, "t.text"), "héllo")
    try Test.eq(try Gguf.strings(h, "t.texts"), ["a", "bc", "", "dé"])
    try Test.eq(try Gguf.ints(h, "t.ints"), [0, -1, 2147483647, -2147483648])
    try Test.eq(try Hub.shape(back, "i.f32"), [2, 2, 2, 2])
    try Test.eq(try Hub.shape(back, "b.f16"), [3, 5])
    for item in spec.tensors
      if item.format == "F32"
        want = try S.floats_of(try Hub.tensor(c, item.src, S.REF))
        try Test.eq(try S.floats_of(try Hub.tensor(back, item.out, S.REF)), want)
      end
    end

    ok()
  end
end

def test_a_value_beyond_f16_a_bad_shape_or_a_name_twice_is_refused() -> Result[Void]
  S.with_fixtures("gguf_write") do |dir|
    c = try Hub.open("#{dir}/src.safetensors")
    spec = try spec_of(dir)
    path = "#{dir}/refused.gguf"
    try S.expect_error(GgufWrite.write(path, [], [try out_of(c, spec.refused)]), "outside what F16 holds")
    try Test.eq(Path.real(path).ok?, false)
    try Test.eq(Path.real(path + ".part").ok?, false)
    a = try out_of(c, spec.tensors[0])
    try S.expect_error(GgufWrite.write(path, [], [a, a]), "appears twice")
    key = GgufWrite.Field(key: "k", value: GgufWrite.Meta.U32(1))
    try S.expect_error(GgufWrite.write(path, [key, key], [a]), "appears twice")
    try S.expect_error(GgufWrite.write(path, [], [reshaped(a, [1, 1, 1, 1, 6])]), "5 dimensions")
    try S.expect_error(GgufWrite.write(path, [], [reshaped(a, [0, 6])]), "dimension of 0")
    quantized = GgufWrite.Out(name: a.name, source: a.source, format: Tensor.FloatFormat.Q8_0)
    try S.expect_error(GgufWrite.write(path, [], [quantized]), "not Q8_0")
    try S.expect_error(Hub.source(c, "n"), "not F32, F16 or BF16")
    S.expect_error(Hub.source(c, "missing"), "no tensor missing")
  end
end
