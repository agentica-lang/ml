## The safetensors writer (`safetensors_write.agc`): its bytes are the format's (the header is pinned here, byte for
## byte), the reader of `safetensors.agc` reads back what it wrote on each backend, and writing what was read gives the
## same file.

import Hub from "../hub"
import S from "support"
import Shards from "../shard"
import Tensor from "pkg/tensor/tensor"
import Write from "../safetensors_write"

## 1.5 and -2.25, then 0.1, 3.0 and -0.0, as little-endian F32.
DATA = [0, 0, 192, 63, 0, 0, 16, 192, 205, 204, 204, 61, 0, 0, 64, 64, 0, 0, 0, 128]

private

def with_dir(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-hub-write-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

## The header of `sample`, before its padding.
def header() -> String
  [
    "{\"__metadata__\":{\"format\":\"pt\"}",
    "\"a.b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}",
    "\"w\":{\"dtype\":\"F32\",\"shape\":[1,3],\"data_offsets\":[8,20]}",
    "\"z\":{\"dtype\":\"F32\",\"shape\":[0,4],\"data_offsets\":[20,20]}}",
  ].join(",")
end

def backends() -> Result[List[Tensor.Backend]] = ok([S.REF].concat(try S.native()))

## The names are given out of order: the file sorts them.
def sample(backend: Tensor.Backend) -> Result[Map[String, Tensor.Tensor]]
  ok({
    "z": try S.f32([], [0, 4], backend),
    "w": try S.f32([0.1, 3.0, -0.0], [1, 3], backend),
    "a.b": try S.f32([1.5, -2.25], [2], backend),
  })
end

def bytes_of(path: String) -> Result[List[Int]]
  size = (try File.stat(path)).size_bytes
  ok((try Shards.read_range(path, 0, size)).to_a)
end

def test_the_file_is_the_format_byte_for_byte_on_each_backend() -> Result[Void]
  with_dir do |dir|
    for backend in try backends()
      path = "#{dir}/out.safetensors"
      try Write.write(path, try sample(backend), {"format": "pt"})
      padded = header() + " " * ((8 - header().bytes.size % 8) % 8)
      want = [padded.bytes.size, 0, 0, 0, 0, 0, 0, 0] + padded.bytes.to_a + DATA
      try Test.eq(try bytes_of(path), want)
      try Test.assert(not Path.real(path + ".part").ok?, "the part file is gone")
    end

    ok()
  end
end

def test_what_is_read_writes_back_the_same_bytes() -> Result[Void]
  with_dir do |dir|
    first = "#{dir}/first.safetensors"
    try Write.write(first, try sample(S.REF), {"format": "pt"})
    for backend in try backends()
      c = try Hub.open(first)
      read = try Hub.tensors(c, backend)
      try Test.eq(read.keys.sort, ["a.b", "w", "z"])
      try Test.eq(Tensor.shape(read["w"]), [1, 3])
      try Test.eq(S.bits(try Tensor.to_list(read["w"])), S.bits(try Tensor.to_list(try Hub.tensor(c, "w", S.REF))))
      second = "#{dir}/second.safetensors"
      try Write.write(second, read, Hub.metadata(c)["first.safetensors"])
      try Test.eq(try bytes_of(second), try bytes_of(first))
    end

    ok()
  end
end

def test_a_file_with_no_tensor_and_no_metadata_opens() -> Result[Void]
  with_dir do |dir|
    path = "#{dir}/empty.safetensors"
    nothing: Map[String, Tensor.Tensor] = {}
    try Write.write(path, nothing)
    try Test.eq(try bytes_of(path), [8, 0, 0, 0, 0, 0, 0, 0] + "{}      ".bytes.to_a)
    Test.eq(Hub.names(try Hub.open(path)), [])
  end
end

def test_a_tensor_that_is_not_f32_is_refused_and_leaves_no_file() -> Result[Void]
  with_dir do |dir|
    path = "#{dir}/bad.safetensors"
    ids = try Tensor.arange(0, 3, 1, Tensor.Dtype.I32, S.REF)
    try S.expect_error(Write.write(path, {"ids": ids}), "tensor ids")
    try S.expect_error(Write.write(path, {"__metadata__": try S.f32([1.0], [1], S.REF)}), "__metadata__")
    try S.expect_error(Write.write("#{dir}/missing/x.safetensors", try sample(S.REF)), "missing")
    try Test.assert(not Path.real(path).ok?, "no file was written")
    Test.assert(not Path.real(path + ".part").ok?, "no part file was left")
  end
end
