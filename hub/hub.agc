## The `hub/` package: checkpoints in safetensors files (`hub/DESIGN.md`).
##
## `open` takes one `.safetensors` file or the `.index.json` of a sharded
## checkpoint; `tensor` reads one tensor by name from the shard that holds
## it. `load.agc` builds `nn/` layers from those tensors.

import Safetensors from "safetensors"
import Tensor from "pkg/tensor/tensor"

## A checkpoint: its shards by file name (no folder) and the file of every
## tensor.
struct Checkpoint
  shards: Map[String, Safetensors.Shard]
  weight_map: Map[String, String]
end

## The JSON form of a sharded checkpoint's index.
struct Index
  weight_map: Map[String, String]
end

## Opens a `.safetensors` file, or every shard of an `.index.json`.
def open(path: String) -> Result[Checkpoint]
  return open_index(path) if path.end_with?(".json")
  shard = try Safetensors.open(path)
  file = path.split("/").last
  weight_map: Map[String, String] = {}
  for name in shard.entries.keys
    weight_map = weight_map.put(name, file)
  end

  ok(Checkpoint(shards: {file: shard}, weight_map: weight_map))
end

def open_index(path: String) -> Result[Checkpoint]
  index = try within(path, JSON.decode(try File.read_text(path), Index))

  parts = path.split("/")
  dir = parts.take(parts.size - 1).push("").join("/")
  shards: Map[String, Safetensors.Shard] = {}
  for name in index.weight_map.keys
    file = index.weight_map[name]
    if not shards.key?(file)
      if file.include?("/") or file.include?("\\") or file == ".." or file.empty?
        return error("#{path}: shard #{file} is not a file name in the index's folder")
      end

      shards = shards.put(file, try Safetensors.open(dir + file))
    end

    return error("#{path}: tensor #{name} is not in its shard #{file}") if not shards[file].entries.key?(name)
  end

  ok(Checkpoint(shards: shards, weight_map: index.weight_map))
end

## Every tensor name, sorted.
def names(c: Checkpoint) -> List[String] = c.weight_map.keys.sort

def has?(c: Checkpoint, name: String) -> Bool = c.weight_map.key?(name)

## The tensor `name` on `backend`: a float one as `F32`, or as `F16` with `dtype: F16` (a half weight, ggml only).
def tensor(
  c: Checkpoint,
  name: String,
  backend: Tensor.Backend,
  dtype: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Tensor.Tensor]
  file = try c.weight_map.get(name).to_result("no tensor #{name} in the checkpoint")
  Safetensors.tensor(c.shards[file], name, backend, dtype)
end

## The shape of the tensor `name`, from the header: nothing is read.
def shape(c: Checkpoint, name: String) -> Result[List[Int]]
  file = try c.weight_map.get(name).to_result("no tensor #{name} in the checkpoint")
  Safetensors.shape(c.shards[file], name)
end

## The `__metadata__` of every shard, by file name.
def metadata(c: Checkpoint) -> Map[String, Map[String, String]]
  out: Map[String, Map[String, String]] = {}
  for file in c.shards.keys
    out = out.put(file, c.shards[file].metadata)
  end

  out
end

private

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
end
