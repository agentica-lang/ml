## Where `layers_on_device` puts the blocks of a model, its embedding table and its head: the rule of llama.cpp's `-ngl`,
## shared by the models that take the option.

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Tensor from "pkg/tensor/tensor"

## Where the embedding table is, and the head with the final norm.
struct Placement
  table: Tensor.Backend
  head: Tensor.Backend
end

## Where the keys and values of a decode are kept: `Device` where the blocks that make them are (the cache of a block
## on the CPU is in host memory, the others stay on the device), `Host` all in host memory, as llama.cpp's
## `--no-kv-offload` does: the attention then reads them on the CPU. The step of the prompt still makes the whole cache
## on the card, at the capacity of the prompt, and copies it over, so that is the card's peak; the steps after it keep
## the cache in host memory, where it grows.
enum Kv
  Device
  Host
end

## What a model puts on the card, in bytes: each block (with its KV cache, if the caller counts it), the embedding table,
## and the head with the final norm. `tied` says the head is the table: `head` is then the size of the copy of the table
## that `layers_on_device` makes on the card (with every block there the table is the head, and `fit` counts it once).
struct Footprint
  blocks: List[Int]
  table: Int
  head: Int
  tied: Bool
  head_widen: Int = 0
end

## The sides of the table and of the head (with the final norm) of a model of `blocks` blocks opened on `backend` with
## `layers_on_device: n`, as llama.cpp's `-ngl` puts them: with `n` below `blocks` on a Vulkan device the table is on the
## CPU, and the head is on the device for an `n` of 1 or more and on the CPU for 0; with `none` or an `n` of `blocks` or
## more, or on any other backend, both are on `backend`.
def placement(backend: Tensor.Backend, blocks: Int, layers_on_device: Optional[Int]) -> Placement
  return Placement(table: backend, head: backend) if layers_on_device.none? or layers_on_device.unwrap! >= blocks

  Placement(
    table: layer_backend(backend, blocks - 1, layers_on_device),
    head: layer_backend(backend, 0, layers_on_device),
  )
end

## The backend of block `i`: the CPU from block `layers_on_device` on when the model is on a Vulkan device.
def layer_backend(backend: Tensor.Backend, i: Int, layers_on_device: Optional[Int]) -> Tensor.Backend
  return backend if layers_on_device.none? or i < layers_on_device.unwrap!
  case backend
    Tensor.Backend.Ggml(Tensor.Device.Vulkan(_)) -> Tensor.Backend.Ggml(Tensor.Device.Cpu)
    _ -> backend
  end
end

## The `Footprint` of the checkpoint `c` under `weights`, counted from its shapes and dtypes (`Load.held_bytes`) without
## reading a value: block `i` is the tensors named `{layers}{i}.` plus `kv` bytes for its KV cache, `table` and `head` are
## tensors by name, and the head is the table when the checkpoint has no `head`. `norm` is the final norm, counted with
## the head.
def footprint(
  c: Hub.Checkpoint,
  layers: String,
  blocks: Int,
  table: String,
  head: String,
  norm: String,
  weights: Tensor.Dtype,
  kv: Int,
  tied_head: Optional[Bool] = none,
) -> Result[Footprint]
  names = Hub.names(c)
  sizes: List[Int] = []
  for i in 0...blocks
    prefix = "#{layers}#{i}."
    bytes = kv
    for name in names.select { |n| n.start_with?(prefix) }
      bytes = bytes + try Load.held_bytes(c, name, weights) + try Load.bf16_widen_bytes(c, name, weights)
    end

    sizes = sizes.push(bytes)
  end

  held = try Load.held_bytes(c, table, weights)
  tied = tied_head.unwrap_or(not Hub.has?(c, head))
  head_name = head
  if tied
    head_name = table
  end

  head_bytes = try Load.held_bytes(c, head_name, weights) + try Load.held_bytes(c, norm, weights)
  head_widen = try Load.bf16_widen_bytes(c, head_name, weights)
  ok(Footprint(blocks: sizes, table: held, head: head_bytes, tied: tied, head_widen: head_widen))
end

## The largest `layers_on_device` whose layout needs at most `budget` bytes on the card, 0 when not even one block fits.
## The layout is the one `placement` makes: with `n` of 0 nothing is on the card; with `n` below the blocks the first `n`
## blocks and the head (a tied head is a copy of the table there); with every block, the blocks and the table, and the
## head too unless it is the table.
def fit(footprint: Footprint, budget: Int) -> Int
  count = footprint.blocks.size
  used = 0
  fitting = 0
  for n in 1..count
    used = used + footprint.blocks[n - 1]
    need = used + footprint.head
    if n == count and not footprint.tied
      need = need + footprint.table
    end

    if need + footprint.head_widen <= budget
      fitting = n
    end
  end

  fitting
end

## The bytes a model may take on the card: `free` less the `reserve` that the compute buffers and the rest of the process
## need, and not more than `cap` when the user set one; never below 0.
def budget(free: Int, reserve: Int, cap: Optional[Int]) -> Int
  room = free - reserve
  if cap.some? and cap.unwrap! < room
    room = cap.unwrap!
  end

  return 0 if room < 0
  room
end

## The `Kv` of the `AGENTICA_ML_KV` variable that the tools read (`host` or `device`, `device` when it is not set).
def kv_from_env() -> Result[Kv] = ok(kv_of(try Load.kv_host_from_env()))

## `Host` for true, `Device` for false.
def kv_of(host: Bool) -> Kv
  return Kv.Host if host
  Kv.Device
end

## The backend a cache tensor made on `backend` is kept on under `kv`: `Host` moves a tensor of a Vulkan device to the
## CPU; any other tensor stays where it is.
def cache_backend(kv: Kv, backend: Tensor.Backend) -> Tensor.Backend
  case kv
    Kv.Device -> backend
    Kv.Host ->
      case backend
        Tensor.Backend.Ggml(Tensor.Device.Vulkan(_)) -> Tensor.Backend.Ggml(Tensor.Device.Cpu)
        _ -> backend
      end
  end
end
