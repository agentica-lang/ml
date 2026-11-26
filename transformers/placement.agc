## Where `layers_on_device` puts the blocks of a model, its embedding table and its head: the rule of llama.cpp's `-ngl`,
## shared by the models that take the option.

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
