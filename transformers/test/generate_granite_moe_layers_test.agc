## `layers_on_device` of `GraniteMoe.open`: on a Vulkan device the layers from the `n`th on run on the CPU, with their
## keys in host memory, and a step gives the logits of the model with every layer on the device
## (`n` of 0, 1 and the number of layers). The model is the tiny one of `granite_moe_golden.py`, whose weights are read from
## its golden and compared with themselves, so it asks nothing of Hugging Face. Run from `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Generate from "../generate_granite_moe"
import GraniteMoe from "../granite_moe"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## ggml computes in `F32`, the two sides of the bus differ by their rounding.
ABS = 0.005
REL = 0.005

HOME = Tensor.Backend.Ggml(Tensor.Device.Cpu)

## The layers of the golden.
LAYERS = 2

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
end

private

def with_model(body: Fn[String, Result[Void]]) -> Result[Void]
  g = try JSON.decode(try File.read_text("test/fixtures/granite_moe_golden.json"), Golden)
  dir = "/tmp/agentica-granite-layers-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  tensors: Map[String, Tensor.Tensor] = {}
  for name in g.weights.keys
    tensors = tensors.put(name, try S.f32(g.weights[name], g.shapes[name]))
  end

  try Write.write("#{dir}/model.safetensors", tensors)
  try File.write_text("#{dir}/config.json", g.config)
  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

## The logits of a prompt of 5 ids and of the 3 tokens after it.
def run(dir: String, backend: Tensor.Backend, n: Optional[Int]) -> Result[List[List[Float]]]
  m = try GraniteMoe.open(dir, backend, Tensor.Dtype.F32, n)
  gen = try Generate.open(m)
  out: List[List[Float]] = []
  cache = Generate.start()
  for fed in [[3, 1, 4, 1, 5], [9], [2], [6]]
    s = try Generate.step_logits(gen, cache, fed)
    cache = s.cache
    out = out.push(s.logits)
  end

  Generate.finish(cache)
  Generate.close(gen)
  GraniteMoe.close_model(m)
  ok(out)
end

def test_layers_on_the_cpu_give_the_logits_of_the_layers_on_the_device() -> Result[Void]
  with_model do |dir|
    for backend in try S.ggml_backends()
      want = try run(dir, backend, none)
      for n in [0, 1, LAYERS]
        got = try run(dir, backend, some(n))
        for i in 0...want.size
          try S.expect_close(got[i], want[i], ABS, REL, "#{backend} with #{n} layers, step #{i}")
        end
      end
    end

    ok()
  end
end

## A layer on the CPU keeps its keys and values in host memory, a layer on the device keeps them there.
def test_the_keys_stay_with_their_layer() -> Result[Void]
  with_model do |dir|
    for backend in try S.ggml_backends()
      next if backend == HOME

      for n in [0, 1, LAYERS]
        m = try GraniteMoe.open(dir, backend, Tensor.Dtype.F32, some(n))
        gen = try Generate.open(m)
        s = try Generate.step_logits(gen, Generate.start(), [3, 1, 4])
        s = try Generate.step_logits(gen, s.cache, [1])
        for layer in 0...LAYERS
          try Test.eq(Tensor.backend(s.cache.keys[layer]), side(backend, layer, n))
          try Test.eq(Tensor.backend(s.cache.values[layer]), side(backend, layer, n))
        end

        Generate.finish(s.cache)
        Generate.close(gen)
        GraniteMoe.close_model(m)
      end
    end

    ok()
  end
end

## The backend of `layer` when the first `n` are on `backend`.
def side(backend: Tensor.Backend, layer: Int, n: Int) -> Tensor.Backend
  return backend if layer < n
  HOME
end

def test_a_negative_count_is_refused() -> Result[Void]
  with_model do |dir| Test.error?(GraniteMoe.open(dir, S.REF, Tensor.Dtype.F32, some(-1))) end
end
