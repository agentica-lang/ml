## `layers_on_device` of `Lfm3.open`: on a Vulkan device the layers from the `n`th on run on the CPU, with their
## convolution states and keys in host memory, and a step gives the logits of the model with every layer on the device
## (`n` of 0, 1, 2 and the number of layers). The model is the tiny one of `lfm3_golden.py`, whose weights are read from
## its golden and compared with themselves, so it asks nothing of Hugging Face. Run from `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Generate from "../generate_lfm3"
import Lfm3 from "../lfm3"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## ggml computes in `F32`, the two sides of the bus differ by their rounding.
ABS = 0.005
REL = 0.005

HOME = Tensor.Backend.Ggml(Tensor.Device.Cpu)

## The layers of the golden: conv, conv, full attention, conv, sliding attention.
LAYERS = 5

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
end

private

def with_model(body: Fn[String, Result[Void]]) -> Result[Void]
  g = try JSON.decode(try File.read_text("test/fixtures/lfm3_golden.json"), Golden)
  dir = "/tmp/agentica-lfm3-layers-#{Random.int(0, 999999999)}"
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
  m = try Lfm3.open(dir, backend, Tensor.Dtype.F32, n)
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
  Lfm3.close_model(m)
  ok(out)
end

def test_layers_on_the_cpu_give_the_logits_of_the_layers_on_the_device() -> Result[Void]
  with_model do |dir|
    for backend in try S.ggml_backends()
      want = try run(dir, backend, none)
      for n in [0, 1, 2, LAYERS]
        got = try run(dir, backend, some(n))
        for i in 0...want.size
          try S.expect_close(got[i], want[i], ABS, REL, "#{backend} with #{n} layers, step #{i}")
        end
      end
    end

    ok()
  end
end

## A layer on the CPU keeps its keys, values and convolution state in host memory, a layer on the device keeps them there.
## The convolutions are the layers 0, 1 and 3 (the states 0 to 2), the attentions the layers 2 and 4 (the keys 0 and 1).
def test_the_states_and_the_keys_stay_with_their_layer() -> Result[Void]
  with_model do |dir|
    for backend in try S.ggml_backends()
      next if backend == HOME

      for n in [0, 2, 4, LAYERS]
        m = try Lfm3.open(dir, backend, Tensor.Dtype.F32, some(n))
        gen = try Generate.open(m)
        s = try Generate.step_logits(gen, Generate.start(), [3, 1, 4])
        s = try Generate.step_logits(gen, s.cache, [1])
        for state in 0...3
          layer = [0, 1, 3][state]
          try Test.eq(Tensor.backend(s.cache.states[state]), side(backend, layer, n))
        end

        for key in 0...2
          layer = [2, 4][key]
          try Test.eq(Tensor.backend(s.cache.keys[key]), side(backend, layer, n))
          try Test.eq(Tensor.backend(s.cache.values[key]), side(backend, layer, n))
        end

        Generate.finish(s.cache)
        Generate.close(gen)
        Lfm3.close_model(m)
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
  with_model do |dir| Test.error?(Lfm3.open(dir, S.REF, Tensor.Dtype.F32, some(-1))) end
end
