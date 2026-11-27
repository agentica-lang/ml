## `layers_on_device` of `Qwen3_5.open`: on a Vulkan device the blocks from the `n`th on run on the CPU, with their
## recurrent states and keys in host memory, and a step gives the logits of the model with every block on the device
## (`n` of 0, 1, 2 and the number of blocks). The model is the tiny one of `qwen35_golden.py` with a gated delta head of 16
## values (the width Vulkan's rule needs) and weights of a fixed pattern, so it asks nothing of Hugging Face. Without a
## Vulkan device the test checks that `n` changes nothing on the CPU.

import Write from "pkg/hub/safetensors_write"
import Generate from "../generate_qwen35"
import Qwen3_5 from "../qwen3_5"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## ggml computes in `F32`, the two sides of the bus differ by their rounding.
ABS = 0.005
REL = 0.005

HOME = Tensor.Backend.Ggml(Tensor.Device.Cpu)

struct Golden
  config: String
  shapes: Map[String, List[Int]]
end

private

## The shapes of the golden, with the gated delta head widened to 16 (`linear_key_head_dim` and `linear_value_head_dim`).
def wide(g: Golden) -> Golden
  shapes = g.shapes
  for name in shapes.keys
    if name.end_with?("conv1d.weight")
      shapes = shapes.put(name, [128, 1, 4])
    elsif name.end_with?("in_proj_qkv.weight")
      shapes = shapes.put(name, [128, 16])
    elsif name.end_with?("in_proj_z.weight")
      shapes = shapes.put(name, [64, 16])
    elsif name.end_with?("linear_attn.out_proj.weight")
      shapes = shapes.put(name, [16, 64])
    elsif name.end_with?("linear_attn.norm.weight")
      shapes = shapes.put(name, [16])
    end
  end

  config = g.config.replace("\"linear_key_head_dim\": 4", "\"linear_key_head_dim\": 16")
  Golden(config: config.replace("\"linear_value_head_dim\": 4", "\"linear_value_head_dim\": 16"), shapes: shapes)
end

## A tensor of `shape` with a fixed pattern of values in `-0.25...0.25`, which differs from tensor to tensor by `salt`.
def pattern(shape: List[Int], salt: Int) -> Result[Tensor.Tensor]
  count = 1
  for n in shape
    count = count * n
  end

  values = (0...count).map { |i| ((i * 7 + salt * 5) % 17 - 8).to_f / 32.0 }
  S.f32(values, shape)
end

def with_model(body: Fn[String, Result[Void]]) -> Result[Void]
  g = wide(try JSON.decode(try File.read_text("test/fixtures/qwen35_golden.json"), Golden))
  dir = "/tmp/agentica-qwen35-layers-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  tensors: Map[String, Tensor.Tensor] = {}
  salt = 0
  for name in g.shapes.keys
    salt = salt + 1
    tensors = tensors.put(name, try pattern(g.shapes[name], salt))
  end

  try Write.write("#{dir}/model.safetensors", tensors)
  try File.write_text("#{dir}/config.json", g.config)
  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

## The logits of a prompt of 5 ids and of the 3 tokens after it, and the cache that is left.
def run(dir: String, backend: Tensor.Backend, n: Optional[Int]) -> Result[List[List[Float]]]
  m = try Qwen3_5.open(dir, backend, Tensor.Dtype.F32, n)
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
  Qwen3_5.close_model(m)
  ok(out)
end

def test_blocks_on_the_cpu_give_the_logits_of_the_blocks_on_the_device() -> Result[Void]
  with_model do |dir|
    for backend in try S.ggml_backends()
      want = try run(dir, backend, none)
      for n in [0, 1, 2, 4]
        got = try run(dir, backend, some(n))
        for i in 0...want.size
          try S.expect_close(got[i], want[i], ABS, REL, "#{backend} with #{n} blocks, step #{i}")
        end
      end
    end

    ok()
  end
end

## A layer on the CPU keeps its keys, values and states in host memory, a layer on the device keeps them there.
def test_the_states_and_the_keys_stay_with_their_layer() -> Result[Void]
  with_model do |dir|
    for backend in try S.ggml_backends()
      next if backend == HOME

      for n in [0, 2, 4]
        m = try Qwen3_5.open(dir, backend, Tensor.Dtype.F32, some(n))
        gen = try Generate.open(m)
        s = try Generate.step_logits(gen, Generate.start(), [3, 1, 4])
        s = try Generate.step_logits(gen, s.cache, [1])
        for linear in 0...3
          side = backend
          if linear >= n
            side = HOME
          end

          try Test.eq(Tensor.backend(s.cache.states[2 * linear]), side)
          try Test.eq(Tensor.backend(s.cache.states[2 * linear + 1]), side)
        end

        full = backend
        if n < 4
          full = HOME
        end

        try Test.eq(Tensor.backend(s.cache.keys[0]), full)
        try Test.eq(Tensor.backend(s.cache.values[0]), full)
        Generate.finish(s.cache)
        Generate.close(gen)
        Qwen3_5.close_model(m)
      end
    end

    ok()
  end
end

def test_a_negative_count_is_refused() -> Result[Void]
  with_model do |dir| Test.error?(Qwen3_5.open(dir, S.REF, Tensor.Dtype.F32, some(-1))) end
end
