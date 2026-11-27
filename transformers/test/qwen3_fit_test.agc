## `Qwen3.layers_within` and `Qwen3.layers_that_fit`: the `layers_on_device` that a byte budget buys, from a checkpoint
## written by the safetensors writer (three blocks of 128 bytes, a table of 320, a norm of 32) and, on a Vulkan device, the
## free memory the device reports. Pure, but for the device check, which passes without a card.

import Qwen3 from "../qwen3"
import Tensor from "pkg/tensor/tensor"
import Write from "pkg/hub/safetensors_write"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
F32 = Tensor.Dtype.F32

## 2 key heads of 2 values, 10 tokens, keys and values in F32: 320 bytes of cache for each block.
CONFIG = [
  "{\"model_type\":\"qwen3\",\"hidden_size\":8,\"num_hidden_layers\":3,\"num_attention_heads\":2,",
  "\"num_key_value_heads\":2,\"head_dim\":2,\"intermediate_size\":8,\"vocab_size\":10,",
  "\"max_position_embeddings\":64,\"rms_norm_eps\":0.000001,\"rope_theta\":10000.0,\"hidden_act\":\"silu\",",
  "\"tie_word_embeddings\":true}",
].join("")

## Each block is 128 + 320 = 448 bytes; with the head (a copy of the table, 320, and the norm, 32) one block needs 800,
## two 1248, and all three, with the table once, 1664.
def with_model(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-qwen3-fit-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  zeros = ->(shape: List[Int]) { Tensor.zeros(shape, F32, Tensor.Backend.Reference) }
  tensors = {
    "model.embed_tokens.weight": try zeros.call([10, 8]),
    "model.norm.weight": try zeros.call([8]),
    "model.layers.0.up.weight": try zeros.call([4, 8]),
    "model.layers.1.up.weight": try zeros.call([4, 8]),
    "model.layers.2.up.weight": try zeros.call([4, 8]),
  }

  try Write.write("#{dir}/model.safetensors", tensors)
  try File.write_text("#{dir}/config.json", CONFIG)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

def test_the_budget_buys_the_most_blocks_whose_layout_fits() -> Result[Void]
  with_model do |dir|
    c = try Qwen3.checkpoint(dir)
    config = try Qwen3.read_config(dir)
    for want in [[799, 0], [800, 1], [1247, 1], [1248, 2], [1663, 2], [1664, 3], [100000, 3]]
      try Test.eq(try Qwen3.layers_within(c, config, F32, want[0], 10), want[1])
    end

    ok()
  end
end

## On a Vulkan device the device's free memory less the reserve, capped by `max_device_bytes`, is the budget; any other
## backend has no layers to place.
def test_a_vulkan_device_gives_the_layers_that_fit_its_memory_and_another_backend_none() -> Result[Void]
  with_model do |dir|
    try Test.eq(try Qwen3.layers_that_fit(dir, CPU, F32, none, 10), none)
    try Test.eq(try Qwen3.layers_that_fit(dir, Tensor.Backend.Reference, F32, none, 10), none)
    return ok() if not Tensor.available(VULKAN)

    try Test.eq(try Qwen3.layers_that_fit(dir, VULKAN, F32, none, 10, 0), some(3))
    try Test.eq(try Qwen3.layers_that_fit(dir, VULKAN, F32, some(1664), 10, 0), some(3))
    try Test.eq(try Qwen3.layers_that_fit(dir, VULKAN, F32, some(1663), 10, 0), some(2))
    try Test.eq(try Qwen3.layers_that_fit(dir, VULKAN, F32, some(799), 10, 0), some(0))
    Test.eq(try Qwen3.layers_that_fit(dir, VULKAN, F32, none, 10, 1000000000000), some(0))
  end
end
