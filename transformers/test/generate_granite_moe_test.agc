## `Generate` of `generate_granite_moe.agc` against Hugging Face's `GraniteMoeForCausalLM` on the tiny random model of
## `granite_moe_golden.py` (`fixtures/granite_moe_golden.json`): the logits of a prompt fed in one padded step and then one
## token at a time equal those of the whole forward (the cache is right, and the padding rows that go through the experts
## change nothing), and a greedy generation from a prompt of 7 ids, 14 new ones, crossing the capacities 8, 16 and 32,
## gives Hugging Face's ids and logits on ggml (CPU). Run from `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Generate from "../generate_granite_moe"
import GraniteMoe from "../granite_moe"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles; ggml keeps its cache in `F16`.
ABS = 0.0001
REL = 0.0001
F32_ABS = 0.005
F32_REL = 0.005

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  rows: List[List[Int]]
  logits: List[Float]
  generated: List[Int]
  scores: List[List[Float]]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/granite_moe_golden.json"), Golden)

## The model folder of the golden and `body` over it.
def with_model(g: Golden, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-granite-moe-gen-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = populated(g, dir, body)
  _ = Path.remove(dir)
  result
end

def populated(g: Golden, dir: String, body: Fn[String, Result[Void]]) -> Result[Void]
  tensors: Map[String, Tensor.Tensor] = {}
  for name in g.weights.keys
    tensors = tensors.put(name, try S.f32(g.weights[name], g.shapes[name]))
  end

  try Write.write("#{dir}/model.safetensors", tensors)
  try File.write_text("#{dir}/config.json", g.config)
  body.call(dir)
end

def tolerance(backend: Tensor.Backend) -> List[Float]
  case backend
    Tensor.Backend.Ggml(_) -> [F32_ABS, F32_REL]
    _ -> [ABS, REL]
  end
end

## The prompt of the first 3 ids of each row, then the rest one id at a time: every step's logits are the row of the
## whole forward at that position.
def cached_steps_match(g: Golden, backend: Tensor.Backend) -> Result[Void]
  tol = tolerance(backend)
  with_model(g) do |dir|
    m = try GraniteMoe.open(dir, backend)
    gen = try Generate.open(m)
    for r in 0...2
      row = g.rows[r]
      cache = Generate.start()
      for t in 2...row.size
        fed = [row[t]]
        if t == 2
          fed = row.take(3)
        end

        s = try Generate.step_logits(gen, cache, fed)
        cache = s.cache
        at = (r * row.size + t) * 16
        try S.expect_close(s.logits, g.logits.drop(at).take(16), tol[0], tol[1], "row #{r} position #{t}")
      end

      Generate.finish(cache)
    end

    Generate.close(gen)
    GraniteMoe.close_model(m)
    ok()
  end
end

## `generate` gives the ids of Hugging Face's greedy `generate`, and the logits of each step (fed with those ids) are its
## scores.
def greedy_matches(g: Golden, backend: Tensor.Backend) -> Result[Void]
  tol = tolerance(backend)
  with_model(g) do |dir|
    m = try GraniteMoe.open(dir, backend)
    gen = try Generate.open(m)
    prompt = g.rows[0]
    try Test.eq(try Generate.generate(gen, prompt, g.generated.size), g.generated)
    cache = Generate.start()
    feed = prompt
    for i in 0...g.generated.size
      s = try Generate.step_logits(gen, cache, feed)
      cache = s.cache
      try S.expect_close(s.logits, g.scores[i], tol[0], tol[1], "step #{i}")
      feed = [g.generated[i]]
    end

    Generate.finish(cache)
    Generate.close(gen)
    GraniteMoe.close_model(m)
    ok()
  end
end

def test_cached_steps_equal_the_whole_forward_on_the_reference() -> Result[Void]
  cached_steps_match(try golden(), S.REF)
end

def test_cached_steps_and_greedy_generation_match_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try cached_steps_match(g, backend)
    try greedy_matches(g, backend)
  end

  ok()
end
