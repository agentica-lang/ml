## A prompt with image features through `generate.agc` against Hugging Face's `Idefics3ForConditionalGeneration` on the
## tiny random SmolVLM of `smolvlm_image_golden.py` (`fixtures/smolvlm_image_golden.json`): the text model is opened by
## `Qwen3.open_idefics3`, the features of the golden are spliced in at the image tokens by a `KvDecoder.Sight` whose
## layout is the index of each token (`KvDecoder.plain_sight`: Idefics3 has no multimodal rope), and the logits of the
## prompt and of the token fed after it, and the greedy ids, are Hugging Face's. Run from `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Lift from "../lift"
import Generate from "../generate"
import KvDecoder from "../kv_decoder"
import Mrope from "../mrope"
import Qwen3 from "../qwen3"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles; ggml keeps its cache in `F16`.
ABS = 0.00001
REL = 0.00001
F32_ABS = 0.0002
F32_REL = 0.0002

HIDDEN = 24
IMAGE = 30

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  ids: List[Int]
  image_token: Int
  features: List[Float]
  next: Int
  prompt_logits: List[Float]
  next_logits: List[Float]
  generated: List[Int]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/smolvlm_image_golden.json"), Golden)

def sight_of(g: Golden, backend: Tensor.Backend) -> Result[KvDecoder.Sight]
  ok(KvDecoder.plain_sight(try S.f32(g.features, [g.features.size / HIDDEN, HIDDEN], backend), IMAGE, g.ids))
end

def with_model[T](g: Golden, body: Fn[String, Result[T]]) -> Result[T]
  dir = "/tmp/agentica-smolvlm-image-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = populated(g, dir, body)
  _ = Path.remove(dir)
  result
end

def populated[T](g: Golden, dir: String, body: Fn[String, Result[T]]) -> Result[T]
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

## The prompt with its features, then the token after it: the logits of both are Hugging Face's.
def steps_match(g: Golden, backend: Tensor.Backend) -> Result[Void]
  tol = tolerance(backend)
  with_model(g) do |dir|
    m = try Qwen3.open_idefics3(dir, backend)
    gen = try Generate.open(m)
    sight = try sight_of(g, backend)
    first = try Generate.step_logits(gen, Generate.start(some(sight)), g.ids)
    try S.expect_close(first.logits, g.prompt_logits, tol[0], tol[1], "the prompt")
    second = try Generate.step_logits(gen, first.cache, [g.next])
    try S.expect_close(second.logits, g.next_logits, tol[0], tol[1], "the token after it")
    Generate.finish(second.cache)
    Generate.close(gen)
    Lift.close([sight.features])
    Qwen3.close_model(m)
    ok()
  end
end

def test_the_plain_layout_counts_the_tokens_and_marks_the_image_ones() -> Result[Void]
  g = try golden()
  layout = Mrope.indexed(g.ids, IMAGE)
  positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  try Test.eq(layout.time, positions)
  try Test.eq(layout.down, positions)
  try Test.eq(layout.across, positions)
  try Test.eq(layout.at, [2, 3, 4, 5])
  try Test.eq(layout.delta, 0)
  ok()
end

def test_the_image_token_is_read_from_the_config() -> Result[Void]
  g = try golden()
  with_model(g) do |dir|
    try Test.eq(try Qwen3.image_token(dir), IMAGE)
    try File.write_text("#{dir}/config.json", "{\"model_type\": \"idefics3\"}")
    Test.error?(Qwen3.image_token(dir))
  end
end

def test_the_prompt_and_the_token_after_it_match_hugging_face_on_the_reference() -> Result[Void]
  steps_match(try golden(), S.REF)
end

def test_the_prompt_and_greedy_generation_match_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try steps_match(g, backend)
    try with_model(g) do |dir|
      m = try Qwen3.open_idefics3(dir, backend)
      gen = try Generate.open(m)
      sight = try sight_of(g, backend)
      made = try Generate.generate(gen, g.ids, g.generated.size, [], none, some(sight))
      try Test.eq(made, g.generated)
      Generate.close(gen)
      Lift.close([sight.features])
      Qwen3.close_model(m)
      ok()
    end
  end

  ok()
end

## The images change the answer: the same ids without a sight are a text prompt, whose logits differ.
def test_a_prompt_without_its_images_answers_otherwise() -> Result[Void]
  g = try golden()
  with_model(g) do |dir|
    m = try Qwen3.open_idefics3(dir, S.REF)
    gen = try Generate.open(m)
    plain = try Generate.step_logits(gen, Generate.start(), g.ids)
    try Test.assert(plain.logits != g.prompt_logits, "a prompt without images gave the logits of the one with them")
    Generate.finish(plain.cache)
    Generate.close(gen)
    Qwen3.close_model(m)
    ok()
  end
end

def test_a_sight_that_does_not_fit_the_prompt_is_an_error() -> Result[Void]
  g = try golden()
  with_model(g) do |dir|
    m = try Qwen3.open_idefics3(dir, S.REF)
    gen = try Generate.open(m)
    sight = try sight_of(g, S.REF)
    few = sight.with(features: try S.f32(g.features.take(3 * HIDDEN), [3, HIDDEN]))
    try S.expect_error(Generate.step_logits(gen, Generate.start(some(few)), g.ids), "features")
    other = sight.with(image_token: 7)
    try S.expect_error(Generate.step_logits(gen, Generate.start(some(other)), g.ids), "not the image token")
    Generate.close(gen)
    Lift.close([sight.features, few.features])
    Qwen3.close_model(m)
    ok()
  end
end

## A model with one position for each token turns by the index, so a layout that moves a token, or has a `delta` as
## Qwen3.5's, is refused before any step runs.
def test_a_layout_with_a_delta_is_an_error_for_a_plain_model() -> Result[Void]
  g = try golden()
  with_model(g) do |dir|
    m = try Qwen3.open_idefics3(dir, S.REF)
    gen = try Generate.open(m)
    sight = try sight_of(g, S.REF)
    layout = sight.layout
    moved = layout.time.take(5) + [20] + layout.time.drop(6)
    cases = [
      sight.with(layout: layout.with(delta: 3)),
      sight.with(layout: layout.with(time: moved)),
      sight.with(layout: layout.with(down: moved)),
      sight.with(layout: layout.with(across: moved)),
    ]

    for c in cases
      try S.expect_error(Generate.step_logits(gen, Generate.start(some(c)), g.ids), "one position for each token")
    end

    Generate.close(gen)
    Lift.close([sight.features])
    Qwen3.close_model(m)
    ok()
  end
end

## A text prompt, with no sight, is the model of `Qwen3.run`: the logits of its last position are the same.
def test_a_prompt_without_a_sight_is_what_run_gives() -> Result[Void]
  g = try golden()
  for backend in [S.REF] + try S.ggml_backends()
    tol = tolerance(backend)
    try with_model(g) do |dir|
      m = try Qwen3.open_idefics3(dir, backend)
      gen = try Generate.open(m)
      ids = g.ids.take(6)
      stepped = try Generate.step_logits(gen, Generate.start(), ids)
      run = try Qwen3.run(m, [ids], true)
      try S.expect_close(stepped.logits, try S.floats_of(run), tol[0], tol[1], "the last position")
      Generate.finish(stepped.cache)
      Generate.close(gen)
      Qwen3.close_model(m)
      ok()
    end
  end

  ok()
end
