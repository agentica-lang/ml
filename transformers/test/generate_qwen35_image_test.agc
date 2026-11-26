## A prompt with images through `generate_qwen35.agc` against Hugging Face's `Qwen3_5ForConditionalGeneration` on the
## tiny random model of `qwen35_image_golden.py` (`fixtures/qwen35_image_golden.json`): `Mrope.layout` gives the
## 3-axis positions and `rope_deltas` of `get_rope_index`; the logits of a prompt with two images (the features of
## the golden spliced in at the image tokens, turned by the three axes) and then of each token fed after it (turned
## by its index plus the delta) are Hugging Face's; and `generate` gives its greedy ids on ggml (CPU; the reference
## is too slow for it). Run from `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Lift from "../lift"
import Generate from "../generate_qwen35"
import KvDecoder from "../kv_decoder"
import Mrope from "../mrope"
import Qwen3_5 from "../qwen3_5"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles; ggml keeps its cache in `F16`.
ABS = 0.0001
REL = 0.0001
F32_ABS = 0.005
F32_REL = 0.005

## The `linear_key_head_dim` and `linear_value_head_dim` of the tiny model (`qwen35_image_golden.py`), and its sizes.
DELTA_WIDTH = 4
VOCAB = 32
HIDDEN = 16
IMAGE = 28

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  ids: List[Int]
  grids: List[List[Int]]
  features: List[Float]
  time: List[Int]
  down: List[Int]
  across: List[Int]
  delta: Int
  logits: List[Float]
  generated: List[Int]
  scores: List[List[Float]]
end

## A sight that the step must refuse, with a phrase of the error.
struct Refusal
  sight: KvDecoder.Sight
  phrase: String
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/qwen35_image_golden.json"), Golden)

def grids_of(g: Golden) -> List[Mrope.Grid] = g.grids.map { |p| Mrope.Grid(height: p[0], width: p[1]) }

## The images of the golden, their features on `backend`.
def sight_of(g: Golden, backend: Tensor.Backend) -> Result[KvDecoder.Sight]
  layout = try Mrope.layout(g.ids, IMAGE, grids_of(g), 2)
  rows = g.features.size / HIDDEN
  ok(KvDecoder.Sight(features: try S.f32(g.features, [rows, HIDDEN], backend), image_token: IMAGE, layout: layout))
end

## The model folder of the golden and `body` over it.
def with_model[T](g: Golden, body: Fn[String, Result[T]]) -> Result[T]
  dir = "/tmp/agentica-qwen35-image-#{Random.int(0, 999999999)}"
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

## The prompt with its images, then the first `steps - 1` ids of `generated` one at a time: the logits of every step are
## the scores of Hugging Face's `generate`, which turns the tokens after the images by their index plus `rope_deltas`.
def steps_match(g: Golden, backend: Tensor.Backend, steps: Int) -> Result[Void]
  tol = tolerance(backend)
  with_model(g) do |dir|
    m = try Qwen3_5.open(dir, backend)
    gen = try Generate.open(m)
    sight = try sight_of(g, backend)
    cache = Generate.start(some(sight))
    feed = g.ids
    for i in 0...steps
      s = try Generate.step_logits(gen, cache, feed)
      cache = s.cache
      try S.expect_close(s.logits, g.scores[i], tol[0], tol[1], "step #{i}")
      feed = [g.generated[i]]
    end

    try Test.eq(cache.delta, g.delta)
    Generate.finish(cache)
    Generate.close(gen)
    Lift.close([sight.features])
    Qwen3_5.close_model(m)
    ok()
  end
end

def test_the_layout_is_get_rope_index_of_hugging_face() -> Result[Void]
  g = try golden()
  layout = try Mrope.layout(g.ids, IMAGE, grids_of(g), 2)
  try Test.eq(layout.time, g.time)
  try Test.eq(layout.down, g.down)
  try Test.eq(layout.across, g.across)
  try Test.eq(layout.delta, g.delta)
  try Test.eq(layout.at, [3, 4, 5, 6, 7, 8, 12, 13, 14])
  ok()
end

## The template writes one `<|image_pad|>` for each image, and the tokens of the image are as many ids: the grid says
## how many, and the id is the one of the config.
def test_the_pad_of_each_image_is_expanded_to_its_tokens() -> Result[Void]
  g = try golden()
  grids = grids_of(g)
  pads = [5, 6, 29, IMAGE, 30, 7, 29, IMAGE, 30, 8, 9]
  try Test.eq(try Mrope.expand(pads, IMAGE, grids, 2), g.ids)
  try Test.error?(Mrope.expand(pads, IMAGE, [grids[0]], 2))
  try Test.error?(Mrope.expand(pads, IMAGE, grids + [grids[0]], 2))
  try Test.error?(Mrope.expand(pads, IMAGE, grids, 0))
  try S.expect_error(
    Mrope.expand(pads, IMAGE, [grids[0], Mrope.Grid(height: 3, width: 6)], 2),
    "multiple of the merge",
  )

  try S.expect_error(
    Mrope.expand(pads, IMAGE, [grids[0], Mrope.Grid(height: 4, width: 3)], 2),
    "multiple of the merge",
  )

  ok()
end

def test_the_image_token_is_read_from_the_config() -> Result[Void]
  g = try golden()
  with_model(g) do |dir|
    try Test.eq(try Qwen3_5.image_token(dir), IMAGE)
    try File.write_text("#{dir}/config.json", "{\"model_type\": \"qwen3_5\"}")
    Test.error?(Qwen3_5.image_token(dir))
  end
end

## A side of 0 is told apart from a grid that the merge does not divide.
def test_a_grid_with_an_empty_side_is_an_error() -> Result[Void]
  for grid in [Mrope.Grid(height: 0, width: 4), Mrope.Grid(height: 4, width: 0), Mrope.Grid(height: -2, width: 4)]
    try S.expect_error(Mrope.expand([IMAGE], IMAGE, [grid], 2), "needs 1 or more")
    try S.expect_error(Mrope.layout([IMAGE], IMAGE, [grid], 2), "needs 1 or more")
  end

  ok()
end

def test_a_prompt_that_does_not_match_its_grids_is_an_error() -> Result[Void]
  g = try golden()
  grids = grids_of(g)
  try S.expect_error(Mrope.layout(g.ids, IMAGE, grids.take(1), 2), "more images")
  try S.expect_error(Mrope.layout(g.ids.take(8), IMAGE, grids, 2), "tokens in the prompt")
  try S.expect_error(Mrope.layout(g.ids.take(3), IMAGE, grids, 2), "2 grids")
  try S.expect_error(Mrope.layout(g.ids, IMAGE, [Mrope.Grid(height: 3, width: 6), grids[1]], 2), "merge")
  ok()
end

def test_the_prompt_and_the_tokens_after_it_match_hugging_face_on_the_reference() -> Result[Void]
  steps_match(try golden(), S.REF, 2)
end

def test_the_prompt_and_greedy_generation_match_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends_for_delta(DELTA_WIDTH)
    try steps_match(g, backend, g.generated.size)
    try with_model(g) do |dir|
      m = try Qwen3_5.open(dir, backend)
      gen = try Generate.open(m)
      sight = try sight_of(g, backend)
      made = try Generate.generate(gen, g.ids, g.generated.size, [], none, some(sight))
      try Test.eq(made, g.generated)
      Generate.close(gen)
      Lift.close([sight.features])
      Qwen3_5.close_model(m)
      ok()
    end
  end

  ok()
end

## The images change the answer: the same ids without a sight are a text prompt, whose logits differ.
def test_a_prompt_without_its_images_answers_otherwise() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends_for_delta(DELTA_WIDTH)
    try with_model(g) do |dir|
      m = try Qwen3_5.open(dir, backend)
      gen = try Generate.open(m)
      plain = try Generate.step_logits(gen, Generate.start(), g.ids)
      try Test.assert(plain.logits != g.scores[0], "a prompt without images gave the logits of the one with them")
      Generate.finish(plain.cache)
      Generate.close(gen)
      Qwen3_5.close_model(m)
      ok()
    end
  end

  ok()
end

## Each way a layout can differ from the prompt is an error, before any step runs.
def test_a_sight_that_does_not_fit_the_prompt_is_an_error() -> Result[Void]
  g = try golden()
  with_model(g) do |dir|
    m = try Qwen3_5.open(dir, S.REF)
    gen = try Generate.open(m)
    sight = try sight_of(g, S.REF)
    layout = sight.layout
    swapped = [layout.at[1], layout.at[0]] + layout.at.drop(2)
    cases = [
      Refusal(sight: sight.with(layout: layout.with(time: layout.time.take(5))), phrase: "positions for"),
      Refusal(sight: sight.with(layout: layout.with(down: layout.down.take(5))), phrase: "positions for"),
      Refusal(sight: sight.with(layout: layout.with(across: layout.across.take(5))), phrase: "positions for"),
      Refusal(sight: sight.with(layout: layout.with(at: swapped)), phrase: "increasing order"),
      Refusal(sight: sight.with(layout: layout.with(at: layout.at.take(8) + [99])), phrase: "outside"),
      Refusal(sight: sight.with(layout: layout.with(at: [0] + layout.at.drop(1))), phrase: "not the image token"),
      Refusal(sight: sight.with(layout: layout.with(at: layout.at.take(8))), phrase: "image rows"),
      Refusal(sight: sight.with(image_token: 7), phrase: "not the image token"),
    ]

    for c in cases
      try S.expect_error(Generate.step_logits(gen, Generate.start(some(c.sight)), g.ids), c.phrase)
    end

    few = sight.with(features: try S.f32(g.features.take(4 * HIDDEN), [4, HIDDEN]))
    try S.expect_error(Generate.step_logits(gen, Generate.start(some(few)), g.ids), "features")
    first = try Generate.step_logits(gen, Generate.start(), [1, 2, 3])
    try S.expect_error(Generate.step_logits(gen, first.cache.with(sight: some(sight)), [1]), "prompt")
    Generate.close(gen)
    Lift.close([sight.features, few.features])
    Qwen3_5.close_model(m)
    ok()
  end
end

## A model that does not read images refuses a sight, before it runs a step.
def test_a_model_that_reads_no_images_refuses_a_sight() -> Result[Void]
  g = try golden()
  stepper = KvDecoder.Stepper(
    layers: 0,
    limit: 64,
    vocab: 32,
    backend: S.REF,
    traced: ->(p: Int) { p },
    stepped: ->(_p: Int, _req: KvDecoder.Request) { error("the step must not run") },
  )

  core = try KvDecoder.open(0, stepper)
  sight = try sight_of(g, S.REF)
  try S.expect_error(KvDecoder.step_logits(core, KvDecoder.start(some(sight)), g.ids), "reads no images")
  Lift.close([sight.features])
  ok()
end

## One step of the prompt `ids` with the first image of the golden at `layout` on `backend`: finite logits, and the
## delta of the layout in the cache.
def runs_with_delta(g: Golden, backend: Tensor.Backend, ids: List[Int], layout: Mrope.Layout) -> Result[Void]
  rows = try S.f32(g.features.take(6 * HIDDEN), [6, HIDDEN], backend)
  sight = KvDecoder.Sight(features: rows, image_token: IMAGE, layout: layout)
  result = with_model(g) do |dir|
    m = try Qwen3_5.open(dir, backend)
    gen = try Generate.open(m)
    s = try Generate.step_logits(gen, Generate.start(some(sight)), ids)
    try Test.assert(s.logits.all? { |v| v > -1000000.0 and v < 1000000.0 }, "a logit is not finite")
    try Test.eq(s.cache.delta, layout.delta)
    Generate.finish(s.cache)
    Generate.close(gen)
    Qwen3_5.close_model(m)
    ok()
  end

  Lift.close([rows])
  result
end

## An image at the very start and at the very end of the prompt: the positions are the ones of the rule worked out by
## hand (a 2 by 3 grid of merged patches counts 3 positions), and the step runs.
def test_an_image_at_the_start_or_at_the_end_of_the_prompt() -> Result[Void]
  g = try golden()
  grids = [Mrope.Grid(height: 4, width: 6)]
  start = [IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, 8, 9]
  at_start = try Mrope.layout(start, IMAGE, grids, 2)
  try Test.eq(at_start.time, [0, 0, 0, 0, 0, 0, 3, 4])
  try Test.eq(at_start.down, [0, 0, 0, 1, 1, 1, 3, 4])
  try Test.eq(at_start.across, [0, 1, 2, 0, 1, 2, 3, 4])
  try Test.eq(at_start.at, [0, 1, 2, 3, 4, 5])
  try Test.eq(at_start.delta, -3)
  finish = [8, 9, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE]
  at_end = try Mrope.layout(finish, IMAGE, grids, 2)
  try Test.eq(at_end.time, [0, 1, 2, 2, 2, 2, 2, 2])
  try Test.eq(at_end.down, [0, 1, 2, 2, 2, 3, 3, 3])
  try Test.eq(at_end.across, [0, 1, 2, 3, 4, 2, 3, 4])
  try Test.eq(at_end.delta, -3)
  for backend in try S.ggml_backends_for_delta(DELTA_WIDTH)
    try runs_with_delta(g, backend, start, at_start)
    try runs_with_delta(g, backend, finish, at_end)
  end

  ok()
end

## The logits after the prompt `ids` of the first image of the golden alone (`layout`, its 6 rows of features) on
## `target`, through a generator that has compiled the step of the two images first when `warmed`.
def one_image_logits(
  g: Golden,
  target: Tensor.Backend,
  ids: List[Int],
  layout: Mrope.Layout,
  warmed: Bool,
) -> Result[List[Float]]
  with_model(g) do |dir|
    m = try Qwen3_5.open(dir, target)
    gen = try Generate.open(m)
    both = try sight_of(g, target)
    if warmed
      warm = try Generate.step_logits(gen, Generate.start(some(both)), g.ids)
      Generate.finish(warm.cache)
    end

    rows = try S.f32(g.features.take(6 * HIDDEN), [6, HIDDEN], target)
    s = try Generate.step_logits(
      gen,
      Generate.start(some(KvDecoder.Sight(features: rows, image_token: IMAGE, layout: layout))),
      ids,
    )

    Generate.finish(s.cache)
    Generate.close(gen)
    Lift.close([both.features, rows])
    Qwen3_5.close_model(m)
    ok(s.logits)
  end
end

## The prompt of the first image alone, 18 tokens as the one of the two images (the same capacity) with 6 image
## tokens in place of 9, and positions of its own: its logits on ggml are the same from a generator that has compiled
## the step of the two images (each number of image tokens is its own graph) as from one that has not.
def test_a_prompt_with_other_images_runs_its_own_step_on_the_same_generator() -> Result[Void]
  g = try golden()
  ids = g.ids.take(10) + [8, 9, 8, 9, 8, 9, 8, 9]
  layout = try Mrope.layout(ids, IMAGE, [Mrope.Grid(height: 4, width: 6)], 2)
  for backend in try S.ggml_backends_for_delta(DELTA_WIDTH)
    fresh = try one_image_logits(g, backend, ids, layout, false)
    warmed = try one_image_logits(g, backend, ids, layout, true)
    try S.expect_close(warmed, fresh, F32_ABS, F32_REL, "one image after two")
  end

  ok()
end
