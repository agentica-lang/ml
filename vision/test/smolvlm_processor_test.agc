## The input of SmolVLM against Hugging Face's `Idefics3Processor` (`smolvlm_processor_golden.py`,
## `fixtures/smolvlm/processor_golden.json`): `Vision.idefics3_tiling` on a table of sizes; the tiles of
## `Vision.idefics3_tiles` of the drawn images of `fixtures/resnet` (each tile 512 by 512, the global image last) through
## `to_tensor` and `normalize` with the `pixel_values` of the processor, which is a resize of libvips against Pillow's;
## and `SmolVlmPrompt.expand`, which gives the ids of the processor from the ids with one `<image>` for each image. No
## checkpoint is needed but for the reading of the ids of the tokenizer, which runs when `SMOLVLM_MODEL` names the
## folder of `HuggingFaceTB/SmolVLM-256M-Instruct`. Run from `vision/`.

import Image from "pkg/image/image"
import SmolVlmPrompt from "pkg/transformers/smolvlm_prompt"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Vision from "../vision"

TILE = 512
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

## What passes for a value of a frame, and for the mean of one: three 8-bit levels (1 / 255 / 0.5 = 0.0078 each; the
## largest gap measured is two) and 0.002 (the largest is 9e-4, for a global image, which shrinks 4 times), since libvips
## resizes as Pillow does up to rounding. The bilinear kernel is 4 levels off and fails; the bicubic one is as close as the
## Lanczos one on these smooth drawings.
PIXEL_ABS = 0.024
PIXEL_MEAN = 0.002

struct Row
  height: Int
  width: Int
  resized: List[Int]
  grid: List[Int]
  rows: Int
  cols: Int
end

struct Case
  name: String
  images: List[String]
  ids_one: List[Int]
  ids: List[Int]
  rows: List[Int]
  cols: List[Int]
  stride: Int
  means: List[Float]
  samples: List[List[Float]]
end

struct Tokens
  fake: Int
  image: Int
  global: Int
  row_cols: List[Int]
  seq_len: Int
  newline: Int
  newlines: Int
end

## The gap of the sampled values of one frame and the one of its mean.
struct Gap
  largest: Float
  mean: Float
end

struct Golden
  tokens: Tokens
  sizes: List[Row]
  cases: List[Case]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/smolvlm/processor_golden.json"), Golden)

def prompt_of(t: Tokens) -> SmolVlmPrompt.Prompt
  SmolVlmPrompt.Prompt(
    fake: t.fake,
    image: t.image,
    global: t.global,
    row_cols: t.row_cols,
    newline: t.newline,
    newlines: t.newlines,
    seq_len: t.seq_len,
  )
end

def grids_of(c: Case) -> List[SmolVlmPrompt.Grid]
  (0...c.rows.size).to_a.map { |i| SmolVlmPrompt.Grid(rows: c.rows[i], cols: c.cols[i]) }
end

def test_the_sizes_are_hugging_faces() -> Result[Void]
  for r in (try golden()).sizes
    t = try Vision.idefics3_tiling(r.height, r.width)
    try Test.eq([t.resized.height, t.resized.width], r.resized)
    try Test.eq([t.grid.height, t.grid.width], r.grid)
    try Test.eq([t.rows, t.cols], [r.rows, r.cols])
  end

  ok()
end

def test_the_tiling_refuses_what_it_cannot_split() -> Result[Void]
  try Test.error?(Vision.idefics3_tiling(0, 10))
  try Test.error?(Vision.idefics3_tiling(10, 0))
  try Test.error?(Vision.idefics3_tiling(10, 10, 0))
  Test.error?(Vision.idefics3_tiling(10, 10, 2048, 0))
end

## The gap of the sampled values of `frame` of `got` (`[3, 512, 512]`) and the one of its mean.
def measure(got: Tensor.Tensor, c: Case, frame: Int) -> Result[Gap]
  total = 3 * TILE * TILE
  want = c.samples[frame]
  at = (0...want.size).to_a.map { |i| Tensor.Values.Int(i * c.stride) }
  ids = try Tensor.from_list(at, [at.size], Tensor.Dtype.I32, Tensor.backend(got))
  outs = try Trace.eval_many([got, ids]) do |xs|
    column = try Tensor.reshape(xs[0], [total, 1])
    ok([try Tensor.embedding(xs[1], column), try Tensor.reshape(try Tensor.mean(xs[0]), [1])])
  end

  values = try S.floats_of(outs[0])
  largest = 0.0
  for i in 0...want.size
    largest = [largest, (values[i] - want[i]).abs].max.unwrap!
  end

  ok(Gap(largest: largest, mean: ((try S.floats_of(outs[1]))[0] - c.means[frame]).abs))
end

## The tiles of each image of each case, normalized as the processor does, one frame after the other.
def test_the_tiles_are_hugging_faces_frames_within_the_resize_of_libvips() -> Result[Void]
  g = try golden()
  backends = try S.backends()
  backend = backends[backends.size - 1]
  for c in g.cases
    frame = 0
    for k in 0...c.images.size
      frame = try image_frames(c, k, frame, backend)
    end

    try Test.eq(frame, c.means.size)
  end

  ok()
end

## The frames of the image `k` of `c`, which start at the frame `first` of the case; the frame after them.
def image_frames(c: Case, k: Int, first: Int, backend: Tensor.Backend) -> Result[Int]
  Native.arena do
    image = try Image.colourspace(try Image.load("test/fixtures/#{c.images[k]}"), Image.Space.Srgb)
    tiles = try Vision.idefics3_tiles(image)
    try Test.eq([tiles.rows, tiles.cols], [c.rows[k], c.cols[k]])
    try Test.eq(tiles.images.size, c.rows[k] * c.cols[k] + 1)
    for i in 0...tiles.images.size
      tile = tiles.images[i]
      try Test.eq([Image.width(tile), Image.height(tile)], [TILE, TILE])
      got = try Vision.normalize(try Vision.to_tensor(tile, backend), MEAN, STD)
      gap = try measure(got, c, first + i)
      try Test.assert(gap.largest <= PIXEL_ABS, "#{c.name} frame #{first + i}: the largest gap is #{gap.largest}")
      try Test.assert(
        gap.mean <= PIXEL_MEAN,
        "#{c.name} frame #{first + i}: the mean is #{gap.mean} from Hugging Face's",
      )
    end

    ok(first + tiles.images.size)
  end
end

def test_the_ids_of_the_processor_are_the_ids_with_the_tiles_expanded() -> Result[Void]
  g = try golden()
  prompt = prompt_of(g.tokens)
  for c in g.cases
    try Test.eq(try SmolVlmPrompt.expand(c.ids_one, prompt, grids_of(c)), c.ids)
  end

  ok()
end

## An image that was not split has its global image alone; the others have tiles first, and the ids fit what the
## expansion counts: 64 `<image>` ids for each tile and each global image.
def test_the_expansion_of_an_image_that_was_not_split_is_its_global_image() -> Result[Void]
  g = try golden()
  prompt = prompt_of(g.tokens)
  t = g.tokens
  ids = try SmolVlmPrompt.expand([7, t.image, 8], prompt, [SmolVlmPrompt.Grid(rows: 0, cols: 0)])
  try Test.eq(ids.size, 5 + t.seq_len)
  try Test.eq(ids.take(4), [7, t.fake, t.global, t.image])
  try Test.eq(ids.drop(ids.size - 2), [t.fake, 8])
  ok()
end

def refused[T](r: Result[T], phrase: String) -> Result[Void]
  case r
    ok(_) -> error("expected an error with \"#{phrase}\", got a value")
    error(e) -> Test.assert(e.message.include?(phrase), "\"#{e.message}\" does not hold \"#{phrase}\"")
  end
end

def test_an_expansion_that_does_not_match_its_grids_is_an_error() -> Result[Void]
  g = try golden()
  prompt = prompt_of(g.tokens)
  one = [SmolVlmPrompt.Grid(rows: 3, cols: 4)]
  ids = [1, g.tokens.image, 2]
  try refused(SmolVlmPrompt.expand(ids, prompt, []), "more images")
  try refused(SmolVlmPrompt.expand(ids, prompt, one + one), "2 grids")
  try refused(SmolVlmPrompt.expand([1, 2], prompt, one), "0 images")
  try refused(SmolVlmPrompt.expand(ids, prompt, [SmolVlmPrompt.Grid(rows: 7, cols: 1)]), "1 to 6")
  try refused(SmolVlmPrompt.expand(ids, prompt, [SmolVlmPrompt.Grid(rows: 0, cols: 2)]), "1 to 6")
  ok()
end

## The ids that `SmolVlmPrompt.open` reads from the files of the checkpoint are the processor's.
def test_the_ids_of_the_checkpoint_are_the_processors() -> Result[Void]
  listed = System.env("SMOLVLM_MODEL")
  if !listed.ok? or try listed == ""
    puts("smolvlm_processor_test: the ids of the checkpoint skipped, SMOLVLM_MODEL names no folder")
    return ok()
  end

  want = prompt_of((try golden()).tokens)
  try Test.eq(try SmolVlmPrompt.open(try listed), want)
  ok()
end
