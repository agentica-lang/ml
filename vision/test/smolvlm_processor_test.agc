## The input of SmolVLM against Hugging Face's `Idefics3Processor` (`smolvlm_processor_golden.py`,
## `fixtures/smolvlm/processor_golden.json`): `Vision.idefics3_tiling` on a table of sizes; the tiles of
## `Vision.idefics3_tiles` of the drawn images of `fixtures/resnet` and of a synthetic one with hard edges
## (`fixtures/smolvlm/checker.png`; each tile 512 by 512, the global image last) through `normalize` against the
## `pixel_values` of the processor; and `SmolVlmPrompt.expand`,
## which gives the ids of the processor from the ids with one `<image>` for each image. No checkpoint is needed. Run from
## `vision/`.

import Image from "pkg/image/image"
import SmolVlmPrompt from "pkg/transformers/smolvlm_prompt"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Vision from "../vision"

TILE = 512
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

## What passes for a frame: a value within 1.4 of the 8-bit levels the processor makes (0.0112 measured: 1 / 255 / 0.5 =
## 0.0078 a level, and our passes are not rounded to a level), its mean within 1e-4 (1e-5 measured), its standard
## deviation within 2e-4 (1.2e-4), its mean absolute difference of neighbours across and down within 1.3e-4 (1.2e-4 and
## 9e-5), and at the edges of the synthetic image with hard edges (`edges`) the values within 0.016 (0.0122) and 0.004 in
## a typical place (0.0031). The same bounds hold for every frame, smooth or with hard edges, a tile (an enlargement) or
## a global image (a shrink): the resize is Pillow's, whatever the kernel and the scale. With libvips' resize the tiles
## of `edges` were 0.28 off in a value and 0.023 in a mean, and the bicubic kernel or a global image taken before the
## resize to the grid were outside the bounds of the global image.
struct Limits
  largest: Float
  mean: Float
  std: Float
  dx: Float
  dy: Float
  places: Float
  typical: Float
end

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
  stds: List[Float]
  dx: List[Float]
  dy: List[Float]
  edge_at: List[List[Int]]
  edge_values: List[List[Float]]
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

## How far a frame is from Hugging Face's: the sampled values, the mean, the standard deviation, the mean absolute
## difference of neighbours across and down, and the values at the places of the largest differences.
struct Gap
  largest: Float
  mean: Float
  std: Float
  dx: Float
  dy: Float
  places: Float
  ## The mean of the gaps at the places, not the largest.
  typical: Float
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

def test_the_tiling_refuses_a_side_under_1() -> Result[Void]
  try Test.error?(Vision.idefics3_tiling(0, 10))
  Test.error?(Vision.idefics3_tiling(10, 0))
end

## The mean absolute difference of neighbours of `x` (`[3, 512, 512]`) across (`axis` 2) or down (1).
def neighbours(x: Tensor.Tensor, axis: Int) -> Result[Tensor.Tensor]
  whole = [3, TILE, TILE]
  after = try Tensor.slice(x, [0, 0, 0].set(axis, 1), whole)
  before = try Tensor.slice(x, [0, 0, 0], whole.set(axis, TILE - 1))
  step = try Tensor.sub(after, before)
  Tensor.reshape(try Tensor.mean(try Tensor.sqrt(try Tensor.mul(step, step))), [1])
end

def typical_gap(values: List[Float], want: List[Float]) -> Float
  total = 0.0
  for i in 0...want.size
    total += (values[i] - want[i]).abs
  end

  total / want.size.to_f
end

def gap_of(values: List[Float], want: List[Float]) -> Float
  largest = 0.0
  for i in 0...want.size
    largest = [largest, (values[i] - want[i]).abs].max.unwrap!
  end

  largest
end

## The gap of the sampled values of `frame` of `got` (`[3, 512, 512]`) from the golden's, and of its statistics.
def measure(got: Tensor.Tensor, c: Case, frame: Int) -> Result[Gap]
  total = 3 * TILE * TILE
  want = c.samples[frame]
  backend = Tensor.backend(got)
  at = (0...want.size).to_a.map { |i| Tensor.Values.Int(i * c.stride) }
  ids = try Tensor.from_list(at, [at.size], Tensor.Dtype.I32, backend)
  places = c.edge_at[frame]
  # A tensor with no elements is not lowered to ggml: a frame without edge places reads the first value, unused.
  read = places
  if places.empty?
    read = [0]
  end

  edge_ids = try Tensor.from_list(read.map { |i| Tensor.Values.Int(i) }, [read.size], Tensor.Dtype.I32, backend)
  outs = try Trace.eval_many([got, ids, edge_ids]) do |xs|
    column = try Tensor.reshape(xs[0], [total, 1])
    mean = try Tensor.mean(xs[0])
    spread = try Tensor.sqrt(try Tensor.sub(try Tensor.mean(try Tensor.mul(xs[0], xs[0])), try Tensor.mul(mean, mean)))
    ok([
      try Tensor.embedding(xs[1], column),
      try Tensor.reshape(mean, [1]),
      try Tensor.reshape(spread, [1]),
      try neighbours(xs[0], 2),
      try neighbours(xs[0], 1),
      try Tensor.embedding(xs[2], column),
    ])
  end

  mean = (try S.floats_of(outs[1]))[0]
  spread = (try S.floats_of(outs[2]))[0]
  across = (try S.floats_of(outs[3]))[0]
  down = (try S.floats_of(outs[4]))[0]
  edge_gap = 0.0
  edge_typical = 0.0
  if places.size > 0
    edge_gap = gap_of(try S.floats_of(outs[5]), c.edge_values[frame])
    edge_typical = typical_gap(try S.floats_of(outs[5]), c.edge_values[frame])
  end

  ok(
    Gap(
      largest: gap_of(try S.floats_of(outs[0]), want),
      mean: (mean - c.means[frame]).abs,
      std: (spread - c.stds[frame]).abs,
      dx: (across - c.dx[frame]).abs,
      dy: (down - c.dy[frame]).abs,
      places: edge_gap,
      typical: edge_typical,
    ),
  )
end

def limits() -> Limits
  Limits(largest: 0.015, mean: 0.0001, std: 0.0002, dx: 0.00013, dy: 0.00013, places: 0.016, typical: 0.004)
end

## An error unless `gap` is within what passes for the frame `frame` of the case `c`.
def within(c: Case, frame: Int, gap: Gap) -> Result[Void]
  max = limits()
  where = "#{c.name} frame #{frame}"
  try Test.assert(gap.largest <= max.largest, "#{where}: the largest gap of a sample is #{gap.largest}")
  try Test.assert(gap.mean <= max.mean, "#{where}: the mean is #{gap.mean} from Hugging Face's")
  try Test.assert(gap.std <= max.std, "#{where}: the standard deviation is #{gap.std} from Hugging Face's")
  try Test.assert(gap.dx <= max.dx, "#{where}: the differences across are #{gap.dx} from Hugging Face's")
  try Test.assert(gap.dy <= max.dy, "#{where}: the differences down are #{gap.dy} from Hugging Face's")
  try Test.assert(gap.places <= max.places, "#{where}: a value at an edge is #{gap.places} from Hugging Face's")
  Test.assert(gap.typical <= max.typical, "#{where}: the values at the edges are #{gap.typical} from Hugging Face's")
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
    tiles = try Vision.idefics3_tiles(image, backend)
    try Test.eq([tiles.rows, tiles.cols], [c.rows[k], c.cols[k]])
    try Test.eq(tiles.tiles.size, c.rows[k] * c.cols[k] + 1)
    for i in 0...tiles.tiles.size
      tile = tiles.tiles[i]
      try Test.eq(Tensor.shape(tile), [3, TILE, TILE])
      got = try Vision.normalize(tile, MEAN, STD)
      try within(c, first + i, try measure(got, c, first + i))
    end

    ok(first + tiles.tiles.size)
  end
end

## An image of 1 band (grey) or 4 (with alpha) is not an sRGB one, which the frames are made of: the caller converts it.
def test_an_image_that_is_not_srgb_is_an_error() -> Result[Void]
  grey = try S.u8([0, 255, 128, 64], 2, 2, 1)
  try refused(Vision.idefics3_tiles(grey), "3 bands")
  alpha = try S.u8((0...16).to_a.map { |i| i * 10 }, 2, 2, 4)
  refused(Vision.idefics3_tiles(alpha), "3 bands")
end

def test_the_ids_of_the_processor_are_the_ids_with_the_tiles_expanded() -> Result[Void]
  g = try golden()
  prompt = prompt_of(g.tokens)
  for c in g.cases
    try Test.eq(try SmolVlmPrompt.expand(c.ids_one, prompt, grids_of(c)), c.ids)
  end

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
  try refused(SmolVlmPrompt.expand(ids, prompt, [SmolVlmPrompt.Grid(rows: 0, cols: 0)]), "1 to 6")
  ok()
end

## The ids that `SmolVlmPrompt.open` reads from a folder with only the files it needs (`fixtures/smolvlm/prompt`, made by
## the golden script from the checkpoint's) are the processor's.
def test_the_ids_of_a_folder_are_the_processors() -> Result[Void]
  want = prompt_of((try golden()).tokens)
  try Test.eq(try SmolVlmPrompt.open("test/fixtures/smolvlm/prompt"), want)
  try refused(SmolVlmPrompt.open("test/fixtures/smolvlm"), "tokenizer.json")
  ok()
end

## The same on the real folder of the checkpoint, when `SMOLVLM_MODEL` names it.
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
