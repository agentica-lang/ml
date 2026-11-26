## The prompt of SmolVLM with images, as Hugging Face's `Idefics3Processor` writes it (`transformers/DESIGN.md`): the
## `<image>` that the chat template writes once for each image becomes the ids of its tiles, then of its global image.
## Each tile is `<fake_token_around_image>`, `<row_r_col_c>` and `image_seq_len` `<image>` ids, a row of tiles ends with
## a newline, and the global image follows after another one: `<fake_token_around_image>`, `<global-img>`, the
## `<image>` ids and `<fake_token_around_image>`. The caller's ids hold one `<image>` id for each image (the tokenizer
## makes it one id), and the ids of the result are what the processor tokenizes, as the special tokens cut the text.
##
## `Vision.idefics3_tiles` (`vision/`) splits an image and gives the rows and columns of the grid.

import Source from "tokenizer/source"

## The rows and the columns of `<row_r_col_c>` tokens that the tokenizer has.
ROW_COLS = 6

## How many rows and columns of 512 by 512 tiles an image was split into (`Vision.Tiles`): both 0 for an image that was
## not split, which has only its global image.
struct Grid
  rows: Int
  cols: Int
end

## The ids of the prompt of a checkpoint.
struct Prompt
  fake: Int
  image: Int
  global: Int
  ## `<row_1_col_1>`, `<row_1_col_2>` ... `<row_6_col_6>`, row by row.
  row_cols: List[Int]
  ## The ids of `"\n"` and of `"\n\n"`.
  newline: Int
  newlines: Int
  ## The `<image>` ids of each image.
  seq_len: Int
end

struct Added
  id: Int
  content: String
end

struct Vocabulary
  vocab: Map[String, Int]
end

struct Tokenizer
  added_tokens: List[Added]
  model: Vocabulary
end

struct Processor
  image_seq_len: Int
end

## The ids of the folder `dir` of an official SmolVLM repository: the special tokens of `tokenizer.json`, its ids for
## `"\n"` and `"\n\n"`, and `image_seq_len` of `processor_config.json`.
def open(dir: String) -> Result[Prompt]
  file = "#{dir}/tokenizer.json"
  text = try Source.within(file, Source.read_text(file))
  tok = try Source.within(file, JSON.decode(text, Tokenizer))
  ids: Map[String, Int] = {}
  for a in tok.added_tokens
    ids = ids.put(a.content, a.id)
  end

  config = "#{dir}/processor_config.json"
  seq = try Source.within(config, JSON.decode(try Source.within(config, Source.read_text(config)), Processor))
  row_cols: List[Int] = []
  for r in 1..ROW_COLS
    for c in 1..ROW_COLS
      row_cols = row_cols.push(try named(ids, "<row_#{r}_col_#{c}>"))
    end
  end

  ok(
    Prompt(
      fake: try named(ids, "<fake_token_around_image>"),
      image: try named(ids, "<image>"),
      global: try named(ids, "<global-img>"),
      row_cols: row_cols,
      newline: try named(tok.model.vocab, String.from_code(266)),
      newlines: try named(tok.model.vocab, String.from_code(266) + String.from_code(266)),
      seq_len: seq.image_seq_len,
    ),
  )
end

## `ids` with each `<image>` replaced by the ids of the `k`th image of `grids`, in order. An error when the number of
## `<image>` ids is not that of `grids`, or a grid is not 0 by 0 or from 1 by 1 up to 6 by 6.
def expand(ids: List[Int], p: Prompt, grids: List[Grid]) -> Result[List[Int]]
  out: List[Int] = []
  next = 0
  for id in ids
    if id != p.image
      out = out.push(id)
    else
      return error("the prompt has more images than the #{grids.size} grids") if next >= grids.size
      out = out + try image_ids(p, grids[next], next)
      next += 1
    end
  end

  return error("the prompt has #{next} images and #{grids.size} grids") if next != grids.size
  ok(out)
end

private

def named(ids: Map[String, Int], name: String) -> Result[Int]
  found = ids.get(name)
  return error("the tokenizer has no #{name}") if found.none?
  ok(found.unwrap!)
end

def repeated(id: Int, n: Int) -> List[Int]
  out: List[Int] = []
  for _ in 0...n
    out = out.push(id)
  end

  out
end

## The ids that stand for one image: its tiles row by row, and its global image.
def image_ids(p: Prompt, g: Grid, index: Int) -> Result[List[Int]]
  fits = g.rows >= 1 and g.rows <= ROW_COLS and g.cols >= 1 and g.cols <= ROW_COLS
  if not fits and (g.rows != 0 or g.cols != 0)
    return error("image #{index}: #{g.rows} by #{g.cols} tiles, a side needs 1 to #{ROW_COLS} (or 0 by 0)")
  end

  pixels = repeated(p.image, p.seq_len)
  out: List[Int] = []
  for r in 0...g.rows
    for c in 0...g.cols
      out = out + [p.fake, p.row_cols[r * ROW_COLS + c]] + pixels
    end

    if r < g.rows - 1
      out = out.push(p.newline)
    end
  end

  if g.rows > 0
    out = out.push(p.newlines)
  end

  ok(out + [p.fake, p.global] + pixels + [p.fake])
end
