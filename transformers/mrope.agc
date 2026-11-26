## The positions of a prompt with images for the multimodal rope of Qwen3.5 (`get_rope_index` of Hugging Face,
## `transformers/DESIGN.md`), worked out on the host: a text token has the same position on the three axes (time, down,
## across), counting up; the tokens of an image have the position where it starts as their time, and that position plus
## their row (down) or column (across) in the merged grid; the text after an image goes on from the start plus the
## larger side of the merged grid, so an image spans fewer positions than it has tokens. `delta` is how far the position
## of the next token is from its index in the prompt, for the tokens that follow.

## The patches of an image before the merge, as the vision tower takes them (`Qwen3_5Vision.encode`).
struct Grid
  height: Int
  width: Int
end

## The position of every token of a prompt on each axis, `at` the indexes of the image tokens, and the `delta`: the
## position that follows the prompt minus its number of tokens.
struct Layout
  time: List[Int]
  down: List[Int]
  across: List[Int]
  at: List[Int]
  delta: Int
end

## `ids` with each `pad_token`, which the template writes once for an image (`Chat.IMAGE`), repeated as many times as
## the image has tokens: the `height / merge * width / merge` of its grid, `grids[k]` for the `k`th pad. An error when
## the number of pads is not that of `grids`, or a grid is not a multiple of `merge`.
def expand(ids: List[Int], pad_token: Int, grids: List[Grid], merge: Int) -> Result[List[Int]]
  return error("merge needs 1 or more, got #{merge}") if merge < 1
  out: List[Int] = []
  next = 0
  for id in ids
    if id != pad_token
      out = out.push(id)
    else
      return error("the prompt has more image tokens than the #{grids.size} images") if next >= grids.size
      try check_grid(grids[next], merge, next)
      for _ in 0...tokens(grids[next], merge)
        out = out.push(pad_token)
      end

      next += 1
    end
  end

  return error("the prompt has #{next} images and #{grids.size} grids") if next != grids.size
  ok(out)
end

## The `Layout` of a model with one position for each token (Idefics3): every token has its index on the three axes,
## `at` the tokens equal to `image_token`, and no `delta`.
def indexed(ids: List[Int], image_token: Int) -> Layout
  index: List[Int] = []
  at: List[Int] = []
  for i in 0...ids.size
    index = index.push(i)
    if ids[i] == image_token
      at = at.push(i)
    end
  end

  Layout(time: index, down: index, across: index, at: at, delta: 0)
end

## The `Layout` of `ids`, whose tokens equal to `image_token` come in runs, one per grid of `grids` in order: each
## run has the `height / merge * width / merge` tokens of its image. Two images need the vision start and end tokens
## between them: adjacent runs are one run, and the count of its tokens then fails.
def layout(ids: List[Int], image_token: Int, grids: List[Grid], merge: Int) -> Result[Layout]
  return error("merge needs 1 or more, got #{merge}") if merge < 1
  time: List[Int] = []
  down: List[Int] = []
  across: List[Int] = []
  at: List[Int] = []
  next = 0
  pos = 0
  i = 0
  while i < ids.size
    if ids[i] != image_token
      time = time.push(pos)
      down = down.push(pos)
      across = across.push(pos)
      pos += 1
      i += 1
    else
      return error("the prompt has more images than the #{grids.size} grids") if next >= grids.size
      grid = grids[next]
      try check_grid(grid, merge, next)
      rows = grid.height / merge
      columns = grid.width / merge
      run = 0
      while i + run < ids.size and ids[i + run] == image_token
        run += 1
      end

      if run != rows * columns
        return error("image #{next} has #{run} tokens in the prompt, its grid has #{rows * columns}")
      end

      for k in 0...run
        time = time.push(pos)
        down = down.push(pos + k / columns)
        across = across.push(pos + k % columns)
        at = at.push(i + k)
      end

      pos += [rows, columns].max.unwrap!
      i += run
      next += 1
    end
  end

  return error("the prompt has #{next} images and #{grids.size} grids") if next != grids.size
  ok(Layout(time: time, down: down, across: across, at: at, delta: pos - ids.size))
end

private

## An error when the sides of the `index`th grid are not 1 or more, or `merge` does not divide them.
def check_grid(grid: Grid, merge: Int, index: Int) -> Result[Void]
  if grid.height < 1 or grid.width < 1
    return error("grid #{index}: #{grid.height} by #{grid.width} patches, a side needs 1 or more")
  end

  if grid.height % merge != 0 or grid.width % merge != 0
    return error("grid #{index}: #{grid.height} by #{grid.width} patches is not a multiple of the merge #{merge}")
  end

  ok()
end

## The image tokens of a grid that `check_grid` accepted: its patches joined by `merge` on each axis.
def tokens(grid: Grid, merge: Int) -> Int = (grid.height / merge) * (grid.width / merge)
