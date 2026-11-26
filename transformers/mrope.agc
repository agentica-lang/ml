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
      if grid.height < 1 or grid.width < 1 or grid.height % merge != 0 or grid.width % merge != 0
        return error("grid #{next}: #{grid.height} by #{grid.width} patches is not a multiple of the merge #{merge}")
      end

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
