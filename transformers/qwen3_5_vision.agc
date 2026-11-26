## The vision tower of Qwen3.5 (`Qwen3_5ForConditionalGeneration`, `model.visual.*` of a `Qwen/Qwen3.5-*` checkpoint;
## `transformers/DESIGN.md`): a patch embedding (a 3-D convolution whose stride is its kernel, run as one dense layer),
## learned positions resampled bilinearly to the grid of the image, pre-norm blocks of full attention with a 2-D rotary
## position (one half of the pairs turns by the row, the other half by the column) and an MLP with `gelu_pytorch_tanh`,
## and a merger that joins each 2 by 2 square of patches into one vector of the text model's width.
##
## The input is `pixel_values` of Hugging Face's image processor: one row for each patch of one image, `[height * width,
## channels * frames * patch * patch]` in `F32`, the patches in the order of the merge squares (all the patches of a
## square together, the squares row by row). The output is one row for each square, `[height * width / 4, out_hidden_size]`:
## what the text model reads at the image tokens. One call is one image.

import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Qwen3 from "qwen3"
import Source from "tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import Weights from "weights"

## The prefix of the tower's tensors.
PREFIX = "model.visual."

## The `eps` of every layer norm of the tower.
EPS = 0.000001

## The `vision_config` fields that the model reads.
struct Config
  depth: Int
  hidden_size: Int
  hidden_act: String
  intermediate_size: Int
  num_heads: Int
  patch_size: Int
  spatial_merge_size: Int
  temporal_patch_size: Int
  out_hidden_size: Int
  num_position_embeddings: Int
  in_channels: Int = 3
  rope_parameters: Optional[Rotary]
end

struct Rotary
  rope_theta: Float = 10000.0
end

## One pre-norm block. `qkv` is the fused projection `[3 * width, width]`: the queries, the keys and the values.
struct Block
  norm1: Nn.LayerNorm
  qkv: Nn.Dense
  proj: Nn.Dense
  norm2: Nn.LayerNorm
  up: Nn.Dense
  down: Nn.Dense
end

struct Vision
  config: Config
  ## The patch embedding: the convolution `[width, channels, frames, patch, patch]` as `[width, channels * frames * patch
  ## * patch]`, since ggml holds tensors of rank 4 at most.
  patch: Nn.Dense
  ## `[num_position_embeddings, width]`, a square grid of positions.
  positions: Nn.Embedding
  blocks: List[Block]
  merger_norm: Nn.LayerNorm
  merger_up: Nn.Dense
  merger_down: Nn.Dense
end

## The vision tower in `dir`, which holds `config.json` and `model.safetensors` (or its `.index.json`). Only the tensors
## `model.visual.*` are read.
def open(dir: String, backend: Tensor.Backend) -> Result[Vision]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  file = try Source.within("#{dir}/config.json", JSON.decode(text, Document))
  return error("model_type is #{file.model_type}, not qwen3_5") if file.model_type != "qwen3_5"
  load(try Qwen3.checkpoint(dir), file.vision_config, backend)
end

## The vision tower of an open checkpoint. The names and shapes of its tensors are checked against `config` before any
## is loaded.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[Vision]
  try check_config(config)
  specs = specs_of(config)
  try Weights.check(c, specs)
  patch = try patch_matrix(c, config, backend)
  loaded = Weights.load(c, specs.drop(1), backend)
  if not loaded.ok?
    Lift.close([patch])
  end

  t = try loaded
  blocks: List[Block] = []
  for i in 0...config.depth
    p = "#{PREFIX}blocks.#{i}."
    blocks = blocks.push(
      Block(
        norm1: Weights.norm_of(t, "#{p}norm1", EPS),
        qkv: Weights.dense_of(t, "#{p}attn.qkv"),
        proj: Weights.dense_of(t, "#{p}attn.proj"),
        norm2: Weights.norm_of(t, "#{p}norm2", EPS),
        up: Weights.dense_of(t, "#{p}mlp.linear_fc1"),
        down: Weights.dense_of(t, "#{p}mlp.linear_fc2"),
      ),
    )
  end

  ok(
    Vision(
      config: config,
      patch: Nn.Dense(w: patch, b: some(Weights.at(t, "#{PREFIX}patch_embed.proj.bias")), out_in: true),
      positions: Nn.Embedding(table: Weights.at(t, "#{PREFIX}pos_embed.weight")),
      blocks: blocks,
      merger_norm: Weights.norm_of(t, "#{PREFIX}merger.norm", EPS),
      merger_up: Weights.dense_of(t, "#{PREFIX}merger.linear_fc1"),
      merger_down: Weights.dense_of(t, "#{PREFIX}merger.linear_fc2"),
    ),
  )
end

## The side of the square of patches that `encode` joins into one feature (`spatial_merge_size` of the checkpoint): the
## `merge` of `Mrope.expand` and `Mrope.layout` and of `Vision.patches`.
def merge(m: Vision) -> Int = m.config.spatial_merge_size

## The features of one image: `patches` is `[height * width, channels * frames * patch * patch]` (see the top of this
## file) for a grid of `height` by `width` patches, both a multiple of `spatial_merge_size`. The result is
## `[height * width / spatial_merge_size^2, out_hidden_size]` and belongs to the caller, who closes it (`Lift.close`).
## One graph on ggml.
def encode(m: Vision, patches: Tensor.Tensor, height: Int, width: Int) -> Result[Tensor.Tensor]
  merge = m.config.spatial_merge_size
  if height < merge or width < merge or height % merge != 0 or width % merge != 0
    return error("the grid #{height} by #{width} is not a multiple of spatial_merge_size #{merge}")
  end

  want = [
    height * width,
    m.config.in_channels * m.config.temporal_patch_size * m.config.patch_size * m.config.patch_size,
  ]

  return error("the patches need #{want}, got #{Tensor.shape(patches)}") if Tensor.shape(patches) != want
  side = try grid_side(m.config)
  layout = try layout_of(side, merge, height, width)
  backend = Tensor.backend(patches)
  inputs = try host_inputs(layout, backend)
  out = Lift.staged(
    m,
    ->(p: Vision) { traced(p) },
    [patches] + inputs,
    ->(p: Vision, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0], xs[1], xs[2], xs[3], xs[4])]) },
  )

  Lift.close(inputs)
  case out
    ok(ts) -> ok(ts[0])
    error(e) -> error(e)
  end
end

## Closes the weights of `m` now. Nothing may use `m` afterwards. The features that `encode` gave belong to the caller.
def close_model(m: Vision) -> Void
  tensors = Lift.dense_parts(m.patch) + [m.positions.table] + Lift.norm_parts(m.merger_norm)
  tensors = tensors + Lift.dense_parts(m.merger_up) + Lift.dense_parts(m.merger_down)
  for b in m.blocks
    tensors = tensors + Lift.norm_parts(b.norm1) + Lift.dense_parts(b.qkv) + Lift.dense_parts(b.proj)
    tensors = tensors + Lift.norm_parts(b.norm2) + Lift.dense_parts(b.up) + Lift.dense_parts(b.down)
  end

  Lift.close(tensors)
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: Vision) -> Vision
  m.with(
    patch: Lift.dense(m.patch),
    positions: Lift.embedding(m.positions),
    blocks: m.blocks.map { |b: Block| traced_block(b) },
    merger_norm: Lift.norm(m.merger_norm),
    merger_up: Lift.dense(m.merger_up),
    merger_down: Lift.dense(m.merger_down),
  )
end

private

## `config.json` of a checkpoint: the tower is in `vision_config`.
struct Document
  model_type: String
  vision_config: Config
end

## What the host works out from the grid, one entry for each patch in the order of the merge squares: the row and the
## column of the patch (the rotary positions), and the four taps of the bilinear resampling of the position table with
## their weights (`taps` and `weights` hold the first tap of every patch, then the second, and so on).
struct Layout
  rows: List[Int]
  cols: List[Int]
  taps: List[Int]
  weights: List[Float]
end

## The two neighbours of a coordinate on the axis of the position table, and the weight of the second.
struct Axis
  first: Int
  second: Int
  weight: Float
end

## The rows, the columns and the taps (`I32`) and the tap weights (`F32`) as tensors. When one fails, the ones made
## before it are closed.
def host_inputs(layout: Layout, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  made: List[Tensor.Tensor] = []
  for ids in [layout.rows, layout.cols, layout.taps]
    case Tensor.from_list(ids.map { |i: Int| Tensor.Values.Int(i) }, [ids.size], Tensor.Dtype.I32, backend)
      ok(t) -> made = made.push(t)
      error(e) ->
        Lift.close(made)
        return error(e)
    end
  end

  case Tensor.floats(layout.weights, [layout.weights.size, 1], backend)
    ok(t) -> ok(made.push(t))
    error(e) ->
      Lift.close(made)
      error(e)
  end
end

def traced_block(b: Block) -> Block
  b.with(
    norm1: Lift.norm(b.norm1),
    qkv: Lift.dense(b.qkv),
    proj: Lift.dense(b.proj),
    norm2: Lift.norm(b.norm2),
    up: Lift.dense(b.up),
    down: Lift.dense(b.down),
  )
end

def check_config(config: Config) -> Result[Void]
  if config.hidden_act != "gelu_pytorch_tanh"
    return error("hidden_act #{config.hidden_act} is not supported, only gelu_pytorch_tanh")
  end

  if config.depth < 1 or config.patch_size < 1 or config.temporal_patch_size < 1 or config.spatial_merge_size < 1
    return error("depth, patch_size, temporal_patch_size and spatial_merge_size need 1 or more")
  end

  if (
    config.num_heads < 1 or
    config.hidden_size % config.num_heads != 0 or
    (config.hidden_size / config.num_heads) % 4 != 0
  )
    return error(
      "hidden_size #{config.hidden_size} needs a head width that is a multiple of 4 over #{config.num_heads} heads",
    )
  end

  try grid_side(config)
  ok()
end

## The side of the square table of positions.
def grid_side(config: Config) -> Result[Int]
  side = 1
  while side * side < config.num_position_embeddings
    side = side + 1
  end

  if side * side != config.num_position_embeddings
    return error("num_position_embeddings #{config.num_position_embeddings} is not a square")
  end

  ok(side)
end

## The patch embedding on `backend`, flattened: read on the reference, whose tensors have any rank, and made again.
def patch_matrix(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  stored = try Load.weight(c, "#{PREFIX}patch_embed.proj.weight", 5, Tensor.Backend.Reference)
  values = try Tensor.to_list(stored)
  Tensor.from_list(values, [config.hidden_size, values.size / config.hidden_size], Tensor.Dtype.F32, backend)
end

## Every tensor that `load` reads, with its shape; the patch weight, which `patch_matrix` loads, is the first.
def specs_of(config: Config) -> List[Weights.Spec]
  width = config.hidden_size
  joined = width * config.spatial_merge_size * config.spatial_merge_size
  patch = [width, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size]
  specs = [
    Weights.Spec(name: "#{PREFIX}patch_embed.proj.weight", shape: patch),
    Weights.Spec(name: "#{PREFIX}patch_embed.proj.bias", shape: [width]),
    Weights.Spec(name: "#{PREFIX}pos_embed.weight", shape: [config.num_position_embeddings, width]),
  ]

  for i in 0...config.depth
    p = "#{PREFIX}blocks.#{i}."
    specs = specs + norm_specs("#{p}norm1", width) + dense_specs("#{p}attn.qkv", 3 * width, width)
    specs = specs + dense_specs("#{p}attn.proj", width, width) + norm_specs("#{p}norm2", width)
    specs = specs + dense_specs("#{p}mlp.linear_fc1", config.intermediate_size, width)
    specs = specs + dense_specs("#{p}mlp.linear_fc2", width, config.intermediate_size)
  end

  specs = specs + norm_specs("#{PREFIX}merger.norm", width)
  specs = specs + dense_specs("#{PREFIX}merger.linear_fc1", joined, joined)
  specs + dense_specs("#{PREFIX}merger.linear_fc2", config.out_hidden_size, joined)
end

def norm_specs(prefix: String, width: Int) -> List[Weights.Spec]
  [Weights.Spec(name: "#{prefix}.weight", shape: [width]), Weights.Spec(name: "#{prefix}.bias", shape: [width])]
end

def dense_specs(prefix: String, out: Int, into: Int) -> List[Weights.Spec]
  [Weights.Spec(name: "#{prefix}.weight", shape: [out, into]), Weights.Spec(name: "#{prefix}.bias", shape: [out])]
end

## The layout of a grid of `height` by `width` patches with merge squares of `merge`, over a table of `side` by `side`
## positions. The bilinear resampling aligns the corners: the first and last coordinate of an axis are the first and
## last position of the table.
def layout_of(side: Int, merge: Int, height: Int, width: Int) -> Result[Layout]
  rows: List[Int] = []
  cols: List[Int] = []
  for square_row in 0...(height / merge)
    for square_col in 0...(width / merge)
      for r in 0...merge
        for c in 0...merge
          rows = rows.push(square_row * merge + r)
          cols = cols.push(square_col * merge + c)
        end
      end
    end
  end

  down = try axes_of(rows, height, side)
  across = try axes_of(cols, width, side)
  taps: List[Int] = []
  weights: List[Float] = []
  for tap in 0...4
    for i in 0...rows.size
      y = down[i]
      x = across[i]
      row = y.first
      col = x.first
      share = 1.0 - y.weight
      if tap >= 2
        row = y.second
        share = y.weight
      end

      if tap % 2 == 1
        col = x.second
        share = share * x.weight
      else
        share = share * (1.0 - x.weight)
      end

      taps = taps.push(row * side + col)
      weights = weights.push(share)
    end
  end

  ok(Layout(rows: rows, cols: cols, taps: taps, weights: weights))
end

## The neighbours of each coordinate of `coords` on an axis of `size` coordinates, in a table of `side` positions.
def axes_of(coords: List[Int], size: Int, side: Int) -> Result[List[Axis]]
  span = 1
  if size > 1
    span = size - 1
  end

  out: List[Axis] = []
  for coord in coords
    at = (coord.to_f * (side - 1).to_f) / span.to_f
    first = try at.to_i
    second = first + 1
    if second > side - 1
      second = side - 1
    end

    out = out.push(Axis(first: first, second: second, weight: at - first.to_f))
  end

  ok(out)
end

## The positions of the table resampled to the grid: the four taps of every patch, weighted, `[n, width]`.
def position_rows(m: Vision, taps: Tensor.Tensor, weights: Tensor.Tensor) -> Result[Tensor.Tensor]
  rows = try Nn.embedding(m.positions, taps)
  shape = Tensor.shape(rows)
  weighted = try Tensor.mul(rows, try Tensor.broadcast_to(weights, shape))
  parted = try Tensor.reshape(weighted, [4, shape[0] / 4, shape[1]])
  Tensor.sum(parted, some([0]))
end

## The whole tower on `patches` `[n, ...]`: `[n / merge^2, out_hidden_size]`.
def graph(
  m: Vision,
  patches: Tensor.Tensor,
  rows: Tensor.Tensor,
  cols: Tensor.Tensor,
  taps: Tensor.Tensor,
  weights: Tensor.Tensor,
) -> Result[Tensor.Tensor]
  n = Tensor.shape(patches)[0]
  width = m.config.hidden_size
  into = Tensor.shape(patches)[1]
  embedded = try Nn.dense(m.patch, try Tensor.reshape(patches, [1, n, into]))
  h = try Tensor.add(embedded, try Tensor.reshape(try position_rows(m, taps, weights), [1, n, width]))
  for b in m.blocks
    h = try block(m, b, h, rows, cols)
  end

  joined = width * m.config.spatial_merge_size * m.config.spatial_merge_size
  squares = try Tensor.reshape(
    try Nn.layer_norm(m.merger_norm, h),
    [n / (m.config.spatial_merge_size * m.config.spatial_merge_size), joined],
  )

  Nn.dense(m.merger_down, try Nn.activate(Nn.Activation.GeluErf, try Nn.dense(m.merger_up, squares)))
end

## One block over `h` `[1, n, width]`: attention over every patch of the image, then the MLP.
def block(m: Vision, b: Block, h: Tensor.Tensor, rows: Tensor.Tensor, cols: Tensor.Tensor) -> Result[Tensor.Tensor]
  n = Tensor.shape(h)[1]
  width = m.config.hidden_size
  qkv = try Nn.dense(b.qkv, try Nn.layer_norm(b.norm1, h))
  query = try rotate(m, try projected(m, qkv, 0), rows, cols)
  key = try rotate(m, try projected(m, qkv, 1), rows, cols)
  context = try Tensor.attention(query, key, try projected(m, qkv, 2))
  merged = try Tensor.reshape(try Tensor.permute(context, [0, 2, 1, 3]), [1, n, width])
  seen = try Tensor.add(h, try Nn.dense(b.proj, merged))
  up = try Nn.dense(b.up, try Nn.layer_norm(b.norm2, seen))
  Tensor.add(seen, try Nn.dense(b.down, try gelu_tanh(up)))
end

## The queries (`i` 0), the keys (1) or the values (2) of the fused projection `qkv` `[1, n, 3 * width]`, as heads
## `[1, heads, n, head]`.
def projected(m: Vision, qkv: Tensor.Tensor, i: Int) -> Result[Tensor.Tensor]
  n = Tensor.shape(qkv)[1]
  width = m.config.hidden_size
  heads_of(m, try Tensor.slice(qkv, [0, 0, i * width], [1, n, (i + 1) * width]))
end

## `gelu_pytorch_tanh`, `x * sigmoid(2 * sqrt(2 / pi) * (x + 0.044715 * x^3))`, from the ops that ggml compiles: it has
## no `Tensor.gelu` (the tanh form) yet (ml#301).
def gelu_tanh(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  by = ->(v: Float) { Tensor.full(shape, Tensor.Values.Float(v), Tensor.dtype(x), Tensor.backend(x)) }
  cube = try Tensor.mul(try Tensor.mul(x, x), x)
  inner = try Tensor.add(x, try Tensor.mul(cube, try by.call(0.044715)))
  Tensor.mul(x, try Tensor.sigmoid(try Tensor.mul(inner, try by.call(1.5957691216057308))))
end

## `[1, n, width]` as `[1, heads, n, head]`.
def heads_of(m: Vision, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  n = Tensor.shape(x)[1]
  heads = m.config.num_heads
  Tensor.permute(try Tensor.reshape(x, [1, n, heads, m.config.hidden_size / heads]), [0, 2, 1, 3])
end

## Rope of the heads `x` `[1, heads, n, head]` by the row of each patch on one half of the pairs and by its column on
## the other, as Hugging Face's `rotate_half` over the whole head does with the angles `[rows, cols, rows, cols]`: the
## pairs are `(i, i + head / 2)`, so the columns of the row angles are `0...head / 4` and `head / 2...3 * head / 4`.
## They are brought together and rotated by `Tensor.rope`, and so are the others; the head comes out with its columns
## in that order, the same for the queries and the keys, so `q k^T` is the same.
def rotate(m: Vision, x: Tensor.Tensor, rows: Tensor.Tensor, cols: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  head = shape[3]
  half = head / 2
  quarter = head / 4
  cut = ->(from: Int, to: Int) { Tensor.slice(x, [0, 0, 0, from], [shape[0], shape[1], shape[2], to]) }
  down = try Tensor.concat([try cut.call(0, quarter), try cut.call(half, half + quarter)], 3)
  across = try Tensor.concat([try cut.call(quarter, half), try cut.call(half + quarter, head)], 3)
  base = m.config.rope_parameters.map { |r: Rotary| r.rope_theta }.unwrap_or(10000.0)
  Tensor.concat([try Tensor.rope(down, rows, base: base), try Tensor.rope(across, cols, base: base)], 3)
end
