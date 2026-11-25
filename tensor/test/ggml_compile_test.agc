## `Trace.evaluate` on `Backend.Ggml` (`src/lower.agc`, `native/ggml_graph.c`):
## a small MLP and a cross-entropy, forward and gradients, against the
## reference; the structural cache; the refusals; and the same parity on
## Vulkan device 0 when there is one.
##
## Without the module the checks pass by asserting nothing, unless
## `TENSOR_REQUIRE_GGML` is set (CI sets it), and then they fail. Without a
## Vulkan device the Vulkan checks pass the same way (note.md §5), unless
## `TENSOR_REQUIRE_VULKAN` is set.

import Lower from "../src/lower"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
REF = Tensor.Backend.Reference
## The largest gap allowed, relative to the largest reference value of a tensor.
TOL = 0.000001

private

## True when `backend` is there; with `TENSOR_REQUIRE_GGML` set, a missing
## ggml CPU backend is an error, never a pass, and with `TENSOR_REQUIRE_VULKAN`
## set so is a missing Vulkan device.
def present(backend: Tensor.Backend) -> Result[Bool]
  try Tensor.vulkan_fp16(false)
  return ok(true) if Tensor.available(backend)
  name = "TENSOR_REQUIRE_VULKAN"
  if backend == CPU
    name = "TENSOR_REQUIRE_GGML"
  end

  flag = System.env(name)
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("the backend is required but not available (#{name} is set)")
end

## `count` floats in `[-scale, scale)` from `seed`.
def uniform(count: Int, seed: Int, scale: Float) -> List[Float]
  s = seed
  out: List[Float] = []
  for _ in 0...count
    s = (s * 1103515245 + 12345) % 2147483648
    out = out.push(((s.to_f / 2147483648.0) * 2.0 - 1.0) * scale)
  end

  out
end

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

def labels(xs: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Int(x) }, [xs.size], Tensor.Dtype.I32, backend)
end

## The mean softmax cross-entropy of `logits` `[batch, classes]` against
## `labels` `[batch]`, as `train/`'s `Loss.cross_entropy` writes it.
def cross_entropy(logits: Tensor.Tensor, labels: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(logits)
  top = try Tensor.stop_grad(try Tensor.max(logits, some([1]), true))
  shifted = try Tensor.sub(logits, try Tensor.broadcast_to(top, shape))
  total = try Tensor.sum(try Tensor.exp(shifted), some([1]), true)
  log_probs = try Tensor.sub(shifted, try Tensor.broadcast_to(try Tensor.log(total), shape))
  classes = try Tensor.arange(0, shape[1], 1, Tensor.dtype(labels), Tensor.backend(logits))
  hit = try Tensor.eq(
    try Tensor.broadcast_to(try Tensor.reshape(classes, [1, shape[1]]), shape),
    try Tensor.broadcast_to(try Tensor.reshape(labels, [shape[0], 1]), shape),
  )

  zeros = try Tensor.full(shape, Tensor.Values.Float(0.0), Tensor.dtype(logits), Tensor.backend(logits))
  picked = try Tensor.where(hit, log_probs, zeros)
  Tensor.neg(try Tensor.mean(try Tensor.sum(picked, some([1]))))
end

## The arguments of the small MLP on `backend`: x `[6, 8]`, then w1, b1, w2,
## b2 of an 8 → 16 → 4 network, from `seed`.
def mlp_args(backend: Tensor.Backend, seed: Int) -> Result[List[Tensor.Tensor]]
  ok([
    try floats(uniform(48, seed, 1.0), [6, 8], backend),
    try floats(uniform(128, seed + 1, 0.35), [8, 16], backend),
    try floats(uniform(16, seed + 2, 0.1), [16], backend),
    try floats(uniform(64, seed + 3, 0.25), [16, 4], backend),
    try floats(uniform(4, seed + 4, 0.1), [4], backend),
  ])
end

## The MLP's loss as a function of its arguments, against `ys`.
def mlp_loss(ys: Tensor.Tensor) -> Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
  ->(a: List[Tensor.Tensor]) {
    h = try Tensor.relu(try Tensor.linear(a[0], a[1], some(a[2])))
    cross_entropy(try Tensor.linear(h, a[3], some(a[4])), Trace.constant(ys))
  }
end

## The sum of `k` times the argument: `k` is a fill of the trace.
def scaled(k: Float) -> Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
  ->(a: List[Tensor.Tensor]) {
    filled = try Tensor.full(Tensor.shape(a[0]), Tensor.Values.Float(k), Tensor.Dtype.F32, Tensor.backend(a[0]))
    Tensor.sum(try Tensor.mul(a[0], filled))
  }
end

## `f` run on `args` gives the floats `want`.
def gives(
  args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
  want: List[Float],
) -> Result[Void]
  Test.eq(try Tensor.to_list(try Trace.eval(args, f)), want.map { |x| Tensor.Values.Float(x) })
end

def sum_of_squares(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = Tensor.sum(try Tensor.mul(a[0], a[0]))

def total(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = Tensor.sum(a[0])

## The MLP's loss on `backend`, `rounds` times, each within `TOL` of `want`.
def repeat_mlp(backend: Tensor.Backend, rounds: Int, want: Float) -> Result[Bool]
  for _ in 0...rounds
    vg = try Trace.value_and_grad(try mlp_args(backend, 7), mlp_loss(try labels([0, 3, 1, 2, 3, 0], backend)))
    got = try float_of((try Tensor.to_list(vg.value))[0])
    return error("a task got the loss #{got}, alone it is #{want}") if (got - want).abs > TOL
  end

  ok(true)
end

## `max |got - want| / max |want|` (floored at 1e-6) of two tensors' values.
def gap(got: Tensor.Tensor, want: Tensor.Tensor) -> Result[Float]
  return error("shape #{Tensor.shape(got)} against #{Tensor.shape(want)}") if Tensor.shape(got) != Tensor.shape(want)
  a = try Tensor.to_list(got)
  b = try Tensor.to_list(want)
  worst = 0.0
  scale = 0.000001
  for i in 0...b.size
    x = try float_of(a[i])
    y = try float_of(b[i])
    if (x - y).abs > worst
      worst = (x - y).abs
    end

    if y.abs > scale
      scale = y.abs
    end
  end

  ok(worst / scale)
end

def float_of(v: Tensor.Values) -> Result[Float]
  case v
    Tensor.Values.Float(x) -> ok(x)
    _ -> error("a float tensor holds #{v}")
  end
end

## The value and every gradient of `f` agree between `backend` and the
## reference, within `TOL`.
def parity(
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  want_args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
  want_f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Void]
  got = try Trace.value_and_grad(args, f)
  want = try Trace.value_and_grad(want_args, want_f)
  outs = [got.value] + got.grads
  wants = [want.value] + want.grads
  for i in 0...outs.size
    g = try gap(outs[i], wants[i])
    return error("output #{i}: gap #{g} is above #{TOL}") if g > TOL
  end

  ok()
end

def mlp_parity(backend: Tensor.Backend) -> Result[Void]
  ys = [0, 3, 1, 2, 3, 0]
  parity(
    backend,
    try mlp_args(backend, 7),
    try mlp_args(REF, 7),
    mlp_loss(try labels(ys, backend)),
    mlp_loss(try labels(ys, REF)),
  )
end

## The cross-entropy alone, differentiated for its logits.
def ce_parity(backend: Tensor.Backend) -> Result[Void]
  xs = uniform(15, 3, 4.0)
  ys = [4, 0, 2]
  got_ys = try labels(ys, backend)
  want_ys = try labels(ys, REF)
  parity(
    backend,
    [try floats(xs, [3, 5], backend)],
    [try floats(xs, [3, 5], REF)],
    ->(a: List[Tensor.Tensor]) { cross_entropy(a[0], Trace.constant(got_ys)) },
    ->(a: List[Tensor.Tensor]) { cross_entropy(a[0], Trace.constant(want_ys)) },
  )
end

## The error `r` answers names `text`.
def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("computed a tensor where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def test_an_mlp_and_its_gradients_match_the_reference_on_the_cpu() -> Result[Void]
  return ok() if !try present(CPU)

  try mlp_parity(CPU)
  try ce_parity(CPU)
  # The opt-out of `present` was fixed by the first ggml call, Vulkan device or not.
  Test.eq(try System.env("GGML_VK_DISABLE_F16"), "1")
end

## A graph evaluated with `keep` false takes no cache slot: the same graph
## compiles again on the next call, and once kept it compiles no more.
def test_a_graph_not_kept_compiles_on_every_call() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats(uniform(21, 9, 1.0), [7, 3], CPU)
  start = try Lower.compiles()
  _ = try Trace.eval_many([x], &sigmoids, false)
  _ = try Trace.eval_many([x], &sigmoids, false)
  try Test.eq(try Lower.compiles(), start + 2)
  _ = try Trace.eval_many([x], &sigmoids)
  _ = try Trace.eval_many([x], &sigmoids)
  Test.eq(try Lower.compiles(), start + 3)
end

def sigmoids(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]] = ok([try Tensor.sigmoid(xs[0])])

def test_a_retrace_with_new_values_compiles_nothing() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats(uniform(6, 1, 1.0), [2, 3], CPU)
  _ = try Trace.eval([x], scaled(2.0))
  before = try Lower.compiles()
  y = try floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], CPU)
  got = try Trace.eval([y], scaled(0.5))
  try Test.eq(try Lower.compiles(), before)
  try Test.eq(try Tensor.to_list(got), [Tensor.Values.Float(10.5)])
  mlp = try mlp_args(CPU, 11)
  loss = mlp_loss(try labels([1, 1, 0, 2, 3, 0], CPU))
  _ = try Trace.value_and_grad(mlp, loss)
  after_one = try Lower.compiles()
  _ = try Trace.value_and_grad(try mlp_args(CPU, 12), mlp_loss(try labels([0, 1, 2, 3, 0, 1], CPU)))
  Test.eq(try Lower.compiles(), after_one)
end

def test_graphs_that_take_turns_keep_their_values() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], CPU)
  try gives([x], scaled(2.0), [42.0])
  try gives([x], &sum_of_squares, [91.0])
  gives([x], scaled(3.0), [63.0])
end

## `slice` and `gather` (one evaluation with three outputs) give the
## reference's values on `backend`: multiples of 1/32, exact in F32.
def windows_match(backend: Tensor.Backend) -> Result[Void]
  xs = (0...24).to_a.map { |i| ((i * 7) % 32).to_f / 32.0 - 0.5 }
  at = [3, 0, 1, 1]
  outs = try Trace.eval_many([try floats(xs, [4, 3, 2], backend), try labels(at, backend)], &windows)
  wants = try Trace.eval_many([try floats(xs, [4, 3, 2], REF), try labels(at, REF)], &windows)
  try Test.eq(outs.size, 3)
  for i in 0...3
    try Test.eq(try Tensor.to_list(outs[i]), try Tensor.to_list(wants[i]))
  end

  ok()
end

## A slice of `a[0]`, its rows `a[1]`, and the rows of a slice.
def windows(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  part = try Tensor.slice(a[0], [1, 1, 0], [3, 3, 2])
  ok([part, try Tensor.gather(a[0], a[1]), try Tensor.gather(part, try Tensor.slice(a[1], [1], [3]))])
end

def test_slice_and_gather_match_the_reference() -> Result[Void]
  return ok() if !try present(CPU)

  try windows_match(CPU)
  return ok() if !Tensor.available(VULKAN)
  windows_match(VULKAN)
end

## Every output of `f` on `backend` is within `TOL` of the reference, for
## the arguments `values` of `shapes` (floats from `uniform`).
def forward_parity(
  backend: Tensor.Backend,
  shapes: List[List[Int]],
  f: Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[Void]
  got_args: List[Tensor.Tensor] = []
  want_args: List[Tensor.Tensor] = []
  for i in 0...shapes.size
    n = shapes[i].reduce(1) do |acc, d| acc * d end
    xs = uniform(n, 31 + i, 1.0)
    got_args = got_args.push(try floats(xs, shapes[i], backend))
    want_args = want_args.push(try floats(xs, shapes[i], REF))
  end

  got = try Trace.eval_many(got_args, f)
  want = try Trace.eval_many(want_args, f)
  for i in 0...want.size
    g = try gap(got[i], want[i])
    return error("output #{i}: gap #{g} is above #{TOL}") if g > TOL
  end

  ok()
end

## `concat` on each axis of rank-3 tensors, the head join on axis 1 that
## attention makes (ggml counts its axes from the other end), and `stack`.
def joins(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ok([
    try Tensor.concat([a[0], a[1]], 0),
    try Tensor.concat([a[0], a[1], a[0]], 1),
    try Tensor.concat([a[1], a[0]], 2),
    try Tensor.concat([try Tensor.reshape(a[0], [6, 4]), try Tensor.reshape(a[1], [6, 4])], 1),
    try Tensor.stack([a[0], a[1]], 0),
    try Tensor.stack([a[0], a[1]], 2),
  ])
end

## Two convolutions, one of a batch, the exact GELU, a layer norm, a softmax
## and a masked attention: what Whisper's forward is made of.
def whisper_ops(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  mask = try Tensor.mul(
    try Tensor.relu(a[4]),
    try Tensor.full([5, 5], Tensor.Values.Float(-1000.0), Tensor.Dtype.F32, Tensor.backend(a[4])),
  )

  ok([
    try Tensor.conv1d(a[0], a[1], 1, 1),
    try Tensor.conv1d(try Tensor.reshape(a[2], [2, 3, 8]), a[1], 2, 1),
    try Tensor.gelu_erf(
      try Tensor.mul(a[0], try Tensor.full([3, 8], Tensor.Values.Float(4.0), Tensor.Dtype.F32, Tensor.backend(a[0]))),
    ),
    try Tensor.layer_norm(a[3], 0.00001),
    try Tensor.softmax(a[3], 1),
    try Tensor.multi_head_attention(a[5], a[5], a[5], 2, some(mask)),
  ])
end

def test_joins_and_whisper_ops_match_the_reference() -> Result[Void]
  return ok() if !try present(CPU)

  for backend in [CPU, VULKAN]
    next if !Tensor.available(backend)
    try forward_parity(backend, [[2, 3, 4], [2, 3, 4]], &joins)
    try forward_parity(backend, [[3, 8], [4, 3, 3], [6, 8], [4, 6], [5, 5], [1, 5, 4]], &whisper_ops)
  end

  ok()
end

## The loss of an embedding lookup: the sum of squares of the rows `ids` name.
def lookup_loss(ids: Tensor.Tensor) -> Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
  ->(a: List[Tensor.Tensor]) {
    rows = try Tensor.gather(a[0], Trace.constant(ids))
    Tensor.sum(try Tensor.mul(rows, rows))
  }
end

## The gradient of a lookup is a `scatter_add`: it sums the rows of a repeated
## index and matches the reference.
def lookup_parity(backend: Tensor.Backend) -> Result[Void]
  xs = uniform(12, 5, 1.0)
  ids = [3, 0, 3, 1, 3]
  parity(
    backend,
    [try floats(xs, [4, 3], backend)],
    [try floats(xs, [4, 3], REF)],
    lookup_loss(try labels(ids, backend)),
    lookup_loss(try labels(ids, REF)),
  )
end

def test_the_gradient_of_a_lookup_matches_the_reference() -> Result[Void]
  return ok() if !try present(CPU)

  try lookup_parity(CPU)
  return ok() if !Tensor.available(VULKAN)
  lookup_parity(VULKAN)
end

## `scatter_add` of rank-2 indices into a rank-2 tail, on `backend` and on the
## reference.
def tail_scatter(backend: Tensor.Backend) -> Result[Tensor.Tensor]
  ids = try Tensor.from_list([1, 0, 1, 2].map { |x| Tensor.Values.Int(x) }, [2, 2], Tensor.Dtype.I32, backend)
  updates = try floats((0...16).to_a.map { |i| (i % 5).to_f / 4.0 - 0.25 }, [2, 2, 2, 2], backend)
  Trace.eval([ids, updates]) do |a| Tensor.scatter_add(a[0], a[1], 3) end
end

def test_scatter_add_takes_rank_two_indices_and_a_rank_two_tail() -> Result[Void]
  return ok() if !try present(CPU)

  got = try tail_scatter(CPU)
  want = try tail_scatter(REF)
  try Test.eq(Tensor.shape(got), [3, 2, 2])
  try Test.eq(try Tensor.to_list(got), try Tensor.to_list(want))
  ok()
end

def test_a_scatter_add_with_one_index_outside_the_rows_is_an_error() -> Result[Void]
  return ok() if !try present(CPU)

  rows = ->(a: List[Tensor.Tensor]) { Tensor.scatter_add(a[0], a[1], 3) }
  updates = try floats([1.0, 2.0], [1, 2], CPU)
  try refused(Trace.eval([try labels([5], CPU), updates], rows), "scatter_add is outside the rows")
  try gives([try labels([2], CPU), updates], rows, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
  ok()
end

## `scatter_add` of `k` rows of `width` into `rows` rows on `backend` and on the reference, the indices repeating over
## `period` rows: the largest gap between them, relative to the largest value.
def scatter_gap(backend: Tensor.Backend, k: Int, rows: Int, width: Int, period: Int) -> Result[Float]
  at = (0...k).map { |i| (i * 7) % period }
  gap(try scattered(backend, at, rows, width), try scattered(REF, at, rows, width))
end

def scattered(backend: Tensor.Backend, at: List[Int], rows: Int, width: Int) -> Result[Tensor.Tensor]
  updates = try floats(uniform(at.size * width, 11, 1.0), [at.size, width], backend)
  Trace.eval([try labels(at, backend), updates]) do |a| Tensor.scatter_add(a[0], a[1], rows) end
end

## On Vulkan the lowering adds the rows of an index with a matmul by a mask of equal indices, a chunk of its columns at
## a time, instead of ggml's `get_rows_back`: one index, heavy repeats, k below and above the rows, and k above the
## 2^24 cells of a chunk (5000 indices, with 4999 or 50 distinct) must all give the reference's sums.
def test_scatter_add_on_vulkan_sums_the_rows_of_an_index_like_the_reference() -> Result[Void]
  return ok() if !try present(VULKAN)

  for shape in [
    [1, 16, 3, 1],
    [5, 16, 3, 3],
    [256, 1000, 8, 7],
    [300, 200, 4, 150],
    [5000, 6000, 4, 4999],
    [5000, 6000, 4, 50],
  ]
    worst = try scatter_gap(VULKAN, shape[0], shape[1], shape[2], shape[3])
    try Test.assert(worst < TOL * 10.0, "k #{shape[0]} into #{shape[1]} rows differs by #{worst}")
  end

  ok()
end

def test_a_scatter_add_index_outside_the_rows_is_an_error_on_vulkan() -> Result[Void]
  return ok() if !try present(VULKAN)

  rows = ->(a: List[Tensor.Tensor]) { Tensor.scatter_add(a[0], a[1], 3) }
  updates = try floats([1.0, 2.0], [1, 2], VULKAN)
  try refused(Trace.eval([try labels([5], VULKAN), updates], rows), "scatter_add is outside the rows")
  gives([try labels([2], VULKAN), updates], rows, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
end

## Every lookup of a chain of `n` makes about 14 ggml tensors, more than the
## 8 per traced node the graph once had room for.
def chain_of_lookups(n: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  table = try floats((0...16).to_a.map { |i| i.to_f }, [4, 4], backend)
  ids = try labels([1, 0, 3, 2], backend)
  Trace.eval([table, ids]) do |a|
    acc = a[0]
    for _ in 0...n
      acc = try Tensor.gather(acc, a[1])
    end

    ok(acc)
  end
end

def test_a_graph_of_many_lookups_fits_its_room() -> Result[Void]
  return ok() if !try present(CPU)

  got = try chain_of_lookups(600, CPU)
  Test.eq(try Tensor.to_list(got), try Tensor.to_list(try chain_of_lookups(600, REF)))
end

def test_scatter_add_sums_repeated_indices_and_refuses_one_outside_the_rows() -> Result[Void]
  return ok() if !try present(CPU)

  rows = ->(a: List[Tensor.Tensor]) { Tensor.scatter_add(a[0], a[1], 4) }
  updates = try floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], CPU)
  try gives([try labels([2, 0, 2], CPU), updates], rows, [3.0, 4.0, 0.0, 0.0, 6.0, 8.0, 0.0, 0.0])
  try refused(Trace.eval([try labels([0, 4, 1], CPU), updates], rows), "outside")
  refused(Trace.eval([try labels([0, -1, 1], CPU), updates], rows), "outside")
end

def test_a_gather_index_outside_the_rows_is_an_error() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], CPU)
  rows = ->(a: List[Tensor.Tensor]) { Tensor.gather(a[0], a[1]) }
  try refused(Trace.eval([x, try labels([0, 3], CPU)], rows), "outside the rows")
  try refused(Trace.eval([x, try labels([-1, 2], CPU)], rows), "outside the rows")
  try gives([x, try labels([2, 0], CPU)], rows, [5.0, 6.0, 1.0, 2.0])
  ten = try floats(uniform(20, 3, 1.0), [10, 2], CPU)
  Test.error?(Trace.eval([ten, try labels([134217728], CPU)], rows))
end

def test_a_zero_fill_and_another_fill_are_two_graphs() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], CPU)
  start = try Lower.compiles()
  try gives([x], scaled(0.0), [0.0])
  try Test.eq(try Lower.compiles(), start + 1)
  try gives([x], scaled(5.0), [105.0])
  try Test.eq(try Lower.compiles(), start + 2)
  try gives([x], scaled(0.0), [0.0])
  Test.eq(try Lower.compiles(), start + 2)
end

def test_a_graph_that_failed_a_run_runs_again() -> Result[Void]
  return ok() if !try present(CPU)

  f = ->(a: List[Tensor.Tensor]) { Tensor.log(a[0]) }
  try refused(Trace.eval([try floats([1.0, 0.0], [2], CPU)], f), "not finite")
  gives([try floats([1.0, 1.0], [2], CPU)], f, [0.0, 0.0])
end

## An output that is the result of an op is written where the caller holds it, and a view of one is copied there; both
## are checked, and the error names the output.
def test_a_value_that_is_not_finite_is_refused_whether_it_is_copied_or_not() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats([1.0, 0.0, 2.0, 3.0], [2, 2], CPU)
  try says(Trace.eval_many([x], &log_after_double), "output 1")
  try says(Trace.eval_many([x], &double_after_log), "output 0")
  big = try floats([1.0e30, 2.0e30, 3.0e30, 4.0e30], [2, 2], CPU)
  outs = try Trace.eval_many([big], &log_after_double)
  Test.eq(outs.size, 2)
end

def log_after_double(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ok([try Tensor.add(a[0], a[0]), try Tensor.reshape(try Tensor.log(a[0]), [4])])
end

def double_after_log(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ok([try Tensor.log(a[0]), try Tensor.reshape(try Tensor.add(a[0], a[0]), [4])])
end

def says(r: Result[List[Tensor.Tensor]], text: String) -> Result[Void]
  case r
    ok(_) -> error("computed outputs where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def test_an_i32_graph_converts_in_and_out() -> Result[Void]
  return ok() if !try present(CPU)

  step: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]] = &plus_positions
  got = try Trace.eval([try labels([1, 1, 5], CPU)], step)
  try Test.eq(Tensor.dtype(got), Tensor.Dtype.I32)
  try Test.eq(try Tensor.to_list(got), [1, 11, 5].map { |x| Tensor.Values.Int(x) })
  refused(Trace.eval([try labels([16777217, 1, 5], CPU)], step), "beyond 2^24")
end

def test_the_least_recently_used_graph_is_evicted_past_32() -> Result[Void]
  return ok() if !try present(CPU)

  first = try floats([4.0], [1], CPU)
  try gives([first], &total, [4.0])
  for n in 2..33
    try gives([try floats(uniform(n, n, 1.0).map { |_| 1.0 }, [n], CPU)], &total, [n.to_f])
  end

  before = try Lower.compiles()
  try gives([first], &total, [4.0])
  Test.eq(try Lower.compiles(), before + 1)
end

def test_two_tasks_run_graphs_at_once() -> Result[Void]
  return ok() if !try present(CPU)

  alone = try Trace.value_and_grad(try mlp_args(CPU, 7), mlp_loss(try labels([0, 3, 1, 2, 3, 0], CPU)))
  want = try float_of((try Tensor.to_list(alone.value))[0])
  other = CPU
  if Tensor.available(VULKAN)
    other = VULKAN
  end

  a = Task.spawn do repeat_mlp(CPU, 5, want) end
  b = Task.spawn do repeat_mlp(other, 5, want) end
  try Task.await(a)
  try Task.await(b)
  ok()
end

def test_what_ggml_does_not_lower_is_an_error_naming_it() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats([1.0, 2.0, 3.0, 4.0], [2, 2], CPU)
  try refused(Trace.eval([x], ->(a: List[Tensor.Tensor]) { Tensor.gelu(a[0]) }), "gelu_tanh_approx is not lowered")
  to_int = ->(a: List[Tensor.Tensor]) { Tensor.to_dtype(a[0], Tensor.Dtype.I32) }
  try refused(Trace.eval([x], to_int), "to_dtype from F32 to I32 is not lowered")
  cube = try floats(uniform(8, 2, 1.0), [2, 2, 2], CPU)
  middle = ->(a: List[Tensor.Tensor]) { Tensor.sum(a[0], some([1])) }
  try refused(Trace.eval([cube], middle), "not a leading or trailing block")
  plain = try floats([1.0, 2.0, 3.0, 4.0], [2, 2], REF)
  mixed = ->(a: List[Tensor.Tensor]) { Tensor.add(a[0], Trace.constant(plain)) }
  try refused(Trace.eval([x], mixed), "a constant on Reference in a trace run on Ggml(Cpu)")
  try refused(
    Trace.eval([try floats(uniform(27, 4, 1.0), [3, 3, 3], CPU)], &transposed),
    "rank-3 tensor is not lowered",
  )

  try refused(Trace.eval([try labels([1, 2, 3], CPU)], &is_countdown), "step <= 0")
  refused(Trace.eval([try labels([1, 2, 3], CPU)], &is_far), "beyond 2^24")
end

def transposed(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = Tensor.transpose(a[0])

## `a[0]` with `10 + i` where element `i` equals `i`.
def plus_positions(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  b = Tensor.backend(a[0])
  hit = try Tensor.eq(a[0], try Tensor.arange(0, 3, 1, Tensor.Dtype.I32, b))
  Tensor.where(hit, try Tensor.arange(10, 13, 1, Tensor.Dtype.I32, b), a[0])
end

def is_countdown(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  Tensor.eq(a[0], try Tensor.arange(3, 0, -1, Tensor.Dtype.I32, Tensor.backend(a[0])))
end

def is_far(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  Tensor.eq(a[0], try Tensor.arange(16777215, 16777218, 1, Tensor.Dtype.I32, Tensor.backend(a[0])))
end

def test_the_refusals_of_ggml_name_their_cause() -> Result[Void]
  return ok() if !try present(CPU)

  zero = try floats([1.0, 0.0, 3.0, 4.0], [2, 2], CPU)
  refused(Trace.eval([zero], ->(a: List[Tensor.Tensor]) { Tensor.log(a[0]) }), "not finite")
end

def test_vulkan_matches_the_reference_or_says_why_not() -> Result[Void]
  try Tensor.vulkan_fp16(false)
  if !Tensor.available(VULKAN)
    IO.warn("SKIP: no Vulkan device 0; the compiled Vulkan parity runs where there is one")
    return ok()
  end

  try mlp_parity(VULKAN)
  try ce_parity(VULKAN)
  try Test.eq(try System.env("GGML_VK_DISABLE_F16"), "1")
  case Tensor.vulkan_fp16(true)
    ok(_) -> error("fp16 was chosen after a Vulkan device opened with F32")
    error(e) -> Test.assert(e.message.include?("chose F32 mul_mat"), "the error said \"#{e.message}\"")
  end
end

## `x + i` for `i` below `count`: one output each.
def shifted(count: Int) -> Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]]
  ->(a: List[Tensor.Tensor]) {
    outs: List[Tensor.Tensor] = []
    for i in 0...count
      step = try Tensor.full([2], Tensor.Values.Float(i.to_f), Tensor.Dtype.F32, Tensor.Backend.Trace)
      outs = outs.push(try Tensor.add(a[0], step))
    end

    ok(outs)
  }
end

## The cap of one graph: `MAX_OUTPUTS` of `ggml_graph.c`.
def test_a_graph_returns_512_outputs_and_refuses_one_more() -> Result[Void]
  for backend in [CPU, VULKAN]
    next if !try present(backend)
    x = try floats([1.0, -2.0], [2], backend)
    outs = try Trace.eval_many([x], shifted(512))
    try Test.eq(outs.size, 512)
    try Test.eq(try Tensor.to_list(outs[511]), [Tensor.Values.Float(512.0), Tensor.Values.Float(509.0)])
    try Test.eq(try Tensor.to_list(outs[300]), [Tensor.Values.Float(301.0), Tensor.Values.Float(298.0)])
    try Test.error?(Trace.eval_many([x], shifted(513)))
  end

  ok()
end
