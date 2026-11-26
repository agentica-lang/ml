## Generation with a cache against the tiny Qwen3 of `qwen_peer.py`: the logits
## of every step match the peer's forward over the whole row, a prompt padded to
## a bucket (of 5, 8 and 9 tokens) and a cache that grows from 8 to 16 to 32 rows
## included, and the greedy ids are those of the forward without a cache. Run from
## `transformers/`.

import Generate from "../generate"
import Stepping from "../stepping"
import Ggml from "pkg/tensor/src/ggml"
import GgmlRaw from "pkg/tensor/src/ggml_native"
import KvDecoder from "../kv_decoder"
import Lift from "../lift"
import Placement from "../placement"
import Qwen3 from "../qwen3"
import S from "support"
import Sample from "../sample"
import Tensor from "pkg/tensor/tensor"

ABS = 0.000000001
REL = 0.000000001

# The cache on ggml is F16: its keys and values are rounded to 11 bits, so the logits of the tiny model drift by up to
# 2.4e-4 (the real Qwen3-0.6B by 7e-3 on logits of 10 to 30); the bound is that gap with a margin, and a
# relative part of 1e-3 only.
HALF_ABS = 0.0006
HALF_REL = 0.001

# The two generators of `reading` read the same F16 cache: only the order of the sums of an attention may differ.
SAME_ABS = 0.00001
SAME_REL = 0.00001

## Prompts of 5 (padded to 8), exactly 8 and 9 (padded to 16).
def test_each_step_matches_the_forward_over_the_whole_row() -> Result[Void]
  with_peer do |dir|
    for prompt in [5, 8, 9]
      try stepping(dir, S.REF, prompt, ABS, REL)
      for backend in try S.ggml_backends()
        try stepping(dir, backend, prompt, HALF_ABS, HALF_REL)
      end
    end

    ok()
  end
end

## `layers_on_device: n` runs the blocks from the `n`th on on the CPU: every step gives the logits of the forward, with
## all the blocks on the device, with one, and with none. On a Vulkan device the cache of a CPU block is in host memory
## (`Ggml(Cpu)`) and the cache of a device block is not; on any other backend `n` changes nothing.
def test_blocks_on_the_cpu_give_the_logits_of_the_blocks_on_the_device() -> Result[Void]
  with_peer do |dir|
    row = try ints(try JSON.get(try expected(dir), "long"))
    for backend in try S.ggml_backends()
      for n in [0, 1, 2]
        try stepping(dir, backend, 5, HALF_ABS, HALF_REL, some(n))
      end

      home = Tensor.Backend.Ggml(Tensor.Device.Cpu)
      next if backend == home

      g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend, Tensor.Dtype.F32, some(1)))
      cache = (try Generate.step_logits(g, Generate.start(), row.take(5))).cache
      try Test.eq(Tensor.backend(cache.keys[0]), backend)
      try Test.eq(Tensor.backend(cache.keys[1]), home)
      try Test.eq(Tensor.backend(cache.values[1]), home)
      cache = (try Generate.step_logits(g, cache, [row[5]])).cache
      try Test.eq(Tensor.backend(cache.keys[0]), backend)
      try Test.eq(Tensor.backend(cache.keys[1]), home)
      Generate.finish(cache)
      Generate.close(g)
    end

    ok()
  end
end

## Every key and value of `cache` is on `home`.
def all_on(cache: Generate.Cache, home: Tensor.Backend) -> Result[Void]
  for t in cache.keys + cache.values
    try Test.eq(Tensor.backend(t), home)
  end

  ok()
end

## One generator with the cache where its blocks make it and one with `kv: Host`, over the same model and the same
## ids: the logits agree after the prompt and after each of the tokens 5 to 10, every key and value of the host one is
## on `home`, the tensors of the prompt stay open through the steps in place and are closed by the step that grows the
## capacity (token 8), whose new tensors are open. With `mixed`, the steps of the host generator go through the
## scheduler (`mixed_runs` grows); with `plain`, those of the device one do not (a model with all its blocks on the
## device).
def host_run(
  m: Qwen3.Qwen3,
  row: List[Int],
  home: Tensor.Backend,
  abs: Float,
  rel: Float,
  mixed: Bool,
  plain: Bool,
) -> Result[Void]
  device = try Generate.open(m)
  host = try Generate.open(m, kv: Placement.Kv.Host)
  a = try Generate.step_logits(device, Generate.start(), row.take(5))
  b = try Generate.step_logits(host, Generate.start(), row.take(5))
  try S.expect_close(b.logits, a.logits, abs, rel, "the prompt")
  try all_on(b.cache, home)
  prompt = [b.cache.keys[0], b.cache.values[1]]
  for t in 5...11
    runs = GgmlRaw.mixed_runs()
    a = try Generate.step_logits(device, a.cache, [row[t]])
    if plain
      try Test.eq(GgmlRaw.mixed_runs(), runs)
    end

    b = try Generate.step_logits(host, b.cache, [row[t]])
    if mixed
      try Test.assert(GgmlRaw.mixed_runs() > runs, "the host step #{t} did not go through the scheduler")
    end

    try S.expect_close(b.logits, a.logits, abs, rel, "token #{t}")
    try all_on(b.cache, home)
    for old in prompt
      try Test.eq(try closed?(old), t >= 8)
    end

    try Test.assert(!try closed?(b.cache.keys[0]), "step #{t} closed the cache it gave")
    try Test.assert(!try closed?(b.cache.values[1]), "step #{t} closed the cache it gave")
  end

  Generate.finish(a.cache)
  Generate.finish(b.cache)
  Generate.close(device)
  Generate.close(host)
  ok()
end

## `kv: Host` keeps the keys and values of every layer in host memory (`Ggml(Cpu)`), the blocks staying on the device.
## The prompt step makes the whole cache on the device first and copies it over; the steps after it find it on the
## host (the step that grows the capacity makes its tensors there already, where the old ones were read). Also with a
## block on the CPU already (`layers_on_device` of 1). Without a Vulkan device nothing runs, and the test says so.
def test_a_cache_in_host_memory_gives_the_logits_of_a_cache_on_the_device() -> Result[Void]
  with_peer do |dir|
    row = try ints(try JSON.get(try expected(dir), "long"))
    home = Tensor.Backend.Ggml(Tensor.Device.Cpu)
    unset: Optional[Int] = none
    checked = 0
    for backend in try S.ggml_backends()
      next if backend == home

      checked += 1
      for layers in [unset, some(1)]
        m = try Qwen3.open("#{dir}/tiny", backend, Tensor.Dtype.F32, layers)
        try host_run(m, row, home, HALF_ABS, HALF_REL, true, layers.none?)
        Qwen3.close_model(m)
      end
    end

    if checked == 0
      IO.warn("SKIP kv Host: no Vulkan device")
    end

    try Test.assert(checked > 0 or !Tensor.available(S.VULKAN), "a Vulkan device was there and no model was checked")

    ok()
  end
end

## On the CPU engine, `kv: Host` changes nothing: the same logits, every cache tensor on the CPU, the prompt's tensors
## open through the steps in place (what CI runs without a card).
def test_a_cache_in_host_memory_on_the_cpu_engine_changes_nothing() -> Result[Void]
  with_peer do |dir|
    row = try ints(try JSON.get(try expected(dir), "long"))
    home = Tensor.Backend.Ggml(Tensor.Device.Cpu)
    for backend in try S.ggml_backends()
      next if backend != home

      m = try Qwen3.open("#{dir}/tiny", backend)
      try host_run(m, row, home, SAME_ABS, SAME_REL, false, false)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## The order of the nodes of a step in place: both projections of a layer come before the writes of its keys and values,
## so that with the cache on the CPU the writes and the attention are one split of the scheduler and the layer is two
## (the card, the CPU), not four. `cache_write_products` counts the products between each write and the first node that
## reads it: 0. On the CPU engine, which runs the attention of the tiny model as ggml's flash attention (on Vulkan this
## model's attention is the longer one, with its query projected after the writes; `ggml_mixed_test` pins the splits of a
## layer there).
def test_a_step_in_place_projects_the_keys_and_the_values_before_writing_either() -> Result[Void]
  with_peer do |dir|
    row = try ints(try JSON.get(try expected(dir), "long"))
    for backend in try S.ggml_backends()
      next if backend != Tensor.Backend.Ggml(Tensor.Device.Cpu)

      m = try Qwen3.open("#{dir}/tiny", backend)
      g = try Generate.open(m)
      s = try Generate.step_logits(g, Generate.start(), row.take(5))
      s = try Generate.step_logits(g, s.cache, [row[5]])
      try Test.eq(GgmlRaw.cache_write_products(), 0)
      Generate.finish(s.cache)
      Generate.close(g)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## A cache in host memory across a reach (768 rows of a capacity of 1024) gives the logits of the cache on the device.
def test_a_cache_in_host_memory_crosses_a_reach_with_the_same_logits() -> Result[Void]
  with_peer do |dir|
    home = Tensor.Backend.Ggml(Tensor.Device.Cpu)
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/roomy", backend)
      host = try Generate.open(m, kv: Placement.Kv.Host)
      device = try Generate.open(m)
      abs = SAME_ABS
      rel = SAME_REL
      if backend != home
        abs = HALF_ABS
        rel = HALF_REL
      end

      try reading_with(host, device, 765, 5, abs, rel)
      Generate.close(host)
      Generate.close(device)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## On ggml a one-token step of the same capacity writes its row into the tensors it was given, which are the keys of the
## cache it returns (they stay open); the step that grows the capacity makes new ones and closes the old.
def test_a_step_writes_the_cache_in_place_until_the_capacity_grows() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend))
      row = try ints(try JSON.get(try expected(dir), "long"))
      cache = (try Generate.step_logits(g, Generate.start(), row.take(5))).cache
      prompt = cache.keys[0]
      try Test.eq(Tensor.dtype(prompt), Tensor.Dtype.F16)
      for t in [5, 6, 7]
        cache = (try Generate.step_logits(g, cache, [row[t]])).cache
        try Test.assert(!try closed?(prompt), "step #{t} closed the cache it was given")
        try Test.eq(cache.length, t + 1)
      end

      # The ninth token needs 16 rows: new tensors, and the old ones are closed.
      cache = (try Generate.step_logits(g, cache, [row[8]])).cache
      try Test.assert(try closed?(prompt), "the old capacity stayed open")
      grown = cache.keys[0]
      for t in [9, 10]
        cache = (try Generate.step_logits(g, cache, [row[t]])).cache
        try Test.assert(!try closed?(grown), "step #{t} closed the cache it was given")
      end

      Generate.finish(cache)
      try Test.assert(try closed?(grown), "finish left the cache open")
      Generate.close(g)
    end

    ok()
  end
end

def closed?(t: Tensor.Tensor) -> Result[Bool] = ok(Native.closed?(try Ggml.handle_of(t)))

## A generator of `m` whose steps read the whole capacity of the cache: its own plans, and the logits that a step
## over the rows in use must give.
def reading_all(m: Qwen3.Qwen3) -> Result[Generate.Generator] = ok((try Generate.open(m)).with(whole: true))

## How many compiled steps `g` keeps now.
def held(g: Generate.Generator) -> Result[Int]
  plans = try g.plans.recv(1000)
  try g.plans.send(plans, 0)
  ok(plans.graphs.size)
end

## On ggml a one-token step reads the rows of the cache in use, rounded up to a reach, and each reach is a compiled
## step of its own. Three sequences each cross one: 768 rows inside a capacity of 1024, 1024 rows where the capacity
## grows to 2048 (reach 1280), and 1280 rows inside it. Every step gives the logits of a step that reads the whole
## capacity, the step of a reach ends the step of the reach before it, and a later sequence at a smaller reach
## takes the step that is there, so a capacity keeps one step.
def test_a_step_reads_the_rows_in_use_and_gives_the_logits_of_the_whole_cache() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/roomy", backend)
      g = try Generate.open(m)
      whole = try reading_all(m)
      # the prompt, a step of each reach: 768 and 1024
      try reading(g, whole, 765, 5)
      try Test.eq(Generate.compiles(g), 3)
      try Test.eq(Generate.compiles(whole), 2)
      try Test.eq(try held(g), 2)
      # nothing new up to 1024 rows; then the step that grows the capacity and the step of reach 1280
      try reading(g, whole, 1022, 4)
      try Test.eq(Generate.compiles(g), 5)
      try Test.eq(try held(g), 4)
      # the prompt of 2048, the reach of 1280 as it is, and the reach of 1536, which ends that one
      try reading(g, whole, 1278, 4)
      try Test.eq(Generate.compiles(g), 7)
      try Test.eq(try held(g), 5)
      # a second sequence at the reach that ended takes the step of the larger one
      try reading(g, whole, 1278, 1)
      try Test.eq(Generate.compiles(g), 7)
      try Test.eq(try held(g), 5)
      Generate.close(g)
      Generate.close(whole)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## One sequence that grows from 760 to 1290 rows, a token at a time: it crosses the reach of 768 rows, the growth of
## the capacity from 1024 to 2048 rows and the reach of 1280, with the logits of the whole cache at every step, and
## it leaves one step per capacity.
def test_a_sequence_that_crosses_three_reaches_gives_the_logits_of_the_whole_cache() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/roomy", backend)
      g = try Generate.open(m)
      whole = try reading_all(m)
      try reading(g, whole, 760, 530)
      # the prompt, the reaches of 768 and 1024, the growth, the reaches of 1280 and 1536
      try Test.eq(Generate.compiles(g), 6)
      try Test.eq(try held(g), 4)
      try Test.eq(Generate.compiles(whole), 4)
      Generate.close(g)
      Generate.close(whole)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## A prompt of `first` ids and `steps` more, one at a time, through `g` and through `whole`: the same logits at each.
def reading(g: Generate.Generator, whole: Generate.Generator, first: Int, steps: Int) -> Result[Void]
  reading_with(g, whole, first, steps, SAME_ABS, SAME_REL)
end

## `reading` within `abs` and `rel`.
def reading_with(
  g: Generate.Generator,
  whole: Generate.Generator,
  first: Int,
  steps: Int,
  abs: Float,
  rel: Float,
) -> Result[Void]
  vocab = g.model.config.vocab_size
  on = Tensor.backend(g.model.tokens.table)
  row = (0...(first + steps)).map { |i| (i * 7 + i / 5) % vocab }
  ours = try Generate.step_logits(g, Generate.start(), row.take(first))
  theirs = try Generate.step_logits(whole, Generate.start(), row.take(first))
  try S.expect_close(ours.logits, theirs.logits, abs, rel, "prompt of #{first} on #{on}")
  for t in first...row.size
    ours = try Generate.step_logits(g, ours.cache, [row[t]])
    theirs = try Generate.step_logits(whole, theirs.cache, [row[t]])
    try S.expect_close(ours.logits, theirs.logits, abs, rel, "row #{t} after a prompt of #{first} on #{on}")
  end

  Generate.finish(ours.cache)
  Generate.finish(theirs.cache)
  ok()
end

def test_generate_is_the_greedy_choice_of_the_forward_on_the_reference() -> Result[Void]
  with_peer do |dir| generating(dir, "tiny", S.REF) end
end

def test_generate_is_the_greedy_choice_of_the_forward_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try generating(dir, "tiny", backend)
    end

    ok()
  end
end

## The Qwen2 layout (no query and key norms, a bias on the query, key and value) through the cache.
def test_generate_is_the_greedy_choice_of_the_forward_for_the_qwen2_layout() -> Result[Void]
  with_peer do |dir|
    try generating(dir, "qwen2", S.REF)
    for backend in try S.ggml_backends()
      try generating(dir, "qwen2", backend)
    end

    ok()
  end
end

## A sampled run is the greedy one when only the largest id can be drawn, the same for the same seed, and not the same
## for every seed.
def test_sampling_is_greedy_when_one_id_is_left_and_seeded_otherwise_on_the_reference() -> Result[Void]
  with_peer do |dir| sampling(dir, S.REF) end
end

def test_sampling_is_greedy_when_one_id_is_left_and_seeded_otherwise_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try sampling(dir, backend)
    end

    ok()
  end
end

## A model whose step gives other tensors than its `Stepper` says (a keys-and-values layer, one tensor too few) is an
## error, on the reference and on ggml.
def test_a_step_with_other_outputs_than_the_stepper_says_is_an_error() -> Result[Void]
  for backend in [S.REF] + try S.ggml_backends()
    stepper = KvDecoder.Stepper(
      layers: 1,
      limit: 8,
      vocab: 4,
      backend: backend,
      traced: ->(m: Int) { m },
      stepped: ->(_m: Int, req: KvDecoder.Request) {
        logits = try Tensor.zeros([4], Tensor.Dtype.F32, Tensor.backend(req.xs[0]))
        ok([try Stepping.headed(logits, req.mode)])
      },
    )

    g = try KvDecoder.open(0, stepper)
    try Test.error?(KvDecoder.step(g, KvDecoder.start(), [1, 2]))
    KvDecoder.close(g)
  end

  ok()
end

def test_bad_steps_are_errors() -> Result[Void]
  with_peer do |dir|
    for backend in [S.REF] + try S.ggml_backends()
      g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend))
      try Test.error?(Generate.step(g, Generate.start(), []))
      try Test.error?(Generate.step(g, Generate.start(), [1, 13]))
      try Test.error?(Generate.step(g, Generate.start(), List.repeat(1, 33)))
      first = (try Generate.step(g, Generate.start(), [1, 2, 3])).cache
      try Test.error?(Generate.step(g, first, [1, 2]))
      # 3 + 31 - 1 positions are more than the 32 of the model.
      try Test.error?(Generate.generate(g, [1, 2, 3], 31))
      Generate.close(g)
    end

    ok()
  end
end

## Ties go to the first index, as `argmax` and Hugging Face's greedy pick do.
def test_the_pick_of_a_tie_is_the_first_index() -> Result[Void]
  for backend in [S.REF] + try S.ggml_backends()
    scores = try Tensor.from_list(
      [1.0, 3.0, 2.0, 3.0, 3.0, 0.0].map { |f| Tensor.Values.Float(f) },
      [6],
      Tensor.Dtype.F32,
      backend,
    )

    got = try Lift.staged(
      0,
      ->(m: Int) { m },
      [scores],
      ->(_: Int, xs: List[Tensor.Tensor]) { ok([try Stepping.picked(xs[0])]) },
    )

    try Test.eq(try S.floats_of(got[0]), [1.0])
  end

  ok()
end

## A row of 1024 logits or more finds its maximum in pieces on ggml (1500 = 23 pieces of 64 and a tail of 28); a tie
## still goes to the first index, across two pieces, at a seam or in the tail.
def test_the_pick_of_a_tie_in_a_long_row_is_the_first_index() -> Result[Void]
  n = 1500
  base = (0...n).map { |i| (i % 13).to_f * 0.25 }
  for backend in [S.REF] + try S.ggml_backends()
    for tie in [[63, 64], [5, 1499], [1471, 1490], [1472, 1499], [0, 63, 1472]]
      scores = try S.f32(tie.reduce(base) do |acc, i| acc.set(i, 9.0) end, [n], backend)
      got = try Lift.staged(
        0,
        ->(m: Int) { m },
        [scores],
        ->(_: Int, xs: List[Tensor.Tensor]) { ok([try Stepping.picked(xs[0])]) },
      )

      try Test.eq(try S.floats_of(got[0]), [tie[0].to_f])
    end
  end

  ok()
end

private

def sampling(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  prompt = [1, 5, 2]
  want = try Generate.generate(g, prompt, 8)
  for s in [Sample.Sampling(temperature: 0.0), Sample.Sampling(top_k: 1, seed: 9)]
    try Test.eq(try Generate.generate(g, prompt, 8, [], some(s)), want)
  end

  warm = Sample.Sampling(temperature: 1.5, seed: 21)
  first = try Generate.generate(g, prompt, 8, [], some(warm))
  try Test.eq(try Generate.generate(g, prompt, 8, [], some(warm)), first)
  differs = false
  for seed in 1...4
    if try Generate.generate(g, prompt, 8, [], some(warm.with(seed: seed))) != first
      differs = true
    end
  end

  try Test.assert(differs, "four seeds drew the same ids")
  try Test.error?(Generate.generate(g, prompt, 8, [], some(Sample.Sampling(top_p: 0.0))))
  try Test.error?(Generate.generate(g, prompt, 8, [], some(Sample.Sampling(temperature: -1.0))))
  try Test.error?(Generate.generate(g, prompt, 8, [], some(Sample.Sampling(top_k: -1, temperature: 0.0))))
  Generate.close(g)
  ok()
end

## Seven ids after a prompt of three: the cache goes from 8 to 16 rows. The ids are those of the forward without a cache.
## A second call finds the steps of the first: it compiles none.
def generating(dir: String, folder: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open("#{dir}/#{folder}", backend)
  ran = generating_on(m, backend)
  Qwen3.close_model(m)
  ran
end

def generating_on(m: Qwen3.Qwen3, backend: Tensor.Backend) -> Result[Void]
  prompt = [1, 5, 2]
  g = try Generate.open(m)
  want = try greedy(m, prompt, 7)
  try Test.eq(try Generate.generate(g, prompt, 7), want)
  made = Generate.compiles(g)
  case backend
    Tensor.Backend.Ggml(_) -> try Test.assert(made >= 3, "the steps compiled: #{made}")
    _ -> try Test.eq(made, 0)
  end

  # A stop id ends it before that id.
  try Test.eq(try Generate.generate(g, prompt, 7, [want[3]]), want.take(first_of(want, want[3])))
  try Test.eq(try Generate.generate(g, prompt, 0), [])
  try Test.eq(Generate.compiles(g), made)
  Generate.close(g)
  # With `usable` the ids from it on are never answered: the first id of the greedy choice is cut off.
  cut = [want[0], 1].max.unwrap!
  try Test.error?(Generate.open(m, m.config.vocab_size + 1))
  few = try Generate.open(m, cut)
  ran = cut_off(few, prompt, cut)
  Generate.close(few)
  ran
end

def cut_off(few: Generate.Generator, prompt: List[Int], cut: Int) -> Result[Void]
  made = try Generate.generate(few, prompt, 7)
  try Test.eq(made.size, 7)
  try Test.assert(made.all? { |id| id < cut }, "an id of #{cut} or more")
  hot = some(Sample.Sampling(temperature: 5.0, seed: 3))
  drawn = try Generate.generate(few, prompt, 7, [], hot)
  try Test.eq(drawn.size, 7)
  try Test.assert(drawn.all? { |id| id < cut }, "a drawn id of #{cut} or more")
  try Test.error?(Generate.step_top(few, Generate.start(), prompt, cut + 1))
  ok()
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def ints(v: JSON.Value) -> Result[List[Int]]
  out: List[Int] = []
  for x in try JSON.list(v)
    out = out.push(try JSON.int(x))
  end

  ok(out)
end

## The first index of `id` in `ids`.
def first_of(ids: List[Int], id: Int) -> Int
  for i in 0...ids.size
    return i if ids[i] == id
  end

  ids.size
end

## The row `long` fed as a prompt of `first` and then one id at a time (the cache goes from 8 to 16 to 32 rows), the
## logits of every step against the peer's forward over the whole row.
def stepping(
  dir: String,
  backend: Tensor.Backend,
  first: Int,
  abs: Float,
  rel: Float,
  layers: Optional[Int] = none,
) -> Result[Void]
  g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend, Tensor.Dtype.F32, layers))
  vocab = g.model.config.vocab_size
  e = try expected(dir)
  row = try ints(try JSON.get(e, "long"))
  want = try S.flat(try JSON.get(e, "tiny_long"))
  scored = try Generate.step_logits(g, Generate.start(), row.take(first))
  try S.expect_close(scored.logits, want.drop((first - 1) * vocab).take(vocab), abs, rel, "prompt of #{first}")
  cache = scored.cache
  for t in first...row.size
    scored = try Generate.step_logits(g, cache, [row[t]])
    cache = scored.cache
    try S.expect_close(scored.logits, want.drop(t * vocab).take(vocab), abs, rel, "position #{t}")
  end

  try Test.eq(cache.length, row.size)
  Generate.finish(cache)
  Generate.close(g)
  ok()
end

## `n` ids after `prompt`, each the largest logit of the forward over everything so far (no cache).
def greedy(m: Qwen3.Qwen3, prompt: List[Int], n: Int) -> Result[List[Int]]
  ids = prompt
  out: List[Int] = []
  for _ in 0...n
    last = try S.floats_of(try Qwen3.run(m, [ids], true))
    best = 0
    for i in 0...last.size
      if last[i] > last[best]
        best = i
      end
    end

    ids = ids.push(best)
    out = out.push(best)
  end

  ok(out)
end
