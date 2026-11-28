## Two Qwen3 decode rows in one paired step on a Vulkan device of ggml: the device the model was opened on, the rows'
## own positions, the ids each row gives when it decodes alone on that same device, and one round of the scheduler that
## pairs them. A Vulkan device batches at most two rows: a direct `step_batch` of three is refused before any cache is
## read, and `Serving` clamps `max_batch` to two, so a third request takes the singleton remainder. Mesa's llvmpipe —
## the software Vulkan device a CI runner without a card has — runs this case as a card does, because software or card
## is a property of the device and not a path of the code: the job names the device it got in its log. The device the
## model was opened on names the width (`Generate.batch_width`). `TENSOR_REQUIRE_VULKAN` makes a job fail when there is
## no device instead of running the case nowhere.
##
## It is a smoke, and it is short on purpose: the software device interprets every shader of a compiled plan, and the
## runner gives a test 30 s, so a case that opens four models and compiles a dozen shapes runs out of it. Each case opens
## one model, takes its rows' own runs and the pair from the same generator (a prefill shape compiles once for both), and
## stops at the rounds that say what it must say. The longer walk — three rounds, the single step between two of them,
## and the room the pair pads when one row outgrows the other — is covered on the CPU engine, which compiles no shader
## (`generate_test.agc`, `test_a_paired_step_matches_the_rows_decoded_alone`).
## Run from `transformers/`.

import Generate from "../generate"
import Qwen3 from "../qwen3"
import S from "support"
import Serving from "../serving"
import Tensor from "pkg/tensor/tensor"

def collect(ch: Channel[Serving.Event]) -> Result[List[Int]]
  out: List[Int] = []
  going = true
  while going
    case ch.recv(60000)
      ok(ev) ->
        case ev
          Serving.Event.Token(id) -> out = out.push(id)
          Serving.Event.Done -> going = false
          _ -> nil
        end

      error(e) -> return error("no event in 60000ms: #{e.message}")
    end
  end

  ok(out)
end

## One paired round over rows of 3 and 6 tokens, on one model and one set of plans: each row's ids are the ones it gives
## alone on `backend`, and the cache the round gives back is on `backend`. A round is what this smoke can afford: every
## distinct shape is a plan, every plan compiles its shaders on the software device, and a runner's shader cache starts
## empty.
def pairing(dir: String, backend: Tensor.Backend) -> Result[Void]
  try Test.eq(Generate.batch_width(backend), 2)
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  # The rows' own runs first, on the generator the pair then uses: a prompt asks for a plan the singleton already made.
  want_a = try Generate.generate(g, [1, 5, 2], 2)
  want_b = try Generate.generate(g, [4, 4, 9, 2, 7, 1], 2)
  a = try Generate.step(g, Generate.start(), [1, 5, 2])
  b = try Generate.step(g, Generate.start(), [4, 4, 9, 2, 7, 1])
  try Test.eq(Tensor.backend(a.cache.keys[0]), backend)
  try Test.eq(Tensor.backend(b.cache.keys[0]), backend)
  # A width of three is refused before it reads any cache on Vulkan; the third cache still decodes alone to the ids of
  # its prompt's singleton run. The same prompt as `a` reuses a plan the first run already compiled.
  c = try Generate.step(g, Generate.start(), [1, 5, 2])
  rows = [
    Generate.BatchRow(cache: a.cache, id: a.id),
    Generate.BatchRow(cache: b.cache, id: b.id),
    Generate.BatchRow(cache: c.cache, id: c.id),
  ]

  try S.expect_error(Generate.step_batch(g, rows), "at most 2")
  fallback = try Generate.step(g, c.cache, [c.id])
  try Test.eq([c.id, fallback.id], want_a)
  Generate.finish(fallback.cache)
  first = try Generate.step_pair(g, a.cache, a.id, b.cache, b.id)
  try Test.eq(Tensor.backend(first.a.cache.keys[0]), backend)
  try Test.eq(Tensor.backend(first.b.cache.keys[0]), backend)
  # A test keeps its own output, so the cases pay nothing for this line; the warm-up run of the job reads it as the
  # device evidence: the cache the pair gave back, from this very path and not from a copy of its shapes.
  IO.print("pair cache on #{Tensor.backend(first.a.cache.keys[0])}\n")
  got_a = [a.id, first.a.id]
  got_b = [b.id, first.b.id]
  Generate.finish(first.a.cache)
  Generate.finish(first.b.cache)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  Test.eq(got_b, want_b)
end

## Three requests on `backend` with `max_batch` ten: the scheduler clamps the width to Vulkan's two, so the rounds are
## a batch of two and a singleton remainder, every answer is its own run alone on that device, and the shared batch
## still counts (`pair_runs` is not zero). No graph wider than two is compiled.
def serving_pair(dir: String, backend: Tensor.Backend) -> Result[Void]
  try Test.eq(Generate.batch_width(backend), 2)
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  want_a = try Generate.generate(g, [1, 5, 2], 2)
  want_b = try Generate.generate(g, [4, 4, 9, 2, 7, 1], 2)
  want_c = want_a
  server = try Serving.start(g, Serving.Limits(max_active: 3, max_pending: 4, max_batch: 10))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  c_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 2, [], a_ch)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 2, [], b_ch)
  try Serving.submit(server, [1, 5, 2], 2, [], c_ch)
  got_a = try collect(a_ch)
  got_b = try collect(b_ch)
  got_c = try collect(c_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  try Test.eq(got_b, want_b)
  try Test.eq(got_c, want_c)
  Test.assert(runs > 0, "the round on #{backend} paired no rows")
end

## The two rows of a pair run on one Vulkan device and give the ids each one gives alone there.
def test_a_paired_step_on_vulkan_gives_the_rows_singleton_ids() -> Result[Void]
  with_peer do |dir|
    for backend in try S.vulkan_backends()
      try pairing(dir, backend)
    end

    ok()
  end
end

## The warm-up of the software-Vulkan job: the fixture and the path of the first case, once, with no test around it and
## so no 30 s bound, so that the timed cases after it find Mesa's shader cache warm. It is the case's own `pairing` — the
## same prompts, the same plans, the same asserts — and not a copy of its shapes: a job step writes a six-line program
## that imports this file and calls this (`agentica run`, not `agentica test`), and reads its device line.
def warm_pair() -> Result[Void]
  with_peer do |dir|
    for backend in try S.vulkan_backends()
      try pairing(dir, backend)
    end

    ok()
  end
end

## A scheduler round on a Vulkan device pairs its two rows and reports it.
def test_a_serving_round_pairs_on_vulkan() -> Result[Void]
  with_peer do |dir|
    for backend in try S.vulkan_backends()
      try serving_pair(dir, backend)
    end

    ok()
  end
end

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end
