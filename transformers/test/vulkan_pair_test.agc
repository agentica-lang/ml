## Two Qwen3 decode rows in one paired step on a Vulkan device of ggml: the device the model was opened on, the rows'
## own positions, the ids each row gives when it decodes alone on that same device, and one round of the scheduler that
## pairs them. Mesa's llvmpipe — the software Vulkan device a CI runner without a card has — runs this case as a card
## does, because software or card is a property of the device and not a path of the code: the job names the device it
## got in its log. `TENSOR_REQUIRE_VULKAN` makes a job fail when there is no device instead of running the case nowhere.
## Run from `transformers/`.

import Generate from "../generate"
import Qwen3 from "../qwen3"
import S from "support"
import Serving from "../serving"
import Tensor from "pkg/tensor/tensor"

## The ids `ids` gives decoded alone for `n` tokens, on `backend`: the row's own greedy run on the same device the pair
## runs on.
def singleton(dir: String, backend: Tensor.Backend, ids: List[Int], n: Int) -> Result[List[Int]]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  out = try Generate.generate(g, ids, n)
  Generate.close(g)
  Qwen3.close_model(m)
  ok(out)
end

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

## Three paired rounds with a single step between two of them, over rows of 3 and 6 tokens: both rows' ids are the ones
## they give alone on `backend`, and every cache the step gives back is on `backend`.
def pairing(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  a = try Generate.step(g, Generate.start(), [1, 5, 2])
  b = try Generate.step(g, Generate.start(), [4, 4, 9, 2, 7, 1])
  try Test.eq(Tensor.backend(a.cache.keys[0]), backend)
  try Test.eq(Tensor.backend(b.cache.keys[0]), backend)
  first = try Generate.step_pair(g, a.cache, a.id, b.cache, b.id)
  second = try Generate.step_pair(g, first.a.cache, first.a.id, first.b.cache, first.b.id)
  # A row the pair gave back steps alone, as the scheduler does with a row whose peer is gone, and pairs again.
  solo = try Generate.step(g, second.a.cache, [second.a.id])
  third = try Generate.step_pair(g, solo.cache, solo.id, second.b.cache, second.b.id)
  try Test.eq(Tensor.backend(third.a.cache.keys[0]), backend)
  try Test.eq(Tensor.backend(third.b.cache.keys[0]), backend)
  got_a = [a.id, first.a.id, second.a.id, solo.id]
  got_b = [b.id, first.b.id, second.b.id]
  Generate.finish(third.a.cache)
  Generate.finish(third.b.cache)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, try singleton(dir, backend, [1, 5, 2], 4))
  Test.eq(got_b, try singleton(dir, backend, [4, 4, 9, 2, 7, 1], 3))
end

## Two requests the scheduler admits together on `backend`: both answers are the ids of their own run alone on that
## device, and the round paired them (`pair_runs` of the run is not zero).
def serving_pair(dir: String, backend: Tensor.Backend) -> Result[Void]
  want_a = try singleton(dir, backend, [1, 5, 2], 3)
  want_b = try singleton(dir, backend, [4, 4, 9, 2, 7, 1], 2)
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 3, [], a_ch)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 2, [], b_ch)
  got_a = try collect(a_ch)
  got_b = try collect(b_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  try Test.eq(got_b, want_b)
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
