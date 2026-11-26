## A tiny random Whisper that the Python peer `peer.py` writes with Hugging
## Face's names: the encoder output and the logits match the peer's forward
## in doubles, and `transcribe` runs end to end. Run from `asr/`.

import Asr from "../asr"
import Greedy from "../greedy"
import Ggml from "pkg/tensor/src/ggml"
import Lower from "pkg/tensor/src/lower"
import Mel from "../mel"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "../tokenizer"
import Trace from "pkg/tensor/trace"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## The reference computes in doubles, as the peer does, with the same GELU.
ABS = 0.000000001
REL = 0.000000001

## libtorch's F32 against the peer's doubles.
F32 = 0.00001

## The compiled steps belong to the model: a second decoding over the same model finds them and traces and plans
## nothing, which the VM steps of the task show, and gives the same logits.
def test_the_compiled_steps_are_shared_by_every_decoding_of_a_model() -> Result[Void]
  with_fixtures do |dir|
    for backend in try ggml_backends()
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      before = System.Profile.steps()
      first = try decoded(m, audio)
      cold = System.Profile.steps() - before
      before = System.Profile.steps()
      second = try decoded(m, audio)
      warm = System.Profile.steps() - before
      try Test.eq(second, first)
      try Test.assert(warm * 2 < cold, "a second decoding took #{warm} steps, the first #{cold}")
    end

    ok()
  end
end

## `weights: F16` is an error off ggml, and on ggml a model whose matrices are all under `Load.HALF_MIN_ELEMENTS` loads
## as `F32` (the half path itself is `hub/test/load_test.agc` and the checkpoints of `tools/`).
def test_f16_weights_need_ggml_and_leave_a_tiny_model_alone() -> Result[Void]
  with_fixtures do |dir|
    try Test.error?(Asr.open("#{dir}/speech/model.safetensors", REF, Tensor.Dtype.F16))
    for backend in try ggml_backends()
      half = try Asr.open("#{dir}/speech/model.safetensors", backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(half.tokens.table), Tensor.Dtype.F32)
      try Test.eq(Tensor.dtype(half.logits.w), Tensor.Dtype.F32)
    end

    ok()
  end
end

## The compiled steps are named by the shapes of their inputs: audio of another length on the same model gets steps of
## its own, and the first audio still gets its own. One test per backend, each with its own 30 s of the runner.
def test_the_compiled_steps_follow_the_audio_length_on_ggml_cpu() -> Result[Void] = steps_follow_the_audio_length(0)

def test_the_compiled_steps_follow_the_audio_length_on_vulkan() -> Result[Void] = steps_follow_the_audio_length(1)

## The steps one task compiled serve the others: after a decoding in a task, the caller's is warm, and three tasks at
## once agree with each other and leave the map whole, so a fourth decoding is warm too.
def test_the_compiled_steps_are_shared_between_tasks() -> Result[Void]
  with_fixtures do |dir|
    for backend in try ggml_backends()
      opened = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(opened, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      cold = try steps_of(try Asr.open("#{dir}/speech/model.safetensors", backend), audio)
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      first = try Task.await(Task.spawn do decoded(m, audio) end)
      try Test.assert(try steps_of(m, audio) * 2 < cold, "the caller did not find what the task compiled")
      other = try Asr.open("#{dir}/speech/model.safetensors", backend)
      tasks = (0...3).to_a.map { |_| Task.spawn do decoded(other, audio) end }
      for t in tasks
        try Test.eq(try Task.await(t), first)
      end

      try Test.assert(try steps_of(other, audio) * 2 < cold, "the map did not survive three tasks")
    end

    ok()
  end
end

## A task cancelled while it holds the map loses it: the next caller puts a new one in, so that the model compiles
## again once and then is warm again.
def test_a_lost_map_is_put_back() -> Result[Void]
  with_fixtures do |dir|
    for backend in try ggml_backends()
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      cold = try steps_of(m, audio)
      try Test.assert(try steps_of(m, audio) * 2 < cold, "no warm decoding before the loss")
      _ = Channel.recv(m.plans, 0)
      try Test.assert(try steps_of(m, audio) * 2 > cold, "the lost map was not rebuilt from empty")
      try Test.assert(try steps_of(m, audio) * 2 < cold, "not warm after the map was put back")
    end

    ok()
  end
end

private

## A vocabulary of 16: text 0..4, `eot` 5, specials 6..9 (`<|notimestamps|>` is 9, 7 is suppressed), timestamps 10..15,
## the first one at most 12; 3 and `eot` do not come first.
RULES = Greedy.Rules(
  suppress: [7],
  begin_suppress: [3, 5],
  eot: 5,
  no_timestamps: 9,
  timestamp_begin: 10,
  timestamps: true,
  max_initial: 2,
)

## A closed model frees its weights and the log-mel plan with its constants: using it afterwards is an error, and a
## model opened again over the same file works.
def test_a_closed_model_is_an_error_to_use() -> Result[Void]
  with_fixtures do |dir|
    for backend in try ggml_backends()
      path = "#{dir}/speech/model.safetensors"
      m = try Asr.open(path, backend)
      mel = try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend)
      before = try floats_of(try Asr.encode(m, mel))
      pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
      try Test.eq(Tensor.shape(try Asr.features(m, pcm, 6)), [1, 3, 6])
      Asr.close_model(m)
      Asr.close_model(m)
      try Test.error?(Asr.encode(m, mel))
      try Test.error?(Asr.features(m, pcm, 6))
      again = try Asr.open(path, backend)
      try Test.eq(try floats_of(try Asr.encode(again, mel)), before)
      Asr.close_model(again)
    end

    ok()
  end
end

## `cache_over` reads the tensors `step_inputs` lists in the same order, across several blocks: a cache of three blocks,
## each tensor of its own value, comes back unchanged, and the order of the list is the blocks' cross key and value, then
## the self-attention keys and values.
def test_a_cache_over_its_own_inputs_is_the_same_cache() -> Result[Void]
  next = 0.0
  cross: List[Asr.Heard] = []
  own: List[Asr.Kv] = []
  for _ in 0...3
    cross = cross.push(Asr.Heard(key: try tagged(next), value: try tagged(next + 1.0)))
    next += 2.0
  end

  for _ in 0...3
    own = own.push(Asr.Kv(key: try tagged(next), value: try tagged(next + 1.0)))
    next += 2.0
  end

  c = Asr.Cache(cross: cross, own: own, length: 2, graphs: {})
  inputs = Asr.step_inputs(c)
  try Test.eq(inputs.size, 12)
  try Test.eq(try firsts(inputs), (0...12).map { |i| i.to_f })
  back = Asr.cache_over(c, inputs)
  try Test.eq(Asr.step_inputs(back).size, 12)
  for i in 0...12
    try Test.eq(try floats_of(Asr.step_inputs(back)[i]), try floats_of(inputs[i]))
  end

  ok()
end

## The first number of each tensor of `ts`.
def firsts(ts: List[Tensor.Tensor]) -> Result[List[Float]]
  out: List[Float] = []
  for t in ts
    out = out.push((try floats_of(t))[0])
  end

  ok(out)
end

## A one-element tensor holding `x`.
def tagged(x: Float) -> Result[Tensor.Tensor] = f32([x], [1])

def test_the_tiny_whisper_matches_the_peer() -> Result[Void]
  with_fixtures do |dir|
    want = try JSON.parse(try File.read_text("#{dir}/expected.json"))
    m = try Asr.open("#{dir}/tiny/model.safetensors", REF)
    try Test.eq([m.encoder.size, m.decoder.size, m.config.vocab_size], [1, 1, 7])
    mel = try rows(try JSON.get(want, "mel"))
    audio = try Asr.encode(m, try f32(mel, [1, 3, 6]))
    try Test.eq(Tensor.shape(audio), [1, 3, 4])
    logits = try Asr.decode(m, try ids(try ints(try JSON.get(want, "tokens")), [1, 4]), audio)
    try Test.eq(Tensor.shape(logits), [1, 4, 7])
    erf = try JSON.get(want, "erf")
    try expect_close(try floats_of(audio), try rows(try JSON.get(erf, "encoder")), ABS, REL, "encoder")
    try expect_close(try floats_of(logits), try rows(try JSON.get(erf, "logits")), ABS, REL, "logits")
    # `gelu_new` in the config is the tanh form.
    tanh = try Asr.open("#{dir}/tanh/model.safetensors", REF)
    approximate = try Asr.encode(tanh, try f32(mel, [1, 3, 6]))
    expect_close(
      try floats_of(approximate),
      try rows(try JSON.get(try JSON.get(want, "tanh"), "encoder")),
      ABS,
      REL,
      "tanh",
    )
  end
end

def test_a_batch_of_two_matches_the_peer_item_by_item() -> Result[Void]
  with_fixtures do |dir|
    want = try JSON.parse(try File.read_text("#{dir}/expected.json"))
    m = try Asr.open("#{dir}/tiny/model.safetensors", REF)
    mel = try rows(try JSON.get(want, "mel")) + try rows(try JSON.get(want, "second_mel"))
    audio = try Asr.encode(m, try f32(mel, [2, 3, 6]))
    tokens = try ints(try JSON.get(want, "tokens")) + try ints(try JSON.get(want, "second_tokens"))
    logits = try Asr.decode(m, try ids(tokens, [2, 4]), audio)
    try Test.eq([Tensor.shape(audio), Tensor.shape(logits)], [[2, 3, 4], [2, 4, 7]])
    first = try JSON.get(want, "erf")
    second = try JSON.get(want, "second")
    audio_want = try rows(try JSON.get(first, "encoder")) + try rows(try JSON.get(second, "encoder"))
    try expect_close(try floats_of(audio), audio_want, ABS, REL, "batch encoder")
    logits_want = try rows(try JSON.get(first, "logits")) + try rows(try JSON.get(second, "logits"))
    expect_close(try floats_of(logits), logits_want, ABS, REL, "batch logits")
  end
end

## 4 frames use 2 of the 3 encoder positions; 8 frames need 4, and 9 tokens
## pass the 8 decoder positions.
def test_positions_are_sliced_up_to_the_end_of_their_tables() -> Result[Void]
  with_fixtures do |dir|
    want = try JSON.parse(try File.read_text("#{dir}/expected.json"))
    m = try Asr.open("#{dir}/tiny/model.safetensors", REF)
    audio = try Asr.encode(m, try f32(try rows(try JSON.get(want, "short_mel")), [1, 3, 4]))
    try Test.eq(Tensor.shape(audio), [1, 2, 4])
    short = try rows(try JSON.get(try JSON.get(want, "short"), "encoder"))
    try expect_close(try floats_of(audio), short, ABS, REL, "short encoder")
    try Test.error?(Asr.encode(m, try f32((0...24).map { |i| i.to_f / 24.0 }, [1, 3, 8])))
    Test.error?(Asr.decode(m, try ids((0...9).map { |i| i % 7 }, [1, 9]), audio))
  end
end

def test_a_missing_tensor_is_named() -> Result[Void]
  with_fixtures do |dir|
    missing = try JSON.string(try JSON.get(try JSON.parse(try File.read_text("#{dir}/expected.json")), "missing"))
    case Asr.open("#{dir}/missing/model.safetensors", REF)
      ok(_) -> error("a checkpoint without #{missing} loaded")
      error(e) -> Test.assert(e.message.include?(missing), "the error does not name #{missing}: #{e.message}")
    end
  end
end

## libtorch computes in F32, the peer in doubles. Without the native module,
## loading is the error.
def test_on_libtorch_the_tiny_whisper_matches_the_peer() -> Result[Void]
  with_fixtures do |dir|
    loaded = Asr.open("#{dir}/tiny/model.safetensors", TORCH)
    return Test.error?(loaded) if not Tensor.available(TORCH)

    want = try JSON.parse(try File.read_text("#{dir}/expected.json"))
    matched: Result[Void] = Native.arena do
      m = try loaded
      audio = try Asr.encode(m, try f32(try rows(try JSON.get(want, "mel")), [1, 3, 6], TORCH))
      tokens = try ids(try ints(try JSON.get(want, "tokens")), [1, 4], TORCH)
      logits = try Asr.decode(m, tokens, audio)
      erf = try JSON.get(want, "erf")
      try expect_close(try floats_of(audio), try rows(try JSON.get(erf, "encoder")), F32, F32, "torch encoder")
      expect_close(try floats_of(logits), try rows(try JSON.get(erf, "logits")), F32, F32, "torch logits")
    end

    matched
  end
end

## On ggml a loaded model carries the compiled log-mel graph (`Model.mel`); the
## features it runs equal those of the same graph traced and run once, which
## is what a model without the plan does. Both have the window's shape.
def test_on_ggml_the_log_mel_plan_and_the_traced_graph_agree() -> Result[Void]
  with_fixtures do |dir|
    pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    for backend in try ggml_backends()
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      try Test.assert(!m.mel.none?, "a ggml model has no log-mel plan")
      frames = 2 * m.config.max_source_positions
      planned = try Asr.features(m, pcm, frames)
      traced = try Asr.features(m.with(mel: none), pcm, frames)
      try Test.eq(Tensor.shape(planned), [1, m.config.num_mel_bins, frames])
      try expect_close(try floats_of(planned), try floats_of(traced), F32, F32, "the log-mel plan")
    end

    ok()
  end
end

## A prompt of two ids, then one id twice (the second step concatenates onto
## a cache that already did), give the last three rows of the peer's logits
## for the same four tokens.
def test_the_kv_cache_steps_match_the_peer() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/tiny/model.safetensors", REF)
    try kv_steps(m, dir, REF, ABS, REL)
    cache = try Asr.listen(m, try Asr.encode(m, try f32(List.repeat(0.5, 18), [1, 3, 6])))
    first = try Asr.step(m, cache, [1])
    try Test.error?(Asr.step(m, first.cache, [1, 2]))
    Test.error?(Asr.step(m, cache, (0...9).map { |i| i % 7 }))
  end
end

def test_on_libtorch_the_kv_cache_steps_match_the_peer() -> Result[Void]
  with_fixtures do |dir|
    loaded = Asr.open("#{dir}/tiny/model.safetensors", TORCH)
    return Test.error?(loaded) if not Tensor.available(TORCH)

    matched: Result[Void] = Native.arena do kv_steps(try loaded, dir, TORCH, F32, F32) end
    matched
  end
end

## On ggml (its CPU backend and Vulkan, where built) each encoder stage and
## each step is one compiled graph: the cached steps match the peer, and
## `transcribe` emits the reference's tokens. The padded cache grows from 8
## to 16 rows on the way, and a second call compiles no graph.
def test_on_ggml_the_steps_and_transcribe_match() -> Result[Void]
  with_fixtures do |dir|
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    want = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", REF), tok, pcm)
    for backend in try ggml_backends()
      try kv_steps(try Asr.open("#{dir}/tiny/model.safetensors", backend), dir, backend, F32, F32)
      speech = try Asr.open("#{dir}/speech/model.safetensors", backend)
      got = try Asr.transcribe(speech, tok, pcm)
      try Test.eq(got.tokens, want.tokens)
      try Test.assert(got.tokens.size > 5, "the padded cache grows past 8 rows: #{got.tokens}")
      compiled = try Lower.compiles()
      again = try Asr.transcribe(speech, tok, pcm)
      try Test.eq(again.tokens, want.tokens)
      try Test.eq(try Lower.compiles(), compiled)
    end

    ok()
  end
end

## Without timestamp rules, greedy decoding on ggml runs the pick in each step's graph (`Asr.step_pick`): only an id
## and a log-probability leave the device. It emits the reference's tokens, in the padded cache's growth too, and its
## `avg_logprob` is the reference's within F32 rounding. `step_pick` also answers on the reference, which computes the
## same graph eagerly.
def test_on_ggml_the_pick_runs_in_the_graph() -> Result[Void]
  with_fixtures do |dir|
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    options = Asr.Options(timestamps: false)
    want = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", REF), tok, pcm, options)
    try Test.assert(want.tokens.size > 9, "the padded cache grows past 8 rows: #{want.tokens}")
    for backend in try ggml_backends()
      speech = try Asr.open("#{dir}/speech/model.safetensors", backend)
      got = try Asr.transcribe(speech, tok, pcm, options)
      try Test.eq(got.tokens, want.tokens)
      try Test.assert(
        (got.avg_logprob - want.avg_logprob).abs < 0.00001,
        "avg_logprob #{got.avg_logprob} against #{want.avg_logprob}",
      )

      compiled = try Lower.compiles()
      again = try Asr.transcribe(speech, tok, pcm, options)
      try Test.eq(again.tokens, want.tokens)
      try Test.eq(try Lower.compiles(), compiled)
    end

    ok()
  end
end

## `step_pick` gives, on every backend, the id and the log-probability that `Greedy` computes from the logits of
## `step`, for a mask that forbids ids and for the first token's.
def test_step_pick_matches_greedy_over_the_logits() -> Result[Void]
  with_fixtures do |dir|
    rules = Greedy.Rules(
      suppress: [1, 3],
      begin_suppress: [2],
      eot: 0,
      no_timestamps: 4,
      timestamp_begin: 32,
      timestamps: false,
      max_initial: 50,
    )

    backends = [REF] + try ggml_backends()
    for backend in backends
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      cache = try Asr.listen(m, audio)
      logits = try Asr.step(m, cache, [5, 6, 7])
      first = try masked(Greedy.mask(rules, m.config.vocab_size, true), backend)
      rest = try masked(Greedy.mask(rules, m.config.vocab_size, false), backend)
      got = try Asr.step_pick(m, cache, [5, 6, 7], first)
      try agrees(got.id, got.logprob, Greedy.allowed(rules, logits.logits, []), "prompt on #{backend}")
      emitted = [got.id]
      plain = logits
      picked = got
      spent: List[Tensor.Tensor] = []
      for _ in 0...10
        plain = try Asr.step(m, plain.cache, [emitted.last])
        before = picked
        picked = try Asr.step_pick(m, before.cache, [emitted.last], rest, spent)
        spent = before.cache.own.flat_map { |kv| [kv.key, kv.value] }
        try agrees(
          picked.id,
          picked.logprob,
          Greedy.allowed(rules, plain.logits, emitted),
          "step #{emitted.size} on #{backend}",
        )

        emitted = emitted.push(picked.id)
      end
    end

    ok()
  end
end

## A step writes its cache into `spare` when the capacity is the same (the tensors stay open and the logits are the same as
## without it), and closes `spare` when the capacity grows.
def test_a_spare_cache_is_reused_at_the_same_capacity_and_closed_at_a_new_one() -> Result[Void]
  with_fixtures do |dir|
    for backend in try ggml_backends()
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      cache = try Asr.listen(m, audio)
      older = (try Asr.step(m, cache, [5, 6, 4])).cache
      newer = (try Asr.step(m, older, [1])).cache
      spare = older.own.flat_map { |kv| [kv.key, kv.value] }
      plain = try Asr.step(m, newer, [2])
      reused = try Asr.step(m, newer, [2], spare)
      try Test.eq(reused.logits, plain.logits)
      try Test.assert(!Native.closed?(try Ggml.handle_of(spare[0])), "a spare of the same capacity was closed")
      try Test.eq(Native.closed?(try Ggml.handle_of(reused.cache.own[0].key)), false)

      grown = reused.cache
      while grown.length < 8
        grown = (try Asr.step(m, grown, [3])).cache
      end

      old = newer.own.flat_map { |kv| [kv.key, kv.value] }
      _ = try Asr.step(m, grown, [3], old)
      try Test.assert(Native.closed?(try Ggml.handle_of(old[0])), "a spare of another capacity stayed open")
    end

    ok()
  end
end

## The timestamp rules in the graph, branch by branch, over logits made by hand (`Asr.scored`, then `Asr.picked`): each
## case gives the id and the log-probability that `Greedy.allowed` gives, and where `differs` the largest logit is another
## id, so the branch decides the pick.
def test_the_graph_applies_each_timestamp_rule() -> Result[Void]
  base = List.repeat(-5.0, 16)
  opening: List[Int] = []
  for backend in [REF] + try ggml_backends()
    # the first token: no text, no timestamp after the first second
    try ruled(backend, opening, base.set(2, 9.0).set(11, 1.0), true, "first, no text")
    try ruled(backend, opening, base.set(14, 9.0).set(11, 1.0), true, "first, the initial maximum")
    # a lone timestamp: no timestamp may follow
    try ruled(backend, [10], base.set(12, 9.0).set(2, 1.0), true, "a lone timestamp")
    # a timestamp after text opens a pair: no text before `eot`, `eot` or a timestamp from the same time on
    try ruled(backend, [1, 10], base.set(2, 9.0).set(5, 4.0), true, "a pair opened, no text")
    try ruled(backend, [2, 12], base.set(11, 9.0).set(12, 5.0).set(5, 4.0), true, "a pair closes on the same time")
    # a pair closed: the next one starts later
    try ruled(backend, [1, 12, 2], base.set(11, 9.0).set(13, 2.0).set(2, 1.0), true, "a new pair starts later")
    # the timestamps together against the best text token
    together = (
      base.set(2, 3.0)
        .set(9, 9.0)
        .set(10, 2.5)
        .set(11, 2.5)
        .set(12, 2.5)
        .set(13, 2.5)
        .set(14, 2.5)
        .set(15, 2.5)
    )

    try ruled(backend, [2], together, true, "timestamps likelier together than the best text")
    try ruled(backend, [2], together.set(2, 5.0).set(9, -5.0), false, "text likelier than the timestamps together")
  end

  ok()
end

## The model's step with the same rules on its own logits: `step_pick` with `Asr.Stamps` after ten histories gives the
## id and log-probability of `Greedy.allowed` over the logits of `step`, on the reference and on ggml.
def test_step_pick_with_stamps_agrees_with_the_host_rules_over_the_logits() -> Result[Void]
  with_fixtures do |dir|
    rules = Greedy.Rules(
      suppress: [0],
      begin_suppress: [3, 1],
      eot: 3,
      no_timestamps: 4,
      timestamp_begin: 5,
      timestamps: true,
      max_initial: 2,
    )

    nothing: List[Int] = []
    histories = [nothing, [5], [6, 7], [1, 5], [5, 1], [1, 5, 6], [5, 6, 2], [1, 2], [6, 6, 2, 7], [9]]
    for backend in [REF] + try ggml_backends()
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      cache = try Asr.listen(m, audio)
      start = try Asr.step(m, cache, [5, 6, 4])
      for i in 0...histories.size
        history = histories[i]
        feed = [(i + 4) % 7]
        plain = try Asr.step(m, start.cache, feed)
        first = try masked(Greedy.mask(rules, m.config.vocab_size, history.empty?), backend)
        state = try masked(Greedy.state(rules, history), backend)
        got = try Asr.step_pick(m, start.cache, feed, first, [], some(Asr.Stamps(rules: rules, state: state)))
        try agrees(
          got.id,
          got.logprob,
          Greedy.allowed(rules, plain.logits, history),
          "history #{history} on #{backend}",
        )
      end
    end

    ok()
  end
end

## that `Greedy.allowed` gives over the logits of `step`.
def test_step_pick_applies_the_timestamp_rules_in_the_graph() -> Result[Void]
  with_fixtures do |dir|
    rules = Greedy.Rules(
      suppress: [0],
      begin_suppress: [1],
      eot: 1,
      no_timestamps: 2,
      timestamp_begin: 3,
      timestamps: true,
      max_initial: 2,
    )

    nothing: List[Int] = []
    histories = [nothing, [3], [4, 5], [0, 3], [3, 0], [0, 3, 4], [3, 4, 0], [0, 0], [4, 4, 0, 5], [6]]
    for backend in [REF] + try ggml_backends()
      m = try Asr.open("#{dir}/speech/model.safetensors", backend)
      audio = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
      cache = try Asr.listen(m, audio)
      start = try Asr.step(m, cache, [5, 6, 4])
      for i in 0...histories.size
        history = histories[i]
        feed = [(i + 4) % 7]
        plain = try Asr.step(m, start.cache, feed)
        first = try masked(Greedy.mask(rules, m.config.vocab_size, history.empty?), backend)
        state = try masked(Greedy.state(rules, history), backend)
        got = try Asr.step_pick(m, start.cache, feed, first, [], some(Asr.Stamps(rules: rules, state: state)))
        try agrees(
          got.id,
          got.logprob,
          Greedy.allowed(rules, plain.logits, history),
          "history #{history} on #{backend}",
        )
      end
    end

    ok()
  end
end

## The pick of the graph over `logits` after `history`, against `Greedy.allowed`; when `differs`, also another id than the
## largest logit.
def ruled(
  backend: Tensor.Backend,
  history: List[Int],
  logits: List[Float],
  differs: Bool,
  what: String,
) -> Result[Void]
  size = logits.size
  mask = try masked(Greedy.mask(RULES, size, history.empty?), backend)
  state = try masked(Greedy.state(RULES, history), backend)
  out = try Trace.eval_many(
    [try f32(logits, [1, size], backend), mask, state],
    ->(xs: List[Tensor.Tensor]) {
      ok([try Asr.picked(try Asr.scored(xs[0], xs[1], some(Asr.Stamps(rules: RULES, state: xs[2]))))])
    },
  )

  pair = try floats_of(out[0])
  try agrees(try pair[0].to_i, pair[1], Greedy.allowed(RULES, logits, history), "#{what} on #{backend}")
  if differs
    try Test.assert(try pair[0].to_i != Greedy.pick(logits), "#{what}: the rules did not change the pick")
  end

  ok()
end

## The VM steps of one decoding in this task.
def steps_of(m: Asr.Model, audio: Tensor.Tensor) -> Result[Int]
  before = System.Profile.steps()
  _ = try decoded(m, audio)
  ok(System.Profile.steps() - before)
end

## The logits of ten steps after a prompt.
def decoded(m: Asr.Model, audio: Tensor.Tensor) -> Result[List[Float]]
  cache = try Asr.listen(m, audio)
  stepped = try Asr.step(m, cache, [5, 6, 4])
  logits: List[Float] = []
  for i in 0...10
    stepped = try Asr.step(m, stepped.cache, [i % 7])
    logits = logits + stepped.logits
  end

  ok(logits)
end

def masked(values: List[Float], backend: Tensor.Backend) -> Result[Tensor.Tensor] = f32(values, [values.size], backend)

## `id` and `logprob` are the first largest of `allowed` and its log-probability.
def agrees(id: Int, logprob: Float, allowed: List[Float], what: String) -> Result[Void]
  want = Greedy.pick(allowed)
  try Test.assert(id == want, "#{what}: picked #{id}, the logits say #{want}")
  Test.assert((logprob - Greedy.log_probability(allowed, want)).abs < 0.00001, "#{what}: logprob #{logprob}")
end

## On ggml the padded cache grows on purpose: 10 single ids take it from 8 to
## 16 rows, and a 9-id prompt on a model of 12 positions takes it from none to
## 12 rows (16, clamped) and then to its last position. Every step gives the
## reference's logits.
def test_on_ggml_the_padded_cache_grows_and_clamps() -> Result[Void]
  with_fixtures do |dir|
    for backend in try ggml_backends()
      try steps_match(dir, "speech", (0...10).map { |i| (i * 7) % 36 }, 1, backend)
      try steps_match(dir, "twelve", (0...12).map { |i| (i * 5) % 36 }, 9, backend)
    end

    ok()
  end
end

## The random `speech/` model says nothing sensible, but whatever it says
## keeps OpenAI's rules: its config suppresses 1 and 2, the first token is a
## timestamp (32 is `<|0.00|>`), the timestamps never decrease, and at most 13
## tokens fill its 16 positions after a prompt of 3. The language, its
## probability, `no_speech_prob`, the tokens and `avg_logprob` match a greedy
## loop over the full `Asr.decode`, without the KV cache.
def test_transcribe_matches_a_loop_over_the_full_decoder() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/speech/model.safetensors", REF)
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    t = try Asr.transcribe(m, tok, pcm)
    audio = try Asr.encode(m, try f32((try Mel.log_mel(pcm, 3, 6)).map { |x| x.to_f }, [1, 3, 6]))
    first = try last_logits(m, [tok.sot], audio)
    scores = tok.languages.map { |l| first[l.id] }
    best = (0...scores.size).to_a.max_by { |i| scores[i] }.unwrap!
    try Test.eq(t.language, tok.languages[best].code)
    try close_to(t.language_probability, Math.exp(scores[best]) / sum_exp(scores), "language probability")
    try close_to(t.no_speech_prob, Math.exp(first[30]) / sum_exp(first), "no_speech_prob")

    prompt = try Tokenizer.prompt(tok, some(t.language), Tokenizer.Job.Transcribe, true)
    rules = Greedy.rules(tok, some([1, 2]), some([4, 21]), true)
    emitted: List[Int] = []
    logprob = 0.0
    while emitted.size < 16 - prompt.size
      allowed = Greedy.allowed(rules, try last_logits(m, prompt + emitted, audio), emitted)
      id = Greedy.pick(allowed)
      logprob += Greedy.log_probability(allowed, id)
      break if id == tok.eot
      emitted = emitted.push(id)
    end

    try Test.eq(t.tokens, emitted)
    try close_to(t.avg_logprob, logprob / (emitted.size + 1).to_f, "avg_logprob")
    try Test.assert(
      not t.tokens.empty? and t.tokens[0] >= 32 and t.tokens.all? { |id| id != 1 and id != 2 },
      "#{t.tokens}",
    )

    stamps = t.tokens.select { |id| id >= 32 }
    try Test.assert((1...stamps.size).all? { |i| stamps[i] >= stamps[i - 1] }, "stamps #{stamps}")
    try Test.eq(t.text, try Tokenizer.decode(tok, t.tokens))
    Test.eq(t.segments, try Greedy.segments(tok, t.tokens, 0.06, t.avg_logprob))
  end
end

def test_transcribe_takes_options_and_refuses_bad_input() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/speech/model.safetensors", REF)
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    given = try Asr.transcribe(m, tok, pcm, Asr.Options(language: some("es"), translate: true, timestamps: false))
    try Test.eq([given.language, given.language_probability.to_s], ["es", "1.0"])
    try Test.eq((try Asr.transcribe(m, tok, pcm, Asr.Options(max_tokens: some(2)))).tokens.size, 2)
    try Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(max_tokens: some(0))))
    try Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(language: some("fr"))))
    try Test.error?(Asr.transcribe(m, tok, pcm + pcm, Asr.Options(timestamps: false)))
    try Test.error?(Asr.transcribe(m, tok, []))
    Test.error?(Asr.transcribe(try Asr.open("#{dir}/tiny/model.safetensors", REF), tok, pcm))
  end
end

## Audio over one window (here 6 frames, 960 samples) is read a window at a time, each one starting where the last
## timestamp of the one before said, on every backend alike, with the times of the segments counted from the start of
## the audio. The language is the first window's.
def test_long_audio_is_read_in_windows_that_seek() -> Result[Void]
  with_fixtures do |dir|
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...3000).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    want = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", REF), tok, pcm)
    try Test.eq(want.text, try Tokenizer.decode(tok, want.tokens))
    try Test.assert(not want.segments.empty?, "no segments")
    try Test.assert(want.segments[want.segments.size - 1].stop > 0.06, "no segment after the first window")
    first = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", REF), tok, pcm.take(960))
    try Test.eq(want.language, first.language)
    backends = try ggml_backends()
    if Tensor.available(TORCH)
      backends = backends.push(TORCH)
    end

    for backend in backends
      got = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", backend), tok, pcm)
      try Test.eq(got.tokens, want.tokens)
      try Test.eq(got.segments.map { |s| [s.start, s.stop] }, want.segments.map { |s| [s.start, s.stop] })
      try Test.eq(got.language, want.language)
    end

    ok()
  end
end

## The window is 2 * `max_source_positions` frames (30 s of 3000 for Whisper, 6 frames and 960 samples here), as Hugging
## Face's `truncation=False` features count them: `samples / 160` frames, so up to 159 samples past the window are still
## one window, which may go without timestamps, and the next frame is audio that seeks, which needs them.
def test_the_window_boundary_is_in_frames() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/speech/model.safetensors", REF)
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    off = Asr.Options(timestamps: false)
    for n in [960, 1119]
      _ = try Asr.transcribe(m, tok, (0...n).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }, off)
    end

    long = (0...1120).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    try Test.error?(Asr.transcribe(m, tok, long, off))
    Test.eq((try Asr.transcribe(m, tok, long)).truncated, false)
  end
end

## At most `max_windows` windows are decoded (by default `2 * ceil(frames / window) + 2`, a budget that stops a model
## that never advances), and the transcript says it stopped short; what was decoded is the start of the whole one.
def test_a_budget_of_windows_stops_a_transcription_short() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/speech/model.safetensors", REF)
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...3000).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    whole = try Asr.transcribe(m, tok, pcm, Asr.Options(max_windows: some(100)))
    try Test.eq(whole.truncated, false)
    one = try Asr.transcribe(m, tok, pcm, Asr.Options(max_windows: some(1)))
    try Test.eq(one.truncated, true)
    try Test.assert(whole.tokens.size > one.tokens.size, "the first window is not all of it")
    try Test.eq(whole.tokens.take(one.tokens.size), one.tokens)
    try Test.eq(whole.segments.take(one.segments.size), one.segments)
    Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(max_windows: some(0))))
  end
end

## An English-only checkpoint has no language token: the language is `en`
## with probability 1.0, and another language or translation is refused.
def test_an_english_only_checkpoint_transcribes_english() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/english/model.safetensors", REF)
    tok = try Tokenizer.open("#{dir}/english/tokenizer.json")
    pcm = (0...960).map { |i| ((i % 37).to_f / 37.0 - 0.5).to_float32 }
    t = try Asr.transcribe(m, tok, pcm, Asr.Options(max_tokens: some(3)))
    try Test.eq([t.language, t.language_probability.to_s, t.tokens.size.to_s], ["en", "1.0", "3"])
    try Test.assert(t.no_speech_prob > 0.0 and t.no_speech_prob < 1.0, "no_speech_prob #{t.no_speech_prob}")
    try Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(language: some("es"))))
    Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(translate: true)))
  end
end

## The logits of the last of `tokens` from the full decoder, over `audio`.
def last_logits(m: Asr.Model, tokens: List[Int], audio: Tensor.Tensor) -> Result[List[Float]]
  all = try floats_of(try Asr.decode(m, try ids(tokens, [1, tokens.size]), audio))
  ok(all.drop(all.size - m.config.vocab_size))
end

def sum_exp(xs: List[Float]) -> Float = xs.reduce(0.0) do |acc, x| acc + Math.exp(x) end

def close_to(got: Float, want: Float, what: String) -> Result[Void]
  Test.assert((got - want).abs <= 0.000000001 * (1.0 + want.abs), "#{what}: #{got}, expected #{want}")
end

## The `at`-th ggml backend of this machine, if it has one: the audio length makes steps of their own on it.
def steps_follow_the_audio_length(at: Int) -> Result[Void]
  backends = try ggml_backends()
  return ok() if backends.size <= at
  backend = backends[at]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/speech/model.safetensors", backend)
    longer = try Asr.encode(m, try f32((0...18).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 6], backend))
    shorter = try Asr.encode(m, try f32((0...12).map { |i| (i % 5).to_f / 5.0 - 0.4 }, [1, 3, 4], backend))
    a = try decoded(m, longer)
    b = try decoded(m, shorter)
    try Test.eq(try decoded(m, shorter), b)
    try Test.eq(try decoded(m, longer), a)
    fresh = try Asr.open("#{dir}/speech/model.safetensors", backend)
    Test.eq(try decoded(fresh, shorter), b)
  end
end

## Runs `body` on the folder the peer wrote, then removes it.
def with_fixtures(body: Fn[String, Result[Void]]) -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", "test/peer.py"])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("peer.py exited with #{code}: #{out.stderr}") if code != 0
  dir = out.stdout.strip.split("\n").last
  if not dir.start_with?("/") or not dir.split("/").last.start_with?("agentica-asr-")
    return error("peer.py printed no fixture folder: #{dir}")
  end

  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

## `Asr.step` over the peer's first mel and tokens on `backend`: a prompt of
## two ids, then the third and the fourth, against rows 1, 2 and 3 of the
## peer's logits.
def kv_steps(m: Asr.Model, dir: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  want = try JSON.parse(try File.read_text("#{dir}/expected.json"))
  cache = try Asr.listen(m, try Asr.encode(m, try f32(try rows(try JSON.get(want, "mel")), [1, 3, 6], backend)))
  tokens = try ints(try JSON.get(want, "tokens"))
  logits = try rows(try JSON.get(try JSON.get(want, "erf"), "logits"))
  prompt = try Asr.step(m, cache, tokens.take(2))
  try expect_close(prompt.logits, logits.drop(7).take(7), abs, rel, "prompt")
  third = try Asr.step(m, prompt.cache, [tokens[2]])
  try expect_close(third.logits, logits.drop(14).take(7), abs, rel, "third")
  fourth = try Asr.step(m, third.cache, [tokens[3]])
  expect_close(fourth.logits, logits.drop(21), abs, rel, "fourth")
end

## The ggml backends this machine has: its CPU backend when the module is
## built, and Vulkan device 0 when ggml sees one. With `TENSOR_REQUIRE_GGML`
## set to anything but "" (CI sets it) a missing module is an error.
def ggml_backends() -> Result[List[Tensor.Backend]]
  try Tensor.vulkan_fp16(false)
  if !Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok([])
  end

  return ok([GGML, VULKAN]) if Tensor.available(VULKAN)
  ok([GGML])
end

## The logits of `ids` fed to the `model` fixture on `backend`, first `prompt`
## of them at once and then one at a time, against the reference's.
def steps_match(dir: String, model: String, ids: List[Int], prompt: Int, backend: Tensor.Backend) -> Result[Void]
  mel = try rows(try JSON.get(try JSON.parse(try File.read_text("#{dir}/expected.json")), "mel"))
  ref = try Asr.open("#{dir}/#{model}/model.safetensors", REF)
  got = try Asr.open("#{dir}/#{model}/model.safetensors", backend)
  want = try Asr.listen(ref, try Asr.encode(ref, try f32(mel, [1, 3, 6])))
  have = try Asr.listen(got, try Asr.encode(got, try f32(mel, [1, 3, 6], backend)))
  at = 0
  while at < ids.size
    n = 1
    if at == 0
      n = prompt
    end

    a = try Asr.step(ref, want, ids.drop(at).take(n))
    b = try Asr.step(got, have, ids.drop(at).take(n))
    try expect_close(b.logits, a.logits, F32, F32, "#{model}, the step at #{at}")
    want = a.cache
    have = b.cache
    at += n
  end

  ok()
end

def ints(json: JSON.Value) -> Result[List[Int]]
  out: List[Int] = []
  for v in try JSON.list(json)
    out = out.push(try JSON.int(v))
  end

  ok(out)
end

def ids(tokens: List[Int], shape: List[Int], backend: Tensor.Backend = REF) -> Result[Tensor.Tensor]
  Tensor.from_list(tokens.map { |t: Int| Tensor.Values.Int(t) }, shape, Tensor.Dtype.I64, backend)
end

def f32(data: List[Float], shape: List[Int], backend: Tensor.Backend = REF) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x: Float| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

## JSON rows of numbers, flattened row after row.
def rows(json: JSON.Value) -> Result[List[Float]]
  out: List[Float] = []
  for row in try JSON.list(json)
    for v in try JSON.list(row)
      out = out.push(try JSON.float(v))
    end
  end

  ok(out)
end

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

## `got` matches `want` within `abs + rel * |want|`, element by element.
def expect_close(got: List[Float], want: List[Float], abs: Float, rel: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > abs + rel * want[i].abs
  end

  ok()
end
