## A tiny random Whisper that the Python peer `peer.py` writes with Hugging
## Face's names: the encoder output and the logits match the peer's forward
## in doubles, and `transcribe` runs end to end. Run from `asr/`.

import Asr from "../asr"
import Greedy from "../greedy"
import Mathx from "../mathx"
import Mel from "../mel"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "../tokenizer"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## The reference computes in doubles, as the peer does, with the same GELU.
ABS = 0.000000001
REL = 0.000000001

## libtorch's F32 against the peer's doubles.
F32 = 0.00001

private

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
## `transcribe` emits the reference's tokens.
def test_on_ggml_the_steps_and_transcribe_match() -> Result[Void]
  with_fixtures do |dir|
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...960).map { |i| (i % 37).to_f / 37.0 - 0.5 }
    want = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", REF), tok, pcm)
    for backend in try ggml_backends()
      try kv_steps(try Asr.open("#{dir}/tiny/model.safetensors", backend), dir, backend, F32, F32)
      got = try Asr.transcribe(try Asr.open("#{dir}/speech/model.safetensors", backend), tok, pcm)
      try Test.eq(got.tokens, want.tokens)
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
    pcm = (0...960).map { |i| (i % 37).to_f / 37.0 - 0.5 }
    t = try Asr.transcribe(m, tok, pcm)
    audio = try Asr.encode(m, try f32(try Mel.log_mel(pcm, 3, 6), [1, 3, 6]))
    first = try last_logits(m, [tok.sot], audio)
    scores = tok.languages.map { |l| first[l.id] }
    best = (0...scores.size).to_a.max_by { |i| scores[i] }.unwrap!
    try Test.eq(t.language, tok.languages[best].code)
    try close_to(t.language_probability, Mathx.exp(scores[best]) / sum_exp(scores), "language probability")
    try close_to(t.no_speech_prob, Mathx.exp(first[30]) / sum_exp(first), "no_speech_prob")

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
    pcm = (0...960).map { |i| (i % 37).to_f / 37.0 - 0.5 }
    given = try Asr.transcribe(m, tok, pcm, Asr.Options(language: some("es"), translate: true, timestamps: false))
    try Test.eq([given.language, given.language_probability.to_s], ["es", "1.0"])
    try Test.eq((try Asr.transcribe(m, tok, pcm, Asr.Options(max_tokens: some(2)))).tokens.size, 2)
    try Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(max_tokens: some(0))))
    try Test.error?(Asr.transcribe(m, tok, pcm, Asr.Options(language: some("fr"))))
    try Test.error?(Asr.transcribe(m, tok, pcm + [0.0]))
    try Test.error?(Asr.transcribe(m, tok, []))
    Test.error?(Asr.transcribe(try Asr.open("#{dir}/tiny/model.safetensors", REF), tok, pcm))
  end
end

## An English-only checkpoint has no language token: the language is `en`
## with probability 1.0, and another language or translation is refused.
def test_an_english_only_checkpoint_transcribes_english() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/english/model.safetensors", REF)
    tok = try Tokenizer.open("#{dir}/english/tokenizer.json")
    pcm = (0...960).map { |i| (i % 37).to_f / 37.0 - 0.5 }
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

def sum_exp(xs: List[Float]) -> Float = xs.reduce(0.0) do |acc, x| acc + Mathx.exp(x) end

def close_to(got: Float, want: Float, what: String) -> Result[Void]
  Test.assert((got - want).abs <= 0.000000001 * (1.0 + want.abs), "#{what}: #{got}, expected #{want}")
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
  if !Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok([])
  end

  return ok([GGML, VULKAN]) if Tensor.available(VULKAN)
  ok([GGML])
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
