## A tiny random Whisper that the Python peer `peer.py` writes with Hugging
## Face's names: the encoder output and the logits match the peer's forward
## in doubles, and `transcribe` runs end to end. Run from `asr/`.

import Asr from "../asr"
import Greedy from "../greedy"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "../tokenizer"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

## The reference computes in doubles, as the peer does, with the same tanh
## GELU.
ABS = 0.000000001
REL = 0.000000001

## Whisper's exact erf GELU against the tanh form here: on this tiny random
## model the peer's two forwards differ by 1.7e-4 on the encoder output and
## 3.8e-5 on the logits; a real checkpoint may drift more.
DRIFT = 0.0003

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
    tanh = try JSON.get(want, "tanh")
    try expect_close(try floats_of(audio), try rows(try JSON.get(tanh, "encoder")), ABS, REL, "encoder")
    try expect_close(try floats_of(logits), try rows(try JSON.get(tanh, "logits")), ABS, REL, "logits")
    erf = try JSON.get(want, "erf")
    try expect_close(try floats_of(audio), try rows(try JSON.get(erf, "encoder")), DRIFT, 0.0, "erf encoder")
    expect_close(try floats_of(logits), try rows(try JSON.get(erf, "logits")), DRIFT, 0.0, "erf logits")
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
    first = try JSON.get(want, "tanh")
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
      tanh = try JSON.get(want, "tanh")
      try expect_close(try floats_of(audio), try rows(try JSON.get(tanh, "encoder")), F32, F32, "torch encoder")
      expect_close(try floats_of(logits), try rows(try JSON.get(tanh, "logits")), F32, F32, "torch logits")
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

## The random `speech/` model says nothing sensible, but whatever it says
## keeps OpenAI's rules: its config suppresses 1 and 2, the first token is a
## timestamp (32 is `<|0.00|>`), the timestamps never decrease, and 13 tokens
## fill its 16 positions after a prompt of 3.
def test_transcribe_keeps_the_rules_end_to_end() -> Result[Void]
  with_fixtures do |dir|
    m = try Asr.open("#{dir}/speech/model.safetensors", REF)
    tok = try Tokenizer.open("test/fixtures/tokenizer.json")
    pcm = (0...960).map { |i| (i % 37).to_f / 37.0 - 0.5 }
    t = try Asr.transcribe(m, tok, pcm)
    try Test.assert(["en", "es", "yue"].include?(t.language), "language #{t.language}")
    try Test.assert(t.language_probability > 0.0 and t.language_probability <= 1.0, "p #{t.language_probability}")
    try Test.assert(not t.tokens.empty? and t.tokens.size <= 13, "tokens #{t.tokens}")
    try Test.assert(t.tokens[0] >= 32 and t.tokens.all? { |id| id != 1 and id != 2 }, "tokens #{t.tokens}")
    stamps = t.tokens.select { |id| id >= 32 }
    try Test.assert((1...stamps.size).all? { |i| stamps[i] >= stamps[i - 1] }, "stamps #{stamps}")
    try Test.eq(t.text, try Tokenizer.decode(tok, t.tokens))
    try Test.eq(t.segments, try Greedy.segments(tok, t.tokens, 0.06))

    given = try Asr.transcribe(m, tok, pcm, Asr.Options(language: some("es"), translate: true, timestamps: false))
    try Test.eq([given.language, given.language_probability.to_s], ["es", "1.0"])
    try Test.error?(Asr.transcribe(m, tok, pcm + [0.0]))
    Test.error?(Asr.transcribe(try Asr.open("#{dir}/tiny/model.safetensors", REF), tok, pcm))
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
  logits = try rows(try JSON.get(try JSON.get(want, "tanh"), "logits"))
  prompt = try Asr.step(m, cache, tokens.take(2))
  try expect_close(prompt.logits, logits.drop(7).take(7), abs, rel, "prompt")
  third = try Asr.step(m, prompt.cache, [tokens[2]])
  try expect_close(third.logits, logits.drop(14).take(7), abs, rel, "third")
  fourth = try Asr.step(m, third.cache, [tokens[3]])
  expect_close(fourth.logits, logits.drop(21), abs, rel, "fourth")
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
