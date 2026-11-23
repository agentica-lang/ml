## A tiny random Whisper that the Python peer `peer.py` writes with Hugging
## Face's names: the encoder output and the logits match the peer's forward
## in doubles. Run from `asr/`.

import Asr from "../asr"
import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

## The reference computes in doubles, as the peer does, with the same tanh
## GELU.
ABS = 0.000000001
REL = 0.000000001

## Whisper's exact erf GELU against the tanh form here: the peer's two
## forwards differ by 1.7e-4 on the encoder output and 3.8e-5 on the logits.
DRIFT = 0.0003

private

def test_the_tiny_whisper_matches_the_peer() -> Result[Void]
  with_fixtures do |dir|
    want = try JSON.parse(try File.read_text("#{dir}/expected.json"))
    m = try Asr.open("#{dir}/tiny/model.safetensors", REF)
    try Test.eq([m.encoder.size, m.decoder.size, m.config.vocab_size], [1, 1, 7])
    mel = try rows(try JSON.get(want, "mel"))
    audio = try Asr.encode(m, try f32(mel, [1, 3, 6]))
    try Test.eq(Tensor.shape(audio), [1, 3, 4])
    ids: List[Tensor.Values] = []
    for t in try JSON.list(try JSON.get(want, "tokens"))
      ids = ids.push(Tensor.Values.Int(try JSON.int(t)))
    end

    logits = try Asr.decode(m, try Tensor.from_list(ids, [1, ids.size], Tensor.Dtype.I64, REF), audio)
    try Test.eq(Tensor.shape(logits), [1, 4, 7])
    tanh = try JSON.get(want, "tanh")
    try expect_close(try floats_of(audio), try rows(try JSON.get(tanh, "encoder")), ABS, REL, "encoder")
    try expect_close(try floats_of(logits), try rows(try JSON.get(tanh, "logits")), ABS, REL, "logits")
    erf = try JSON.get(want, "erf")
    try expect_close(try floats_of(audio), try rows(try JSON.get(erf, "encoder")), DRIFT, 0.0, "erf encoder")
    expect_close(try floats_of(logits), try rows(try JSON.get(erf, "logits")), DRIFT, 0.0, "erf logits")
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

## libtorch has no `gelu_tanh_approx` yet (`tensor/README.md`): the model
## loads there and the encoder stops at the first gelu. Without the native
## module, loading is the error.
def test_on_libtorch_the_encoder_stops_at_gelu() -> Result[Void]
  with_fixtures do |dir|
    loaded = Asr.open("#{dir}/tiny/model.safetensors", TORCH)
    return Test.error?(loaded) if not Tensor.available(TORCH)

    stopped: Result[Void] = Native.arena do
      m = try loaded
      case Asr.encode(
        m,
        try f32(
          [0.5, -1.0, 0.25, 1.0, 0.0, -0.5, 0.5, -1.0, 0.25, 1.0, 0.0, -0.5, 0.5, -1.0, 0.25, 1.0, 0.0, -0.5],
          [1, 3, 6],
          TORCH,
        ),
      )
        ok(_) -> error("the encoder ran on libtorch: drop this test and compare with the peer")
        error(e) -> Test.assert(e.message.include?("gelu_tanh_approx"), "unexpected error: #{e.message}")
      end
    end

    stopped
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
