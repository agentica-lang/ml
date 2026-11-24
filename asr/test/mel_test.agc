## The log-mel frontend against `mel_peer.py`, which computes the same
## spectrogram in doubles with a direct DFT. Run from `asr/`.

import Mel from "../mel"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## Hugging Face's values for the first second of this file
## (`test/hf_golden.py`).
WAV = "../whisper/live/fixtures/es-hola.wav"
GOLDEN = "test/fixtures/hf_mel.json"
## The golden keeps 7 decimals of Hugging Face's float32 output.
HF_CLOSE = 0.000001

## Both sides compute in doubles; the FFT and the direct DFT round
## differently (the gap is about 1e-13).
CLOSE = 0.000000001

## The graph computes in `F32`; its gap to the doubles is about 1e-5.
GRAPH_CLOSE = 0.0001
## The golden values are in the first second: 100 frames.
FIRST = 100

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

## 80 mels over 21 frames of a short signal (padded, with silent frames and an
## odd last frame), and 128 mels over 10 frames of a longer one (trimmed).
def test_the_log_mel_matches_the_peer() -> Result[Void]
  cases = try JSON.list(try JSON.parse(try peer()))
  try Test.eq(cases.size, 2)
  for c in cases
    mels = try JSON.int(try JSON.get(c, "mels"))
    frames = try JSON.int(try JSON.get(c, "frames"))
    got = try Mel.log_mel(try floats(try JSON.get(c, "samples")), mels, frames)
    want = try floats(try JSON.get(c, "mel"))
    try Test.eq(got.size, mels * frames)
    for i in 0...want.size
      gap = (got[i] - want[i]).abs
      return error("#{mels} mels: value #{i} is #{got[i]}, expected #{want[i]}") if gap > CLOSE
    end
  end

  ok()
end

## `spectrogram`, the graph that every backend but `Reference` runs, against
## the FFT of `log_mel` (which the peer checks) on a signal longer than its 2
## frames (trimmed; the padded one is the backend test below). The
## sizes are small: `Reference` runs the matmuls in the interpreter.
def test_the_graph_matches_the_fft() -> Result[Void]
  samples = (0...1000).map { |i| (0.5 * ((i * 7919) % 1000).to_f) / 1000.0 - 0.25 }
  out = try Mel.spectrogram(samples, 3, 2, REF)
  try Test.eq(Tensor.shape(out), [1, 3, 2])
  expect_close(try tensor_floats(out), try Mel.log_mel(samples, 3, 2), "1000 samples")
end

def test_the_graph_refuses_a_bad_size() -> Result[Void]
  try Test.error?(Mel.spectrogram([0.5], 0, 4, REF))
  Test.error?(Mel.spectrogram([0.5], 3, 1, REF))
end

## `features` on the backends this machine has (ggml, Vulkan, libtorch)
## against Hugging Face, over the 30 s of 3000 frames.
def test_the_features_on_every_backend_match_hugging_face() -> Result[Void]
  golden = try JSON.parse(try String.from_bytes(try read_bytes(GOLDEN)))
  samples = (try pcm16(try read_bytes(WAV))).take(try JSON.int(try JSON.get(golden, "samples")))
  nan = 1.0e308 * 10.0 - 1.0e308 * 10.0
  all = try backends()
  for at in 0...all.size
    backend = all[at]
    try Test.error?(Mel.features([0.5, nan], 4, 8, backend))
    # 128 mels (large-v3) on the first backend only: the graph is the same.
    sizes = [80]
    if at == 0
      sizes = [80, 128]
    end

    for mels in sizes
      features = try Mel.features(samples, mels, Mel.FRAMES, backend)
      try Test.eq(Tensor.shape(features), [1, mels, Mel.FRAMES])
      got = try tensor_floats(try Trace.eval([features], ->(xs: List[Tensor.Tensor]) { first_frames(xs[0], mels) }))
      for entry in try JSON.list(try JSON.get(golden, "#{mels}"))
        m = try JSON.int(try JSON.at(entry, 0))
        f = try JSON.int(try JSON.at(entry, 1))
        v = try JSON.float(try JSON.at(entry, 2))
        value = got[m * FIRST + f]
        if (value - v).abs > GRAPH_CLOSE
          return error("#{backend}, #{mels} mels: [#{m}, #{f}] is #{value}, Hugging Face has #{v}")
        end
      end
    end
  end

  ok()
end

def backends() -> Result[List[Tensor.Backend]]
  out: List[Tensor.Backend] = []
  if Tensor.available(TORCH)
    out = out.push(TORCH)
  end

  if !Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok(out)
  end

  out = out.push(GGML)
  if Tensor.available(VULKAN)
    out = out.push(VULKAN)
  end

  ok(out)
end

## The first `FIRST` frames of `[1, mels, frames]` features: reading all of
## them back would cost the test most of its steps.
def first_frames(features: Tensor.Tensor, mels: Int) -> Result[Tensor.Tensor]
  Tensor.slice(features, [0, 0, 0], [1, mels, FIRST])
end

def tensor_floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(x) -> out = out.push(x)
      _ -> return error("a log-mel tensor holds floats")
    end
  end

  ok(out)
end

def expect_close(got: List[Float], want: List[Float], what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if (got[i] - want[i]).abs > GRAPH_CLOSE
  end

  ok()
end

## Silence is the floor everywhere: log10(1e-10) = -10, so (-10 + 4) / 4.
def test_silence_is_the_floor() -> Result[Void]
  got = try Mel.log_mel([], 3, 4)
  Test.eq(got, List.repeat(-1.5, 12))
end

def test_bad_sizes_and_samples_are_errors() -> Result[Void]
  try Test.error?(Mel.log_mel([0.5], 0, 4))
  try Test.error?(Mel.log_mel([0.5], 80, 1))
  inf = 1.0e308 * 10.0
  try Test.error?(Mel.log_mel([0.5, inf], 3, 4))
  try Test.error?(Mel.log_mel([0.5, -1.0e151], 3, 4))
  try Test.ok?(Mel.log_mel([0.5, 1.0e150], 3, 4))
  Test.error?(Mel.log_mel([inf - inf, 0.5], 3, 4))
end

def test_the_first_second_of_speech_matches_hugging_face() -> Result[Void]
  golden = try JSON.parse(try String.from_bytes(try read_bytes(GOLDEN)))
  samples = (try pcm16(try read_bytes(WAV))).take(try JSON.int(try JSON.get(golden, "samples")))
  for mels in [80, 128]
    got = try Mel.log_mel(samples, mels)
    want = try JSON.list(try JSON.get(golden, "#{mels}"))
    try Test.eq(want.size, 100)
    for entry in want
      m = try JSON.int(try JSON.at(entry, 0))
      f = try JSON.int(try JSON.at(entry, 1))
      v = try JSON.float(try JSON.at(entry, 2))
      value = got[m * Mel.FRAMES + f]
      return error("#{mels} mels: [#{m}, #{f}] is #{value}, Hugging Face has #{v}") if (value - v).abs > HF_CLOSE
    end
  end

  ok()
end

## 0.5 s at the default 30 s: 3000 frames, and every frame that starts after
## the last sample (from frame 52, at sample 8320 of the padded signal) is the
## floor, the smallest value.
def test_half_a_second_fills_30_seconds_with_the_floor() -> Result[Void]
  samples = (0...8000).map { |i| (0.5 * ((i * 7919) % 1000).to_f) / 1000.0 - 0.25 }
  got = try Mel.log_mel(samples, 4)
  try Test.eq(got.size, 4 * Mel.FRAMES)
  floor = got.reduce(got[0]) do |a, v| if_less(v, a) end

  try Test.assert(floor < got[0], "the first frame is the floor")
  for m in 0...4
    for f in 52...Mel.FRAMES
      if got[m * Mel.FRAMES + f] != floor
        return error("mel #{m} frame #{f} is #{got[m * Mel.FRAMES + f]}, not the floor #{floor}")
      end
    end
  end

  ok()
end

def peer() -> Result[String]
  p = try Process.spawn_captured("python3", ["-I", "test/mel_peer.py"])
  out = try Process.output(p, 60000)
  code = try Process.wait(p, 60000)
  return error("mel_peer.py exited with #{code}: #{out.stderr}") if code != 0
  ok(out.stdout)
end

def floats(json: JSON.Value) -> Result[List[Float]]
  out: List[Float] = []
  for v in try JSON.list(json)
    out = out.push(try JSON.float(v))
  end

  ok(out)
end

def if_less(a: Float, b: Float) -> Float
  return a if a < b
  b
end

def read_bytes(path: String) -> Result[Bytes]
  File.open(path) do |f|
    parts: List[Bytes] = []
    while true
      chunk = try File.read_bytes(f, 1048576)
      break if chunk.empty?
      parts = parts.push(chunk)
    end

    ok(Bytes.concat(parts))
  end
end

## The samples of a 16-bit PCM WAV, scaled to -1..1.
def pcm16(wav: Bytes) -> Result[List[Float]]
  at = 12
  while at + 8 <= wav.size
    size = u16(wav, at + 4) + u16(wav, at + 6) * 65536
    if wav.slice(at, 4) == "data".bytes
      out: List[Float] = []
      for i in 0...(size / 2)
        v = u16(wav, at + 8 + 2 * i)
        if v >= 32768
          v -= 65536
        end

        out = out.push(v.to_f / 32768.0)
      end

      return ok(out)
    end

    at += 8 + size
  end

  error("no data chunk")
end

def u16(b: Bytes, i: Int) -> Int = b[i] + b[i + 1] * 256
