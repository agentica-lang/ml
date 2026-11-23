## The log-mel frontend against `mel_peer.py`, which computes the same
## spectrogram in doubles with a direct DFT. Run from `asr/`.

import Mel from "../mel"

## Both sides compute in doubles; the FFT and the direct DFT round
## differently (the gap is about 1e-13).
CLOSE = 0.000000001

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

## Silence is the floor everywhere: log10(1e-10) = -10, so (-10 + 4) / 4.
def test_silence_is_the_floor() -> Result[Void]
  got = try Mel.log_mel([], 3, 4)
  Test.eq(got, List.repeat(-1.5, 12))
end

def test_bad_sizes_are_errors() -> Result[Void]
  try Test.error?(Mel.log_mel([0.5], 0, 4))
  Test.error?(Mel.log_mel([0.5], 80, 1))
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
