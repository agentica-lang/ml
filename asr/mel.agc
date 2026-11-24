## Whisper's log-mel frontend: 16 kHz mono samples to the `[mels, frames]`
## log-mel spectrogram the encoder takes (`asr/DESIGN.md`).

import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## Samples per second.
SAMPLE_RATE = 16000
## The STFT window and its hop, in samples.
N_FFT = 400
HOP = 160
## 30 s of audio.
FRAMES = 3000

## The log-mel spectrogram of `samples` (16 kHz, mono, about -1..1), row-major
## `[mels, frames]`: the samples are padded with zeros or trimmed to
## `frames * 160` (the samples after those, up to 200, are still read as the
## end of the signal, for the last frame), then a reflect-padded STFT (Hann
## window of 400, hop 160)
## gives `frames` power spectra, the Slaney mel filterbank, `log10`, a floor of
## the maximum minus 8 and `(x + 4) / 4`, as Whisper and Hugging Face do.
## 3000 frames are Hugging Face's input features; fewer are the spectrogram of
## that shorter segment (its own reflection and maximum), not a prefix of them.
def log_mel(samples: List[Float], mels: Int, frames: Int = FRAMES) -> Result[List[Float]]
  return error("log_mel needs mels >= 1, got #{mels}") if mels < 1
  return error("log_mel needs frames >= 2, got #{frames}") if frames < 2
  size = frames * HOP
  x = samples.take(size)
  heard = 0
  for i in 0...samples.size
    # NaN fails `x - x == 0`; a bound far above the -1..1 range keeps the
    # power spectrum from overflowing to Inf and then NaN.
    if samples[i] - samples[i] != 0.0 or samples[i].abs > LIMIT
      return error("log_mel needs finite samples of about -1..1, got #{samples[i]} at #{i}")
    end

    if samples[i] != 0.0 and i < size
      heard = i + 1
    end
  end

  x = x + List.repeat(0.0, size - x.size)
  half = N_FFT / 2
  padded = x.drop(1).take(half).reverse + x + right_context(samples, x, half)
  # `padded` is zero from `half + heard` on, unless the end reflection holds a
  # sample: then every frame that starts there or later is silent.
  silent = frames
  if heard <= size - half - 1 and samples.size <= size
    silent = (heard + half + HOP - 1) / HOP
  end

  table = cos_sin(N_FFT)
  order = input_order(N_FFT)
  window = (0...N_FFT).map { |i| 0.5 - 0.5 * table.re[i] }
  bank = filters(mels)
  columns: List[List[Float]] = []
  top = -10.0
  i = 0
  while i < frames
    if i >= silent
      columns = columns.push(List.repeat(-10.0, mels))
      i += 1
      next
    end

    a = frame(padded, window, i)
    b = List.repeat(0.0, N_FFT)
    if i + 1 < frames and i + 1 < silent
      b = frame(padded, window, i + 1)
    end

    z = fft(Spectrum(re: a, im: b), order, table)
    pair = powers(z)
    for power in pair.take(frames - i)
      column: List[Float] = []
      for f in bank
        v = log10(mel_energy(f, power))
        top = larger(top, v)
        column = column.push(v)
      end

      columns = columns.push(column)
    end

    i += 2
  end

  low = top - 8.0
  out: List[Float] = []
  for m in 0...mels
    for c in columns
      out = out.push((larger(c[m], low) + 4.0) / 4.0)
    end
  end

  ok(out)
end

## The features the encoder takes, `[1, mels, frames]` `F32` on `backend`. On
## `Reference` they are `log_mel` (the exact FFT in doubles); on every other
## backend `spectrogram`, one graph that stays on the device. With a `plan`
## (`Trace.compile` of `graph` over `specs`, on ggml) the graph runs as
## compiled, with its constants made once; without one it is traced and run
## once, uncached (`keep: false`), so it does not evict the graphs of a
## decoder. The result is the caller's to close.
def features(
  samples: List[Float],
  mels: Int,
  frames: Int,
  backend: Tensor.Backend,
  plan: Optional[Trace.Compiled] = none,
) -> Result[Tensor.Tensor]
  case backend
    Tensor.Backend.Reference -> Tensor.floats(try log_mel(samples, mels, frames), [1, mels, frames], backend)
    # No arena here: the result must outlive the call, and `run` closes the
    # inputs and the check.
    Tensor.Backend.Ggml(_) -> run(samples, mels, frames, backend, plan)
    _ -> Native.arena do run(samples, mels, frames, backend, plan) end
  end
end

## The same spectrogram as `log_mel`, as one traced graph on `backend`: the
## signal padded by reflection, read as blocks of one hop, the windowed DFT as
## a matmul with a `[N_FFT, 2 * bins]` cosine and sine basis, the power, the
## mel filters as a second matmul, then the log, the floor and the scale. It
## computes in the backend's `F32` (the gap to `log_mel` is about 1e-5), and
## an `Inf` or `NaN` sample, or one so large that the power overflows (about
## 1.8e19), is an error.
def spectrogram(samples: List[Float], mels: Int, frames: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  run(samples, mels, frames, backend, none)
end

## The specs of the three tensors of `signal`, for `Trace.compile`.
def specs(frames: Int, backend: Tensor.Backend) -> List[Trace.Spec]
  half = N_FFT / 2
  [
    Trace.Spec(shape: [half], dtype: Tensor.Dtype.F32, backend: backend),
    Trace.Spec(shape: [frames * HOP], dtype: Tensor.Dtype.F32, backend: backend),
    Trace.Spec(shape: [half], dtype: Tensor.Dtype.F32, backend: backend),
  ]
end

## The graph of `spectrogram` over the tensors of `signal`: the features, and
## the sum of the features, of the samples and the largest power, which is not
## finite when a sample was not or its power overflowed (`relu` hides a `NaN`
## or an `Inf` inside the features). A caller that runs it often compiles it
## once (`Trace.compile`), which keeps the constants for as long as the task
## that built them lives.
def graph(xs: List[Tensor.Tensor], mels: Int, frames: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  bins = N_FFT / 2 + 1
  basis = Trace.constant(try Tensor.floats(dft_basis(), [N_FFT, 2 * bins], backend))
  bank = Trace.constant(try Tensor.floats(filter_matrix(mels), [bins, mels], backend))
  signal = try Tensor.concat(xs, 0)
  # Frame `i` is the blocks `i`, `i + 1` and the first 80 samples of `i + 2`.
  blocks = try Tensor.reshape(try Tensor.slice(signal, [0], [(frames + 2) * HOP]), [frames + 2, HOP])
  head = try Tensor.slice(blocks, [0, 0], [frames, HOP])
  middle = try Tensor.slice(blocks, [1, 0], [frames + 1, HOP])
  tail = try Tensor.slice(blocks, [2, 0], [frames + 2, N_FFT - 2 * HOP])
  spectrum = try Tensor.matmul(try Tensor.concat([head, middle, tail], 1), basis)
  re = try Tensor.slice(spectrum, [0, 0], [frames, bins])
  im = try Tensor.slice(spectrum, [0, bins], [frames, 2 * bins])
  power = try Tensor.add(try Tensor.mul(re, re), try Tensor.mul(im, im))
  energy = try at_least(try Tensor.matmul(power, bank), FLOOR)
  logs = try Tensor.mul(try Tensor.log(energy), try filled(LOG10_E, energy))
  top = try Tensor.max(logs, none, true)
  low = try Tensor.broadcast_to(try Tensor.sub(top, try filled(8.0, top)), Tensor.shape(logs))
  floored = try Tensor.add(low, try Tensor.relu(try Tensor.sub(logs, low)))
  scaled = try Tensor.div(try Tensor.add(floored, try filled(4.0, floored)), try filled(4.0, floored))
  mel = try Tensor.reshape(try Tensor.transpose(scaled), [1, mels, frames])
  seen = try Tensor.add(try Tensor.sum(mel), try Tensor.sum(signal))
  ok([mel, try Tensor.add(seen, try Tensor.max(power))])
end

private

PI = 3.141592653589793
## cos and sin of 2π/5 and 4π/5, for the radix-5 butterfly.
C1 = 0.30901699437494745
C2 = -0.8090169943749475
S1 = 0.9510565162951535
S2 = 0.5877852522924731
## The largest sample magnitude `log_mel` takes.
LIMIT = 1.0e150
## The power floor before `log10`.
FLOOR = 0.0000000001
## `log10(e)`: `log10(x) = ln(x) * LOG10_E`.
LOG10_E = 0.4342944819032518

## A complex sequence as two lists.
struct Spectrum
  re: List[Float]
  im: List[Float]
end

## One mel filter: its non-zero weights from bin `start` on.
struct Filter
  start: Int
  weights: List[Float]
end

## The `half` values after `x`, the samples as `log_mel` reads them: the
## samples that `x` cut off when there are some, then the reflection of the end
## of `samples`; the reflection of the end of `x` when nothing was cut off.
def right_context(samples: List[Float], x: List[Float], half: Int) -> List[Float]
  return x.drop(x.size - half - 1).take(half).reverse if samples.size <= x.size

  (samples.drop(x.size) + samples.drop(samples.size - half - 1).take(half).reverse).take(half)
end

## The `mels` triangular filters over the `N_FFT / 2 + 1` bins, on the Slaney
## mel scale with Slaney's area norm (librosa's `filters.mel`, which Whisper's
## `mel_filters.npz` and Hugging Face's feature extractor use), from 0 Hz to
## the Nyquist frequency.
def filters(mels: Int) -> List[Filter]
  bins = N_FFT / 2 + 1
  top = hz_to_mel((SAMPLE_RATE / 2).to_f)
  # The last point is `top` itself, as numpy's `linspace` ends.
  points: List[Float] = []
  for i in 0...(mels + 1)
    points = points.push(mel_to_hz((i.to_f * top) / (mels + 1).to_f))
  end

  points = points.push(mel_to_hz(top))

  out: List[Filter] = []
  for m in 0...mels
    lo = points[m]
    mid = points[m + 1]
    hi = points[m + 2]
    norm = 2.0 / (hi - lo)
    start = -1
    weights: List[Float] = []
    for k in 0...bins
      hz = (k * SAMPLE_RATE).to_f / N_FFT.to_f
      w = larger(0.0, smaller((hz - lo) / (mid - lo), (hi - hz) / (hi - mid)))
      next if w <= 0.0
      if start < 0
        start = k
      end

      weights = weights.push(w * norm)
    end

    out = out.push(Filter(start: start.clamp(0, bins), weights: weights))
  end

  out
end

## The discrete Fourier transform of the `N_FFT` points `z`: a mixed-radix
## Cooley-Tukey FFT (decimation in time), radix 2 while the length is even,
## then radix 5 (400 = 2^4 · 5^2), as whisper.cpp splits it, with butterflies
## for both. `order` is `input_order(N_FFT)`; `table` holds `cos` and `sin` of
## `2πj / N_FFT`.
def fft(z: Spectrum, order: List[Int], table: Spectrum) -> Spectrum
  re: List[Float] = []
  im: List[Float] = []
  for j in order
    re = re.push(z.re[j])
    im = im.push(z.im[j])
  end

  cur = Spectrum(re: re, im: im)
  n = 1
  for p in radices(N_FFT).reverse
    n = n * p
    cur = case p
      2 -> radix2(cur, n, table)
      _ -> radix5(cur, n, table)
    end
  end

  cur
end

## `features` once the plan is chosen: the three inputs, the graph (the plan,
## or traced and run once), and the check.
def run(
  samples: List[Float],
  mels: Int,
  frames: Int,
  backend: Tensor.Backend,
  plan: Optional[Trace.Compiled],
) -> Result[Tensor.Tensor]
  inputs = try signal(samples, mels, frames, backend)
  ran = case plan
    some(p) -> Trace.run(p, inputs)
    none -> Trace.eval_many(inputs, ->(xs: List[Tensor.Tensor]) { graph(xs, mels, frames, backend) }, false)
  end

  close(inputs)
  checked(try ran)
end

## The three inputs of `graph`, `F32` on `backend`: the reflection of the first
## samples, the `frames * HOP` samples (padded with zeros or trimmed), and the
## reflection of the last ones.
def signal(samples: List[Float], mels: Int, frames: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  return error("log_mel needs mels >= 1, got #{mels}") if mels < 1
  return error("log_mel needs frames >= 2, got #{frames}") if frames < 2
  size = frames * HOP
  half = N_FFT / 2
  x = samples.take(size)
  x = x + List.repeat(0.0, size - x.size)
  left = x.drop(1).take(half).reverse
  right = right_context(samples, x, half)
  ok([
    try Tensor.floats(left, [half], backend),
    try Tensor.floats(x, [size], backend),
    try Tensor.floats(right, [half], backend),
  ])
end

## The features of the answer of `graph`, once their sum is finite; it closes the check.
def checked(out: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  seen = Tensor.to_list(out[1])
  close([out[1]])
  case seen
    ok(values) ->
      case values[0]
        Tensor.Values.Float(total) ->
          return ok(out[0]) if total - total == 0.0

          close([out[0]])
          error("log_mel needs finite samples of about -1..1")
        _ ->
          close([out[0]])
          error("the log-mel check is not a float")
      end

    error(e) ->
      close([out[0]])
      error(e)
  end
end

## The mel filters as a row-major `[N_FFT / 2 + 1, mels]` matrix: bin `k`, filter `m`.
def filter_matrix(mels: Int) -> List[Float]
  bank = filters(mels)
  out: List[Float] = []
  for k in 0...(N_FFT / 2 + 1)
    for f in bank
      at = k - f.start
      if at >= 0 and at < f.weights.size
        out = out.push(f.weights[at])
      else
        out = out.push(0.0)
      end
    end
  end

  out
end

## `max(x, floor)` from the primitives the backends lower.
def at_least(x: Tensor.Tensor, floor: Float) -> Result[Tensor.Tensor]
  f = try filled(floor, x)
  Tensor.add(f, try Tensor.relu(try Tensor.sub(x, f)))
end

def filled(v: Float, like: Tensor.Tensor) -> Result[Tensor.Tensor]
  Tensor.full(Tensor.shape(like), Tensor.Values.Float(v), Tensor.dtype(like), Tensor.backend(like))
end

## The Hann window times the DFT, row-major `[N_FFT, 2 * bins]`: row `n` holds
## `window[n] * cos(2π n k / N_FFT)` for the bins `k`, then the same with
## `sin` (the sign of the imaginary part does not change the power).
def dft_basis() -> List[Float]
  table = cos_sin(N_FFT)
  bins = N_FFT / 2 + 1
  out: List[Float] = []
  for n in 0...N_FFT
    window = 0.5 - 0.5 * table.re[n]
    for k in 0...bins
      out = out.push(window * table.re[(n * k) % N_FFT])
    end

    for k in 0...bins
      out = out.push(window * table.im[(n * k) % N_FFT])
    end
  end

  out
end

## Frame `i` of the padded signal, windowed.
def frame(padded: List[Float], window: List[Float], i: Int) -> List[Float]
  offset = i * HOP
  out: List[Float] = []
  for j in 0...N_FFT
    out = out.push(window[j] * padded[offset + j])
  end

  out
end

## The power spectra (`N_FFT / 2 + 1` bins) of the two real frames packed as
## the real and the imaginary part of `z`, the FFT of `a + i·b`.
def powers(z: Spectrum) -> List[List[Float]]
  n = z.re.size
  a: List[Float] = []
  b: List[Float] = []
  for k in 0...(n / 2 + 1)
    j = (n - k) % n
    sr = z.re[k] + z.re[j]
    dr = z.re[k] - z.re[j]
    si = z.im[k] + z.im[j]
    di = z.im[k] - z.im[j]
    a = a.push((sr * sr + di * di) / 4.0)
    b = b.push((si * si + dr * dr) / 4.0)
  end

  [a, b]
end

def mel_energy(f: Filter, power: List[Float]) -> Float
  total = 0.0
  for j in 0...f.weights.size
    total += f.weights[j] * power[f.start + j]
  end

  total
end

## The radices that split `n`, outermost first: 2 while even, then 5.
def radices(n: Int) -> List[Int]
  out: List[Int] = []
  m = n
  while m > 1
    p = 5
    if m % 2 == 0
      p = 2
    end

    out = out.push(p)
    m = m / p
  end

  out
end

## The input indices in the order the leaves of the FFT read them.
def input_order(n: Int) -> List[Int] = split((0...n).to_a, radices(n))

def split(xs: List[Int], radices: List[Int]) -> List[Int]
  return xs if radices.empty?
  p = radices[0]
  out: List[Int] = []
  for r in 0...p
    part: List[Int] = []
    i = r
    while i < xs.size
      part = part.push(xs[i])
      i += p
    end

    out = out + split(part, radices.drop(1))
  end

  out
end

## One radix-2 stage: each block of `n` holds the transforms of its even and
## odd halves; the butterfly joins them.
def radix2(z: Spectrum, n: Int, table: Spectrum) -> Spectrum
  h = n / 2
  step = N_FFT / n
  re: List[Float] = []
  im: List[Float] = []
  base = 0
  while base < N_FFT
    lr: List[Float] = []
    li: List[Float] = []
    hr: List[Float] = []
    hi: List[Float] = []
    for k in 0...h
      wr = table.re[k * step]
      wi = table.im[k * step]
      er = z.re[base + k]
      ei = z.im[base + k]
      orr = z.re[base + h + k]
      oi = z.im[base + h + k]
      tr = wr * orr + wi * oi
      ti = wr * oi - wi * orr
      lr = lr.push(er + tr)
      li = li.push(ei + ti)
      hr = hr.push(er - tr)
      hi = hi.push(ei - ti)
    end

    re = re + lr + hr
    im = im + li + hi
    base += n
  end

  Spectrum(re: re, im: im)
end

## One radix-5 stage: each block of `n` holds the transforms of its five
## interleaved fifths; a five-point butterfly joins them.
def radix5(z: Spectrum, n: Int, table: Spectrum) -> Spectrum
  m = n / 5
  step = N_FFT / n
  re: List[Float] = []
  im: List[Float] = []
  base = 0
  while base < N_FFT
    o0r: List[Float] = []
    o0i: List[Float] = []
    o1r: List[Float] = []
    o1i: List[Float] = []
    o2r: List[Float] = []
    o2i: List[Float] = []
    o3r: List[Float] = []
    o3i: List[Float] = []
    o4r: List[Float] = []
    o4i: List[Float] = []
    for k in 0...m
      x0r = z.re[base + k]
      x0i = z.im[base + k]
      w = k * step
      x1r = table.re[w] * z.re[base + m + k] + table.im[w] * z.im[base + m + k]
      x1i = table.re[w] * z.im[base + m + k] - table.im[w] * z.re[base + m + k]
      w = 2 * k * step
      x2r = table.re[w] * z.re[base + 2 * m + k] + table.im[w] * z.im[base + 2 * m + k]
      x2i = table.re[w] * z.im[base + 2 * m + k] - table.im[w] * z.re[base + 2 * m + k]
      w = 3 * k * step
      x3r = table.re[w] * z.re[base + 3 * m + k] + table.im[w] * z.im[base + 3 * m + k]
      x3i = table.re[w] * z.im[base + 3 * m + k] - table.im[w] * z.re[base + 3 * m + k]
      w = 4 * k * step
      x4r = table.re[w] * z.re[base + 4 * m + k] + table.im[w] * z.im[base + 4 * m + k]
      x4i = table.re[w] * z.im[base + 4 * m + k] - table.im[w] * z.re[base + 4 * m + k]
      a1r = x1r + x4r
      a1i = x1i + x4i
      b1r = x1r - x4r
      b1i = x1i - x4i
      a2r = x2r + x3r
      a2i = x2i + x3i
      b2r = x2r - x3r
      b2i = x2i - x3i
      p1r = x0r + C1 * a1r + C2 * a2r
      p1i = x0i + C1 * a1i + C2 * a2i
      p2r = x0r + C2 * a1r + C1 * a2r
      p2i = x0i + C2 * a1i + C1 * a2i
      dr = S1 * b1r + S2 * b2r
      di = S1 * b1i + S2 * b2i
      er = S2 * b1r - S1 * b2r
      ei = S2 * b1i - S1 * b2i
      o0r = o0r.push(x0r + a1r + a2r)
      o0i = o0i.push(x0i + a1i + a2i)
      o1r = o1r.push(p1r + di)
      o1i = o1i.push(p1i - dr)
      o2r = o2r.push(p2r + ei)
      o2i = o2i.push(p2i - er)
      o3r = o3r.push(p2r - ei)
      o3i = o3i.push(p2i + er)
      o4r = o4r.push(p1r - di)
      o4i = o4i.push(p1i + dr)
    end

    re = re + o0r + o1r + o2r + o3r + o4r
    im = im + o0i + o1i + o2i + o3i + o4i
    base += n
  end

  Spectrum(re: re, im: im)
end

## `cos` and `sin` of `2πj / n` for `j` in `0...n`.
def cos_sin(n: Int) -> Spectrum
  angles = (0...n).map { |j| (2.0 * PI * j.to_f) / n.to_f }
  Spectrum(re: angles.map { |x| Math.cos(x) }, im: angles.map { |x| Math.sin(x) })
end

def larger(a: Float, b: Float) -> Float
  return a if a > b
  b
end

def smaller(a: Float, b: Float) -> Float
  return a if a < b
  b
end

## The Slaney mel scale: linear below 1 kHz, logarithmic above.
def hz_to_mel(hz: Float) -> Float
  return (hz * 3.0) / 200.0 if hz < 1000.0
  15.0 + (Math.log(hz / 1000.0) * 27.0) / Math.log(6.4)
end

def mel_to_hz(mel: Float) -> Float
  return (mel * 200.0) / 3.0 if mel < 15.0
  1000.0 * Math.exp(((mel - 15.0) * Math.log(6.4)) / 27.0)
end

## `log10(max(x, 1e-10))`.
def log10(x: Float) -> Float
  return -10.0 if x <= FLOOR
  Math.log10(x)
end

## Closes the ggml tensors of `ts` now; the other backends' are the arena's.
def close(ts: List[Tensor.Tensor]) -> Void
  for t in ts
    case t.handle
      Tensor.Handle.Ggml(h) -> Native.close(h)
      _ -> nil
    end
  end
end
