## Whisper's log-mel frontend in pure Agentica: 16 kHz mono samples to the
## `[mels, frames]` log-mel spectrogram the encoder takes (`asr/DESIGN.md`).

## Samples per second.
SAMPLE_RATE = 16000
## The STFT window and its hop, in samples.
N_FFT = 400
HOP = 160
## 30 s of audio.
FRAMES = 3000

## The log-mel spectrogram of `samples` (16 kHz, mono, about -1..1), row-major
## `[mels, frames]`: the samples are padded with zeros or trimmed to
## `frames * 160`, then a reflect-padded STFT (Hann window of 400, hop 160)
## gives `frames` power spectra, the Slaney mel filterbank, `log10`, a floor of
## the maximum minus 8 and `(x + 4) / 4`, as Whisper and Hugging Face do.
def log_mel(samples: List[Float], mels: Int, frames: Int = FRAMES) -> Result[List[Float]]
  return error("log_mel needs mels >= 1, got #{mels}") if mels < 1
  return error("log_mel needs frames >= 2, got #{frames}") if frames < 2
  size = frames * HOP
  x = samples.take(size)
  x = x + List.repeat(0.0, size - x.size)
  half = N_FFT / 2
  padded = x.drop(1).take(half).reverse + x + x.drop(size - half - 1).take(half).reverse
  # A frame that starts after the last sample is silent, as is every frame
  # after it, unless the end reflection holds samples.
  heard = samples.size.clamp(0, size)
  silent = frames
  if heard < size - half
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

private

PI = 3.141592653589793
LN2 = 0.6931471805599453
LN10 = 2.302585092994046
SQRT2 = 1.4142135623730951
## cos and sin of 2π/5 and 4π/5, for the radix-5 butterfly.
C1 = 0.30901699437494745
C2 = -0.8090169943749475
S1 = 0.9510565162951535
S2 = 0.5877852522924731
## The power floor before `log10`.
FLOOR = 0.0000000001

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

## The `mels` triangular filters over the `N_FFT / 2 + 1` bins, on the Slaney
## mel scale with Slaney's area norm (librosa's `filters.mel`, which Whisper's
## `mel_filters.npz` and Hugging Face's feature extractor use), from 0 Hz to
## the Nyquist frequency.
def filters(mels: Int) -> List[Filter]
  bins = N_FFT / 2 + 1
  top = hz_to_mel((SAMPLE_RATE / 2).to_f)
  points = (0...(mels + 2)).map do |i|
    return mel_to_hz(top) if i == mels + 1

    mel_to_hz((i.to_f * top) / (mels + 1).to_f)
  end

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
  re: List[Float] = []
  im: List[Float] = []
  for j in 0...n
    x = (2.0 * PI * j.to_f) / n.to_f
    if x > PI
      x -= 2.0 * PI
    end

    term = 1.0
    c = 1.0
    s = 0.0
    for t in 1...40
      term = (term * x) / t.to_f
      case t % 4
        0 -> c += term
        1 -> s += term
        2 -> c -= term
        _ -> s -= term
      end
    end

    re = re.push(c)
    im = im.push(s)
  end

  Spectrum(re: re, im: im)
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
  15.0 + (ln(hz / 1000.0) * 27.0) / ln(6.4)
end

def mel_to_hz(mel: Float) -> Float
  return (mel * 200.0) / 3.0 if mel < 15.0
  1000.0 * exp(((mel - 15.0) * ln(6.4)) / 27.0)
end

## `log10(max(x, 1e-10))`.
def log10(x: Float) -> Float
  return -10.0 if x <= FLOOR
  ln(x) / LN10
end

## The natural logarithm of a positive finite `x`: `x = m · 2^e` with `m` in
## `[√½, √2)` (a ladder of powers of two), then the `atanh` series of
## `(m - 1) / (m + 1)` (|y| < 0.172, so nine terms reach double precision).
def ln(x: Float) -> Float
  m = x
  e = 0
  while m >= 65536.0
    m = m / 65536.0
    e += 16
  end

  while m < 1.0 / 65536.0
    m = m * 65536.0
    e -= 16
  end

  if m >= 256.0
    m = m / 256.0
    e += 8
  end

  if m >= 16.0
    m = m / 16.0
    e += 4
  end

  if m >= 4.0
    m = m / 4.0
    e += 2
  end

  if m >= 2.0
    m = m / 2.0
    e += 1
  end

  if m < 1.0 / 128.0
    m = m * 256.0
    e -= 8
  end

  if m < 1.0 / 8.0
    m = m * 16.0
    e -= 4
  end

  if m < 0.5
    m = m * 4.0
    e -= 2
  end

  if m < 1.0
    m = m * 2.0
    e -= 1
  end

  if m >= SQRT2
    m = m / 2.0
    e += 1
  end

  if m < SQRT2 / 2.0
    m = m * 2.0
    e -= 1
  end

  y = (m - 1.0) / (m + 1.0)
  z = y * y
  series = (
    1.0 +
    z *
    (
      1.0 / 3.0 +
      z *
      (
        1.0 / 5.0 +
        z * (1.0 / 7.0 + z * (1.0 / 9.0 + z * (1.0 / 11.0 + z * (1.0 / 13.0 + z * (1.0 / 15.0 + z / 17.0)))))
      )
    )
  )

  2.0 * y * series + e.to_f * LN2
end

## `e^x` for a small `|x|` (the mel scale needs up to about 2.1): its Taylor
## series.
def exp(x: Float) -> Float
  term = 1.0
  total = 1.0
  for t in 1...40
    term = (term * x) / t.to_f
    total += term
  end

  total
end
