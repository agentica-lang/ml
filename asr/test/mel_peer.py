"""The independent peer of the log-mel tests: Whisper's log-mel spectrogram in
plain Python doubles, with a direct DFT and librosa's Slaney mel filters.
Standard library only.

Usage: python3 mel_peer.py. Prints a JSON list of cases, each with its
synthetic samples, mel count, frame count and expected `[mels, frames]` values
(row-major).
"""

import json
import math
import random
import struct

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160


def hz_to_mel(hz):
    if hz < 1000.0:
        return hz * 3.0 / 200.0
    return 15.0 + math.log(hz / 1000.0) / (math.log(6.4) / 27.0)


def mel_to_hz(mel):
    if mel < 15.0:
        return mel * 200.0 / 3.0
    return 1000.0 * math.exp((math.log(6.4) / 27.0) * (mel - 15.0))


def filters(mels):
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels=mels), norm "slaney"."""
    top = hz_to_mel(SAMPLE_RATE / 2)
    points = [mel_to_hz(top * i / (mels + 1)) for i in range(mels + 2)]
    freqs = [k * SAMPLE_RATE / N_FFT for k in range(N_FFT // 2 + 1)]
    bank = []
    for m in range(mels):
        lo, mid, hi = points[m], points[m + 1], points[m + 2]
        norm = 2.0 / (hi - lo)
        bank.append([max(0.0, min((f - lo) / (mid - lo), (hi - f) / (hi - mid))) * norm for f in freqs])
    return bank


def log_mel(samples, mels, frames):
    size = frames * HOP
    # Hugging Face's feature extractor pads the whole signal by reflection and
    # reads frame `i` at `i * HOP`: samples past `size` stay in the signal, the
    # last frame reads 40 of them, and shorter samples are padded with zeros.
    x = samples if len(samples) >= size else samples + [0.0] * (size - len(samples))
    half = N_FFT // 2
    padded = x[1:half + 1][::-1] + x + x[len(x) - half - 1:len(x) - 1][::-1]
    window = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / N_FFT) for i in range(N_FFT)]
    cos = [math.cos(2.0 * math.pi * j / N_FFT) for j in range(N_FFT)]
    sin = [math.sin(2.0 * math.pi * j / N_FFT) for j in range(N_FFT)]
    bank = filters(mels)
    spec = []
    for i in range(frames):
        seg = [window[j] * padded[i * HOP + j] for j in range(N_FFT)]
        power = []
        for k in range(N_FFT // 2 + 1):
            re = sum(seg[j] * cos[(k * j) % N_FFT] for j in range(N_FFT))
            im = sum(seg[j] * sin[(k * j) % N_FFT] for j in range(N_FFT))
            power.append(re * re + im * im)
        spec.append([math.log10(max(sum(w * p for w, p in zip(row, power)), 1e-10)) for row in bank])
    low = max(max(column) for column in spec) - 8.0
    return [(max(spec[i][m], low) + 4.0) / 4.0 for m in range(mels) for i in range(frames)]


def signal(n, seed):
    """Three tones and some noise, then silence for the last fifth, each sample
    rounded to binary32, as the package takes them (a `List[Float32]`)."""
    rng = random.Random(seed)
    out = []
    for t in range(n):
        s = t / SAMPLE_RATE
        v = 0.3 * math.sin(2 * math.pi * 440 * s) + 0.2 * math.sin(2 * math.pi * 1234.5 * s)
        v += 0.1 * math.sin(2 * math.pi * 5000 * s) + rng.uniform(-0.05, 0.05)
        v = v if t < n * 4 // 5 else 0.0
        out.append(struct.unpack("f", struct.pack("f", v))[0])
    return out


def main():
    cases = []
    # The last two: 5 frames and 37 or 159 samples over, as every clip that is not a
    # multiple of 160 samples is (the last frame reads the samples past `frames * HOP`).
    for mels, frames, n, seed in [(80, 21, 2500, 1), (128, 10, 2000, 2), (80, 5, 5 * HOP + 37, 3), (80, 5, 5 * HOP + 159, 4)]:
        samples = signal(n, seed)
        cases.append({"mels": mels, "frames": frames, "samples": samples, "mel": log_mel(samples, mels, frames)})
    print(json.dumps(cases))


main()
