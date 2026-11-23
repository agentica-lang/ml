"""Writes `fixtures/hf_mel.json`, the golden of `mel_test.agc`: values of
Hugging Face's Whisper log-mel for the first second of
`whisper/live/fixtures/es-hola.wav`, at 80 and at 128 mels.

Manual, not run by the tests: it needs `transformers` and `numpy` (the
extractor's numpy path). From `asr/`: `python3 test/hf_golden.py > test/fixtures/hf_mel.json`.
"""

import json
import wave

import numpy as np
from transformers import WhisperFeatureExtractor

WAV = "../whisper/live/fixtures/es-hola.wav"
SAMPLES = 16000
FRAMES = range(0, 100, 5)


def main():
    with wave.open(WAV) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = np.frombuffer(w.readframes(SAMPLES), dtype="<i2").astype(np.float32) / 32768.0
    out = {"wav": WAV, "samples": SAMPLES}
    for mels in (80, 128):
        fe = WhisperFeatureExtractor(feature_size=mels)
        feats = fe(pcm, sampling_rate=16000, return_tensors="np").input_features[0]
        assert feats.shape == (mels, 3000)
        rows = [0, mels // 4, mels // 2, 3 * mels // 4, mels - 1]
        out[str(mels)] = [[m, f, round(float(feats[m, f]), 7)] for m in rows for f in FRAMES]
    print(json.dumps(out))


main()
