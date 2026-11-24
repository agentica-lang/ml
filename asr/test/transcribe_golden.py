"""Prints what Hugging Face's `WhisperForConditionalGeneration.generate`
(greedy, temperature 0) answers for whisper-tiny on two clips, the content of
`test/fixtures/transcribe_golden.json`, for `tools/transcribe_check.agc` to
compare. Per case: the prompt and the ids after it (without `eot`; for the
two long clips, over 30 s, the ids of the segments of all the windows, as
long-form `generate` returns them), the text,
the detected language with its probability (the softmax over the language
tokens one step after `<|startoftranscript|>`) and the segments, made from
the timestamp tokens of the ids (50 a second) around Hugging Face's decoded
text.

Manual, not run by the tests: it needs `torch` and `transformers`
(4.57.6 was used) and the checkpoint. Run it in a throwaway venv, from
`asr/`, with Python's isolated mode:

    python -I test/transcribe_golden.py MODEL_DIR ES_HOLA_WAV JFK_WAV \
        > test/fixtures/transcribe_golden.json

MODEL_DIR holds `openai/whisper-tiny` (`config.json`, `model.safetensors`,
`tokenizer.json`, `preprocessor_config.json`, `generation_config.json`).
"""

import json
import sys
import wave

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

# (clip, language, task, timestamps); a language of None is detected.
CASES = [
    ("es-hola", None, "transcribe", True),
    ("es-hola", None, "transcribe", False),
    ("es-hola", "es", "translate", True),
    ("jfk", None, "transcribe", True),
    ("jfk", "en", "transcribe", False),
    ("jfk-long", None, "transcribe", True),
    ("es-long", None, "transcribe", True),
]


def pcm(path):
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def silence(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def long_clips(es_hola, jfk):
    """The two long clips, built from the short ones: jfk six times with gaps
    of silence (73.5 s), and es-hola 24 times with half a second between
    (75.5 s). `tools/transcribe_check.agc` builds them the same way."""
    parts = [jfk]
    for gap in [1.0, 3.0, 0.5, 2.0, 1.0]:
        parts += [silence(gap), jfk]
    es = []
    for _ in range(24):
        es += [es_hola, silence(0.5)]
    return {"jfk-long": np.concatenate(parts), "es-long": np.concatenate(es)}


def segments(tok, ids, begin, duration):
    """A timestamp opens a segment and the next one closes it; text with no
    timestamps is one segment from 0 to the end of the clip."""
    out, start, text = [], None, []
    for i in ids:
        if i < begin:
            text.append(i)
            continue
        if text:
            out.append({"start": start or 0.0, "stop": (i - begin) / 50, "text": tok.decode(text)})
        start = None if start is not None else (i - begin) / 50
        text = []
    if text:
        out.append({"start": start or 0.0, "stop": duration, "text": tok.decode(text)})
    return out


def main(model_dir, es_hola, jfk):
    clips = {"es-hola": pcm(es_hola), "jfk": pcm(jfk)}
    clips.update(long_clips(clips["es-hola"], clips["jfk"]))
    features = WhisperFeatureExtractor.from_pretrained(model_dir)
    tok = WhisperTokenizerFast.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).eval()
    config = model.generation_config
    languages = sorted(config.lang_to_id.values())
    sot = config.decoder_start_token_id
    out = []
    for clip, language, task, timestamps in CASES:
        long = clip.endswith("-long")
        if long:
            # More than 30 s: the log-mel of the whole clip, with no padding;
            # `generate` cuts the 30 s windows out of it and seeks.
            extracted = features(
                clips[clip],
                sampling_rate=16000,
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                return_tensors="pt",
            )
            feats = extracted.input_features
        else:
            feats = features(clips[clip], sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            # Language detection as Hugging Face's `detect_language` does: one
            # step after <|startoftranscript|>, the argmax over the language
            # tokens; the probability is the softmax over those tokens.
            audio = model.model.encoder(feats[:, :, :3000]).last_hidden_state
            logits = model(encoder_outputs=(audio,), decoder_input_ids=torch.tensor([[sot]])).logits[0, -1]
            probs = torch.softmax(logits[languages].double(), dim=-1)
            best = int(torch.argmax(probs))
            kwargs = {"task": task, "return_timestamps": timestamps}
            if language is not None:
                kwargs["language"] = language
            if long:
                made = model.generate(
                    feats,
                    attention_mask=extracted.attention_mask,
                    return_segments=True,
                    num_beams=1,
                    do_sample=False,
                    **kwargs,
                )
                # The ids of the segments, in order, with the timestamps of
                # their own window and no prompt.
                ids = made["sequences"][0].tolist()
                spans = [
                    {
                        "start": float(s["start"]),
                        "stop": float(s["end"]),
                        "text": tok.decode(s["tokens"], skip_special_tokens=True),
                    }
                    for s in made["segments"][0]
                ]
            else:
                ids = model.generate(feats, num_beams=1, do_sample=False, **kwargs)[0].tolist()
        detected = tok.convert_ids_to_tokens(languages[best])[2:-2]
        lang_id = config.lang_to_id[f"<|{language}|>"] if language else languages[best]
        prompt = [sot, lang_id, config.task_to_id[task]] + ([] if timestamps else [config.no_timestamps_token_id])
        # Older versions of `generate` (4.46.3) answer the prompt too, newer
        # ones (4.57.6) only what follows it.
        generated = ids[len(prompt):] if ids[: len(prompt)] == prompt else ids
        out.append({
            "clip": clip,
            "language": language,
            "task": task,
            "timestamps": timestamps,
            "prompt": prompt,
            "tokens": generated,
            "text": tok.decode(generated, skip_special_tokens=True),
            "detected": detected,
            "probability": float(probs[best]),
            "segments": spans if long else segments(tok, generated, config.no_timestamps_token_id + 1, len(clips[clip]) / 16000),
        })
    rows = ",\n".join(json.dumps(c, ensure_ascii=False) for c in out)
    print('{"model": "openai/whisper-tiny", "cases": [\n%s\n]}' % rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
