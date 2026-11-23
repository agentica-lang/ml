---
title: asr
description: Whisper in Agentica over tensor, nn and hub, from a Hugging Face checkpoint to a transcript.
weight: 5
---

`asr` is the Whisper model written in Agentica: the conv stem, the encoder, the decoder and the tied logits, for inference. It depends on `tensor`, `nn` and `hub`.

It also has the log-mel frontend, which turns 16 kHz samples into the spectrogram the encoder takes, and the tokenizer, which turns the decoder's token ids back into text and builds the prompt that starts it. The frontend uses `mathx`, a small native module over the C math library, until Agentica's `Math` has `exp`, `log` and the rest.

`Asr.transcribe` puts them together for up to 30 s of audio: the log-mel, the encoder, and greedy decoding with a KV cache and OpenAI's rules for the suppressed tokens and the timestamps. It detects the language when none is given, with its probability, translates to English on request, and returns the text, the token ids and the timestamped segments. On whisper-tiny it gives the same token ids as Hugging Face's `generate`. It runs on the reference backend and on libtorch.
