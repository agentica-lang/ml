---
title: asr
description: Whisper's model forward over tensor, nn and hub, loaded from a Hugging Face checkpoint.
weight: 5
---

`asr` is the Whisper model written in Agentica: the conv stem, the encoder, the decoder and the tied logits, for inference. It depends on `tensor`, `nn` and `hub`.

It also has the log-mel frontend, which turns 16 kHz samples into the spectrogram the encoder takes, and the tokenizer, which turns the decoder's token ids back into text and builds the prompt that starts it. The frontend uses `mathx`, a small native module over the C math library, until Agentica's `Math` has `exp`, `log` and the rest.
