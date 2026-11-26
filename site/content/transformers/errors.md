---
title: errors
description: What the tokenizers refuse instead of silently giving other ids than Hugging Face's.
weight: 4
---

## Errors

A file with another normalizer, pre-tokenizer, post-processor or template, decoder or model, with `truncation`, `padding` or BPE `dropout`, or with an added token that matches normalized text or by words, is an error from `open` and `parse`, not a silent difference. Ids out of range are an error from `decode`.
