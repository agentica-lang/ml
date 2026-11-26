---
title: transformers
description: BERT, Qwen3, Qwen3.5, Granite MoE, LFM3, Qwen2.5, ResNet, ViT and CLIP in Agentica from Hugging Face checkpoints, and the tokenizers that give Hugging Face's ids.
weight: 7
---

`transformers` is BERT, Qwen3, Qwen3.5 (text), Granite MoE, LFM3, Qwen2.5, ResNet, ViT and CLIP in Agentica, for inference. It has BERT (`bert-base-uncased` and its relatives), Qwen3 and Qwen2.5 (the forward, generation with a KV cache, and the chat template), ResNet-18 and ResNet-50 and ViT-B/16 (image classification), CLIP ViT-B/32 (image and text embeddings, zero-shot classification), and the tokenizers of BERT, Qwen and CLIP: they read the `tokenizer.json` that sits beside a Hugging Face checkpoint and give the same ids as Hugging Face `tokenizers`.

The sections are on their own pages:

- [Models](models.md): BERT, DistilBERT, ResNet, ViT, Qwen3, Qwen3.5, Granite MoE and LFM3.
- [Generation](generation.md): generation with a KV cache, the chat template, and zero-shot classification with CLIP.
- [Tokenizers](tokenizers.md): the tokenizer example, WordPiece, byte-level BPE, CLIP's BPE and the Unicode tables.
- [Errors](errors.md): what the tokenizers refuse instead of silently giving other ids.
