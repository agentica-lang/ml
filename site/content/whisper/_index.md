---
title: whisper
description: A binding over the pinned whisper.cpp engine, with WAV reading, microphone capture and long-audio chunking.
weight: 6
---

`whisper` transcribes 16 kHz mono audio with whisper.cpp, which is a pinned git submodule. It needs a C/C++ compiler and cmake to build the engine, and a ggml model that its script downloads.
