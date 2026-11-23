---
title: tensor
description: The tensor API, the backend interface, the pure Agentica reference backend and libtorch on the CPU, tracing and gradients.
weight: 1
---

`tensor` has the public API (`Tensor.*`), 45 backend primitives and the ops derived from them (`softmax`, `layer_norm`, attention), and `Trace.value_and_grad` for gradients. Every op returns a `Result`. A tensor carries its backend, so the reference and libtorch can run side by side in one process.

It needs nothing native for the reference backend. libtorch 2.4.1 for the CPU is optional: without it everything still checks and tests, and `Backend.Torch` answers `error("backend not available")`.

The guide covers the files, what libtorch implements, the setup and the cost. The design is the contract.
