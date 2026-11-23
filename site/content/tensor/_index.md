---
title: tensor
description: The tensor API, the backend interface, the pure Agentica reference backend, libtorch on the CPU, ggml on the CPU and Vulkan, tracing and gradients.
weight: 1
---

`tensor` has the public API (`Tensor.*`), 45 backend primitives and the ops derived from them (`softmax`, `layer_norm`, attention), and `Trace.value_and_grad` for gradients. Every op returns a `Result`. A tensor carries its backend, so the reference and libtorch can run side by side in one process.

It needs nothing native for the reference backend. libtorch 2.4.1 for the CPU is optional: without it everything still checks and tests, and `Backend.Torch` answers `error("backend not available")`. ggml, built from a pinned submodule, is optional the same way: `Backend.Ggml(Device.Cpu)` and `Backend.Ggml(Device.Vulkan(i))` create tensors and read them back, and run a traced function (`Trace.eval`, `Trace.value_and_grad`) as one compiled ggml graph, cached by its structure.

The guide covers the files, what libtorch implements, the setup and the cost. The design is the contract.
