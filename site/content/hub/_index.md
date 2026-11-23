---
title: hub
description: Safetensors checkpoints read in Agentica, and nn layers loaded from them by name.
weight: 4
---

`hub` opens a `.safetensors` file or a sharded checkpoint with its index, and loads `nn` layers from it by tensor name. It reads local files only. It depends on `tensor` and `nn`.
