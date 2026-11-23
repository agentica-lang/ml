---
title: nn
description: Functional layers over tensor, forward only, with seeded initializers.
weight: 2
---

`nn` has dense, layer norm, embedding, convolution, attention, MLP, and transformer encoder and decoder blocks. A layer is a struct of parameters and a function `layer(params, x)`. It depends on `tensor` and runs on every backend `tensor` has.
