---
title: train
description: Losses, SGD and Adam, and a training loop over the gradients of tensor.
weight: 3
---

`train` has `mse`, binary and softmax cross-entropy, the SGD and Adam optimizers, and `Train.fit`, which runs a loop over `Trace.value_and_grad`. It depends on `tensor`; `nn` is used by its tests.
