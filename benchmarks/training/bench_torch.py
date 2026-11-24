#!/usr/bin/env python3
# One training step of the MLP of train/tools/bench_train.agc (784-512-512-10, relu, softmax cross-entropy) in PyTorch,
# the peer of benchmarks/training/README.md. Never run in CI.
#
#   python bench_torch.py MODE OPTIMIZER BATCH [STEPS]
#
# MODE is `eager` (autograd and torch.optim), `compile` (torch.compile of the loss and of the update) or `oracle`
# (eager in float64, the reference of the loss curve). Prints one JSON line like bench_train.agc.
#
# Pinned: Python 3.13, torch 2.6.0+cpu (the venv of `run.sh setup`). Threads: OMP_NUM_THREADS and set_num_threads.
import json
import os
import sys
import time

import torch

SIZES = [784, 512, 512, 10]
CLASSES = 10
WARM = 3
LOSSES = 10
SGD_LR = 0.01
ADAM = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8}


def values(n, seed):
    i = torch.arange(n, dtype=torch.int64)
    return ((i * (2 * seed + 7) + seed) % 1009).double() / 504.5 - 1.0


def init(dtype):
    params = []
    for i in range(len(SIZES) - 1):
        scale = 1.0 / SIZES[i] ** 0.5
        w = (values(SIZES[i] * SIZES[i + 1], 10 + i) * scale).reshape(SIZES[i], SIZES[i + 1])
        b = values(SIZES[i + 1], 20 + i) * 0.01
        params += [w.to(torch.float32).to(dtype).requires_grad_(), b.to(torch.float32).to(dtype).requires_grad_()]
    return params


def batch(n, dtype):
    x = values(n * SIZES[0], 1).reshape(n, SIZES[0]).to(torch.float32).to(dtype)
    y = (torch.arange(n, dtype=torch.int64) * 7) % CLASSES
    return x, y


def loss_fn(params, x, y):
    h = x
    layers = len(SIZES) - 1
    for i in range(layers):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < layers - 1:
            h = torch.relu(h)
    return torch.nn.functional.cross_entropy(h, y)


def scalars(step):
    b1, b2 = ADAM["betas"]
    return torch.tensor([ADAM["lr"] / (1.0 - b1**step), 1.0 / (1.0 - b2**step)], dtype=torch.float32)


def update_sgd(params, grads, first, second, scalars):
    return [p - SGD_LR * g for p, g in zip(params, grads)], first, second


def update_adam(params, grads, first, second, scalars):
    b1, b2 = ADAM["betas"]
    lr_c, c2 = scalars[0], scalars[1]
    m = [b1 * a + (1.0 - b1) * g for a, g in zip(first, grads)]
    v = [b2 * a + (1.0 - b2) * g * g for a, g in zip(second, grads)]
    new = [p - (lr_c * a) / (torch.sqrt(b * c2) + ADAM["eps"]) for p, a, b in zip(params, m, v)]
    return new, m, v


def stepper(mode, optimizer, params):
    if mode == "compile":
        loss_c = torch.compile(loss_fn)
        update_c = torch.compile(update_sgd if optimizer == "sgd" else update_adam)
        state = {"m": [torch.zeros_like(p) for p in params], "v": [torch.zeros_like(p) for p in params], "t": 0}

        def step(x, y):
            loss = loss_c(params, x, y)
            grads = torch.autograd.grad(loss, params)
            state["t"] += 1
            with torch.no_grad():
                new, state["m"], state["v"] = update_c([p.detach() for p in params], list(grads), state["m"], state["v"], scalars(state["t"]))
                for p, n in zip(params, new):
                    p.copy_(n)
            return loss.item()

        return step

    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=SGD_LR)
    else:
        opt = torch.optim.Adam(params, **ADAM)

    def step(x, y):
        opt.zero_grad()
        loss = loss_fn(params, x, y)
        loss.backward()
        opt.step()
        return loss.item()

    return step


def main():
    mode, optimizer, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if threads:
        torch.set_num_threads(threads)
    dtype = torch.float64 if mode == "oracle" else torch.float32
    params = init(dtype)
    x, y = batch(n, dtype)
    step = stepper("eager" if mode == "oracle" else mode, optimizer, params)

    losses = []
    times = []
    first = 0
    for i in range(1 + WARM + steps):
        t0 = time.perf_counter_ns()
        losses.append(step(x, y))
        t1 = time.perf_counter_ns()
        if i == 0:
            first = t1 - t0
        elif i > WARM:
            times.append(t1 - t0)

    times.sort()
    ms = lambda ns: round(ns / 1e6, 3)
    print(json.dumps({
        "first_ms": ms(first),
        "median_ms": ms(times[len(times) // 2]),
        "min_ms": ms(times[0]),
        "max_ms": ms(times[-1]),
        "losses": [round(v, 7) for v in losses[:LOSSES]],
    }))


if __name__ == "__main__":
    main()
