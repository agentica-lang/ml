#!/usr/bin/env python3
# One training step of the MLP of train/tools/bench_train.agc (784-512-512-10, relu, softmax cross-entropy) in JAX, the
# peer of benchmarks/training/README.md. Never run in CI.
#
#   python bench_jax.py OPTIMIZER BATCH [STEPS]
#
# Prints one JSON line like bench_train.agc. The whole step (loss, gradients, update) is one `jax.jit`.
#
# Pinned (venv of `run.sh setup`, installed with --only-binary :all:): Python 3.13.5, jax 0.12.0.1, jaxlib 0.12.0.1 and, as installed on the run date,
# numpy 2.5.5, ml_dtypes 0.7.0, scipy 1.18.1, opt_einsum 3.4.0. Threads: XLA's own default for the CPU (all the cores);
# its pool size is not checked here.
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

SIZES = [784, 512, 512, 10]
CLASSES = 10
WARM = 3
LOSSES = 10
SGD_LR = 0.01
LR, B1, B2, EPS = 0.001, 0.9, 0.999, 1e-8


def values(n, seed):
    i = np.arange(n, dtype=np.int64)
    return ((i * (2 * seed + 7) + seed) % 1009).astype(np.float64) / 504.5 - 1.0


def init():
    params = []
    for i in range(len(SIZES) - 1):
        scale = 1.0 / SIZES[i] ** 0.5
        w = (values(SIZES[i] * SIZES[i + 1], 10 + i) * scale).reshape(SIZES[i], SIZES[i + 1])
        b = values(SIZES[i + 1], 20 + i) * 0.01
        params += [jnp.asarray(w, dtype=jnp.float32), jnp.asarray(b, dtype=jnp.float32)]
    return params


def loss_fn(params, x, y):
    h = x
    layers = len(SIZES) - 1
    for i in range(layers):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < layers - 1:
            h = jax.nn.relu(h)
    logp = jax.nn.log_softmax(h)
    return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=1))


def sgd_step(params, first, second, scalars, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    return loss, [p - SGD_LR * g for p, g in zip(params, grads)], first, second


def adam_step(params, first, second, scalars, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    lr_c, c2 = scalars[0], scalars[1]
    m = [B1 * a + (1.0 - B1) * g for a, g in zip(first, grads)]
    v = [B2 * a + (1.0 - B2) * g * g for a, g in zip(second, grads)]
    new = [p - (lr_c * a) / (jnp.sqrt(b * c2) + EPS) for p, a, b in zip(params, m, v)]
    return loss, new, m, v


def main():
    optimizer, n = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    step_fn = jax.jit(sgd_step if optimizer == "sgd" else adam_step)
    params = init()
    first = [jnp.zeros_like(p) for p in params]
    second = [jnp.zeros_like(p) for p in params]
    x = jnp.asarray(values(n * SIZES[0], 1).reshape(n, SIZES[0]), dtype=jnp.float32)
    y = jnp.asarray((np.arange(n, dtype=np.int64) * 7) % CLASSES, dtype=jnp.int32)

    losses = []
    times = []
    first_ns = 0
    for i in range(1 + WARM + steps):
        t0 = time.perf_counter_ns()
        loss, params, first, second = step_fn(params, first, second, jnp.asarray([LR / (1.0 - B1 ** (i + 1)), 1.0 / (1.0 - B2 ** (i + 1))], dtype=jnp.float32), x, y)
        losses.append(float(loss))
        t1 = time.perf_counter_ns()
        if i == 0:
            first_ns = t1 - t0
        elif i > WARM:
            times.append(t1 - t0)

    times.sort()
    ms = lambda ns: round(ns / 1e6, 3)
    print(json.dumps({
        "first_ms": ms(first_ns),
        "median_ms": ms(times[len(times) // 2]),
        "min_ms": ms(times[0]),
        "max_ms": ms(times[-1]),
        "losses": [round(v, 7) for v in losses[:LOSSES]],
    }))


if __name__ == "__main__":
    main()
