"""The PyTorch peer of `schedule_test.agc`: the learning rate of `torch.optim.lr_scheduler.LambdaLR` with the schedules
of Hugging Face's `Trainer`, and two short runs of a one-hidden-layer MLP on fixed rows, trained with the schedule
(SGD with a linear warmup, Adam with a cosine one). The factor of a cosine schedule stays 0 after `total`, which is
the one place where this peer differs from Hugging Face's lambda. It prints the factors, the starting weights, the rows,
the mean loss of every epoch and the weights at the end, in `float32`.

Manual, not run by the tests: it needs `torch` (2.6.0+cpu was used). Run it from `train/`, in Python's isolated mode:

    python -I test/schedule_golden.py > test/fixtures/schedule_golden.json
"""

import json
import math

import torch

EPOCHS = 3
BATCH = 2
ROWS = 8
STEPS = 16


def linear(warmup, total):
    def factor(t):
        if t < warmup:
            return t / max(1, warmup)
        return max(0.0, (total - t) / max(1, total - warmup))

    return factor


def cosine(warmup, total):
    def factor(t):
        if t < warmup:
            return t / max(1, warmup)
        progress = min(1.0, (t - warmup) / max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return factor


def rates(factor, base):
    p = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([p], lr=base)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    out = []
    for _ in range(STEPS):
        out.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    return out


def forward(w, x):
    return torch.relu(x @ w[0] + w[1]) @ w[2] + w[3]


def run(start, inputs, targets, make, factor):
    weights = [p.clone().requires_grad_() for p in start]
    optimizer = make(weights)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    history = []
    for _ in range(EPOCHS):
        total = 0.0
        for at in range(0, ROWS, BATCH):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(forward(weights, inputs[at : at + BATCH]), targets[at : at + BATCH])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss)
        history.append(total / (ROWS // BATCH))
    return {"history": history, "trained": [[float(v) for v in w.detach().flatten()] for w in weights]}


def main():
    torch.manual_seed(190)
    torch.set_num_threads(1)
    shapes = [[3, 4], [4], [4, 2], [2]]
    start = [torch.randn(*s) * 0.5 for s in shapes]
    inputs = torch.randn(ROWS, 3)
    targets = torch.randn(ROWS, 2)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "linear": rates(linear(2, 10), 0.1),
                "cosine": rates(cosine(3, 12), 0.1),
                "linear_no_warmup": rates(linear(0, 4), 0.1),
                "cosine_all_warmup": rates(cosine(4, 4), 0.1),
                "shapes": shapes,
                "start": [[float(v) for v in w.flatten()] for w in start],
                "inputs": [float(v) for v in inputs.flatten()],
                "targets": [float(v) for v in targets.flatten()],
                "sgd_linear": run(start, inputs, targets, lambda ps: torch.optim.SGD(ps, lr=0.1), linear(2, 10)),
                "adam_cosine": run(start, inputs, targets, lambda ps: torch.optim.Adam(ps, lr=0.05), cosine(3, 12)),
            },
            indent=1,
        )
    )


main()
