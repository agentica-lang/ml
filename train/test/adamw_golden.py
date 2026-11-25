"""The PyTorch peer of `adamw_test.agc`: a short run of a one-hidden-layer MLP on fixed rows with `torch.optim.AdamW`
in two parameter groups, as a fine-tune sets them: the weights decay and the biases do not. One run has a constant
learning rate and one a linear warmup (`LambdaLR`, the lambda of Hugging Face's `Trainer`). It prints the starting
weights, the rows, the mean loss of every epoch and the weights at the end, in `float32`.

Manual, not run by the tests: it needs `torch` (2.6.0+cpu was used). Run it from `train/`, in Python's isolated mode:

    python -I test/adamw_golden.py > test/fixtures/adamw_golden.json
"""

import json

import torch

EPOCHS = 3
BATCH = 2
ROWS = 8
LR = 0.05
DECAY = 0.3


def linear(warmup, total):
    def factor(t):
        if t < warmup:
            return t / max(1, warmup)
        return max(0.0, (total - t) / max(1, total - warmup))

    return factor


def forward(w, x):
    return torch.relu(x @ w[0] + w[1]) @ w[2] + w[3]


def run(start, inputs, targets, factor):
    weights = [p.clone().requires_grad_() for p in start]
    optimizer = torch.optim.AdamW(
        [
            {"params": [weights[0], weights[2]], "weight_decay": DECAY},
            {"params": [weights[1], weights[3]], "weight_decay": 0.0},
        ],
        lr=LR,
    )
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
    torch.manual_seed(182)
    torch.set_num_threads(1)
    shapes = [[3, 4], [4], [4, 2], [2]]
    start = [torch.randn(*s) * 0.5 for s in shapes]
    inputs = torch.randn(ROWS, 3)
    targets = torch.randn(ROWS, 2)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "shapes": shapes,
                "start": [[float(v) for v in w.flatten()] for w in start],
                "inputs": [float(v) for v in inputs.flatten()],
                "targets": [float(v) for v in targets.flatten()],
                "constant": run(start, inputs, targets, lambda t: 1.0),
                "warmup": run(start, inputs, targets, linear(2, 10)),
            },
            indent=1,
        )
    )


main()
