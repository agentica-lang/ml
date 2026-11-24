"""The PyTorch peer of `freeze_test.agc`: a small MLP whose first layer is frozen (`requires_grad = False`), trained by
Adam on fixed rows, as `Train.fit` trains the same split. It prints the starting weights, the rows, the loss of every
epoch and the weights at the end, in `float32`.

Manual, not run by the tests: it needs `torch` (2.6.0+cpu was used). Run it from `train/`, in Python's isolated mode:

    python -I test/freeze_golden.py > test/fixtures/freeze_golden.json
"""

import json

import torch

EPOCHS = 3
BATCH = 4
ROWS = 8
FROZEN = "body.0."


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU()
        )
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.body(x))


def named(model):
    return {
        name: {"shape": list(p.shape), "values": [float(v) for v in p.detach().flatten()]}
        for name, p in model.named_parameters()
    }


def main():
    torch.manual_seed(136)
    torch.set_num_threads(1)
    model = Model()
    inputs = torch.randn(ROWS, 3)
    targets = torch.randn(ROWS, 2)
    start = named(model)
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(FROZEN)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=0.01)
    history = []
    for _ in range(EPOCHS):
        total = 0.0
        for at in range(0, ROWS, BATCH):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs[at : at + BATCH]), targets[at : at + BATCH])
            loss.backward()
            optimizer.step()
            total += float(loss)
        history.append(total / (ROWS // BATCH))
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "frozen": FROZEN,
                "state": [name for name, p in model.named_parameters() if p.requires_grad],
                "inputs": [float(v) for v in inputs.flatten()],
                "targets": [float(v) for v in targets.flatten()],
                "start": start,
                "history": history,
                "trained": named(model),
            },
            indent=1,
        )
    )


main()
