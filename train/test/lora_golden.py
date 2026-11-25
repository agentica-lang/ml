"""The PyTorch + PEFT peer of `lora_test.agc`: a two-layer MLP whose two `Linear` layers get a LoRA adapter from
`peft.get_peft_model` (r 2, alpha 4, no dropout). With fixed base weights, a fixed adapter whose `B` is not zero, and
fixed rows, it prints the output of the adapted model, the loss, PEFT's gradients of every `A` and `B`, and a short
run of `torch.optim.AdamW` on the adapter alone (the loss of every step and the adapter at the end), in `float32`.

It also saves that adapter as PEFT does, with a `lora_dropout` of 0.1 in its config, into `test/fixtures/lora_peft/`
(`adapter_config.json`, with its `target_modules` sorted so that the file is the same on every run, and
`adapter_model.safetensors`), which `lora_peft_test.agc` reads. With `check DIR` it does
the other way: it loads the adapter folder DIR, which `Lora.save` wrote from that same adapter
(`agentica tools/lora_save.agc DIR`), with `PeftModel.from_pretrained` and compares the model's output.

Manual, not run by the tests: it needs `torch` and `peft` (torch 2.6.0+cpu and peft 0.21.2 were used, with accelerate
1.15.0 and psutil 7.2.2, which peft imports). Run it from `train/`, in Python's isolated mode:

    python -I test/lora_golden.py > test/fixtures/lora_golden.json
    python -I test/lora_golden.py check DIR
"""

import json
import os
import sys

import peft
import torch

R = 2
ALPHA = 4
STEPS = 5
LR = 0.05
LAYERS = ("first", "second")
SAVED = "test/fixtures/lora_peft"


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 4)
        self.second = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


def flat(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.detach().reshape(-1)]


def adapter(model: peft.PeftModel) -> list[torch.Tensor]:
    """`A` then `B` of each layer, in the order `Lora.params` gives them."""
    layers = [getattr(model.base_model.model, name) for name in LAYERS]
    return [m.weight for layer in layers for m in (layer.lora_A["default"], layer.lora_B["default"])]


def fresh() -> dict:
    """What PEFT starts a layer with: A's largest value and whether B is all zero (64 inputs, r 8)."""
    torch.manual_seed(5)
    config = peft.LoraConfig(r=8, target_modules=["0"])
    model = peft.get_peft_model(torch.nn.Sequential(torch.nn.Linear(64, 32)), config)
    a, b = adapter_of(model)
    return {"inputs": 64, "r": 8, "a_max": float(a.abs().max()), "b_zero": bool((b == 0).all())}


def adapter_of(model: peft.PeftModel) -> list[torch.Tensor]:
    layer = model.base_model.model[0]
    return [layer.lora_A["default"].weight, layer.lora_B["default"].weight]


def saved(start: list[torch.Tensor], inputs: torch.Tensor, output: torch.Tensor) -> None:
    """The adapter `start` as PEFT writes it, from a config with dropout, which a run does not apply."""
    torch.manual_seed(137)
    config = peft.LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.1, target_modules=list(LAYERS))
    model = peft.get_peft_model(Mlp(), config).eval()
    with torch.no_grad():
        for p, value in zip(adapter(model), start):
            p.copy_(value)
        if not torch.equal(model(inputs), output):
            raise SystemExit("the saved adapter does not give the output of the run")
    model.save_pretrained(SAVED)
    os.remove(f"{SAVED}/README.md")
    # PEFT keeps the targets in a set, whose order changes from run to run: sorted, the file is the same every time.
    path = f"{SAVED}/adapter_config.json"
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config["target_modules"] = sorted(config["target_modules"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, sort_keys=True))


def check(folder: str) -> None:
    """Loads the adapter of `folder` into the base model with PEFT and compares its output with the fixture's."""
    with open("test/fixtures/lora_golden.json", encoding="utf-8") as f:
        golden = json.load(f)
    torch.manual_seed(137)
    model = peft.PeftModel.from_pretrained(Mlp(), folder).eval()
    loaded = [flat(p) for p in adapter(model)]
    if loaded != golden["start"]:
        raise SystemExit("PEFT did not load the tensors of the adapter")
    with torch.no_grad():
        output = model(torch.tensor(golden["inputs"]).reshape(6, 3))
    gap = max(abs(a - b) for a, b in zip(flat(output), golden["output"]))
    if gap > 1e-6:
        raise SystemExit(f"the output differs by {gap}")
    print(f"ok: PEFT loads {folder} and gives the output of the fixture (largest gap {gap:.1e})")


def main() -> None:
    if len(sys.argv) > 1:
        if sys.argv[1:2] != ["check"] or len(sys.argv) != 3:
            raise SystemExit("usage: lora_golden.py [check DIR]")
        check(sys.argv[2])
        return
    torch.manual_seed(137)
    base = Mlp()
    weights = {name: flat(p) for name, p in base.named_parameters()}
    shapes = {name: list(p.shape) for name, p in base.named_parameters()}
    model = peft.get_peft_model(base, peft.LoraConfig(r=R, lora_alpha=ALPHA, target_modules=list(LAYERS)))
    with torch.no_grad():
        for p in adapter(model):
            p.copy_(torch.randn(p.shape) * 0.3)
    start = [flat(p) for p in adapter(model)]
    kept = [p.detach().clone() for p in adapter(model)]
    inputs, targets = torch.randn(6, 3), torch.randn(6, 2)

    output = model(inputs)
    loss = torch.nn.functional.mse_loss(output, targets)
    loss.backward()
    saved(kept, inputs, output.detach())
    gradients = [flat(p.grad) for p in adapter(model)]

    optimizer = torch.optim.AdamW(adapter(model), lr=LR, weight_decay=0.0)
    history = []
    for _ in range(STEPS):
        optimizer.zero_grad()
        step = torch.nn.functional.mse_loss(model(inputs), targets)
        step.backward()
        optimizer.step()
        history.append(float(step))

    print(
        json.dumps(
            {
                "shapes": shapes,
                "weights": weights,
                "adapter_shapes": [list(p.shape) for p in adapter(model)],
                "start": start,
                "inputs": flat(inputs),
                "targets": flat(targets),
                "output": flat(output),
                "loss": float(loss),
                "gradients": gradients,
                "history": history,
                "trained": [flat(p) for p in adapter(model)],
                "fresh": fresh(),
            }
        )
    )


if __name__ == "__main__":
    main()
