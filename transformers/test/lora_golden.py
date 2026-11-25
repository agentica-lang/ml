"""The PyTorch + PEFT peer of `lora_test.agc`: LoRA adapters on the tiny models that the standard-library peers
write, loaded by Hugging Face's own classes.

- `qwen_peer.py`'s `untied` as `AutoModelForCausalLM` (`Qwen3ForCausalLM`), with an adapter on `q_proj` and
  `v_proj` (r 2, alpha 4): the logits of `IDS` with the adapter, and with it merged (`merge_and_unload`).
- `distilbert_peer.py`'s `tiny` as `DistilBertModel`, with an adapter on `q_lin` and `v_lin`: the last hidden state
  of `IDS`.

Every `A` and `B` is drawn from a seed, so `B` is not zero. It writes each adapter as PEFT does
(`PeftModel.save_pretrained`, with `target_modules` sorted and `base_model_name_or_path`, a temporary folder, set to
null, so that the file is the same on every run) into
`test/fixtures/lora_qwen/` and `test/fixtures/lora_distilbert/`, and the outputs into `test/fixtures/lora_golden.json`.

Manual, not run by the tests: it needs `torch`, `transformers` and `peft` (torch 2.6.0+cpu, transformers 4.57.6 and
peft 0.21.2 were used, with accelerate 1.15.0 and psutil 7.2.2, which peft imports). Run it from `transformers/`, in
Python's isolated mode:

    python -I test/lora_golden.py
"""

import json
import os
import shutil
import subprocess
import sys

import peft
import torch
from transformers import AutoModelForCausalLM, DistilBertModel

R = 2
ALPHA = 4
FIXTURES = "test/fixtures"


def peer(script: str) -> tuple[str, dict]:
    """The folder that the standard-library peer `script` writes, and its `expected.json`."""
    folder = subprocess.run(
        [sys.executable, "-I", script], check=True, capture_output=True, text=True
    ).stdout.strip().split("\n")[-1]
    with open(f"{folder}/expected.json", encoding="utf-8") as f:
        return folder, json.load(f)


def adapted(model: torch.nn.Module, targets: list[str], task: str | None, seed: int) -> peft.PeftModel:
    """`model` with an adapter on `targets` whose `A` and `B` are drawn from `seed`."""
    config = peft.LoraConfig(r=R, lora_alpha=ALPHA, target_modules=targets, task_type=task)
    with_adapter = peft.get_peft_model(model, config).eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(with_adapter.named_parameters()):
            if "lora_" in name:
                p.copy_(torch.randn(p.shape) * 0.3)
    return with_adapter


def saved(model: peft.PeftModel, name: str) -> None:
    folder = f"{FIXTURES}/{name}"
    model.save_pretrained(folder)
    os.remove(f"{folder}/README.md")
    path = f"{folder}/adapter_config.json"
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config["target_modules"] = sorted(config["target_modules"])
    config["base_model_name_or_path"] = None
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, sort_keys=True))


def flat(t: torch.Tensor) -> list[float]:
    return [float(v) for v in t.detach().reshape(-1)]


def main() -> None:
    out = {}
    folder, expected = peer("test/qwen_peer.py")
    try:
        base = AutoModelForCausalLM.from_pretrained(f"{folder}/untied", dtype=torch.float32)
        ids = torch.tensor(expected["ids"])
        with torch.no_grad():
            plain = base(ids).logits
            model = adapted(base, ["q_proj", "v_proj"], "CAUSAL_LM", 137)
            logits = model(ids).logits
            saved(model, "lora_qwen")
            merged = model.merge_and_unload()(ids).logits
        out["qwen"] = {
            "ids": expected["ids"],
            "logits": flat(logits),
            "merged": flat(merged),
            "moved": float((logits - plain).abs().max()),
        }
    finally:
        shutil.rmtree(folder)

    folder, expected = peer("test/distilbert_peer.py")
    try:
        base = DistilBertModel.from_pretrained(f"{folder}/tiny", dtype=torch.float32)
        seq = max(len(row) for row in expected["ids"])
        ids = torch.tensor([row + [0] * (seq - len(row)) for row in expected["ids"]])
        mask = (ids != 0).long()
        with torch.no_grad():
            plain = base(input_ids=ids, attention_mask=mask).last_hidden_state
            model = adapted(base, ["q_lin", "v_lin"], None, 138)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            saved(model, "lora_distilbert")
        out["distilbert"] = {
            "ids": expected["ids"],
            "hidden": flat(hidden),
            "moved": float((hidden - plain).abs().max()),
        }
    finally:
        shutil.rmtree(folder)

    with open(f"{FIXTURES}/lora_golden.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(out))


if __name__ == "__main__":
    main()
