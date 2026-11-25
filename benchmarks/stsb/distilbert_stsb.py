#!/usr/bin/env python3
"""The PyTorch peer of `transformers/tools/distilbert_stsb.agc` (ml#182): the same fine-tune of a sentence-transformers
DistilBERT on scored sentence pairs, on the CPU, with the same rules. Never run in CI.

    python -I distilbert_stsb.py \
        MODEL_DIR TRAIN_JSONL VALIDATION_JSONL [PAIRS [BATCH [PAD_TO [WEIGHT_DECAY [ADAPTER_DIR]]]]]

The rules: the first PAIRS pairs (1000) in order, BATCH pairs a step (16), one epoch; the checkpoint's tokenizer with a
limit of 128 tokens; both sentences of a batch go through the model as one batch of `2 BATCH` rows, padded to the
longest of them rounded up to a multiple of PAD_TO (1: the longest); mean pooling by hand (no sentence-transformers),
and the mean squared gap of the cosine of a pair from its score; `torch.optim.AdamW` (lr 2e-5, betas 0.9 and 0.999, eps
1e-8, weight decay WEIGHT_DECAY (0.01), none on a bias or a layer-norm weight: two parameter groups), no warmup, no
clipping, no dropout (`eval()`, with gradients); float32; `torch.set_num_threads` of the CPUs of the process, which
`run.sh` checks are every online CPU, as ggml takes them.

With ADAPTER_DIR the run is a LoRA fine-tune (ml#137) with PEFT: the model is frozen and the adapter of that
folder, which the Agentica side wrote before its first step (`Lora.save`: `q_lin` and `v_lin`, r 8, alpha 16, `B`
zero), is loaded with `PeftModel.from_pretrained(is_trainable=True)` and trained, every tensor decayed, lr 2e-4. So
both sides start from the same tensors. It needs `peft`, which the other run does not import.

It prints the JSON lines of the Agentica side: `checks` (how many weights decay, a SHA-256 of their names and one of
every token id), `spearman` before and after, and one `step` per step with its loss, its wall time in ms (the batch,
forward, backward, update and the loss read back; `batch_ms` of it is building the batch), the padded length and
`new_shape`, true for the first step of a shape: PyTorch compiles nothing, but the median leaves the same steps out on
both sides. It also prints `threads`, the count PyTorch runs with.

The venv of `run.sh setup` (every wheel `--only-binary :all:`): Python 3.13.5, torch 2.6.0+cpu, transformers 4.57.6,
tokenizers 0.22.1, huggingface_hub 0.36.0, safetensors 0.6.2, numpy 2.2.6, regex 2025.9.18. Only `model.safetensors`
is read, never a pickle.
"""

import hashlib
import json
import math
import os
import sys
import time

import torch
from tokenizers import Tokenizer
from transformers import DistilBertModel

MAX_LENGTH = 128
LR = 2e-5
LORA_LR = 2e-4
EMBED_BATCH = 32
COSINE_EPS = 1e-16
MIN_COUNT = 1e-9


def tokenized(tok, path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pairs.append((tok.encode(row["sentence1"]).ids, tok.encode(row["sentence2"]).ids, row["score"]))
    return pairs


def padded(rows, seq, pad):
    ids = torch.tensor([r + [pad] * (seq - len(r)) for r in rows], dtype=torch.long)
    return ids, (ids != pad).to(torch.float32)


def pooled(model, ids, mask):
    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    total = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    return total / mask.sum(dim=1, keepdim=True).clamp(min=MIN_COUNT)


def cosine_loss(a, b, scores):
    dot, aa, bb = (a * b).sum(dim=1), (a * a).sum(dim=1), (b * b).sum(dim=1)
    return ((dot / torch.sqrt(aa * bb + COSINE_EPS) - scores) ** 2).mean()


def no_decay(name):
    return name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def checks(decayed, pairs):
    """What both sides must agree on before a loss means anything: the names that decay and every token id."""
    decayed = sorted(decayed)
    ids = "".join(f"{','.join(map(str, left))}\n{','.join(map(str, right))}\n" for left, right, _ in pairs)
    names = digest("\n".join(decayed))
    return {"kind": "checks", "decayed": len(decayed), "decayed_names": names, "token_ids": digest(ids)}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    at = 0
    while at < len(order):
        stop = at
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[at]]:
            stop += 1
        for k in range(at, stop + 1):
            out[order[k]] = (at + stop) / 2 + 1
        at = stop + 1
    return out


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    both = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return both / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def spearman(model, pairs, pad):
    sentences = [s for left, right, _ in pairs for s in (left, right)]
    units = []
    with torch.no_grad():
        for at in range(0, len(sentences), EMBED_BATCH):
            part = sentences[at : at + EMBED_BATCH]
            ids, mask = padded(part, max(len(s) for s in part), pad)
            units.append(torch.nn.functional.normalize(pooled(model, ids, mask), dim=1))
    unit = torch.cat(units)
    cosines = (unit[0::2] * unit[1::2]).sum(dim=1).tolist()
    return pearson(ranks(cosines), ranks([score for _, _, score in pairs]))


def number(i, default):
    return int(sys.argv[i]) if len(sys.argv) > i else default


def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: MODEL_DIR TRAIN_JSONL VALIDATION_JSONL [PAIRS [BATCH [PAD_TO [WEIGHT_DECAY [ADAPTER_DIR]]]]]"
        )
    path = sys.argv[1]
    count, batch, pad_to = number(4, 1000), number(5, 16), number(6, 1)
    weight_decay = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    print(json.dumps({"kind": "threads", "value": torch.get_num_threads()}), flush=True)
    tok = Tokenizer.from_file(f"{path}/tokenizer.json")
    tok.enable_truncation(max_length=MAX_LENGTH)
    tok.no_padding()
    model = DistilBertModel.from_pretrained(
        path, use_safetensors=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    pad = model.config.pad_token_id
    train = tokenized(tok, sys.argv[2])[:count]
    if len(train) < count:
        raise SystemExit(f"{sys.argv[2]} has {len(train)} pairs, fewer than {count}")
    validation = tokenized(tok, sys.argv[3])
    if len(sys.argv) > 8:
        import peft

        model = peft.PeftModel.from_pretrained(model, sys.argv[8], is_trainable=True).eval()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        # PEFT's names in the model, as in its file: no `base_model.model.` and no adapter name.
        decayed = [n.removeprefix("base_model.model.").replace(".default.", ".") for n, _ in named]
        groups, lr = [{"params": [p for _, p in named], "weight_decay": weight_decay}], LORA_LR
    else:
        named = list(model.named_parameters())
        decayed = [n for n, _ in named if not no_decay(n)]
        groups, lr = [
            {"params": [p for n, p in named if not no_decay(n)], "weight_decay": weight_decay},
            {"params": [p for n, p in named if no_decay(n)], "weight_decay": 0.0},
        ], LR
    print(json.dumps(checks(decayed, train + validation)), flush=True)
    print(json.dumps({"kind": "spearman", "when": "before", "value": spearman(model, validation, pad)}), flush=True)
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    seen = set()
    for step, at in enumerate(range(0, len(train), batch), 1):
        picked = train[at : at + batch]
        longest = max(max(len(left), len(right)) for left, right, _ in picked)
        seq = (longest + pad_to - 1) // pad_to * pad_to
        started = time.perf_counter_ns()
        ids, mask = padded([s for left, right, _ in picked for s in (left, right)], seq, pad)
        scores = torch.tensor([score for _, _, score in picked], dtype=torch.float32)
        batch_ms = (time.perf_counter_ns() - started) / 1e6
        optimizer.zero_grad()
        sides = pooled(model, ids, mask)
        loss = cosine_loss(sides[0::2], sides[1::2], scores)
        loss.backward()
        optimizer.step()
        value = loss.item()
        ms = (time.perf_counter_ns() - started) / 1e6
        shape = (len(picked), seq)
        line = {"kind": "step", "step": step, "loss": value, "ms": ms, "batch_ms": batch_ms, "length": seq}
        print(json.dumps(line | {"new_shape": shape not in seen}), flush=True)
        seen.add(shape)
    print(json.dumps({"kind": "spearman", "when": "after", "value": spearman(model, validation, pad)}), flush=True)


if __name__ == "__main__":
    main()
