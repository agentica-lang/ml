"""Writes `test/fixtures/lfm3_golden.json`: a tiny random LFM3 model (5 layers: three `conv`, a `full_attention` and a
`sliding_attention` with a window of 4; 16 wide, a head of 16 and the real `partial_rotary_factor` 0.5 and thetas of
the checkpoint, tied head) as the official code runs it in `float32` with the eager attention: its `config.json` (as
the `LiquidAI/LFM3-*` repositories write it), its weights under the names of the official checkpoint, two sequences of 9
token ids and the logits of every position. `test/lfm3_test.agc` writes the weights as a safetensors file, opens the
folder with `Lfm3.open` and compares the logits. No model is stored, only this small file of random weights and its
outputs.

The code is `modeling_lfm3.py`, `configuration_lfm3.py` and `lfm3_conv.py` of `LiquidAI/LFM3-1.7B`, revision
1da06bec6a819da10b54c67e5c90972b46d81871, under the LFM Open License v1.0. It is not copied here. The script imports
those files and runs them as code (`trust_remote_code`): read them first, run in a throwaway venv with no network, and it
refuses files whose sha256 is not the one of that revision. Download them with:

    hf download LiquidAI/LFM3-1.7B modeling_lfm3.py configuration_lfm3.py lfm3_conv.py \\
        --revision 1da06bec6a819da10b54c67e5c90972b46d81871 --local-dir ~/.cache/agentica-ml/hf/lfm3-1.7b-code

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions)
and no `triton` or `causal-conv1d`. Run it from `transformers/`, with Python's isolated mode:

    python -I test/lfm3_golden.py ~/.cache/agentica-ml/hf/lfm3-1.7b-code > test/fixtures/lfm3_golden.json
"""

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys

import torch
import transformers

SHAPE = dict(
    hidden_size=16, intermediate_size=24, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, vocab_size=32, max_position_embeddings=64, norm_eps=1e-5, conv_L_cache=7, conv_bias=False,
    layer_types=["conv", "conv", "full_attention", "conv", "sliding_attention"], sliding_window=4,
    use_sandwich_norm=True, tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
)
ROPE = {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}
ROWS = [[3, 1, 4, 1, 5, 9, 2, 6, 5], [8, 9, 7, 9, 3, 2, 3, 8, 4]]
SHA256 = {
    "configuration_lfm3.py": "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5",
    "lfm3_conv.py": "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4",
    "modeling_lfm3.py": "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab",
}


def check_environment(folder):
    if not torch.__version__.startswith("2.15.0"):
        raise SystemExit(f"torch 2.15.0 is needed, got {torch.__version__}")
    if transformers.__version__ != "5.22.2":
        raise SystemExit(f"transformers 5.22.2 is needed, got {transformers.__version__}")
    for name in ("triton", "causal_conv1d"):
        if importlib.util.find_spec(name) is not None:
            raise SystemExit(f"{name} is installed: its kernel is not the reference")
    for name, digest in SHA256.items():
        with open(f"{folder}/{name}", "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                raise SystemExit(f"{name} is not the file of the pinned revision")


def load_remote(folder):
    spec = importlib.machinery.ModuleSpec("lfm3_code", None, is_package=True)
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [folder]
    sys.modules["lfm3_code"] = package
    return (importlib.import_module("lfm3_code.configuration_lfm3"), importlib.import_module("lfm3_code.modeling_lfm3"))


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment(sys.argv[1])
    configuration, modeling = load_remote(sys.argv[1])
    torch.manual_seed(7)
    config = configuration.Lfm3Config(**SHAPE, rope_parameters=ROPE, attn_implementation="eager")
    model = modeling.Lfm3ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn_like(p) * (0.2 if "norm" in name else 0.4)
            p.copy_(((1.0 + noise if "norm" in name else noise).double().round(decimals=8)).float())
        model.lm_head.weight = model.model.embed_tokens.weight
        logits = model(torch.tensor(ROWS)).logits
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        weights[name] = flat(p)
        shapes[name] = list(p.shape)
    document = dict(SHAPE, architectures=["Lfm3ForCausalLM"], model_type="lfm3", rope_parameters=ROPE, dtype="bfloat16")
    json.dump({"config": json.dumps(document), "weights": weights, "shapes": shapes, "rows": ROWS,
               "logits": flat(logits)}, sys.stdout)
    sys.stdout.write("\n")


main()
