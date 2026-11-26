"""Writes `test/fixtures/lfm3_golden.json`: what the official LFM3 code answers for a tiny random model of one layer of
each kind (`conv`, `sliding_attention` with a window of 3, `full_attention`), in `float32` and the eager attention, for
`test/lfm3_test.agc` to compare with `ShortConv.short_conv`, `Nn.attention` with a `head_gate`, and `Nn.causal_mask` with
a window:

- the short convolution over 10 tokens with no cache, then a prefill of 6 and 4 decode steps with a `DynamicCache`, and
  the state the cache holds after them (`[width, taps]` in Hugging Face, the last `taps - 1` columns are ours);
- the two attention layers over the same 10 tokens (rotary positions and masks from the model's own code), and the
  0/1 grid of the keys each query sees in the sliding layer.

The rotary embedding here is the plain one (`partial_rotary_factor` 1.0): the zero-tail layout of the real model is
the loader's (`transformers/`).

The code is `modeling_lfm3.py`, `configuration_lfm3.py` and `lfm3_conv.py` of `LiquidAI/LFM3-1.7B`, revision
1da06bec6a819da10b54c67e5c90972b46d81871, under the LFM Open License v1.0. It is not copied here, and the fixture holds
only outputs on random weights. The script imports those files and runs them as code (`trust_remote_code`): read them
first, run in a throwaway venv, and it refuses files whose sha256 is not the one of that revision. Download them with:

    hf download LiquidAI/LFM3-1.7B modeling_lfm3.py configuration_lfm3.py lfm3_conv.py \\
        --revision 1da06bec6a819da10b54c67e5c90972b46d81871 --local-dir ~/.cache/agentica-ml/hf/lfm3-1.7b-code

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions)
and no `triton` or `causal-conv1d`. Run it from `nn/`, with Python's isolated mode:

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
from transformers import DynamicCache

TOKENS = 10
PREFILL = 6
HIDDEN = 16
TAPS = 7
WINDOW = 3
SHAPE = dict(
    hidden_size=HIDDEN, intermediate_size=24, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, vocab_size=32, max_position_embeddings=64, norm_eps=1e-5, conv_L_cache=TAPS, conv_bias=False,
    layer_types=["conv", "sliding_attention", "full_attention"], sliding_window=WINDOW, use_sandwich_norm=True,
    tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=2,
)
ROPE = {"full_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}


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
    torch.manual_seed(5)
    config = configuration.Lfm3Config(**SHAPE, rope_parameters=ROPE, attn_implementation="eager")
    model = modeling.Lfm3Model(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.2 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        x = (torch.randn(1, TOKENS, HIDDEN)).double().round(decimals=8).float()
        ids = torch.arange(TOKENS)[None]
        masks = modeling.create_masks_by_layer_type(config, x, None, None, ids)
        if masks["sliding_attention"] is None or masks["full_attention"] is None:
            raise SystemExit("a mask is None: the attention would run unmasked")
        conv = model.layers[0].conv
        full = conv(x)
        cache = DynamicCache(config=config)
        prefill = conv(x[:, :PREFILL], past_key_values=cache)
        decode = torch.cat([conv(x[:, t:t + 1], past_key_values=cache) for t in range(PREFILL, TOKENS)], dim=1)
        state = cache.layers[0].conv_states[0]
        short_conv = {
            "tokens": TOKENS, "hidden": HIDDEN, "taps": TAPS, "prefill": PREFILL,
            "input": flat(conv.in_proj.weight), "conv": flat(conv.conv.weight.squeeze(1)),
            "output": flat(conv.out_proj.weight), "x": flat(x[0]), "out": flat(full[0]),
            "prefill_out": flat(prefill[0]), "decode_out": flat(decode[0]),
            "state_shape": list(state.shape), "state": flat(state[0]),
        }
        attentions = []
        for index, kind in ((1, "sliding_attention"), (2, "full_attention")):
            layer = model.layers[index].self_attn
            cos_sin = model.rotary_emb(x, position_ids=ids, layer_type=kind)
            out, _ = layer(x, position_embeddings=cos_sin, attention_mask=masks[kind])
            attentions.append({
                "kind": kind, "tokens": TOKENS, "hidden": HIDDEN, "heads": 4, "kv_heads": 2, "head": 8,
                "theta": ROPE[kind]["rope_theta"], "window": WINDOW if kind == "sliding_attention" else 0,
                "query": flat(layer.q_proj.weight), "key": flat(layer.k_proj.weight),
                "value": flat(layer.v_proj.weight), "output": flat(layer.o_proj.weight),
                "query_norm": flat(layer.q_norm.weight), "key_norm": flat(layer.k_norm.weight),
                "gate": flat(layer.gate_proj.weight), "x": flat(x[0]), "out": flat(out[0]),
                "eps": config.norm_eps,
            })
        visible = (masks["sliding_attention"][0, 0] > -1.0).int() if masks["sliding_attention"].dtype.is_floating_point \
            else masks["sliding_attention"][0, 0].int()
    json.dump({"short_conv": short_conv, "attentions": attentions, "visible": visible.reshape(-1).tolist(),
               "window": WINDOW}, sys.stdout)
    sys.stdout.write("\n")


main()
