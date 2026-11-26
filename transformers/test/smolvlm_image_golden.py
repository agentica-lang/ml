"""Writes `test/fixtures/smolvlm_image_golden.json`: the tiny random SmolVLM of `smolvlm_text_golden.py` (the same
config and seed, plus `image_token_id` 30) on a prompt of 10 token ids with 4 image tokens, to which Hugging Face's
`Idefics3ForConditionalGeneration` gets 4 random feature rows of 24 (what the connector gives; the tower is not run) in
`float32`. The golden holds its `config.json`, the text weights and head under the official names, the ids, the
features, the logits of the last prompt position and of the position after `NEXT` (the token fed once the prompt is in
the cache), and the 4 ids it picks greedily after the prompt. `test/smolvlm_image_test.agc` compares them with
`Generate.step_logits` and `Generate.generate` over a `KvDecoder.Sight`.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions).
Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/smolvlm_image_golden.py > test/fixtures/smolvlm_image_golden.json
"""

import json
import sys

import torch
import transformers
from transformers import Idefics3Config, Idefics3ForConditionalGeneration

IMAGE = 30
TEXT = dict(
    model_type="llama", vocab_size=31, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
    num_attention_heads=6, num_key_value_heads=2, head_dim=4, max_position_embeddings=64, rms_norm_eps=1e-5,
    rope_theta=100000.0, hidden_act="silu", tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
    rope_scaling=None,
)
VISION = dict(
    hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, num_channels=3, image_size=16,
    patch_size=4, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
)
IDS = [3, 17, IMAGE, IMAGE, IMAGE, IMAGE, 5, 12, 0, 9]
NEXT = 21
NEW = 4


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def logits_of(model, ids, features):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), mm_encoder_outputs=features).logits[0]


def main():
    check_environment()
    torch.manual_seed(13)
    config = Idefics3Config(
        text_config=dict(TEXT, attn_implementation="eager"), vision_config=dict(VISION, attn_implementation="eager"),
        scale_factor=2, tie_word_embeddings=False, image_token_id=IMAGE)
    model = Idefics3ForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.3 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        features = (torch.randn(IDS.count(IMAGE), TEXT["hidden_size"]) * 0.5).double().round(decimals=8).float()
    prompt = logits_of(model, IDS, features)
    after = logits_of(model, IDS + [NEXT], features)
    ids = list(IDS)
    generated = []
    for _ in range(NEW):
        picked = int(logits_of(model, ids, features)[-1].argmax())
        generated.append(picked)
        ids.append(picked)
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name.startswith(("model.text_model.", "lm_head.")) and not name.endswith("inv_freq"):
            weights[name] = flat(p)
            shapes[name] = list(p.shape)
    document = {
        "architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3", "scale_factor": 2,
        "tie_word_embeddings": False, "image_token_id": IMAGE, "text_config": TEXT,
        "vision_config": dict(VISION, model_type="idefics3"),
    }
    json.dump(
        {
            "config": json.dumps(document), "weights": weights, "shapes": shapes, "ids": IDS, "image_token": IMAGE,
            "features": flat(features), "next": NEXT, "prompt_logits": flat(prompt[-1]),
            "next_logits": flat(after[-1]), "generated": generated,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


main()
