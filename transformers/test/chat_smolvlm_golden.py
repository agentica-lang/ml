"""Writes `test/fixtures/chat_smolvlm_golden.json`: what Hugging Face's `apply_chat_template` makes (as text) of a few
conversations for the chat template of `HuggingFaceTB/SmolVLM-256M-Instruct` (some with images), or that it refuses
them. `test/chat_test.agc` compares `Chat.apply("idefics3", ...)` to it.

Manual, not run by the tests: it needs `transformers` 5.22.2 (the script stops on another version) and the
`chat_template.json`, `tokenizer_config.json` and `tokenizer.json` of the checkpoint. Run it in a throwaway venv, from
`transformers/`, with Python's isolated mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/chat_smolvlm_golden.py SMOLVLM_DIR \
        > test/fixtures/chat_smolvlm_golden.json
"""

import hashlib
import json
import sys

import transformers
from transformers import AutoTokenizer

assert transformers.__version__ == "5.22.2", transformers.__version__

# The files of the official `HuggingFaceTB/SmolVLM-256M-Instruct` repository that the golden is made from.
SHA256 = {
    "chat_template.json": "12462caec000542ca1bb73cd0ba25aaf320d90335fc46731fa6a8355ea34bd26",
    "tokenizer_config.json": "8a68fb58021bc3df79d70f3d5fa79d2e709083872be1f6965cc9bcd12f3ba66a",
    "tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
}

# What `Chat.SMOLVLM_IMAGE` is: the template's text for an image.
IMAGE = "<image>"


def m(role, content):
    return {"role": role, "content": content}


def as_parts(messages):
    """The messages as the template takes images: a list of parts, `{"type": "image"}` where `IMAGE` is."""
    out = []
    for message in messages:
        parts = []
        for k, piece in enumerate(message["content"].split(IMAGE)):
            if k > 0:
                parts.append({"type": "image"})
            if piece:
                parts.append({"type": "text", "text": piece})
        out.append({"role": message["role"], "content": parts})
    return out


CONVERSATIONS = [
    ([m("user", "Hi")], True),
    ([m("user", "Hi")], False),
    ([m("user", IMAGE + "What is in the picture?")], True),
    ([m("user", "What is in " + IMAGE + "?")], True),
    ([m("user", "Look at this: " + IMAGE)], True),
    ([m("user", IMAGE + IMAGE + "Which is bigger?")], True),
    ([m("user", "Compare " + IMAGE + " and " + IMAGE + ".")], True),
    ([m("user", IMAGE)], True),
    ([m("system", "You answer in French."), m("user", "What is 2 + 2?")], True),
    ([m("system", "Be brief."), m("user", IMAGE + "Describe it."), m("assistant", "A red disc."), m("user", "And now?")], True),
    ([m("user", "Hi"), m("assistant", "Hello! How can I help?"), m("user", "Why is the sky blue?")], True),
    ([m("user", IMAGE + "Describe."), m("assistant", "A disc."), m("user", IMAGE + "And this?")], False),
    ([m("user", "Hi"), m("assistant", "Hello!")], False),
    ([m("user", "Look."), m("assistant", IMAGE + "A disc.")], False),
    ([m("user", "Naïve café: 東京\n\nline two  "), m("assistant", "  ok\n")], True),
    ([m("user", "  " + IMAGE + "\n\nWhat color?\n")], True),
    ([m("user", "a"), m("system", "Now be formal."), m("user", "b")], True),
    ([m("assistant", "I speak first.")], True),
    ([m("system", "Only a system message.")], True),
    ([m("user", "")], True),
    ([], True),
]

for name, digest in SHA256.items():
    with open(f"{sys.argv[1]}/{name}", "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"

cases = []
tok = AutoTokenizer.from_pretrained(sys.argv[1])
for messages, generation in CONVERSATIONS:
    try:
        text = tok.apply_chat_template(as_parts(messages), tokenize=False, add_generation_prompt=generation)
        error = False
    except Exception:
        text = ""
        error = True
    cases.append({"messages": messages, "add_generation_prompt": generation, "error": error, "text": text})

print(json.dumps({"cases": cases}))
