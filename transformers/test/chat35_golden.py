"""Writes `test/fixtures/chat35_golden.json`: what Hugging Face's `apply_chat_template` makes (as text, no tools) of a few
conversations for the chat template of `Qwen/Qwen3.5-0.8B` (some with images), or that it refuses them. `test/chat_test.agc` compares
`Chat.apply("qwen3_5", ...)` to it.

Manual, not run by the tests: it needs `transformers` 5.22.2 (the script stops on another version) and the
`chat_template.jinja` and `tokenizer.json` of the checkpoint. Run it in a throwaway venv, from `transformers/`, with
Python's isolated mode:

    python -I test/chat35_golden.py QWEN35_DIR > test/fixtures/chat35_golden.json
"""

import hashlib
import json
import sys

import transformers
from transformers import AutoTokenizer

assert transformers.__version__ == "5.22.2", transformers.__version__

# The files of the official `Qwen/Qwen3.5-0.8B` repository that the golden is made from.
SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
}


# What `Chat.IMAGE` is: the template's text for an image.
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


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
        out.append({"role": message["role"], "content": parts if IMAGE in message["content"] else message["content"]})
    return out


THINK = "<think>\nI should greet.\n</think>\n\nHello!"

CONVERSATIONS = [
    ([m("user", "Hi")], True, None),
    ([m("user", "Hi")], True, True),
    ([m("user", "Hi")], True, False),
    ([m("user", "Hi")], False, None),
    ([m("system", "You answer in French."), m("user", "What is 2 + 2?")], True, True),
    ([m("system", "You answer in French."), m("user", "What is 2 + 2?")], True, False),
    ([m("user", "Hi"), m("assistant", "Hello! How can I help?"), m("user", "Why is the sky blue?")], True, True),
    ([m("user", "Hi"), m("assistant", "Hello!"), m("user", "Tell me more"), m("assistant", "There is")], False, None),
    ([m("system", "Be brief."), m("user", "Naïve café: 東京\n\nline two"), m("assistant", "ok"), m("user", "next")], True, True),
    ([m("user", "Hi"), m("assistant", THINK), m("user", "Again")], True, True),
    ([m("user", "Hi"), m("assistant", THINK)], False, None),
    ([m("user", "Hi"), m("assistant", THINK), m("user", "Again"), m("assistant", THINK)], False, None),
    ([m("user", "Hi"), m("assistant", "plain answer")], False, None),
    ([m("user", "Hi"), m("assistant", "<think>x</think>")], False, None),
    ([m("user", "Hi"), m("assistant", "a</think>b</think>c")], False, None),
    ([m("user", "Hi"), m("assistant", "<think>\nnever closed")], False, None),
    ([m("user", "\u3000Hi\u00a0"), m("assistant", "\u0085ok\u0085")], False, None),
    ([m("system", "\u00a0Be brief.\u3000"), m("user", "Hi\u0085")], True, None),
    ([m("user", "\x1cHi\x1f"), m("assistant", "\x1dok")], False, None),
    ([m("user", "Hi"), m("assistant", "\n\nanswer")], False, None),
    ([m("user", "Hi"), m("assistant", "plan\n\n</think>\n\n\nanswer")], False, None),
    ([m("user", "Hi"), m("assistant", "<think>\nstep\n</think>\n\nfirst"), m("assistant", "second")], False, None),
    ([m("system", "  Be brief.\n\n"), m("user", "\n  Hi  \n"), m("assistant", "  ok \n")], True, True),
    (
        [
            m("user", "q"),
            m("assistant", "a"),
            m("user", "<tool_response>\n42\n</tool_response>"),
            m("assistant", "done"),
        ],
        False,
        None,
    ),
    ([m("system", "Only a system message.")], True, True),
    ([m("user", "a"), m("system", "Now be formal."), m("user", "b")], True, True),
    ([m("assistant", "I speak first.")], True, True),
    ([], True, True),
    ([m("user", IMAGE + "Describe this picture.")], True, None),
    ([m("user", "What is in " + IMAGE + " and in " + IMAGE + "?")], True, True),
    ([m("user", "  " + IMAGE + "\n\nWhat color?\n")], True, False),
    ([m("system", "Be brief."), m("user", IMAGE + IMAGE), m("assistant", "Two pictures."), m("user", "Which first?")], True, None),
    ([m("user", IMAGE + "Describe."), m("assistant", "A disc."), m("user", IMAGE + "And this?")], False, None),
    ([m("system", "Look: " + IMAGE), m("user", "Hi")], True, None),
]

for name, digest in SHA256.items():
    with open(f"{sys.argv[1]}/{name}", "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"

cases = []
tok = AutoTokenizer.from_pretrained(sys.argv[1])
for messages, generation, thinking in CONVERSATIONS:
    extra = {} if thinking is None else {"enable_thinking": thinking}
    try:
        text = tok.apply_chat_template(as_parts(messages), tokenize=False, add_generation_prompt=generation, **extra)
        error = False
    except Exception:
        text = ""
        error = True
    cases.append(
        {
            "messages": messages,
            "add_generation_prompt": generation,
            "enable_thinking": thinking,
            "error": error,
            "text": text,
        }
    )

print(json.dumps({"cases": cases}))
