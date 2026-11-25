"""Writes `test/fixtures/chat35_golden.json`: what Hugging Face's `apply_chat_template` makes (as text, no tools) of a few
conversations for the chat template of `Qwen/Qwen3.5-0.8B`, or that it refuses them. `test/chat_test.agc` compares
`Chat.apply("qwen3_5", ...)` to it.

Manual, not run by the tests: it needs `transformers` 5.22.2 (the script stops on another version) and the
`chat_template.jinja` and `tokenizer.json` of the checkpoint. Run it in a throwaway venv, from `transformers/`, with
Python's isolated mode:

    python -I test/chat35_golden.py QWEN35_DIR > test/fixtures/chat35_golden.json
"""

import json
import sys

import transformers
from transformers import AutoTokenizer

assert transformers.__version__ == "5.22.2", transformers.__version__


def m(role, content):
    return {"role": role, "content": content}


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
]

cases = []
tok = AutoTokenizer.from_pretrained(sys.argv[1])
for messages, generation, thinking in CONVERSATIONS:
    extra = {} if thinking is None else {"enable_thinking": thinking}
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, **extra)
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
