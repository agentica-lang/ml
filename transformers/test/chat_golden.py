"""Writes `test/fixtures/chat_golden.json`: what Hugging Face's `apply_chat_template` makes (as text, no tools) of a few
conversations, for the chat template of `Qwen/Qwen2.5-0.5B-Instruct` (family `qwen2`) and of `Qwen/Qwen3-0.6B` (family
`qwen3`). `test/chat_test.agc` compares `Chat.apply` to it.

Manual, not run by the tests: it needs `transformers` (4.57.6 was used) and the `tokenizer_config.json` of the two
checkpoints. Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/chat_golden.py QWEN25_DIR QWEN3_DIR > test/fixtures/chat_golden.json
"""

import json
import sys

from transformers import AutoTokenizer


def m(role, content):
    return {"role": role, "content": content}


CONVERSATIONS = [
    ([m("user", "Hi")], True, None),
    ([m("user", "Hi")], False, None),
    ([m("system", "You answer in French."), m("user", "What is 2 + 2?")], True, None),
    ([m("system", "You answer in French."), m("user", "What is 2 + 2?")], True, False),
    ([m("user", "Hi"), m("assistant", "Hello! How can I help?"), m("user", "Why is the sky blue?")], True, None),
    ([m("user", "Hi"), m("assistant", "Hello!"), m("user", "Tell me more"), m("assistant", "There is")], False, None),
    ([m("system", "Be brief."), m("user", "Naïve café: 東京\n\nline two"), m("assistant", "ok"), m("user", "next")], True, True),
    ([m("user", "a"), m("system", "Now be formal."), m("user", "b")], True, None),
    (
        [
            m("user", "Hi"),
            m("assistant", "<think>\nI should greet.\n</think>\n\nHello!"),
            m("user", "Again"),
        ],
        True,
        None,
    ),
    ([m("user", "Hi"), m("assistant", "<think>\nI should greet.\n</think>\n\nHello!")], False, None),
    ([m("user", "Hi"), m("assistant", "<think>\nstep\n</think>\n\nfirst"), m("assistant", "second")], False, None),
    ([m("user", "Hi"), m("assistant", "\n\nanswer")], False, None),
    ([m("user", "Hi"), m("assistant", "plan\n\n</think>\n\n\nanswer")], False, None),
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
]

cases = []
for family, directory in (("qwen2", sys.argv[1]), ("qwen3", sys.argv[2])):
    tok = AutoTokenizer.from_pretrained(directory)
    for index, (messages, generation, thinking) in enumerate(CONVERSATIONS):
        extra = {} if thinking is None or family == "qwen2" else {"enable_thinking": thinking}
        if family == "qwen2" and thinking is False:
            continue
        cases.append(
            {
                "family": family,
                "messages": messages,
                "add_generation_prompt": generation,
                "enable_thinking": thinking,
                "text": tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, **extra),
            }
        )

print(json.dumps({"cases": cases}))
