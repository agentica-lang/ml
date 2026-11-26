"""Writes `test/fixtures/chat_lfm3_golden.json`: what Hugging Face's `apply_chat_template` makes (as text) of a few
conversations, with and without tools, for the chat template of `LiquidAI/LFM3-1.7B` (`chat_template.jinja`, a Jinja
template under the LFM Open License v1.0, not copied here). `test/chat_test.agc` compares `Chat.apply("lfm3", ...)` to it.

Manual, not run by the tests: it needs `transformers` 5.22.2 (the script stops on another version), the four files
below of revision 1da06bec6a819da10b54c67e5c90972b46d81871 (it refuses files with another sha256) and no network. It
only renders the template and reads the tokenizer files; no model code runs. Download and run it in a throwaway venv,
from `transformers/`, with Python's isolated mode:

    hf download LiquidAI/LFM3-1.7B chat_template.jinja tokenizer.json tokenizer_config.json special_tokens_map.json \\
        --revision 1da06bec6a819da10b54c67e5c90972b46d81871 --local-dir LFM3_DIR
    python -I test/chat_lfm3_golden.py LFM3_DIR > test/fixtures/chat_lfm3_golden.json
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import json
import sys

import transformers
from transformers import AutoTokenizer

if transformers.__version__ != "5.22.2":
    raise SystemExit(f"transformers 5.22.2 is needed, got {transformers.__version__}")

SHA256 = {
    "chat_template.jinja": "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818",
    "tokenizer.json": "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701",
    "tokenizer_config.json": "bf028f813efc6137bc07ed045cafe71820df9a5e26a38e39954706ed1e30a457",
    "special_tokens_map.json": "7946f864cbfb62a1b041bda937244210a1a87f4a6cab3a48b791131b2cd57f88",
}


def m(role, content="", **rest):
    return {"role": role, "content": content, **rest}


def call(name, **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def calls(*items):
    return m("assistant", "", tool_calls=list(items))


WEATHER = {
    "name": "get_weather",
    "description": "Gets the weather of a city, in °C.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "A city: \"Zürich\"\n"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
}
TIME = {"name": "now", "description": "The time.", "parameters": {"type": "object", "properties": {}}}
SEARCH = {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}, "strict": True}

HI = [m("user", "Hi")]
ARGS = dict(
    text="a \"quoted\" line\nand 東京 é",
    n=42,
    big=12345678901234567890,
    f=0.25,
    exp=1e16,
    small=2.5e-7,
    whole=100.0,
    yes=True,
    no=False,
    nothing=None,
    items=[1, "two", [3.5], {"k": None}],
    nested={"a": {"b": [True, "c\u0001"]}, "empty": {}, "none": []},
)
ANSWERED = [m("user", "Weather?"), calls(call("get_weather", city="Madrid", days=2)), m("tool", "21 °C")]

# (messages, tools, add_generation_prompt, enable_thinking, preserve_thinking)
CONVERSATIONS = [
    (HI, None, True, None, False),
    (HI, None, True, True, False),
    (HI, None, True, False, False),
    (HI, None, False, None, False),
    ([m("system", "You answer in French."), m("user", "What is 2 + 2?")], None, True, None, False),
    ([m("system", ""), m("user", "Hi")], None, True, None, False),
    ([m("user", "Hi"), m("assistant", "Hello!"), m("user", "Naïve café: 東京\n\nline two")], None, True, None, False),
    ([m("user", "Hi"), m("assistant", "Hello!")], None, False, None, False),
    ([m("user", ""), m("assistant", "")], None, True, None, False),
    ([m("user", " Hi　"), m("assistant", "\x1cok\x1f</think>  \n　done \n")], None, True, None, False),
    ([m("assistant", "I speak first.")], None, True, None, False),
    ([m("system", "Only a system message.")], None, True, None, False),
    ([m("user", "a"), m("system", "Now be formal."), m("user", "b"), m("narrator", "It rained.")], None, True, None, False),
    # tools
    (HI, [WEATHER], True, None, False),
    ([m("system", "Be brief."), m("user", "Hi")], [WEATHER, TIME], True, None, False),
    ([m("system", ""), m("user", "Hi")], [{"type": "function", "function": WEATHER}, SEARCH], True, None, False),
    (HI, [dict(TIME, defer_loading=True), dict(SEARCH, defer_loading=False)], True, None, False),
    (HI, [dict(TIME, defer_loading=True)], True, None, False),
    (HI, [{"type": "function", "function": WEATHER, "defer_loading": True}, TIME], True, None, False),
    # calls and answers
    (ANSWERED, [WEATHER], True, None, False),
    (ANSWERED + [m("assistant", "It is 21 °C.")], [WEATHER], False, None, False),
    ([m("user", "x"), calls(call("a_tool", **ARGS))], None, False, None, False),
    ([m("user", "x"), calls(call("one"), call("two", q="x"), call("three", **{"k y": "v"}))], None, False, None, False),
    ([m("user", "x"), m("assistant", "Let me look.", tool_calls=[call("search", q="agentica")])], None, False, None, False),
    ([m("user", "x"), m("assistant", "", tool_calls=[{"name": "bare", "arguments": {"a": 1}}])], None, False, None, False),
    ([m("user", "x"), calls(call("noargs")), m("tool", "")], None, True, None, False),
    ([m("user", "x"), calls(call("noargs")), m("tool", "<tool_response>\nraw\n</tool_response>")], None, True, None, False),
    ([m("user", "x"), calls(call("a"), call("b")), m("tool", "one", tool_call_id="1"), m("tool", "two", tool_call_id="1")], None, True, None, False),
    (
        [
            m("user", "x"),
            calls(call("a"), call("b"), call("c")),
            m("tool", "one", tool_call_id="1"),
            m("tool", "", tool_call_id="1"),
            m("tool", "three", tool_call_id="1"),
            m("tool", "other", tool_call_id="2"),
            m("tool", "no id"),
            m("tool", "no id either"),
            m("assistant", "done"),
        ],
        None,
        False,
        None,
        False,
    ),
    ([m("user", "x"), m("tool", "", tool_call_id="1"), m("tool", "", tool_call_id="1")], None, True, None, False),
    ([m("user", "x"), m("tool", "a", tool_call_id="1"), m("tool", "<tool_response>b</tool_response>", tool_call_id="1")], None, True, None, False),
    # thinking
    ([m("user", "Hi"), m("assistant", "Hello!", thinking="I should greet.")], None, False, None, False),
    ([m("user", "Hi"), m("assistant", "Hello!", thinking=""), m("user", "More")], None, True, None, False),
    ([m("user", "Hi"), m("assistant", "Hello!", thinking="I should greet."), m("user", "More")], None, True, None, False),
    ([m("user", "Hi"), m("assistant", "Hello!", thinking="I should greet."), m("user", "More")], None, True, None, True),
    ([m("user", "Hi"), m("assistant", "Hello!"), m("user", "More")], None, True, None, True),
    ([m("user", "Hi"), m("assistant", "plan\n\n</think>\n\n\nanswer"), m("user", "More")], None, True, None, False),
    ([m("user", "Hi"), m("assistant", "plan\n\n</think>\n\n\nanswer"), m("user", "More")], None, True, None, True),
    ([m("user", "Hi"), m("assistant", "plan\n\n</think>\n\n\nanswer")], None, False, None, False),
    ([m("user", "Hi"), m("assistant", "a</think>b</think>c"), m("user", "Hi"), m("assistant", "a</think>b</think>c")], None, False, None, False),
    ([m("user", "Hi"), m("assistant", "x</think>"), m("user", "Hi"), m("assistant", "x</think>")], None, False, None, False),
    ([m("user", "Hi"), m("assistant", "<think>\nsteps\n</think>\nanswer", thinking="own"), m("user", "q")], None, True, None, False),
    ([m("assistant", "no user message", thinking="t")], None, False, None, False),
    ([m("system", "s"), m("assistant", "no user message")], None, True, None, False),
    ([m("user", "Hi"), m("assistant", "Let me look.", thinking="think", tool_calls=[call("search", q="x")])], [SEARCH], False, None, False),
    (
        [m("user", "Hi"), calls(call("search", q="x")), m("tool", "found"), m("assistant", "Found it.", thinking="why")],
        [SEARCH],
        False,
        None,
        False,
    ),
    # a message that goes on
    ([m("user", "Hi"), m("assistant", "The answer is CONTINUE_FINAL_MESSAGE_TAG ")], None, False, None, False),
    ([m("user", "Hi"), m("assistant", "abc CONTINUE_FINAL_MESSAGE_TAG ", tool_calls=[call("a", x=1)])], None, False, None, False),
    ([m("user", "Hi"), m("assistant", "CONTINUE_FINAL_MESSAGE_TAG"), m("user", "q"), m("assistant", "a CONTINUE_FINAL_MESSAGE_TAG ")], None, False, None, False),
]

for name, digest in SHA256.items():
    with open(f"{sys.argv[1]}/{name}", "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != digest:
            raise SystemExit(f"{name} is not the file of the golden")

cases = []
tok = AutoTokenizer.from_pretrained(sys.argv[1])
for messages, tools, generation, thinking, preserve in CONVERSATIONS:
    extra = {} if thinking is None else {"enable_thinking": thinking}
    if preserve:
        extra["preserve_thinking"] = True
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=generation, **extra)
    cases.append(
        {
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": generation,
            "enable_thinking": thinking,
            "preserve_thinking": preserve,
            "text": text,
        }
    )

print(json.dumps({"cases": cases}, ensure_ascii=False))
