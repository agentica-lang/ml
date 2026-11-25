## `Chat.apply` against Hugging Face's `apply_chat_template` (`chat_golden.py`, `fixtures/chat_golden.json`) for the
## templates of Qwen2.5-Instruct and Qwen3, and against `chat35_golden.py` (`fixtures/chat35_golden.json`) for the one
## of Qwen3.5. Run from `transformers/`.

import Chat from "../chat"

struct Turn
  role: String
  content: String
end

struct Case
  family: String
  messages: List[Turn]
  add_generation_prompt: Bool
  enable_thinking: Optional[Bool]
  text: String
end

struct Golden
  cases: List[Case]
end

struct Case35
  messages: List[Turn]
  add_generation_prompt: Bool
  enable_thinking: Optional[Bool]
  error: Bool
  text: String
end

struct Golden35
  cases: List[Case35]
end

def test_the_text_is_that_of_the_templates() -> Result[Void]
  golden = try JSON.decode(try File.read_text("test/fixtures/chat_golden.json"), Golden)
  try Test.eq(golden.cases.size, 27)
  for c in golden.cases
    messages = c.messages.map { |t| Chat.Message(role: t.role, content: t.content) }
    thinking = c.enable_thinking.unwrap_or(true)
    try Test.eq(try Chat.apply(c.family, messages, c.add_generation_prompt, thinking), c.text)
  end

  ok()
end

## Hugging Face's template reads an `enable_thinking` that is not given as false.
def test_the_text_is_that_of_the_template_of_qwen35() -> Result[Void]
  golden = try JSON.decode(try File.read_text("test/fixtures/chat35_golden.json"), Golden35)
  try Test.eq(golden.cases.size, 22)
  for c in golden.cases
    messages = c.messages.map { |t| Chat.Message(role: t.role, content: t.content) }
    made = Chat.apply("qwen3_5", messages, c.add_generation_prompt, c.enable_thinking.unwrap_or(false))
    if c.error
      try Test.error?(made)
    else
      try Test.eq(try made, c.text)
    end
  end

  ok()
end

def test_what_the_templates_do_not_cover_is_an_error() -> Result[Void]
  hi = [Chat.Message(role: "user", content: "Hi")]
  try Test.error?(Chat.apply("qwen3", []))
  try Test.error?(Chat.apply("llama", hi))
  try Test.error?(Chat.apply("qwen2", hi, true, false))
  try Test.error?(Chat.apply("qwen3", hi + [Chat.Message(role: "tool", content: "42")]))
  try Test.error?(Chat.apply("qwen3_5", hi + [Chat.Message(role: "tool", content: "42")]))
  Test.error?(Chat.apply("qwen3", hi + [Chat.Message(role: "narrator", content: "42")]))
end
