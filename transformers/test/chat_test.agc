## `Chat.apply` against Hugging Face's `apply_chat_template` (`chat_golden.py`, `fixtures/chat_golden.json`) for the
## templates of Qwen2.5-Instruct and Qwen3, against `chat35_golden.py` (`fixtures/chat35_golden.json`) for the one
## of Qwen3.5 and against `chat_lfm3_golden.py` (`fixtures/chat_lfm3_golden.json`) for the one of LFM3, with tools.
## Run from `transformers/`.

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
    try Test.eq(try Chat.apply(c.family, messages, c.add_generation_prompt, c.enable_thinking), c.text)
  end

  ok()
end

## A case without `enable_thinking` passes none: the template's own default.
def test_the_text_is_that_of_the_template_of_qwen35() -> Result[Void]
  golden = try JSON.decode(try File.read_text("test/fixtures/chat35_golden.json"), Golden35)
  try Test.eq(golden.cases.size, 28)
  for c in golden.cases
    messages = c.messages.map { |t| Chat.Message(role: t.role, content: t.content) }
    made = Chat.apply("qwen3_5", messages, c.add_generation_prompt, c.enable_thinking)
    if c.error
      try Test.error?(made)
    else
      try Test.eq(try made, c.text)
    end
  end

  ok()
end

## The golden of LFM3 is in the form Hugging Face takes: tools and calls as JSON objects.
def test_the_text_is_that_of_the_template_of_lfm3() -> Result[Void]
  golden = try JSON.parse(try File.read_text("test/fixtures/chat_lfm3_golden.json"))
  cases = try JSON.list(try JSON.get(golden, "cases"))
  try Test.eq(cases.size, 49)
  for c in cases
    messages = try messages_of(try JSON.list(try JSON.get(c, "messages")))
    tools: List[Map[String, JSON.Value]] = []
    if given(c, "tools").some?
      tools = try tools_of(try JSON.list(given(c, "tools").unwrap!))
    end

    thinking: Optional[Bool] = none
    if given(c, "enable_thinking").some?
      thinking = some(try JSON.bool(given(c, "enable_thinking").unwrap!))
    end

    made = try Chat.apply(
      "lfm3",
      messages,
      try JSON.bool(try JSON.get(c, "add_generation_prompt")),
      thinking,
      tools,
      try JSON.bool(try JSON.get(c, "preserve_thinking")),
    )

    try Test.eq(made, try JSON.string(try JSON.get(c, "text")))
  end

  ok()
end

## Left alone, `enable_thinking` is what each template does when it is not told: Qwen3 thinks (the Qwen3.5 golden has
## the cases of `none` for its template).
def test_the_default_of_enable_thinking_is_that_of_the_template() -> Result[Void]
  hi = [Chat.Message(role: "user", content: "Hi")]
  try Test.eq(try Chat.apply("qwen3", hi), try Chat.apply("qwen3", hi, true, some(true)))
  try Test.assert(not (try Chat.apply("qwen3", hi)).include?("</think>"), "qwen3 does not think by default")
  Test.assert((try Chat.apply("qwen2", hi)).end_with?("assistant\n"), "qwen2 has no think block")
end

def test_what_the_templates_do_not_cover_is_an_error() -> Result[Void]
  hi = [Chat.Message(role: "user", content: "Hi")]
  try Test.error?(Chat.apply("qwen3", []))
  try Test.error?(Chat.apply("llama", hi))
  try Test.error?(Chat.apply("qwen2", hi, true, some(false)))
  try Test.error?(Chat.apply("qwen3", hi + [Chat.Message(role: "tool", content: "42")]))
  Test.error?(Chat.apply("qwen3", hi + [Chat.Message(role: "narrator", content: "42")]))
end

def test_what_only_lfm3_reads_is_an_error_for_the_others() -> Result[Void]
  hi = [Chat.Message(role: "user", content: "Hi")]
  try Test.error?(Chat.apply("qwen3", hi, true, none, [{"name": JSON.from("f")}]))
  try Test.error?(Chat.apply("qwen3", hi, true, none, [], true))
  try Test.error?(Chat.apply("qwen3_5", [Chat.Message(role: "user", content: "Hi", thinking: some("t"))]))
  try Test.error?(Chat.apply("lfm3", []))
  Test.error?(Chat.apply("lfm3", hi, true, none, [{"name": JSON.from("f"), "defer_loading": JSON.from("yes")}]))
end

private

## The value at `key`, or none when the key is missing or null.
def given(v: JSON.Value, key: String) -> Optional[JSON.Value]
  case JSON.get(v, key)
    ok(found) -> return none if JSON.null?(found)
    error(_) -> return none
  end

  some(JSON.get(v, key).unwrap!)
end

def message_of(v: JSON.Value) -> Result[Chat.Message]
  thinking: Optional[String] = none
  if given(v, "thinking").some?
    thinking = some(try JSON.string(given(v, "thinking").unwrap!))
  end

  calls: List[Chat.ToolCall] = []
  if given(v, "tool_calls").some?
    calls = try calls_of(try JSON.list(given(v, "tool_calls").unwrap!))
  end

  id = ""
  if given(v, "tool_call_id").some?
    id = try JSON.string(given(v, "tool_call_id").unwrap!)
  end

  ok(
    Chat.Message(
      role: try JSON.string(try JSON.get(v, "role")),
      content: try JSON.string(try JSON.get(v, "content")),
      thinking: thinking,
      tool_calls: calls,
      tool_call_id: id,
    ),
  )
end

## A call as Hugging Face writes it: `{"type": "function", "function": {...}}`, or the function alone.
def call_of(v: JSON.Value) -> Result[Chat.ToolCall]
  fn = v
  if given(v, "function").some?
    fn = given(v, "function").unwrap!
  end

  ok(
    Chat.ToolCall(
      name: try JSON.string(try JSON.get(fn, "name")),
      arguments: try fields_of(try JSON.get(fn, "arguments")),
    ),
  )
end

def fields_of(v: JSON.Value) -> Result[Map[String, JSON.Value]]
  case JSON.node(v)
    JSON.Node.Object(fields) -> ok(fields)
    _ -> error("not an object: #{v}")
  end
end

def messages_of(items: List[JSON.Value]) -> Result[List[Chat.Message]]
  out: List[Chat.Message] = []
  for item in items
    out = out.push(try message_of(item))
  end

  ok(out)
end

def calls_of(items: List[JSON.Value]) -> Result[List[Chat.ToolCall]]
  out: List[Chat.ToolCall] = []
  for item in items
    out = out.push(try call_of(item))
  end

  ok(out)
end

def tools_of(items: List[JSON.Value]) -> Result[List[Map[String, JSON.Value]]]
  out: List[Map[String, JSON.Value]] = []
  for item in items
    out = out.push(try fields_of(item))
  end

  ok(out)
end
