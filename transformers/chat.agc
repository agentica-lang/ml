## The chat template of the Qwen2.5, Qwen3 and Qwen3.5 instruct models and of LFM3 (ChatML), as text: what Hugging
## Face's `apply_chat_template` makes of the same messages (`test/chat_golden.py`; with tools for LFM3 only).
## The text goes through the tokenizer, which knows `<|im_start|>` and `<|im_end|>`.

import Unicode from "tokenizer/unicode"
import Pyjson from "pyjson"

## What the Qwen3.5 template writes for an image of a message: put it in the `content` where the image goes. The
## tokenizer makes three ids of it (`<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>`), and `Mrope.expand` gives
## the middle one the tokens of the image. It is only text: a `content` that holds it has an image, whoever wrote it (the
## system message refuses it), and the `qwen2` and `qwen3` templates, which have no images, write it as the tags it is.
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"

## What the SmolVLM template writes for an image of a message: put it in the `content` where the image goes. The
## tokenizer makes one id of it (`<image>`), and `SmolVlmPrompt.expand` gives the tokens of the image's tiles.
SMOLVLM_IMAGE = "<image>"

## One turn: `role` is `system`, `user` or `assistant` (and `tool` for LFM3, which also reads the other fields).
struct Message
  role: String
  content: String
  ## The reasoning of an assistant message, as the template's `thinking` (LFM3).
  thinking: Optional[String] = none
  ## The calls of an assistant message (LFM3).
  tool_calls: List[ToolCall] = []
  ## The id of the call a `tool` message answers: consecutive answers with one id are one `<tool_response>` (LFM3).
  tool_call_id: String = ""
end

## A call of a tool: its name and its arguments in the order given.
struct ToolCall
  name: String
  arguments: Map[String, JSON.Value] = {}
end

## An assistant message of qwen3 cut at its last `</think>`.
struct Reply
  reasoning: String
  content: String
end

## The prompt of `messages` for the model family of `model_type` (`qwen2`, `qwen3`, `qwen3_5`, `lfm3` or `idefics3`,
## the `model_type` of its `config.json`). With `add_generation_prompt` it ends with the opening of the assistant's turn.
## `enable_thinking` (Qwen3 and Qwen3.5) decides whether that turn opens with `<think>` (`some(true)`) or with an
## empty think block, so the model answers without thinking (`some(false)`). Left at `none` it is what Hugging Face's
## template does when it is not told: thinking for `qwen3`, no thinking for `qwen3_5` (whose template thinks only
## when the option is exactly true).
##
## `qwen2` puts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." as the system turn when the first
## message is not one, and writes every message as it is (a `<think>` in one is text). `qwen3` puts none, and reads an
## assistant message as the template does: the text before a `</think>` is its reasoning and the text after it, without
## its leading newlines, its content. After the last user message the reasoning stays, in a think block (empty for the
## message that ends the conversation); before it the reasoning is dropped.
##
## `qwen3_5` trims every message, wants the system message first, puts a think block in every assistant message after
## the last user message (empty or not) and none in the earlier ones, and ends the prompt with `<think>\n` under
## `some(true)` or with the empty think block otherwise. It is an error to have no user message (a tool answer is not
## one). An image is `IMAGE` in the `content` of a user or an assistant message, not of the system one.
##
## `lfm3` (`LiquidAI/LFM3-1.7B`) starts with `<|startoftext|>`, takes `tools` (each a function as the template reads it:
## `{"name": ..., "description": ..., "parameters": ...}`, or that under a `function` key) into the system turn, writes
## the calls of an assistant message and the answers of `tool` messages, and thinks unless `enable_thinking` is
## `some(false)`. An assistant message keeps its reasoning (`thinking`, or an empty think block) after the last user
## message, or in every message with `preserve_thinking`; before it the text up to a `</think>` is dropped. Any role is
## written as it is, as the template does.
##
## `idefics3` (SmolVLM) starts with `<|im_start|>` once, writes each message as its role with a capital (`User`, `Assistant`,
## `System`), a colon, the content and `<end_of_utterance>` with a newline, and ends the prompt with `Assistant:`. Its
## template takes a list of parts for a message with images and `content` is the text of that list, `SMOLVLM_IMAGE` where
## an image goes: a content that starts with the image has no space after the colon (`User:<image>What is it?`), any
## other has one (`User: What is in <image>?`), and an empty content is an error (the template reads its first part). No
## message is trimmed and the roles are the three below, in any order.
##
## Errors: no messages, a role other than the three, a `tool` role (except for `lfm3`), `enable_thinking:
## some(false)` for `qwen2`, tools, `preserve_thinking`, reasoning, calls or a `tool_call_id` for a template other
## than `lfm3`'s.
def apply(
  model_type: String,
  messages: List[Message],
  add_generation_prompt: Bool = true,
  enable_thinking: Optional[Bool] = none,
  tools: List[Map[String, JSON.Value]] = [],
  preserve_thinking: Bool = false,
) -> Result[String]
  if not ["qwen2", "qwen3", "qwen3_5", "lfm3", "idefics3"].include?(model_type)
    return error("no chat template for model_type #{model_type}, only qwen2, qwen3, qwen3_5, lfm3 and idefics3")
  end

  return error("a conversation needs at least one message") if messages.empty?
  if model_type == "lfm3"
    return lfm3(messages, tools, add_generation_prompt, enable_thinking.unwrap_or(true), preserve_thinking)
  end

  if not tools.empty? or preserve_thinking or messages.any? { |m| lfm3_only?(m) }
    return error("tools, tool calls, reasoning fields and preserve_thinking are options of lfm3")
  end

  if (model_type == "qwen2" or model_type == "idefics3") and enable_thinking == some(false)
    return error("enable_thinking is an option of qwen3 and qwen3_5, not of #{model_type}")
  end

  thinking = enable_thinking.unwrap_or(model_type != "qwen3_5")

  for m in messages
    return error("tools are not supported: a message with the role tool") if m.role == "tool"

    if m.role != "system" and m.role != "user" and m.role != "assistant"
      return error("a message role is system, user or assistant, got #{m.role}")
    end
  end

  return qwen35(messages, add_generation_prompt, thinking) if model_type == "qwen3_5"
  return idefics3(messages, add_generation_prompt) if model_type == "idefics3"

  last_user = last_user_index(messages)
  text = ""
  if messages[0].role == "system"
    text = turn("system", messages[0].content)
  elsif model_type == "qwen2"
    text = turn("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
  end

  for i in 0...messages.size
    next if i == 0 and messages[0].role == "system"
    m = messages[i]
    if m.role == "assistant" and model_type == "qwen3"
      reply = split_reply(m.content)
      if i > last_user and (i == messages.size - 1 or not reply.reasoning.empty?)
        thought = "<think>\n#{strip_newlines(reply.reasoning)}\n</think>\n\n"
        text = text + "<|im_start|>assistant\n#{thought}#{trim_start(reply.content)}<|im_end|>\n"
      else
        text = text + turn("assistant", reply.content)
      end
    else
      text = text + turn(m.role, m.content)
    end
  end

  if add_generation_prompt
    text = text + "<|im_start|>assistant\n"
    if not thinking
      text = text + "<think>\n\n</think>\n\n"
    end
  end

  ok(text)
end

private

CONTINUE_TAG = "CONTINUE_FINAL_MESSAGE_TAG "

def lfm3_only?(m: Message) -> Bool = m.thinking.some? or not m.tool_calls.empty? or not m.tool_call_id.empty?

## The prompt of the Qwen3.5 template: no tools; an image is the text `IMAGE`.
def idefics3(messages: List[Message], add_generation_prompt: Bool) -> Result[String]
  text = "<|im_start|>"
  for m in messages
    return error("a message of idefics3 needs content (the template reads its first part)") if m.content.empty?
    colon = ": "
    if m.content.start_with?(SMOLVLM_IMAGE)
      colon = ":"
    end

    text = text + "#{capital(m.role)}#{colon}#{m.content}<end_of_utterance>\n"
  end

  if add_generation_prompt
    text = text + "Assistant:"
  end

  ok(text)
end

def capital(role: String) -> String = role.slice(0, 1).upcase + role.slice(1, role.size - 1)

def qwen35(messages: List[Message], add_generation_prompt: Bool, thinking: Bool) -> Result[String]
  query = -1
  for i in 0...messages.size
    if messages[i].role == "user" and not tool_text?(strip(messages[i].content))
      query = i
    end
  end

  return error("no user query found in the messages") if query < 0
  text = ""
  for i in 0...messages.size
    m = messages[i]
    content = strip(m.content)
    if m.role == "system"
      return error("the system message must be the first one") if i != 0
      return error("a system message cannot contain images") if content.include?(IMAGE)
      text = text + turn("system", content)
    elsif m.role == "user"
      text = text + turn("user", content)
    else
      reply = split_reply(content)
      if i > query
        thought = "<think>\n#{strip(reply.reasoning)}\n</think>\n\n"
        text = text + "<|im_start|>assistant\n#{thought}#{reply.content}<|im_end|>\n"
      else
        text = text + turn("assistant", reply.content)
      end
    end
  end

  if add_generation_prompt
    text = text + "<|im_start|>assistant\n<think>\n"
    if not thinking
      text = text + "\n</think>\n\n"
    end
  end

  ok(text)
end

## Python's `str.strip()`, which jinja's `|trim` calls: White_Space and U+001C to U+001F off both ends.
def strip(text: String) -> String
  chars = Unicode.codepoints(text)
  start = 0
  stop = chars.size
  while start < stop and blank?(chars[start])
    start += 1
  end

  while stop > start and blank?(chars[stop - 1])
    stop -= 1
  end

  Unicode.text_of(Unicode.span(chars, start, stop))
end

def blank?(c: Int) -> Bool = Unicode.space?(c) or (c >= 28 and c <= 31)

def turn(role: String, content: String) -> String = "<|im_start|>#{role}\n#{content}<|im_end|>\n"

## An assistant message of qwen3 as its template splits it at the last `</think>`.
def split_reply(text: String) -> Reply
  return Reply(reasoning: "", content: text) if not text.include?("</think>")

  parts = text.split("</think>")
  opened = trim_end(parts[0]).split("<think>")
  reasoning = ""
  if not opened.empty?
    reasoning = trim_start(opened[opened.size - 1])
  end

  Reply(reasoning: reasoning, content: trim_start(parts[parts.size - 1]))
end

def trim_start(text: String) -> String
  start = 0
  while start < text.size and text.slice(start, 1) == "\n"
    start += 1
  end

  text.slice(start, text.size - start)
end

def trim_end(text: String) -> String
  stop = text.size
  while stop > 0 and text.slice(stop - 1, 1) == "\n"
    stop -= 1
  end

  text.slice(0, stop)
end

def strip_newlines(text: String) -> String = trim_end(trim_start(text))

## A user message that carries a tool's answer, which the template does not count as the user's turn.
def tool_answer?(m: Message) -> Bool = tool_text?(m.content)

def tool_text?(content: String) -> Bool
  content.start_with?("<tool_response>") and content.end_with?("</tool_response>")
end

## The index of the last `user` message that is not a tool answer, or the last message when there is none (as the
## template does).
def last_user_index(messages: List[Message]) -> Int
  found = messages.size - 1
  for i in 0...messages.size
    if messages[i].role == "user" and not tool_answer?(messages[i])
      found = i
    end
  end

  found
end

## The prompt of LFM3's template.
def lfm3(
  messages: List[Message],
  tools: List[Map[String, JSON.Value]],
  add_generation_prompt: Bool,
  thinking: Bool,
  preserve_thinking: Bool,
) -> Result[String]
  turns = messages
  system = ""
  if messages[0].role == "system"
    system = messages[0].content
    turns = messages.drop(1)
  end

  lines: List[String] = []
  for tool in tools
    case try tool_line(tool)
      some(line) -> lines = lines.push(line)
      none -> 0
    end
  end

  if not lines.empty?
    if not system.empty?
      system = system + "\n\n"
    end

    system = system + "<tools>\n" + lines.join("\n") + "\n</tools>"
  end

  text = "<|startoftext|>"
  if not system.empty?
    text = text + turn("system", system)
  end

  last_user = -1
  for i in 0...turns.size
    if turns[i].role == "user"
      last_user = i
    end
  end

  answers: List[String] = []
  for i in 0...turns.size
    m = turns[i]
    items = answer_items(m)
    if joined_answer?(turns, i)
      answers = answers + items
    elsif m.role == "assistant"
      text = text + "<|im_start|>assistant\n" + assistant_text(m, preserve_thinking or i > last_user)
    elsif m.role == "tool"
      text = text + "<|im_start|>tool\n" + tool_response(m, answers) + "<|im_end|>\n"
      answers = []
    else
      text = text + turn(m.role, m.content)
    end
  end

  if add_generation_prompt
    text = text + "<|im_start|>assistant\n<think>"
    if not thinking
      text = text + "</think>"
    end
  end

  ok(text)
end

## The text of a tool as the template writes it, or none for a tool with `defer_loading`.
def tool_line(tool: Map[String, JSON.Value]) -> Result[Optional[String]]
  fn = tool
  if tool.key?("function")
    case JSON.node(tool["function"])
      JSON.Node.Object(fields) -> fn = fields
      _ -> return error("the function of a tool is not an object")
    end
  end

  return ok(none) if try deferred?(tool) or try deferred?(fn)

  ok(some(Pyjson.dumps(JSON.object(fn.delete("defer_loading")))))
end

## Whether a tool has `defer_loading: true`; a value that is not a Boolean is an error.
def deferred?(fields: Map[String, JSON.Value]) -> Result[Bool]
  case fields.get("defer_loading")
    some(v) -> JSON.bool(v)
    none -> ok(false)
  end
end

## The text of a `tool` message as a list item for a joined answer: none for an empty one.
def answer_items(m: Message) -> List[String]
  return [] if m.role != "tool" or m.content.empty?
  [m.content]
end

## Whether a `tool` message is the start or the middle of answers to one call, which are written as one.
def joined_answer?(turns: List[Message], i: Int) -> Bool
  m = turns[i]
  return false if m.role != "tool" or m.tool_call_id.empty? or i + 1 >= turns.size
  turns[i + 1].role == "tool" and turns[i + 1].tool_call_id == m.tool_call_id
end

## The `<tool_response>` of a message: its content, or the joined `pending` answers and its own when some are pending.
def tool_response(m: Message, pending: List[String]) -> String
  return "<tool_response>\n#{(pending + answer_items(m)).join("\n")}\n</tool_response>" if not pending.empty?

  return m.content if m.content.include?("<tool_response>")
  "<tool_response>\n#{m.content}\n</tool_response>"
end

## An assistant turn after its `<|im_start|>assistant\n`.
def assistant_text(m: Message, keep_thinking: Bool) -> String
  text = ""
  if keep_thinking
    case m.thinking
      some(thought) -> text = "<think>#{thought}</think>"
      none -> text = unthought(m.content)
    end
  end

  content = m.content
  if not keep_thinking
    if content.include?("</think>")
      parts = content.split("</think>")
      content = strip(parts[parts.size - 1])
    end

    content = "<think></think>" + content
  end

  continued = content.end_with?(CONTINUE_TAG)
  if continued
    content = content.slice(0, content.size - CONTINUE_TAG.size)
  end

  text = text + content + m.tool_calls.map { |c| call_text(c) }.join("\n\n")
  if continued
    text = text + CONTINUE_TAG
  end

  text + "<|im_end|>\n"
end

def call_text(call: ToolCall) -> String
  text = "<tool_call>\n<function=#{call.name}>\n"
  for name in call.arguments.keys
    text = text + "<parameter=#{name}>\n#{argument_text(call.arguments[name])}\n</parameter>\n"
  end

  text + "</function>\n</tool_call>"
end

## A string as it is, anything else as `json.dumps(ensure_ascii=False)` writes it.
def argument_text(value: JSON.Value) -> String
  case JSON.node(value)
    JSON.Node.String(s) -> s
    _ -> Pyjson.dumps(value)
  end
end

## The empty think block of a message that has no reasoning and none in its text.
def unthought(content: String) -> String
  return "" if content.include?("</think>")
  "<think></think>"
end
