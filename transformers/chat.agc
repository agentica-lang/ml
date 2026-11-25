## The chat template of the Qwen2.5, Qwen3 and Qwen3.5 instruct models (ChatML), as text: what Hugging Face's
## `apply_chat_template` makes of the same messages, without tools (`test/chat_golden.py`).
## The text goes through the tokenizer, which knows `<|im_start|>` and `<|im_end|>`.

## One turn: `role` is `system`, `user` or `assistant`.
struct Message
  role: String
  content: String
end

## An assistant message of qwen3 cut at its last `</think>`.
struct Reply
  reasoning: String
  content: String
end

## The prompt of `messages` for the model family of `model_type` (`qwen2`, `qwen3` or `qwen3_5`, the `model_type` of its
## `config.json`). With `add_generation_prompt` it ends with the opening of the assistant's turn. `enable_thinking: false`
## (Qwen3 only) opens it with an empty think block, so the model answers without thinking.
##
## `qwen2` puts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." as the system turn when the first
## message is not one, and writes every message as it is (a `<think>` in one is text). `qwen3` puts none, and reads an
## assistant message as the template does: the text before a `</think>` is its reasoning and the text after it, without
## its leading newlines, its content. After the last user message the reasoning stays, in a think block (empty for the
## message that ends the conversation); before it the reasoning is dropped.
##
## `qwen3_5` (text only) trims every message, wants the system message first, puts a think block in every assistant
## message after the last user message (empty or not) and none in the earlier ones, and ends the prompt with
## `<think>\n` under `enable_thinking: true` or with the empty think block under `false`. Hugging Face's template
## answers `false` when it is not told (`enable_thinking` undefined), where this takes `true` by default as `qwen3`.
## It is an error to have no user message (a tool answer is not one).
##
## Errors: no messages, a role other than the three, a `tool` role (tools are not supported), `enable_thinking: false`
## for `qwen2`.
def apply(
  model_type: String,
  messages: List[Message],
  add_generation_prompt: Bool = true,
  enable_thinking: Bool = true,
) -> Result[String]
  if model_type != "qwen2" and model_type != "qwen3" and model_type != "qwen3_5"
    return error("no chat template for model_type #{model_type}, only qwen2, qwen3 and qwen3_5")
  end

  return error("a conversation needs at least one message") if messages.empty?
  return error("enable_thinking is an option of qwen3, not of qwen2") if model_type == "qwen2" and not enable_thinking

  for m in messages
    return error("tools are not supported: a message with the role tool") if m.role == "tool"

    if m.role != "system" and m.role != "user" and m.role != "assistant"
      return error("a message role is system, user or assistant, got #{m.role}")
    end
  end

  return qwen35(messages, add_generation_prompt, enable_thinking) if model_type == "qwen3_5"

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
    if not enable_thinking
      text = text + "<think>\n\n</think>\n\n"
    end
  end

  ok(text)
end

private

## The prompt of the Qwen3.5 template: no tools and no images.
def qwen35(messages: List[Message], add_generation_prompt: Bool, enable_thinking: Bool) -> Result[String]
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
    if not enable_thinking
      text = text + "\n</think>\n\n"
    end
  end

  ok(text)
end

## Python's `str.strip()` on the ASCII white space.
def strip(text: String) -> String = text.strip

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
