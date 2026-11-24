## The chat template of the Qwen2.5 and Qwen3 instruct models (ChatML), as text: what Hugging Face's
## `apply_chat_template` makes of the same messages, without tools (`test/chat_golden.py`).
## The text goes through the tokenizer, which knows `<|im_start|>` and `<|im_end|>`.

## One turn: `role` is `system`, `user` or `assistant`.
struct Message
  role: String
  content: String
end

## The prompt of `messages` for the model family of `model_type` (`qwen2` or `qwen3`, the `model_type` of its
## `config.json`). With `add_generation_prompt` it ends with the opening of the assistant's turn. `enable_thinking: false`
## (Qwen3 only) opens it with an empty think block, so the model answers without thinking.
##
## `qwen2` puts "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." as the system turn when the first
## message is not one. `qwen3` puts none, and writes an empty think block before an assistant message that ends the
## conversation.
##
## Errors: no messages, a role other than the three, a `tool` role (tools are not supported), a think block inside an
## assistant message, `enable_thinking: false` for `qwen2`.
def apply(
  model_type: String,
  messages: List[Message],
  add_generation_prompt: Bool = true,
  enable_thinking: Bool = true,
) -> Result[String]
  if model_type != "qwen2" and model_type != "qwen3"
    return error("no chat template for model_type #{model_type}, only qwen2 and qwen3")
  end

  return error("a conversation needs at least one message") if messages.empty?
  return error("enable_thinking is an option of qwen3, not of qwen2") if model_type == "qwen2" and not enable_thinking

  for m in messages
    return error("tools are not supported: a message with the role tool") if m.role == "tool"

    if m.role != "system" and m.role != "user" and m.role != "assistant"
      return error("a message role is system, user or assistant, got #{m.role}")
    end

    if m.role == "assistant" and (m.content.include?("<think>") or m.content.include?("</think>"))
      return error("an assistant message with a think block is not supported")
    end
  end

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
    if m.role == "assistant" and model_type == "qwen3" and i > last_user and i == messages.size - 1
      text = text + "<|im_start|>assistant\n<think>\n\n</think>\n\n#{m.content}<|im_end|>\n"
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

def turn(role: String, content: String) -> String = "<|im_start|>#{role}\n#{content}<|im_end|>\n"

## The index of the last `user` message, or the last message when there is none (as the template does).
def last_user_index(messages: List[Message]) -> Int
  found = messages.size - 1
  for i in 0...messages.size
    if messages[i].role == "user"
      found = i
    end
  end

  found
end
