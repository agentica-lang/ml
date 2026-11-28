## The OpenAI-shaped chat API of the `server` package (ml#90): `GET /health` and
## streaming-only `POST /v1/chat/completions` over the transformers scheduler (`Serving`), with the Qwen3 chat
## template (`Chat`) and the byte-level BPE tokenizer (`Bpe`). No sampling, no non-streaming answer: anything but
## greedy streaming is a bounded 400, and a queue with no room is a 503, never a wait.
##
## The shape of the code keeps the contract:
##
## - One loaded generator. The scheduler task owns it (`Serving.start` lends it until a `stop` that returns `ok()`);
##   the model owner outlives every relay: routes and pumps only borrow the scheduler and the tokenizer. A pump
##   task owns only its two channels (the reply and the events): it never owns scheduler handles, so its end cannot
##   close the scheduler's inbox.
## - Bounded everything. At most `MAX_ACTIVE` sequences step and `MAX_PENDING` more wait (`Serving.Limits`); every
##   channel send and receive in the relay carries a timeout, so a slow or gone consumer ends its own sequence
##   without stalling the others, and a stalled engine ends the stream with an error event instead of hanging it.
## - No silent success. A sequence that fails after the head went out ends with an error event and no `[DONE]`; a
##   `[DONE]` always closes a stream whose deltas concatenate to the decoded answer.
## - UTF-8 across token boundaries. A character split between two tokens decodes with a scrubbed replacement
##   character (`Bpe.decode` scrubs); the relay holds back a trailing replacement character until the next token
##   completes it, so the deltas stay exact and only complete characters go out (a stream that ends on one flushes
##   it whole).
## - Disconnects are observed on the next write. Bond/`std/http` closes the events channel when the client leaves;
##   the relay's next send then fails, it closes the reply channel, and the scheduler's next send to that closed
##   channel drops the sequence (finishing its cache and freeing the slot). A client that disconnects between two
##   tokens is therefore noticed at most one step later; there is no read of the socket to notice it sooner, by
##   design of the server.

import Bond from "pkg/bond/bond"
import Serving from "pkg/transformers/serving"
import Chat from "pkg/transformers/chat"
import Bpe from "pkg/transformers/tokenizer/bpe"

## How many tokens a request gets when it names none, and the most it may name: the reply channel holds the whole
## answer (`Serving` never waits on it), so the cap also bounds that channel.
DEFAULT_MAX_TOKENS = 128
MAX_MAX_TOKENS = 512

## The most input characters a request may carry: the encoded prompt must still fit the model's context beside the
## answer, which the budget check enforces against `max_positions`.
MAX_PROMPT_CHARS = 8192

## At most two sequences step at once (so two streams advance together) and four more wait; the inbox holds one more
## command, and `Serving.submit` refuses past that instead of waiting.
MAX_ACTIVE = 2
MAX_PENDING = 4

## The head of a stream is refused while the scheduler cannot take it; afterwards every wait is bounded.
ANSWER_TIMEOUT_MS = 120_000
PIECE_WAIT_MS = 60_000
EVENT_SEND_TIMEOUT_MS = 10_000

## The replacement character `Bpe.decode` writes for bytes that are not complete UTF-8 yet: held back, never edited.
REPLACEMENT = String.from_code(65533)

## The stop words of a chat answer, as added tokens of the tokenizer: the end of the assistant's turn, and the end
## of the text. Absent tokens are absent stops.
STOP_WORDS = ["<|im_end|>", "<|endoftext|>"]

## What the routes share: the tokenizer (to encode the prompt and decode the tokens), the scheduler's inbox (through
## its handle), the stop ids, and the bounds. The counters name the streamed answers.
struct Api
  tok: Bpe.ByteLevelBpe
  serving: Serving.Server
  model_name: String
  model_type: String
  max_positions: Int
  stop: List[Int]
  default_max_tokens: Int
  max_max_tokens: Int
  counters: ATS[String, Int]
end

## What the next delta owes the stream: the text to emit now, the text sent so far (holding back a trailing
## replacement character that the next token may complete), and how many tokens the answer holds.
struct Owed
  emit: String
  sent: String
  count: Int
end

## The stop ids of `tok`: the added tokens whose content ends an answer.
def stop_ids(tok: Bpe.ByteLevelBpe) -> List[Int]
  ids: List[Int] = []
  for a in tok.added
    if STOP_WORDS.include?(a.content)
      ids = ids.push(a.id)
    end
  end

  ids
end

## The routes: `GET /health` and streaming-only `POST /v1/chat/completions`.
def app(api: Api) -> Bond.App
  (
    Bond.app()
    |> Bond.get("/health", &health)
    |> Bond.post("/v1/chat/completions") do |conn| chat(api, conn) end
  )
end

## The text the decoded `generated` adds after `sent`: `final` flushes everything, otherwise a trailing replacement
## character (a character the next token may still complete) is held back. An error means the decoded text no longer
## extends what was sent, which is a bug in the relay, never bad input: the caller ends the stream loudly.
def emit_text(tok: Bpe.ByteLevelBpe, sent: String, generated: List[Int], final: Bool) -> Result[Owed]
  full = try Bpe.decode(tok, generated)
  # `sent` is the decoded prefix the stream already owes, minus a held-back tail at most: the bytes before the
  # trailing partial character scrub the same however the character completes, so the new text always extends it.
  # Anything else is a bug in the relay, never bad input: the caller ends the stream loudly.
  return error("the decoded answer no longer extends the sent text") if not full.start_with?(sent)

  rest = full.slice(sent.length, full.length - sent.length)
  if not final and rest.end_with?(REPLACEMENT) and rest.length > 0
    held = rest.slice(0, rest.length - 1)
    return ok(Owed(emit: held, sent: sent + held, count: generated.size))
  end

  ok(Owed(emit: rest, sent: full, count: generated.size))
end

private

struct StreamOptions
  include_usage: Optional[Bool] = none
end

# One turn of the conversation, as the client sends it: `Chat.Message` carries the same two fields the template
# reads (plus the LFM3-only ones, which stay empty here).
struct InMessage
  role: String
  content: String
end

# What the route reads: the messages, the budget, and the streaming switch. The sampling knobs (`temperature`,
# `top_p`, `seed`, `n`) and `stop` are read only to be refused: this server is greedy with fixed stops.
struct ChatRequest
  messages: List[InMessage]
  max_tokens: Optional[Int] = none
  stream: Optional[Bool] = none
  stream_options: Optional[StreamOptions] = none
  temperature: Optional[Float] = none
  top_p: Optional[Float] = none
  seed: Optional[Int] = none
  n: Optional[Int] = none
  stop: Optional[List[String]] = none
end

struct Usage
  prompt_tokens: Int
  completion_tokens: Int
  total_tokens: Int
end

# What a `chat.completion.chunk` carries: a key that is absent stays out of the JSON (the final delta is `{}`),
# `finish_reason` is `null` until the last chunk.
struct Delta
  role: JSON.Field[String] = JSON.Field.Absent
  content: JSON.Field[String] = JSON.Field.Absent
end

struct ChunkChoice
  index: Int
  delta: Delta
  finish_reason: JSON.Field[String] = JSON.Field.Null
end

struct Chunk
  id: String
  object: String
  created: Int
  model: String
  choices: List[ChunkChoice]
  usage: JSON.Field[Usage] = JSON.Field.Absent
end

# What the task that turns scheduler events into SSE needs; it cannot take the `Conn`. It holds no scheduler
# handle (`Serving.Server` stays with the route): the task owns the two channels, and closing the reply is what
# drops the sequence.
struct Pump
  tok: Bpe.ByteLevelBpe
  reply: Channel[Serving.Event]
  events: Channel[Bytes]
  generated: List[Int]
  sent: String
  id: String
  created: Int
  model: String
  include_usage: Bool
  prompt_tokens: Int
  max_tokens: Int
end

def health(conn: Bond.Conn) -> Result[Bond.Conn] = ok(Bond.json(conn, 200, JSON.from({"status": "ok"})))

def chat(api: Api, conn: Bond.Conn) -> Result[Bond.Conn]
  body = try Bond.json_body(conn, ChatRequest.codec)
  return Bond.bad_request("messages must not be empty") if body.messages.empty?

  return Bond.bad_request("stream must be true: this server only streams") if body.stream != some(true)

  return Bond.bad_request("temperature is not supported: this server is greedy") if body.temperature.some?

  return Bond.bad_request("top_p is not supported: this server is greedy") if body.top_p.some?

  return Bond.bad_request("seed is not supported: this server is greedy") if body.seed.some?

  return Bond.bad_request("n is not supported: this server answers once") if body.n.some?

  return Bond.bad_request("stop is not supported: the stops are fixed") if body.stop.some?

  max_tokens = body.max_tokens.unwrap_or(api.default_max_tokens)
  return Bond.bad_request("max_tokens is #{max_tokens}, needs at least 1") if max_tokens < 1

  if max_tokens > api.max_max_tokens
    return Bond.bad_request("max_tokens is #{max_tokens}, at most #{api.max_max_tokens}")
  end

  chars = 0
  for m in body.messages
    chars += m.content.length
  end

  if chars > MAX_PROMPT_CHARS
    return Bond.bad_request("the messages hold #{chars} characters, at most #{MAX_PROMPT_CHARS}")
  end

  turns = body.messages.map do |m| Chat.Message(role: m.role, content: m.content) end
  text = try bad_request("the messages do not fit the chat template", Chat.apply(api.model_type, turns))
  prompt = try bad_request("the prompt is not encodable", Bpe.encode(api.tok, text))
  return Bond.bad_request("the prompt encodes to no token") if prompt.empty?

  if prompt.size + max_tokens > api.max_positions
    return Bond.bad_request(
      "the prompt of #{prompt.size} tokens and #{max_tokens} new ones do not fit #{api.max_positions} positions",
    )
  end

  reply: Channel[Serving.Event] = Channel.new(max_tokens + 1)
  case Serving.submit(api.serving, prompt, max_tokens, api.stop, reply)
    ok(_) -> nil
    error(e) -> return ok(Bond.halt(conn, 503, e.message))
  end

  stream_answer(api, conn, reply, prompt.size, max_tokens, wants_usage(body))
end

# `Chat.apply` and `Bpe.encode` fail with their own reasons; the route reports them as the bad input they are.
def bad_request[T](message: String, result: Result[T]) -> Result[T]
  case result
    ok(v) -> ok(v)
    error(e) -> Bond.bad_request("#{message}: #{e.message}")
  end
end

def wants_usage(body: ChatRequest) -> Bool
  case body.stream_options
    some(options) -> options.include_usage.unwrap_or(false)
    none -> false
  end
end

# Waits for the first event, so a request the scheduler refuses is still its status (the head of a stream cannot be
# taken back), then hands the rest to a task: the route returns and Bond sends each event as it is made.
def stream_answer(
  api: Api,
  conn: Bond.Conn,
  reply: Channel[Serving.Event],
  prompt_tokens: Int,
  max_tokens: Int,
  include_usage: Bool,
) -> Result[Bond.Conn]
  case reply.recv(ANSWER_TIMEOUT_MS)
    ok(first) ->
      case first
        Serving.Event.Failed(message) ->
          reply.close
          return ok(Bond.halt(conn, 503, message)) if message.include?("saturated") or message.include?("stopped")

          return ok(Bond.halt(conn, 500, message))
        _ -> nil
      end

      events: Channel[Bytes] = Channel.new(16)
      pump = Pump(
        tok: api.tok,
        reply: reply,
        events: events,
        generated: [],
        sent: "",
        id: "chatcmpl-#{api.counters.add("requests", 1)}",
        created: Time.now_ms() / 1000,
        model: api.model_name,
        include_usage: include_usage,
        prompt_tokens: prompt_tokens,
        max_tokens: max_tokens,
      )

      # Owning: the task owns `reply` and `events`, so a crash of the pump closes them and the response ends.
      Task.spawn_owning(&pump_events, pump_after(pump, first))
      ok(Bond.sse(conn, events))
    error(_) ->
      reply.close
      ok(Bond.halt(conn, 504, "the scheduler did not answer"))
  end
end

# The first event belongs to the pump, not to the route: it is already consumed, so it goes in before the task.
def pump_after(pump: Pump, first: Serving.Event) -> Pump
  case first
    Serving.Event.Token(id) -> pump.with(generated: [id])
    _ -> pump
  end
end

# Turns the scheduler's events into `chat.completion.chunk` events: the role first, a content chunk per piece, then
# the finish reason (and the usage when asked) and `[DONE]`. When an event cannot be sent (the client left, or
# stopped reading) it closes the reply channel — the scheduler's next send to it fails and drops the sequence,
# freeing the slot — and closes the stream: the next write would fail the same way, so this is where a disconnect
# is observed.
def pump_events(pump: Pump) -> Void
  if send_chunk(pump, Delta(role: JSON.Field.Value("assistant"), content: JSON.Field.Value("")), JSON.Field.Null)
    relay(pump)
  end

  pump.reply.close
  pump.events.close
end

def relay(pump: Pump) -> Void
  sent = pump.sent
  generated = pump.generated
  if not generated.empty?
    case emit_text(pump.tok, sent, generated, false)
      ok(owed) ->
        sent = owed.sent
        if not owed.emit.empty? and not send_chunk(pump, Delta(content: JSON.Field.Value(owed.emit)), JSON.Field.Null)
          return
        end

      error(e) ->
        send_error(pump, e.message)
        return
    end
  end

  while true
    case pump.reply.recv(PIECE_WAIT_MS)
      ok(event) ->
        case event
          Serving.Event.Token(id) ->
            generated = generated.push(id)
            case emit_text(pump.tok, sent, generated, false)
              ok(owed) ->
                sent = owed.sent
                if (
                  not owed.emit.empty? and
                  not send_chunk(pump, Delta(content: JSON.Field.Value(owed.emit)), JSON.Field.Null)
                )
                  return
                end

              error(e) ->
                send_error(pump, e.message)
                return
            end

          Serving.Event.Done ->
            case emit_text(pump.tok, sent, generated, true)
              ok(owed) ->
                if (
                  not owed.emit.empty? and
                  not send_chunk(pump, Delta(content: JSON.Field.Value(owed.emit)), JSON.Field.Null)
                )
                  return
                end

                finish(pump, owed.count, reason(pump, owed.count))
                return
              error(e) ->
                send_error(pump, e.message)
                return
            end

          Serving.Event.Failed(message) ->
            # The head went out as a 200: all that is left is an error event, and no `[DONE]`.
            send_error(pump, message)
            return
        end

      error(_) ->
        # The scheduler stalled: say so, since the head already went out as a 200.
        send_error(pump, "the scheduler did not answer")
        return
    end
  end
end

# Why the answer ended: the stop id cut it, or the budget ran out (`Done` carries no reason, but a stop id
# never becomes a token, so an answer shorter than the budget stopped).
def reason(pump: Pump, completion_tokens: Int) -> String
  return "length" if completion_tokens >= pump.max_tokens

  "stop"
end

def finish(pump: Pump, completion_tokens: Int, reason: String) -> Void
  return if not send_chunk(pump, Delta(), JSON.Field.Value(reason))

  return if pump.include_usage and not send_usage(pump, completion_tokens)

  _ = pump.events.send(Bond.sse_event("[DONE]"), EVENT_SEND_TIMEOUT_MS)
end

def send_chunk(pump: Pump, delta: Delta, finish: JSON.Field[String]) -> Bool
  chunk = Chunk(
    id: pump.id,
    object: "chat.completion.chunk",
    created: pump.created,
    model: pump.model,
    choices: [ChunkChoice(index: 0, delta: delta, finish_reason: finish)],
  )

  pump.events.send(Bond.sse_event(JSON.encode(chunk)), EVENT_SEND_TIMEOUT_MS).ok?
end

# The usage chunk of `stream_options.include_usage`: no choices, only the counts.

def send_usage(pump: Pump, completion_tokens: Int) -> Bool
  usage = Usage(
    prompt_tokens: pump.prompt_tokens,
    completion_tokens: completion_tokens,
    total_tokens: pump.prompt_tokens + completion_tokens,
  )

  chunk = Chunk(
    id: pump.id,
    object: "chat.completion.chunk",
    created: pump.created,
    model: pump.model,
    choices: [],
    usage: JSON.Field.Value(usage),
  )

  pump.events.send(Bond.sse_event(JSON.encode(chunk)), EVENT_SEND_TIMEOUT_MS).ok?
end

def send_error(pump: Pump, message: String) -> Void
  failure = JSON.from({"error": {"message": message, "type": "server_error"}})
  _ = pump.events.send(Bond.sse_event(JSON.encode(failure)), EVENT_SEND_TIMEOUT_MS)
end
