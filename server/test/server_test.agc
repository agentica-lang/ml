## The server package against the chat-capable tiny Qwen3 of `chat_peer.py` (same builder and config as
## `transformers/test/qwen_peer.py`, with a vocabulary of every byte so the Qwen3 chat template encodes): the
## existing `tiny` fixture cannot serve chat, because its nine-token vocabulary has neither the template's markers
## nor the prompt's bytes. Every wait is a channel wait with a generous bound, never a sleep. Run from `server/`.
##
## The answers are short on purpose: a reference step of this fixture costs about two million interpreter steps, and
## CI gives a test file 10M x (its tests + 1) steps, which this file spends whole. The tests that script the scheduler
## (`Fake`, `BurstJob`) need the tokenizer alone: no model step, and what a refusing, breaking, out-of-vocabulary or
## gone scheduler leaves a client is still real Bond and real HTTP.

import Api from "../api"
import Server from "../server"
import Bond from "pkg/bond/bond"
import Bpe from "pkg/transformers/tokenizer/bpe"
import Chat from "pkg/transformers/chat"
import Generate from "pkg/transformers/generate"
import Qwen3 from "pkg/transformers/qwen3"
import Serving from "pkg/transformers/serving"
import HTTP from "std/http"
import Tensor from "pkg/tensor/tensor"

## How long one wait may take: the reference steps are quick, and loading the model takes the longest; the bound
## only fails a stuck run.
WAIT_MS = 180_000

## How long a stream that has nothing to wait for may take: one prefill step of the fixture, so a stream that stalls
## shows up long before the minute the relay would spend waiting for a piece that cannot come.
SETTLE_WAIT_MS = 15_000

## How long a sequence whose client left may take to be dropped: the relay notices on a write that fails or times out
## (10 s of send timeout at worst), so this bound only fails a sequence that is never dropped.
DROP_WAIT_MS = 30_000

## How long one scripted scheduler send or receive may take: the fake answers at once, so the bound only fails a task
## that was left behind.
FAKE_MS = 10_000

## What a running server under test offers: the fixture folder and the port.
struct Live
  dir: String
  port: Int
end

## One streamed answer, as the test reads it: the payload of every `data:` line in order (`[DONE]` last).
struct Streamed
  payloads: List[String]
end

## A wire chunk, as the test reads it: `finish_reason` is `none` until the last content chunk, and `usage` is set
## only on the usage chunk of `include_usage`.
struct TestDelta
  role: Optional[String] = none
  content: Optional[String] = none
end

struct TestChoice
  index: Int
  delta: Optional[TestDelta] = none
  finish_reason: Optional[String] = none
end

struct TestUsage
  prompt_tokens: Int
  completion_tokens: Int
  total_tokens: Int
end

struct TestChunk
  id: String
  object: String
  created: Int
  model: String
  choices: List[TestChoice]
  usage: Optional[TestUsage] = none
end

## `GET /health` answers 200 with the status.
def test_health_answers_ok() -> Result[Void]
  with_server do |live|
    response = try HTTP.get_response("http://127.0.0.1:#{live.port}/health")
    try Test.eq(response.status, 200)
    try Test.eq(try HTTP.text(response.body, response.headers), "{\"status\":\"ok\"}")
    ok()
  end
end

## Bad requests are bounded 400s naming the problem: no messages, no stream, a sampling knob, a budget outside
## 1..512, a prompt past the character cap or the context, or a body that is not JSON.
def test_validation_rejects_bad_requests() -> Result[Void]
  with_server do |live|
    for c in cases().push(long_body()).push(long_chars())
      response = try post(live, c.body)
      try Test.eq(response.status, 400)
      text = try HTTP.text(response.body, response.headers)
      try Test.assert(text.include?(c.part), "body #{c.body} gave #{text}, without #{c.part}")
    end

    response = try HTTP.post_response(
      "http://127.0.0.1:#{live.port}/v1/chat/completions",
      "not json".bytes,
      [HTTP.Header(name: "Content-Type", value: "application/json")],
    )

    try Test.eq(response.status, 400)
    ok()
  end
end

## One streamed answer has the OpenAI shape: the role first, content deltas that concatenate to the standalone
## greedy text, the `length` finish reason, and `[DONE]`; its id names the request and its model the server's, and
## without `stream_options.include_usage` no chunk carries usage.
def test_stream_matches_standalone() -> Result[Void]
  with_server do |live|
    streamed = try complete(live, "hi", 3)
    chunks = try decode_chunks(streamed)
    try Test.eq(chunks.size >= 3, true)
    try Test.eq(try delta_content(chunks[0]), "")
    try Test.eq(try delta_role(chunks[0]), "assistant")
    try Test.eq(last_finish(chunks), some("length"))
    try Test.eq(streamed.payloads.last, "[DONE]")
    try Test.eq(chunks[0].id.start_with?("chatcmpl-"), true)
    try Test.eq(chunks[0].model, "chat-test")

    parts: List[String] = []
    for chunk in chunks
      parts = parts.push(try delta_content(chunk))
      try Test.eq(chunk.usage, none)
    end

    want = try standalone(live.dir, "hi", 3)
    try Test.eq(parts.join(""), want)
    ok()
  end
end

## Two simultaneous clients both hear content before either hears `[DONE]`, and each content is the standalone greedy
## text of the one prompt they share.
def test_two_clients_advance_together() -> Result[Void]
  with_server do |live|
    texts = try interleaved(live, "hi", 6, "hi", 6)
    want = try standalone(live.dir, "hi", 6)
    for text in texts
      try Test.eq(text, want)
    end

    ok()
  end
end

## A client that leaves after one delta neither stalls the next one nor leaks its slot: the next client streams to
## `[DONE]` and matches the standalone greedy text, and two clients opened after that advance together, which needs
## both scheduler slots, so the dropped sequence freed its own.
def test_disconnect_neither_stalls_nor_leaks() -> Result[Void]
  with_server do |live|
    leaving = try HTTP.start("http://127.0.0.1:#{live.port}")
    head = try HTTP.send_request(
      leaving,
      "POST",
      "/v1/chat/completions",
      request_body("hi", 8).bytes,
      [HTTP.Header(name: "Content-Type", value: "application/json")],
    )

    try Test.eq(head.status, 200)
    leaving_first = try read_until_content_conn(leaving, 1)
    try Test.eq(leaving_first.size >= 2, true)
    # The disconnect is a true close: the server's next write fails and the sequence is dropped.
    HTTP.finish(leaving)
    peer = try open_stream(live, "hi", 6)
    peer_first = try read_until_content(peer, 1)
    try Test.eq(peer_first.size >= 2, true)
    peer_rest = try drain(peer, "".bytes, peer_first)
    done = Task.await(peer.done)
    try done
    peer_text = try join_text(try decode_chunks(Streamed(payloads: peer_rest)))
    want = try standalone(live.dir, "hi", 6)
    try Test.eq(peer_text, want)
    texts = try interleaved(live, "hi", 6, "hi", 6)
    for text in texts
      try Test.eq(text, want)
    end

    ok()
  end
end

## `include_usage` appends a choiceless chunk with the counts (a stream without it carries none, see the shape test).
def test_usage_chunk_reports_counts() -> Result[Void]
  with_server do |live|
    streamed = try complete_usage(live, "hi", 3)
    chunks = try decode_chunks(streamed)
    usages: List[TestUsage] = []
    for chunk in chunks
      case chunk.usage
        some(u) -> usages = usages.push(u)
        none -> nil
      end
    end

    try Test.eq(usages.size, 1)
    prompt = try prompt_ids(live.dir, "hi")
    try Test.eq(usages[0].prompt_tokens, prompt.size)
    try Test.eq(usages[0].completion_tokens, 3)
    try Test.eq(usages[0].total_tokens, prompt.size + 3)
    try Test.eq(chunks.last.choices, [])
    ok()
  end
end

## The incremental decoder holds back the whole trailing run of replacement characters until the next tokens resolve
## it, flushes it whole on the final call, and recovers when the completion replaces part of what was held.
def test_emit_text_holds_partial_characters() -> Result[Void]
  with_chat do |dir|
    tok = try Bpe.open("#{dir}/chat/tokenizer.json")
    # Byte 195 alone is half of `é`: held back, then completed by byte 169.
    held = try Api.emit_text(tok, "", [195], false)
    try Test.eq(held.emit, "")
    try Test.eq(held.sent, "")
    done = try Api.emit_text(tok, held.sent, [195, 169], false)
    try Test.eq(done.emit, "é")
    try Test.eq(done.sent, "é")
    try Test.eq(done.count, 2)
    # The final call flushes even the half character.
    flushed = try Api.emit_text(tok, "", [195], true)
    try Test.eq(flushed.emit, "�")
    # Two halves scrub to two replacements and the run of both waits: the second may still complete, the first is
    # one byte of a character of its own. Nothing of the run goes out while it is trailing.
    two = try Api.emit_text(tok, "", [195, 195], false)
    try Test.eq(two.emit, "")
    try Test.eq(two.sent, "")
    healed = try Api.emit_text(tok, two.sent, [195, 195, 169], false)
    try Test.eq(healed.emit, "�é")
    try Test.eq(healed.sent, "�é")
    # Plain text passes through, and the count is the tokens so far.
    plain = try Api.emit_text(tok, "", [104, 105], false)
    try Test.eq(plain.emit, "hi")
    try Test.eq(plain.count, 2)
    more = try Api.emit_text(tok, plain.sent, [104, 105, 33], false)
    try Test.eq(more.emit, "!")
    try Test.eq(more.sent, "hi!")
    # What was never sent cannot extend it.
    try Test.error?(Api.emit_text(tok, "bye", [104, 105], false))
    ok()
  end
end

## A three- and a four-byte character arrive one byte per token: nothing of either goes out before its last byte,
## and what the stream owes at every step is a prefix of the answer.
def test_emit_text_holds_a_split_multibyte_character() -> Result[Void]
  with_chat do |dir|
    tok = try Bpe.open("#{dir}/chat/tokenizer.json")
    three = try emitted_split(tok, [227, 129, 130])
    try Test.eq(three, "hiあ")
    four = try emitted_split(tok, [240, 159, 152, 128])
    try Test.eq(four, "hi😀")
    ok()
  end
end

## An answer whose first pick is a stop id ends before its first token (`Serving` sends `Done` alone): the stream
## settles at once with the role, `finish_reason` `stop`, the usage when asked and `[DONE]`, never an error after a
## minute of waiting for a token that cannot come.
def test_a_stop_first_answer_settles_at_once() -> Result[Void]
  with_chat do |dir|
    stop = [try first_pick(dir, "hi")]
    with_stop_server(dir, stop) do |live|
      streamed = try complete_at_once(live, request_body("hi", 4))
      chunks = try decode_chunks(streamed)
      # The role chunk and the finish, nothing between them: the answer holds no token.
      try Test.eq(chunks.size, 2)
      try Test.eq(try delta_role(chunks[0]), "assistant")
      try Test.eq(try delta_content(chunks[0]), "")
      try Test.eq(last_finish(chunks), some("stop"))
      try Test.eq(streamed.payloads.last, "[DONE]")
      counted = try decode_chunks(try complete_at_once(live, usage_body("hi", 4)))
      usage = try only_usage(counted)
      try Test.eq(usage.completion_tokens, 0)
      prompt = try prompt_ids(dir, "hi")
      try Test.eq(usage.prompt_tokens, prompt.size)
      try Test.eq(usage.total_tokens, prompt.size)
      ok()
    end
  end
end

## An id the tokenizer has no piece for (the model's vocabulary outrunning its tokenizer) is never streamed as text
## nor dropped: the answer that already went out stands and the stream ends with the reason, and no `[DONE]`.
def test_an_undecodable_id_after_content_is_an_error_event() -> Result[Void]
  with_chat do |dir|
    try with_fake(dir, Fake.BadAfter(1000)) do |live|
      stream = try open_body(live, request_body("hi", 4))
      try Test.eq(stream.status, 200)
      payloads = try read_to_close(stream)
      try Test.eq(payloads.last == "[DONE]", false)
      chunks = try decode_chunks(Streamed(payloads: payloads.take(payloads.size - 1)))
      try Test.eq(try join_text(chunks), "h")
      Test.eq(
        try JSON.parse(payloads.last),
        JSON.from({"error": {"message": "no token has the id 1000", "type": "server_error"}}),
      )
    end

    ok()
  end
end

## The same id as the very first token leaves no content at all: the role chunk and the error event, never a
## half-written answer, and no `[DONE]`.
def test_an_undecodable_first_id_is_an_error_event() -> Result[Void]
  with_chat do |dir|
    try with_fake(dir, Fake.BadFirst(1000)) do |live|
      stream = try open_body(live, request_body("hi", 4))
      try Test.eq(stream.status, 200)
      payloads = try read_to_close(stream)
      try Test.eq(payloads.size, 2)
      try Test.eq(payloads.last == "[DONE]", false)
      chunks = try decode_chunks(Streamed(payloads: payloads.take(1)))
      try Test.eq(try delta_role(chunks[0]), "assistant")
      Test.eq(
        try JSON.parse(payloads.last),
        JSON.from({"error": {"message": "no token has the id 1000", "type": "server_error"}}),
      )
    end

    ok()
  end
end

## A client that leaves mid-stream ends its sequence on the relay's next failed write: the pump closes the reply, the
## scheduler hears the closed channel on its next send, and no `[DONE]` follows, so nothing claims a success that did
## not happen.
def test_a_client_that_leaves_mid_stream_drops_its_sequence() -> Result[Void]
  with_chat do |dir|
    closed: Channel[Bool] = Channel.new(1)
    serving = burst_server(200, closed)
    result = with_scripted(dir, serving) do |live|
      leaving = try HTTP.start("http://127.0.0.1:#{live.port}")
      head = try HTTP.send_request(
        leaving,
        "POST",
        "/v1/chat/completions",
        request_body("hi", 8).bytes,
        [HTTP.Header(name: "Content-Type", value: "application/json")],
      )

      try Test.eq(head.status, 200)
      first = try read_until_content_conn(leaving, 1)
      try Test.eq(first.size >= 2, true)
      # A true close: the relay's next write to this client fails and the sequence is dropped.
      HTTP.finish(leaving)
      try Test.eq(try closed.recv(DROP_WAIT_MS), true)
      ok()
    end

    try close_fake(serving)
    result
  end
end

## The stop ids are the added tokens that end an answer.
def test_stop_ids_names_the_end_tokens() -> Result[Void]
  with_chat do |dir|
    tok = try Bpe.open("#{dir}/chat/tokenizer.json")
    try Test.eq(Api.stop_ids(tok), [257, 258])
    ok()
  end
end

## Starting refuses a non-loopback host and a folder with no tokenizer, before anything listens or loads.
def test_start_refuses_a_bad_host_or_folder() -> Result[Void]
  with_chat do |dir|
    try Test.error?(Server.start(dir, "chat-test", "0.0.0.0", 0))
    try Test.error?(Server.start("/none-such-dir", "chat-test", "127.0.0.1", 0))
    ok()
  end
end

## A port already taken is an error from `start`, which closes the tokenizer, the model, the generator and the
## scheduler it opened, so the server on that port is still the only one.
def test_start_refuses_a_taken_port() -> Result[Void]
  with_chat do |dir|
    running = try Server.start("#{dir}/chat", "chat-test", "127.0.0.1", 0)
    taken = Server.start("#{dir}/chat", "chat-test", "127.0.0.1", Server.local_port(running))
    try Test.error?(taken)
    response = try HTTP.get_response("http://127.0.0.1:#{Server.local_port(running)}/health")
    try Test.eq(response.status, 200)
    try Server.stop(running)
    ok()
  end
end

## A scheduler that refuses a request is answered before the head, so the client reads a plain body and never a 200
## stream: a saturated or stopped engine is a 503, anything else a 500, each with the scheduler's own reason.
def test_a_refused_request_is_a_status_not_a_stream() -> Result[Void]
  with_chat do |dir|
    try with_fake(dir, Fake.Refuse("serving: saturated, no room for another request")) do |live|
      response = try post(live, request_body("hi", 4))
      try Test.eq(response.status, 503)
      text = try HTTP.text(response.body, response.headers)
      try Test.eq(text, "serving: saturated, no room for another request")
      Test.eq(header_value(response, "content-type").include?("text/event-stream"), false)
    end

    try with_fake(dir, Fake.Refuse("qwen3: the step failed")) do |live|
      response = try post(live, request_body("hi", 4))
      try Test.eq(response.status, 500)
      text = try HTTP.text(response.body, response.headers)
      Test.eq(text, "qwen3: the step failed")
    end

    ok()
  end
end

## A queue that takes nothing is a 503 at once, with no wait for an answer no scheduler will send: the inbox is
## closed, so `Serving.submit` says the server is stopped before the head.
def test_a_closed_queue_is_a_503_at_once() -> Result[Void]
  with_chat do |dir|
    serving = fake_server(Fake.Refuse("unused"))
    serving.inbox.close
    result = with_scripted(dir, serving) do |live|
      response = try post(live, request_body("hi", 4))
      try Test.eq(response.status, 503)
      ok()
    end

    try close_fake(serving)
    result
  end
end

## A scheduler that breaks after a token is never a silent success: the content that went out stands, the stream ends
## with an `error` event that names the reason, and there is no `[DONE]` after it.
def test_a_broken_scheduler_ends_the_stream_with_an_error() -> Result[Void]
  with_chat do |dir|
    try with_fake(dir, Fake.TokenThen("qwen3: the step failed")) do |live|
      stream = try open_body(live, request_body("hi", 4))
      try Test.eq(stream.status, 200)
      payloads = try read_to_close(stream)
      try Test.eq(payloads.size >= 3, true)
      try Test.eq(payloads.last == "[DONE]", false)
      chunks = try decode_chunks(Streamed(payloads: payloads.take(payloads.size - 1)))
      try Test.eq(try join_text(chunks), "h")
      failure = try JSON.parse(payloads.last)
      Test.eq(failure, JSON.from({"error": {"message": "qwen3: the step failed", "type": "server_error"}}))
    end

    ok()
  end
end

private

struct Case
  body: String
  part: String
end

struct Taken
  lines: List[String]
  rest: Bytes
end

# What a client of a pair reports on the pair's channel, in order: its first content arrived, then how its stream
# ended (its joined text, or why it did not end). The order is the proof of the overlap: a stream that ended before
# the other made content sends `Finished` before its peer's `Content`.
enum Note
  Content(Int)
  Finished(Int, String)
end

struct Reader
  live: Live
  tag: Int
  content: String
  max_tokens: Int
  inbox: Channel[Note]
end

# What the fake scheduler answers a request with: a `Failed` before any token, one token and then a `Failed`, or an id
# the tokenizer has no piece for (a model whose vocab outruns its tokenizer, which `Bpe.decode` must refuse). Nothing
# steps a model: each is a thing a scheduler can say that the relay owes an honest answer to.
enum Fake
  Refuse(String)
  TokenThen(String)
  BadFirst(Int)
  BadAfter(Int)
end

struct FakeJob
  inbox: Channel[Serving.Command]
  script: Fake
end

# A scheduler that answers with a long run of one token and reports on `closed` when a send fails: the relay stopped
# taking events, which is what a client that left looks like from the scheduler's side.
struct BurstJob
  inbox: Channel[Serving.Command]
  count: Int
  closed: Channel[Bool]
end

def cases() -> List[Case]
  [
    Case(body: "{\"messages\":[]}", part: "messages must not be empty"),
    Case(body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", part: "stream must be true"),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}",
      part: "stream must be true",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"temperature\":0.7}",
      part: "temperature is not supported",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"top_p\":0.9}",
      part: "top_p is not supported",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"seed\":7}",
      part: "seed is not supported",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"n\":2}",
      part: "n is not supported",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"stop\":[\"x\"]}",
      part: "stop is not supported",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"max_tokens\":0}",
      part: "needs at least 1",
    ),
    Case(
      body: "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"max_tokens\":100000}",
      part: "at most 512",
    ),
    Case(body: "{\"messages\":[{\"role\":\"wizard\",\"content\":\"hi\"}],\"stream\":true}", part: "role"),
    Case(
      body: (
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"max_tokens\":1,\"stream_options\":{\"include_usage\":\"yes\"}}"
      ),
      part: "invalid body",
    ),
  ]
end

# A prompt that fits the character cap but not the 64 positions beside one new token.
def long_body() -> Case
  content = List.repeat("e", 50).join("")
  Case(
    body: "{\"messages\":[{\"role\":\"user\",\"content\":\"#{content}\"}],\"stream\":true,\"max_tokens\":1}",
    part: "do not fit",
  )
end

# A prompt past the character cap: refused before it is encoded.
def long_chars() -> Case
  content = List.repeat("e", Api.MAX_PROMPT_CHARS + 1).join("")
  Case(body: request_body(content, 1), part: "characters, at most")
end

# The folder the `chat_peer.py` script (run from `server/`) writes, removed after.
def with_chat(body: Fn[String, Result[Void]]) -> Result[Void]
  spawned = try Process.spawn_captured("python3", ["-I", "test/chat_peer.py"])
  out = try Process.output(spawned, 60_000)
  code = try Process.wait(spawned, 60_000)
  return error("test/chat_peer.py exited with #{code}: #{out.stderr}") if code != 0

  dir = out.stdout.strip.split("\n").last
  if not dir.start_with?("/") or not dir.split("/").last.start_with?("agentica-chat-")
    return error("test/chat_peer.py printed no fixture folder: #{dir}")
  end

  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

def with_server(body: Fn[Live, Result[Void]]) -> Result[Void]
  with_chat do |dir|
    running = try Server.start("#{dir}/chat", "chat-test", "127.0.0.1", 0)
    result = body.call(Live(dir: dir, port: Server.local_port(running)))
    try Server.stop(running)
    result
  end
end

# The routes over a scheduler the test scripts (real Bond and real HTTP around it, no model): what a `Failed` before
# the head, a `Failed` after a token, or a queue that takes nothing looks like to a client.
def with_fake(dir: String, script: Fake, body: Fn[Live, Result[Void]]) -> Result[Void]
  serving = fake_server(script)
  result = with_scripted(dir, serving, body)
  try close_fake(serving)
  result
end

def with_scripted(dir: String, serving: Serving.Server, body: Fn[Live, Result[Void]]) -> Result[Void]
  tok = try Bpe.open("#{dir}/chat/tokenizer.json")
  api = Api.Api(
    tok: tok,
    serving: serving,
    model_name: "chat-test",
    model_type: "qwen3",
    max_positions: 64,
    stop: Api.stop_ids(tok),
    default_max_tokens: Api.DEFAULT_MAX_TOKENS,
    max_max_tokens: Api.MAX_MAX_TOKENS,
    counters: ATS.new(),
  )

  server = try Bond.start(Api.app(api), "127.0.0.1", 0)
  result = body.call(Live(dir: dir, port: Bond.local_port(server)))
  Bond.stop(server)
  result
end

# A `Serving.Server` whose answers are scripted: one task takes the requests off the inbox and answers each on its own
# reply channel, so the relay's contract is exercised with no model and no step.
def fake_server(script: Fake) -> Serving.Server
  inbox: Channel[Serving.Command] = Channel.new(8)
  Serving.Server(
    inbox: inbox,
    task: Task.spawn(&fake_serve, FakeJob(inbox: inbox, script: script)),
    limits: Serving.Limits(max_active: Api.MAX_ACTIVE, max_pending: Api.MAX_PENDING),
    pairs: Channel.new(1),
  )
end

# Ends the fake scheduler and joins its task, in the order `Serving.stop` uses.
def close_fake(serving: Serving.Server) -> Result[Void]
  serving.inbox.close
  Task.await(serving.task)
end

def fake_serve(job: FakeJob) -> Result[Void]
  while true
    case job.inbox.recv(FAKE_MS)
      ok(Serving.Command.Request(request)) -> answer(request.reply, job.script)
      ok(Serving.Command.Stop) -> return ok()
      ok(Serving.Command.Cancel(_)) -> nil
      error(_) -> return ok()
    end
  end

  ok()
end

# A `Serving.Server` whose answer is a run of `count` tokens: the client reads the first, leaves, and the run has to be
# cut by the relay's own failed writes.
def burst_server(count: Int, closed: Channel[Bool]) -> Serving.Server
  inbox: Channel[Serving.Command] = Channel.new(8)
  Serving.Server(
    inbox: inbox,
    task: Task.spawn(&burst_serve, BurstJob(inbox: inbox, count: count, closed: closed)),
    limits: Serving.Limits(max_active: Api.MAX_ACTIVE, max_pending: Api.MAX_PENDING),
    pairs: Channel.new(1),
  )
end

def burst_serve(job: BurstJob) -> Result[Void]
  while true
    case job.inbox.recv(FAKE_MS)
      ok(Serving.Command.Request(request)) -> burst_answer(request.reply, job.count, job.closed)
      ok(Serving.Command.Stop) -> return ok()
      ok(Serving.Command.Cancel(_)) -> nil
      error(_) -> return ok()
    end
  end

  ok()
end

# Sends `count` tokens, one at a time: `true` on `closed` when a send fails (the relay closed the reply), `false` when
# every one went out.
def burst_answer(reply: Address[Serving.Event], count: Int, closed: Channel[Bool]) -> Void
  for _i in 0...count
    if not reply.send(Serving.Event.Token(104), FAKE_MS).ok?
      _ = closed.send(true, FAKE_MS)
      return
    end
  end

  _ = closed.send(false, FAKE_MS)
end

def answer(reply: Address[Serving.Event], script: Fake) -> Void
  case script
    Fake.Refuse(message) -> _ = reply.send(Serving.Event.Failed(message), FAKE_MS)
    Fake.TokenThen(message) ->
      _ = reply.send(Serving.Event.Token(104), FAKE_MS)
      _ = reply.send(Serving.Event.Failed(message), FAKE_MS)
    Fake.BadFirst(id) -> _ = reply.send(Serving.Event.Token(id), FAKE_MS)
    Fake.BadAfter(id) ->
      _ = reply.send(Serving.Event.Token(104), FAKE_MS)
      _ = reply.send(Serving.Event.Token(id), FAKE_MS)
  end
end

# The one value of `name` in a response, or "" when it has none.
def header_value(response: HTTP.Response, name: String) -> String = HTTP.header_values(response.headers, name).join("")

# The routes over an `Api` whose stop ids are `stop` (`Server.start` takes the tokenizer's): what a sequence that
# ends before its first token needs.
def with_stop_server(dir: String, stop: List[Int], body: Fn[Live, Result[Void]]) -> Result[Void]
  tok = try Bpe.open("#{dir}/chat/tokenizer.json")
  model = try Qwen3.open("#{dir}/chat", Tensor.Backend.Reference)
  generator = try Generate.open(model)
  limits = Serving.Limits(max_active: Api.MAX_ACTIVE, max_pending: Api.MAX_PENDING)
  serving = try Serving.start(generator, limits)
  api = Api.Api(
    tok: tok,
    serving: serving,
    model_name: "chat-test",
    model_type: model.config.model_type,
    max_positions: model.config.max_position_embeddings,
    stop: stop,
    default_max_tokens: Api.DEFAULT_MAX_TOKENS,
    max_max_tokens: Api.MAX_MAX_TOKENS,
    counters: ATS.new(),
  )

  server = try Bond.start(Api.app(api), "127.0.0.1", 0)
  result = body.call(Live(dir: dir, port: Bond.local_port(server)))
  Bond.stop(server)
  try Serving.stop(serving)
  Generate.close(generator)
  Qwen3.close_model(model)
  result
end

def post(live: Live, body: String) -> Result[HTTP.Response]
  HTTP.post_response(
    "http://127.0.0.1:#{live.port}/v1/chat/completions",
    body.bytes,
    [HTTP.Header(name: "Content-Type", value: "application/json")],
  )
end

def request_body(content: String, max_tokens: Int) -> String
  (
    "{\"model\":\"qwen3\",\"messages\":[{\"role\":\"user\",\"content\":\"#{content}\"}],\"stream\":true,\"max_tokens\":#{max_tokens}}"
  )
end

def usage_body(content: String, max_tokens: Int) -> String
  (
    "{\"model\":\"qwen3\",\"messages\":[{\"role\":\"user\",\"content\":\"#{content}\"}],\"stream\":true,\"max_tokens\":#{max_tokens},\"stream_options\":{\"include_usage\":true}}"
  )
end

# A whole streamed answer: opens the stream and reads it to `[DONE]`.
def complete(live: Live, content: String, max_tokens: Int) -> Result[Streamed]
  complete_body(live, request_body(content, max_tokens))
end

def complete_usage(live: Live, content: String, max_tokens: Int) -> Result[Streamed]
  complete_body(live, usage_body(content, max_tokens))
end

def complete_body(live: Live, body: String) -> Result[Streamed]
  stream = try open_body(live, body)
  try Test.eq(stream.status, 200)
  read_whole(stream)
end

# A stream that must settle with no wait: the head at once, then every piece within `SETTLE_WAIT_MS`, so a relay that
# waits for a piece that cannot come fails the test instead of holding it.
def complete_at_once(live: Live, body: String) -> Result[Streamed]
  stream = try HTTP.open(
    "POST",
    "http://127.0.0.1:#{live.port}/v1/chat/completions",
    body.bytes,
    [HTTP.Header(name: "Content-Type", value: "application/json")],
    HTTP.ClientOptions(timeout_ms: SETTLE_WAIT_MS, read_timeout_ms: SETTLE_WAIT_MS),
  )

  try Test.eq(stream.status, 200)
  read_whole(stream)
end

def open_stream(live: Live, content: String, max_tokens: Int) -> Result[HTTP.Stream]
  stream = try open_body(live, request_body(content, max_tokens))
  try Test.eq(stream.status, 200)
  ok(stream)
end

def open_body(live: Live, body: String) -> Result[HTTP.Stream]
  HTTP.open(
    "POST",
    "http://127.0.0.1:#{live.port}/v1/chat/completions",
    body.bytes,
    [HTTP.Header(name: "Content-Type", value: "application/json")],
    HTTP.ClientOptions(timeout_ms: WAIT_MS, read_timeout_ms: WAIT_MS),
  )
end

# The `data:` payloads of `stream` until it closes, `[DONE]` or not: what a stream that fails owes its client.
def read_to_close(stream: HTTP.Stream) -> Result[List[String]]
  buf = "".bytes
  payloads: List[String] = []
  while true
    taken = try take_lines(buf)
    buf = taken.rest
    for line in taken.lines
      if line.start_with?("data: ")
        payloads = payloads.push(line.slice(6, line.length - 6))
      end
    end

    case stream.body.recv(WAIT_MS)
      ok(piece) -> buf = buf + piece
      error(_) -> return ok(payloads)
    end
  end

  ok(payloads)
end

# Reads `stream` to `[DONE]`: the payloads in order. A close without `[DONE]` is a cut stream, never a pass.
def read_whole(stream: HTTP.Stream) -> Result[Streamed]
  streamed = try drain(stream, "".bytes, [])
  done = Task.await(stream.done)
  try done
  ok(Streamed(payloads: streamed))
end

# The `data:` payloads of `stream` until `[DONE]`, reading to the closed error: the server closes the stream right
# after `[DONE]`, so a close without it means the stream was cut.
def drain(stream: HTTP.Stream, rest: Bytes, out: List[String]) -> Result[List[String]]
  buf = rest
  payloads = out
  while true
    taken = try take_lines(buf)
    buf = taken.rest
    for line in taken.lines
      if line.start_with?("data: ")
        payloads = payloads.push(line.slice(6, line.length - 6))
      end
    end

    return ok(payloads) if not payloads.empty? and payloads.last == "[DONE]"
    case stream.body.recv(WAIT_MS)
      ok(piece) -> buf = buf + piece
      error(_) -> return error("the stream closed without [DONE]")
    end
  end

  ok(payloads)
end

# The complete lines of `buf` (each one whole UTF-8: no multi-byte character holds a newline byte), and the rest.
def take_lines(buf: Bytes) -> Result[Taken]
  lines: List[String] = []
  start = 0
  for i in 0...buf.size
    if buf.at(i) == 10
      line = try String.from_bytes(buf.slice(start, i - start))
      lines = lines.push(strip_cr(line))
      start = i + 1
    end
  end

  ok(Taken(lines: lines, rest: buf.drop(start)))
end

def strip_cr(line: String) -> String
  return line.slice(0, line.length - 1) if line.end_with?("\r")

  line
end

# The `data:` payloads of `conn` until `want` non-empty content deltas arrived, leaving it open: the head is
# already read, the pieces arrive as they are sent.
def read_until_content_conn(conn: HTTP.Connection, want: Int) -> Result[List[String]]
  buf = "".bytes
  payloads: List[String] = []
  while true
    taken = try take_lines(buf)
    buf = taken.rest
    for line in taken.lines
      if line.start_with?("data: ")
        payloads = payloads.push(line.slice(6, line.length - 6))
      end
    end

    return ok(payloads) if content_count(payloads) >= want
    piece = try HTTP.read_body(conn, 65536)
    return error("the stream closed before #{want} content deltas") if piece.empty?
    buf = buf + piece
  end

  ok(payloads)
end

# Reads until `want` non-empty content deltas arrived (the role chunk is not one), leaving the stream open.
def read_until_content(stream: HTTP.Stream, want: Int) -> Result[List[String]]
  buf = "".bytes
  payloads: List[String] = []
  while true
    taken = try take_lines(buf)
    buf = taken.rest
    for line in taken.lines
      if line.start_with?("data: ")
        payloads = payloads.push(line.slice(6, line.length - 6))
      end
    end

    return ok(payloads) if content_count(payloads) >= want
    case stream.body.recv(WAIT_MS)
      ok(piece) -> buf = buf + piece
      error(_) -> return error("the stream closed before #{want} content deltas")
    end
  end

  ok(payloads)
end

def content_count(payloads: List[String]) -> Int
  n = 0
  for payload in payloads
    if payload != "[DONE]" and has_content(payload)
      n += 1
    end
  end

  n
end

def has_content(payload: String) -> Bool
  case TestChunk.codec.decode(payload)
    ok(chunk) ->
      case delta_content(chunk)
        ok(text) -> not text.empty?
        error(_) -> false
      end

    error(_) -> false
  end
end

def decode_chunks(streamed: Streamed) -> Result[List[TestChunk]]
  chunks: List[TestChunk] = []
  for payload in streamed.payloads
    next if payload == "[DONE]"

    chunks = chunks.push(try TestChunk.codec.decode(payload))
  end

  ok(chunks)
end

def delta_content(chunk: TestChunk) -> Result[String]
  return error("no choice") if chunk.choices.empty?

  delta = try chunk.choices[0].delta.to_result("no delta")
  ok(delta.content.unwrap_or(""))
end

def delta_role(chunk: TestChunk) -> Result[String]
  return error("no choice") if chunk.choices.empty?

  delta = try chunk.choices[0].delta.to_result("no delta")
  delta.role.to_result("no role")
end

def last_finish(chunks: List[TestChunk]) -> Optional[String]
  finishes: List[String] = []
  for chunk in chunks
    if not chunk.choices.empty?
      case chunk.choices[0].finish_reason
        some(r) -> finishes = finishes.push(r)
        none -> nil
      end
    end
  end

  return none if finishes.empty?

  some(finishes.last)
end

def join_text(chunks: List[TestChunk]) -> Result[String]
  parts: List[String] = []
  for chunk in chunks
    parts = parts.push(try delta_content(chunk))
  end

  ok(parts.join(""))
end

def only_usage(chunks: List[TestChunk]) -> Result[TestUsage]
  for chunk in chunks
    return chunk.usage.to_result("no usage") if chunk.usage.some?
  end

  error("no chunk carries usage")
end

# Two streams at once, each opened by the task that reads it, so both requests are written before either head comes
# back: the joined text of each, by tag. Both advanced while the other ran only when the two notes that arrive first
# are the two contents.
def interleaved(live: Live, first: String, first_n: Int, second: String, second_n: Int) -> Result[List[String]]
  inbox: Channel[Note] = Channel.new(4)
  Task.spawn(&reader, Reader(live: live, tag: 0, content: first, max_tokens: first_n, inbox: inbox))
  Task.spawn(&reader, Reader(live: live, tag: 1, content: second, max_tokens: second_n, inbox: inbox))
  first_note = try inbox.recv(WAIT_MS)
  second_note = try inbox.recv(WAIT_MS)
  ahead = try announced(first_note)
  behind = try announced(second_note)
  try Test.eq(ahead != behind, true)
  third = try inbox.recv(WAIT_MS)
  fourth = try inbox.recv(WAIT_MS)
  notes = [third, fourth]
  zero = try ended(notes, 0)
  one = try ended(notes, 1)
  ok([zero, one])
end

# The tag of a `Content` note: any other note means a stream ended before both clients made content.
def announced(note: Note) -> Result[Int]
  case note
    Note.Content(tag) -> ok(tag)
    Note.Finished(_, text) -> error("a stream ended before both streamed: #{text}")
  end
end

# The joined text of the stream tagged `tag`, among the notes that end the pair.
def ended(notes: List[Note], tag: Int) -> Result[String]
  for note in notes
    case note
      Note.Finished(t, text) -> return ok(text) if t == tag

      Note.Content(_) -> nil
    end
  end

  error("the stream of #{tag} never ended")
end

def reader(job: Reader) -> Void
  text = content_note(job)
  _ = job.inbox.send(Note.Finished(job.tag, text), WAIT_MS)
end

# Announces `job`'s first content on the pair's channel (its place in the order is what proves the overlap) and reads
# its stream to the end: the joined text, or the failure, which the test then compares and reports.
def content_note(job: Reader) -> String
  case open_stream(job.live, job.content, job.max_tokens)
    ok(stream) -> first_note(job, stream)
    error(e) -> "FAILED: #{e.message}"
  end
end

def first_note(job: Reader, stream: HTTP.Stream) -> String
  case read_until_content(stream, 1)
    ok(first) ->
      return "FAILED: the pair stopped reading" if not job.inbox.send(Note.Content(job.tag), WAIT_MS).ok?

      rest_text(stream, first)

    error(e) -> "FAILED: #{e.message}"
  end
end

def rest_text(stream: HTTP.Stream, first: List[String]) -> String
  case drained(stream, first)
    ok(text) -> text
    error(e) -> "FAILED: #{e.message}"
  end
end

def drained(stream: HTTP.Stream, first: List[String]) -> Result[String]
  payloads = try drain(stream, "".bytes, first)
  try Task.await(stream.done)
  join_text(try decode_chunks(Streamed(payloads: payloads)))
end

# One character whose bytes arrive one token at a time, after `hi`: nothing of it may go out before its last byte,
# and what the stream owes at every step stays a prefix of the answer. Answers the text the stream emitted.
def emitted_split(tok: Bpe.ByteLevelBpe, bytes: List[Int]) -> Result[String]
  ids: List[Int] = [104, 105]
  sent = ""
  acc = ""
  steps: List[String] = []
  for i in 0...bytes.size
    ids = ids.push(bytes[i])
    owed = try Api.emit_text(tok, sent, ids, false)
    acc = acc + owed.emit
    sent = owed.sent
    steps = steps.push(acc)
    # Until the character's last byte arrives, only what is before it is whole: `hi`, and no replacement.
    if i < bytes.size - 1
      try Test.eq(acc, "hi")
    end
  end

  final = try Api.emit_text(tok, sent, ids, true)
  acc = acc + final.emit
  full = try Bpe.decode(tok, ids)
  try Test.eq(acc, full)
  for step in steps
    try Test.eq(full.start_with?(step), true)
  end

  ok(acc)
end

# The first id the model picks for `content`: the stop id that ends an answer before its first token.
def first_pick(dir: String, content: String) -> Result[Int]
  m = try Qwen3.open("#{dir}/chat", Tensor.Backend.Reference)
  g = try Generate.open(m)
  ids = try Generate.generate(g, try prompt_ids(dir, content), 1, [])
  Generate.close(g)
  Qwen3.close_model(m)
  return error("the model picked no token") if ids.empty?

  ok(ids[0])
end

# The prompt ids the server encodes, and the standalone greedy answer beside them: the stream's content must match.
def prompt_ids(dir: String, content: String) -> Result[List[Int]]
  tok = try Bpe.open("#{dir}/chat/tokenizer.json")
  text = try Chat.apply("qwen3", [Chat.Message(role: "user", content: content)])
  ok(try Bpe.encode(tok, text))
end

def standalone(dir: String, content: String, max_new: Int) -> Result[String]
  tok = try Bpe.open("#{dir}/chat/tokenizer.json")
  stops = Api.stop_ids(tok)
  m = try Qwen3.open("#{dir}/chat", Tensor.Backend.Reference)
  g = try Generate.open(m)
  ids = try Generate.generate(g, try prompt_ids(dir, content), max_new, stops)
  text = try Bpe.decode(tok, ids)
  Generate.close(g)
  Qwen3.close_model(m)
  ok(text)
end
