# server

A local OpenAI-shaped chat server over one Qwen3 model (ml#90), on the transformers scheduler: `GET /health` and a
streaming-only `POST /v1/chat/completions`, served by Bond on a loopback address. It is the first step of the serving
work: greedy, one model, the existing `Serving` scheduler and the reference CPU backend. Later steps (paged KV,
speculative decoding, sampling, other backends, the Responses API, metrics) are not here.

- `server.agc` — the owner chain: `load` opens the tokenizer, the model and the generator in that order, `start` lends
  the generator to the scheduler and starts the HTTP server, `stop` ends the server, then the scheduler, then the
  generator and the model. Also the command line.
- `api.agc` — the routes, the validation of a request, and the relay that turns scheduler events into SSE
  `chat.completion.chunk` events.
- `test/server_test.agc` — the tests, against the fixture `test/chat_peer.py` writes.
- `test/chat_peer.py` — a tiny random Qwen3 (the builder and config of `transformers/test/qwen_peer.py`) with a
  vocabulary of every byte, so the chat template encodes; standard library only.

## Run it

```sh
cd server
agentica install                     # the lock is committed; this fetches the pinned packages
python3 -I test/chat_peer.py         # prints a temporary folder that holds chat/
agentica server.agc --model /tmp/agentica-chat-XXXXXXXX/chat --port 8080
```

`--model` is a folder with `config.json`, `model.safetensors` and `tokenizer.json`; `--name` is the model name in
the chunks (`qwen3` by default), `--host` is `127.0.0.1` or `localhost` (anything else is refused before anything
loads), and `--port 0` picks a free one, printed at startup.

```sh
curl -s http://127.0.0.1:8080/health
# {"status":"ok"}

curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":8}'
# data: {"id":"chatcmpl-1","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
# ...
# data: {"id":"chatcmpl-1",...,"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}
# data: [DONE]
```

## The contract

- `messages` (the Qwen3 chat template), `max_tokens` (1 to 512, 128 when absent) and `stream: true` are read, and
  `stream_options.include_usage` adds the choiceless chunk with the token counts. Every other JSON field is ignored,
  `model` included: there is one model, so a stock OpenAI body decodes. What a greedy streaming server cannot honour
  is a 400 that names it: `temperature`, `top_p`, `seed` and `n`, `stop`, a body without `stream: true`, a budget
  outside the range, a prompt past 8192 characters or past the model's context.
- The deltas concatenate to the answer, one character at a time at most: a character split between tokens is held
  back until its last byte arrives, and flushed whole at the end, so no replacement character goes out for a
  character that is still arriving.
- A `[DONE]` always closes a stream whose deltas are the answer, and an answer that stops before its first token is
  one of them: the role chunk, `finish_reason` `stop`, the usage when asked, and `[DONE]`. A failure after the head
  (a 200) ends the stream with an `error` event and no `[DONE]`; a request the scheduler refuses outright is a 503
  before the head.
- At most two sequences step at once and four more wait; past that a request is refused as `saturated` with a 503,
  immediately and not as a stall.
- A client that disconnects is noticed on the next write that fails or times out, never sooner: the sequence is then
  dropped and its slot freed. There is no read of the socket to notice it earlier, so a sequence whose client left
  may hold its slot for a bounded while (the events channel holds 16 pieces, each write waits 10 s at most).
- No authentication: the server listens on loopback only and refuses any other host.

## Tests

```sh
cd server
agentica check
agentica fmt --check
agentica test --max-steps 10000000
```

The tests spawn `test/chat_peer.py` for the fixture, so they need `python3`; every streamed answer is compared with
the same greedy answer from `Generate.generate`, and two clients stream at once to check that they advance together.
