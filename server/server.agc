## The loopback chat server of the `server` package (ml#90): one loaded Qwen3 generator behind the
## `Serving` scheduler, served by the `Api` routes on Bond. Greedy only, streaming only, localhost only.
##
## The owner chain, which the shape of the code keeps: `load` opens the tokenizer, the model and the generator (in
## that order, undoing what it opened when the next step fails); `start` lends the generator to the scheduler and
## serves the app; `stop` ends the HTTP server, then the scheduler (ownership of the generator returns to the
## caller), then the generator and the model. The model lives as long as any relay can name the scheduler.

import Bond from "pkg/bond/bond"
import HTTP from "std/http"
import OP from "std/option_parser"
import {Opt, Kind} from "std/option_parser"
import Api from "api"
import Bpe from "pkg/transformers/tokenizer/bpe"
import Generate from "pkg/transformers/generate"
import Qwen3 from "pkg/transformers/qwen3"
import Serving from "pkg/transformers/serving"
import Tensor from "pkg/tensor/tensor"

## What the server borrows while it runs: everything `stop` undoes, in reverse.
struct Running
  server: HTTP.Server
  serving: Serving.Server
  generator: Generate.Generator
  model: Qwen3.Qwen3
end

## Starts the server over the model in `dir` (a `config.json`, a `model.safetensors` and a `tokenizer.json`, as
## `transformers/test/qwen_peer.py` writes): the reference backend, `model_name` in the chunks, on `host:port`
## (port 0 picks a free one, read back with `local_port`). Only loopback hosts are served.
def start(dir: String, model_name: String, host: String, port: Int) -> Result[Running]
  return error("the host is #{host}, only loopback hosts are served") if host != "127.0.0.1" and host != "localhost"

  tok = try Bpe.open("#{dir}/tokenizer.json")
  model = try Qwen3.open(dir, Tensor.Backend.Reference)
  generator = try undone(model, Generate.open(model))
  limits = Serving.Limits(max_active: Api.MAX_ACTIVE, max_pending: Api.MAX_PENDING)
  serving = try undone2(model, generator, Serving.start(generator, limits))
  api = Api.Api(
    tok: tok,
    serving: serving,
    model_name: model_name,
    model_type: model.config.model_type,
    max_positions: model.config.max_position_embeddings,
    stop: Api.stop_ids(tok),
    default_max_tokens: Api.DEFAULT_MAX_TOKENS,
    max_max_tokens: Api.MAX_MAX_TOKENS,
    counters: ATS.new(),
  )

  case Bond.start(Api.app(api), host, port)
    ok(server) -> ok(Running(server: server, serving: serving, generator: generator, model: model))
    error(e) ->
      _ = Serving.stop(serving)
      Generate.close(generator)
      Qwen3.close_model(model)
      error(e.message)
  end
end

## The port the server listens on (for port 0, the one the system picked).
def local_port(running: Running) -> Int = Bond.local_port(running.server)

## Ends the server: the HTTP listener first (no new requests), then the scheduler (draining what is live), then the
## generator and the model. The generator belongs to the caller again only when this returns `ok()`.
def stop(running: Running) -> Result[Void]
  Bond.stop(running.server)
  try Serving.stop(running.serving)
  Generate.close(running.generator)
  Qwen3.close_model(running.model)
  ok()
end

# `Generate.open` failed after the model was opened: closes the model and reports the reason.
def undone[T](model: Qwen3.Qwen3, result: Result[T]) -> Result[T]
  case result
    ok(v) -> ok(v)
    error(e) ->
      Qwen3.close_model(model)
      error(e.message)
  end
end

# `Serving.start` failed after the model and the generator were opened: closes both and reports the reason.
def undone2[T](model: Qwen3.Qwen3, generator: Generate.Generator, result: Result[T]) -> Result[T]
  case result
    ok(v) -> ok(v)
    error(e) ->
      Generate.close(generator)
      Qwen3.close_model(model)
      error(e.message)
  end
end

## The command line: `server.agc --model DIR [--name NAME] [--host HOST] [--port PORT]`.
def main() -> Void
  parsed = OP.parse(parser(), System.args())
  options = OP.run!(parsed)
  running = start(
    OP.string(options, "model"),
    OP.string(options, "name"),
    OP.string(options, "host"),
    OP.int(options, "port"),
  ).unwrap!

  model_dir = OP.string(options, "model")
  host = OP.string(options, "host")
  puts("serving #{model_dir} on #{host}:#{local_port(running)}")
  IO.flush()
  served = Bond.wait(running.server)
  # The server, the scheduler, the generator and the model close either way; a failed wait is what the caller hears
  # (a failed stop would hide it).
  stopped = stop(running)
  return stopped.unwrap! if served.ok?

  _ = stopped
  served.unwrap!
end

private

def parser() -> OP.Parser
  OP.new(
    "server.agc",
    "A loopback OpenAI-shaped chat server over a Qwen3 model (greedy, streaming only)",
    [],
    [
      Opt(
        name: "model",
        metavar: "DIR",
        default: "model",
        help: "model folder: config.json, model.safetensors, tokenizer.json",
      ),
      Opt(name: "name", metavar: "NAME", default: "qwen3", help: "model name in the chunk bodies"),
      Opt(name: "host", default: "127.0.0.1", help: "address to listen on: 127.0.0.1 or localhost only"),
      Opt(name: "port", kind: Kind.Int, default: "8080", help: "port to listen on, 0 for a free one"),
    ],
  )
end
