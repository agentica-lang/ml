## Times the BERT forward on ggml: a batch of `BATCH` sentences of `SEQ` tokens
## (ids cycling through the vocabulary, one warm-up run first), the median of
## `REPS` runs. Manual; run from `transformers/`:
##
##   agentica tools/bench_bert.agc MODEL_DIR [cpu|vulkan] [SEQ] [BATCH] [REPS]

import Bert from "../bert"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "../tokenizer/wordpiece"

def main() -> Void
  args = System.args
  if args.empty?
    puts("usage: bench_bert.agc MODEL_DIR [cpu|vulkan] [SEQ] [BATCH] [REPS]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  seq = 128
  batch = 1
  reps = 5
  if args.size > 2
    seq = args[2].to_i!
  end

  if args.size > 3
    batch = args[3].to_i!
  end

  if args.size > 4
    reps = args[4].to_i!
  end

  case measure(args[0], Tensor.Backend.Ggml(device), seq, batch, reps)
    ok(line) -> puts(line)
    error(e) ->
      puts("bench_bert: #{e.message}")
      System.exit(1)
  end
end

private

def measure(dir: String, backend: Tensor.Backend, seq: Int, batch: Int, reps: Int) -> Result[String]
  started = Time.monotonic_ms
  m = try Bert.open(dir, backend)
  load = Time.monotonic_ms - started
  ids = (0...seq).to_a.map { |i| 1000 + (i * 37) % 20000 }
  rows = List.repeat(WordPiece.Encoding(ids: ids, type_ids: List.repeat(0, seq)), batch)
  _ = try Bert.run(m, rows)
  times: List[Int] = []
  for _ in 0...reps
    started = Time.monotonic_ms
    _ = try Bert.run(m, rows)
    times = times.push(Time.monotonic_ms - started)
  end

  median = times.sort[times.size / 2]
  ok(
    "load #{load} ms; batch #{batch} x #{seq} tokens: median #{median} ms of #{reps} runs, #{batch * seq * 1000 / [median, 1].max.unwrap!} tokens/s",
  )
end
