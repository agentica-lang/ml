## Writes a Qwen3 model folder (`config.json`, `model.safetensors`, `tokenizer.json`) as a GGUF file that llama.cpp and
## `Qwen3.open` read: the matrices in `F32`, `F16` (the default) or `BF16`, the norms in `F32`, the tokenizer and the
## chat template as key-values. Needs the ggml module (the weights are converted in C). Run from `transformers/`:
##
##   agentica tools/to_gguf.agc MODEL_DIR OUT.gguf [F32|F16|BF16]

import Tensor from "pkg/tensor/tensor"
import Qwen3Gguf from "../qwen3_gguf"

def main() -> Void
  args = System.args
  if args.size < 2 or args.size > 3
    puts("usage: to_gguf.agc MODEL_DIR OUT.gguf [F32|F16|BF16]")
    System.exit(2)
  end

  format = Tensor.FloatFormat.F16
  if args.size == 3
    case args[2]
      "F32" -> format = Tensor.FloatFormat.F32
      "F16" -> format = Tensor.FloatFormat.F16
      "BF16" -> format = Tensor.FloatFormat.BF16
      _ ->
        puts("to_gguf: the format is F32, F16 or BF16, not #{args[2]}")
        System.exit(2)
    end
  end

  started = Time.monotonic_ms
  case Qwen3Gguf.convert(args[0], args[1], format)
    ok(_) -> puts("wrote #{args[1]} in #{Time.monotonic_ms - started} ms")
    error(e) ->
      puts("to_gguf: #{e.message}")
      System.exit(1)
  end
end
