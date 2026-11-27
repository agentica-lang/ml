# Memory, reading back, and errors

## Where a tensor lives

A tensor has a `shape`, a `dtype` and a `backend`. On ggml, libtorch and XLA its values sit in native memory (on the GPU
for Vulkan), and the Agentica value only holds a handle. A model keeps its weights there. Nothing crosses to the host
until you read it.

- `Tensor.to_list(t)` reads a tensor back as a list of `Values` (`Float`, `Int` or `Bool`). It copies every element
  through the VM, so read small results: a score, a few ids, one embedding row. For the largest values of a vector, use
  `Tensor.top_ids(x, k)` inside the traced function so only `k` ids cross.
- Read an integer result by matching on the variant:

```agentica
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  scores = Tensor.floats([0.1, 0.9, 0.3, 0.7, 0.2, 0.1], [2, 3]).unwrap!
  best = Tensor.argmax(scores, some([1])).unwrap!
  for v in Tensor.to_list(best).unwrap!
    case v
      Tensor.Values.Int(id) -> puts(id)
      _ -> puts("not an id")
    end
  end
end
```

```output
1
0
```

- `Tensor.shape(t)`, `dtype` and `backend` read fields with no backend call.
- `Tensor.to_device(t, device)` moves a ggml tensor between ggml devices (CPU and Vulkan).

## Closing

Native memory is freed when the garbage collector drops the handle, which can be late for a 1 GB tensor or the VRAM of a
GPU. Close what is big, as soon as you are done:

| What | Close with |
|---|---|
| a Qwen3 model, DistilBERT, BERT, Whisper | `Qwen3.close_model(m)`, `DistilBert.close_model(m)`, `Bert.close_model(m)`, `Asr.close_model(model)` |
| a generator | `Generate.close(g)` |
| the output of a model call (logits, embeddings, a hidden state) | `Lift.close([t])` from `pkg/transformers/lift`; `Bert.close(out)` for a `Bert.Output` |
| a compiled function | `Trace.close(compiled)` |
| a tensor of your own | `Data.free(t)` from `pkg/train/data` |

Using a closed model or tensor is an error (`RUNTIME-NATIVE-CLOSED`), never a wrong number. The reference backend holds
plain data and has nothing to close, so the same code runs there. Tensors you give to a model call stay yours; the call
does not close them. `embed` frees its intermediates, and its result is yours.

`Native.arena do ... end` closes every native tensor made inside the block when it ends, except what the block returns.
Use it around a loop of ops on libtorch.

## Half and quantized weights

`weights: Tensor.Dtype.F16` (on `Qwen3.open`, `Asr.open`, the `Load` functions) keeps the big matrices in 16 bits on
ggml: half the memory, and Qwen3-0.6B decodes about 1.6 times faster on Vulkan. A value beyond 65504 is an error that names
it (on XLA too, from a file or a list). `Dtype.Q8_0` keeps the blocks of a Q8_0 GGUF file, a quarter of the memory, and `Dtype.Q4_K` those of a Q4_K_M file (its Q4_K and Q6_K matrices, each at its own bit width), about a sixth. Half and quantized weights are for
inference only: they have no gradient. Half weights also load on XLA (`Backend.Xla`), where the program widens them to `F32` for each product; quantized weights are ggml only. Any other backend refuses them with an error naming the option.

## Errors

Every op answers a `Result`; read the message.

| You see | It means | Do |
|---|---|---|
| `warning: native module ggml not built` at start, then `error("... backend not available ...")` | the module was not built in this shell | build ggml and `eval` its exports in the shell that runs the app (`backends.md`); check with `Tensor.available` |
| `... backend not available ... run agentica tools/install_xla.agc` | the XLA plugin is missing | install it (`backends.md`) |
| `X does not run eagerly on the ggml backend` (or XLA) | a lone op on a compiled-only backend | put the op in `Trace.eval` or `Trace.compile`, or call the model API |
| an error about two backends | a tensor of another backend met yours | create every tensor on one backend; `Tensor.to_device` for ggml devices |
| `RUNTIME-NATIVE-CLOSED` | a closed model or tensor was used | do not use it after closing; load again |
| `PKG-SOURCE-CONFLICT` at `dependencies.tensor` | two tags or commits for one package (an ml package of another release) | the tags of one release for every ml package |
| `PKG-LOCK-STALE` or `CHECK-IMPORT-PACKAGE` | the lock or `deps/` is behind `agentica.toml` | `agentica update`, then `agentica install` |
| `PKG-AGENTICA-TOO-OLD` | the Agentica binary is older than a package asks | upgrade Agentica (0.38.0 or newer) |
| `no tensor X in the checkpoint`, a shape error from `Qwen3.open` or `Asr.open` | the folder is another model, or another layout | check `config.json` and `Hub.names(c)` |
| an error naming `model_type`, an activation, `rope_scaling` or `attention_bias` | the model is outside what is implemented | use a supported model (`models.md`) |
| a value beyond 65504 for `F16` | the weight does not fit half precision | load as `F32` |
| a ggml output "is not finite" | an op overflowed or divided by zero | check the inputs; `Trace.diff` finds the first node where two backends disagree |
| `Tensor.vulkan_fp16` answers an error | it was called after the first ggml call | call it at the start of `main` (`backends.md`, Settings) |
| `RUNTIME-STEP-LIMIT` in `agentica test` | the test ran out of steps (1M per test file; a test that loads a model or trains needs more) | `agentica test --max-steps 10000000` |
| greedy ids differ from Hugging Face on Vulkan | fp16 products | `Tensor.vulkan_fp16(false)` (`backends.md`, Settings) |

When two backends disagree, `Trace.diff(args, backend_a, backend_b) do |xs| ... end` runs one traced function on both and
names the first node whose output differs.

Page: `https://www.agentica-lang.com/ml/tensor/`.
