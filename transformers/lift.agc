## Running a model on ggml, which computes only whole traced graphs: the
## weights of the model become `Trace` constants (the graph reads the buffers
## they hold, no copy), and `staged` runs one function of the lifted model as
## one graph. Elsewhere the function runs eagerly on the model as it is, in an
## arena that frees the intermediates of the backend. `close` frees tensors at
## once, and the `*_parts` lists name every weight of a layer, for closing a
## model.

import Delta from "pkg/nn/delta"
import Moe from "pkg/nn/moe"
import Nn from "pkg/nn/nn"
import ShortConv from "pkg/nn/short_conv"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## `f` of the model and `args`: one compiled graph of `lift(m)` on `Ggml` and `Xla`; on
## another backend the call itself, in an arena that keeps the results alone.
def staged[M](
  m: M,
  lift: Fn[M, M],
  args: List[Tensor.Tensor],
  f: Fn[M, List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[List[Tensor.Tensor]]
  case Tensor.backend(args[0])
    Tensor.Backend.Ggml(_) -> Trace.eval_many(args, ->(xs: List[Tensor.Tensor]) { f.call(lift.call(m), xs) })
    Tensor.Backend.Xla(_) -> Trace.eval_many(args, ->(xs: List[Tensor.Tensor]) { f.call(lift.call(m), xs) })
    _ -> Native.arena do f.call(m, args) end
  end
end

## Closes the libtorch and ggml tensors of `ts` now, not at the end of an
## arena; the reference has nothing to close.
def close(ts: List[Tensor.Tensor]) -> Void
  for t in ts
    case t.handle
      Tensor.Handle.Native(h) -> Native.close(h)
      Tensor.Handle.Ggml(h) -> Native.close(h)
      _ -> nil
    end
  end
end

def stack(s: Nn.EncoderStack) -> Nn.EncoderStack
  s.with(blocks: s.blocks.map { |b| block(b) }, final_norm: s.final_norm.map { |n| norm(n) })
end

def embedding(p: Nn.Embedding) -> Nn.Embedding = Nn.Embedding(table: Trace.constant(p.table))

def dense(p: Nn.Dense) -> Nn.Dense = p.with(w: Trace.constant(p.w), b: p.b.map { |b| Trace.constant(b) })

def conv2d(p: Nn.Conv2d) -> Nn.Conv2d
  p.with(weight: Trace.constant(p.weight), bias: p.bias.map { |b| Trace.constant(b) })
end

def norm(p: Nn.LayerNorm) -> Nn.LayerNorm = p.with(gamma: Trace.constant(p.gamma), beta: Trace.constant(p.beta))

def mlp(p: Nn.Mlp) -> Nn.Mlp = p.with(up: dense(p.up), down: dense(p.down), gate: p.gate.map { |g| dense(g) })

def attention(a: Nn.Attention) -> Nn.Attention
  a.with(
    query: dense(a.query),
    key: dense(a.key),
    value: dense(a.value),
    output: dense(a.output),
    query_norm: a.query_norm.map { |n| norm(n) },
    key_norm: a.key_norm.map { |n| norm(n) },
    head_gate: a.head_gate.map { |g| dense(g) },
  )
end

def short_conv(p: ShortConv.ShortConv) -> ShortConv.ShortConv
  p.with(input: dense(p.input), conv: Trace.constant(p.conv), output: dense(p.output))
end

def delta(p: Delta.GatedDeltaNet) -> Delta.GatedDeltaNet
  p.with(
    qkv: dense(p.qkv),
    z: dense(p.z),
    b: dense(p.b),
    a: dense(p.a),
    conv: Trace.constant(p.conv),
    dt_bias: Trace.constant(p.dt_bias),
    a_log: Trace.constant(p.a_log),
    norm: Trace.constant(p.norm),
    output: dense(p.output),
  )
end

def moe(p: Moe.Moe) -> Moe.Moe
  p.with(router: dense(p.router), gate_up: Trace.constant(p.gate_up), down: Trace.constant(p.down))
end

def block(b: Nn.EncoderBlock) -> Nn.EncoderBlock
  b.with(norm1: norm(b.norm1), norm2: norm(b.norm2), attention: attention(b.attention), mlp: mlp(b.mlp))
end

def dense_parts(p: Nn.Dense) -> List[Tensor.Tensor]
  out = [p.w]
  case p.b
    some(b) -> out = out.push(b)
    none -> nil
  end

  out
end

def conv2d_parts(p: Nn.Conv2d) -> List[Tensor.Tensor]
  case p.bias
    some(b) -> [p.weight, b]
    none -> [p.weight]
  end
end

def norm_parts(p: Nn.LayerNorm) -> List[Tensor.Tensor] = [p.gamma, p.beta]

def mlp_parts(p: Nn.Mlp) -> List[Tensor.Tensor]
  out = dense_parts(p.up) + dense_parts(p.down)
  case p.gate
    some(g) -> out + dense_parts(g)
    none -> out
  end
end

def attention_parts(a: Nn.Attention) -> List[Tensor.Tensor]
  out: List[Tensor.Tensor] = []
  for d in [a.query, a.key, a.value, a.output]
    out = out + dense_parts(d)
  end

  case a.head_gate
    some(g) -> out = out + dense_parts(g)
    none -> nil
  end

  for n in [a.query_norm, a.key_norm]
    case n
      some(x) -> out = out + norm_parts(x)
      none -> nil
    end
  end

  out
end

def short_conv_parts(p: ShortConv.ShortConv) -> List[Tensor.Tensor]
  [p.conv] + dense_parts(p.input) + dense_parts(p.output)
end

def delta_parts(p: Delta.GatedDeltaNet) -> List[Tensor.Tensor]
  out = [p.conv, p.dt_bias, p.a_log, p.norm]
  for d in [p.qkv, p.z, p.b, p.a, p.output]
    out = out + dense_parts(d)
  end

  out
end

def moe_parts(p: Moe.Moe) -> List[Tensor.Tensor] = dense_parts(p.router) + [p.gate_up, p.down]

def block_parts(b: Nn.EncoderBlock) -> List[Tensor.Tensor]
  attention_parts(b.attention) + mlp_parts(b.mlp) + norm_parts(b.norm1) + norm_parts(b.norm2)
end
