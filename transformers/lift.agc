## Running a model on ggml, which computes only whole traced graphs: the
## weights of the model become `Trace` constants (the graph reads the buffers
## they hold, no copy), and `staged` runs one function of the lifted model as
## one graph. Elsewhere the function runs eagerly on the model as it is, in an
## arena that frees the intermediates of the backend. `close` frees tensors at
## once, and the `*_parts` lists name every weight of a layer, for closing a
## model.

import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## `f` of the model and `args`: one compiled graph of `lift(m)` on `Ggml`; on
## another backend the call itself, in an arena that keeps the results alone.
def staged[M](
  m: M,
  lift: Fn[M, M],
  args: List[Tensor.Tensor],
  f: Fn[M, List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[List[Tensor.Tensor]]
  case Tensor.backend(args[0])
    Tensor.Backend.Ggml(_) -> Trace.eval_many(args, ->(xs: List[Tensor.Tensor]) { f.call(lift.call(m), xs) })
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

def embedding(p: Nn.Embedding) -> Nn.Embedding = Nn.Embedding(table: Trace.constant(p.table))

def dense(p: Nn.Dense) -> Nn.Dense = p.with(w: Trace.constant(p.w), b: p.b.map { |b| Trace.constant(b) })

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
  )
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

def norm_parts(p: Nn.LayerNorm) -> List[Tensor.Tensor] = [p.gamma, p.beta]

def block_parts(b: Nn.EncoderBlock) -> List[Tensor.Tensor]
  a = b.attention
  denses = [a.query, a.key, a.value, a.output, b.mlp.up, b.mlp.down]
  case b.mlp.gate
    some(g) -> denses = denses.push(g)
    none -> nil
  end

  norms = [b.norm1, b.norm2]
  case a.query_norm
    some(n) -> norms = norms.push(n)
    none -> nil
  end

  case a.key_norm
    some(n) -> norms = norms.push(n)
    none -> nil
  end

  out: List[Tensor.Tensor] = []
  for d in denses
    out = out + dense_parts(d)
  end

  for n in norms
    out = out + norm_parts(n)
  end

  out
end
