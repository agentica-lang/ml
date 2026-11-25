## What crossing the boundary costs, op by op: the same public op, the same
## shape and the same input values on both backends — `Reference` in Agentica
## lists, and `Torch` on libtorch over `extern module "torch"`.
##
## Every `Torch` number includes the free of the tensor the op made
## (`Native.close`), because a training step frees its intermediates too.
##
## From the package folder:
##
## ```sh
## eval "$(sh tools/build-torch.sh)"
## agentica tools/bench.agc
## ```
##
## The numbers this prints are the ones `README.md` quotes. They are one
## machine and they are noise, not a benchmark suite: they exist to show what
## the boundary costs relative to the work it carries, and to say plainly that
## a [`2, 2`] op is dominated by the crossing while a [`64, 64`] one is not.

import Tensor from "../tensor"

## The backend the reference runs on.
REF = Tensor.Backend.Reference
## The backend the native libtorch module runs on.
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
## Rounds per measurement: enough that a clock read is noise, few enough that
## the run stays around a second at the reference backend's speed.
ROUNDS = 2000
## A tensor with real work behind it: 64 by 64 floats across the boundary.
WIDE: List[Int] = [64, 64]
## The elementwise pair is [2, 2]: small enough that the fixed cost of a call
## is all that is left of the op.
PAIR: List[Int] = [2, 2]
## The rank-2 product the derived ops of `tensor/DESIGN.md` §3.2 are built on:
## `[4, 3] . [3, 2]` is one `linear` of a small block.
LEFT: List[Int] = [4, 3]
RIGHT: List[Int] = [3, 2]

## Deterministic values for `n` elements: no random source, so two runs print
## the same table.
def values(n: Int) -> List[Float]
  out: List[Float] = []
  i = 0
  while i < n
    out = out.push(((i * 7 + 3) % 11).to_f / 11.0 - 0.4)
    i += 1
  end

  out
end

## Releases the native tensor behind `t`, as a step would when it drops an
## intermediate; a `Plain` tensor has nothing to free.
def free(t: Tensor.Tensor) -> Void
  case t.handle
    Tensor.Handle.Native(h) -> Native.close(h)
    Tensor.Handle.Plain(_) -> puts("")
    Tensor.Handle.Traced(_) -> puts("")
    Tensor.Handle.Ggml(h) -> Native.close(h)
    Tensor.Handle.Xla(h) -> Native.close(h)
  end
end

def main() -> Void
  case run()
    ok(_) -> puts("")
    error(e) ->
      puts("bench: #{e.message}")
      System.exit(1)
  end
end

def run() -> Result[Void]
  start = Time.monotonic_ns()
  enabled = Tensor.available(TORCH)
  load_ns = Time.monotonic_ns() - start
  puts("# bench: #{ROUNDS} rounds per measurement")
  puts(format("native module load: %.3f ms", load_ns.to_f / 1000000.0))
  if !enabled
    puts("the native torch module is not built: no numbers to compare")
    return ok()
  end

  puts(format("%-30s %12s %12s", "op (shape)", "reference", "torch"))
  row("add [2, 2]", try reference_add(), try torch_add())
  row("add [64, 64]", try reference_add_wide(), try torch_add_wide())
  row("matmul [4, 3]x[3, 2]", try reference_matmul(), try torch_matmul())
  row("matmul [64, 64]x[64, 64]", try reference_matmul_wide(), try torch_matmul_wide())
  row("from_list [64, 64] (4096 values)", try reference_create(), try torch_create())
  row("exp [64, 64]", try reference_exp(), try torch_exp())
  row("sum [64, 64]", try reference_sum(), try torch_sum())
  row("linear+relu step, [4, 3]", try reference_step(), try torch_step())
  puts(format("%-30s %12s %12s", "", "ns/op", "ns/op"))
  ok()
end

## One row of the table: nanoseconds per op on each backend.
def row(name: String, reference_ns: Int, torch_ns: Int) -> Void
  puts(format("%-30s %12d %12d", name, reference_ns, torch_ns))
end

## `values_of` on `backend`, the input every timed loop runs on.
def input_on(shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(
    values(shape_x(shape)).map do |x: Float| Tensor.Values.Float(x) end,
    shape,
    Tensor.Dtype.F32,
    backend,
  )
end

def reference_add() -> Result[Int]
  a = try input_on(PAIR, REF)
  b = try input_on(PAIR, REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.add(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_add_wide() -> Result[Int]
  a = try input_on(WIDE, REF)
  b = try input_on(WIDE, REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.add(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_matmul() -> Result[Int]
  a = try input_on(LEFT, REF)
  b = try input_on(RIGHT, REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.matmul(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_matmul_wide() -> Result[Int]
  a = try input_on(WIDE, REF)
  b = try input_on(WIDE, REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.matmul(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_create() -> Result[Int]
  data = values(4096).map do |x: Float| Tensor.Values.Float(x) end
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.from_list(data, WIDE, Tensor.Dtype.F32, REF)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_exp() -> Result[Int]
  a = try input_on(WIDE, REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.exp(a)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_sum() -> Result[Int]
  a = try input_on(WIDE, REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Tensor.sum(a)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_step() -> Result[Int]
  x = try input_on(LEFT, REF)
  w = try input_on(RIGHT, REF)
  b = try input_on([LEFT[0], RIGHT[1]], REF)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    y = try Tensor.matmul(x, w)
    z = try Tensor.add(y, b)
    _ = try Tensor.relu(z)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_add() -> Result[Int]
  a = try input_on(PAIR, TORCH)
  b = try input_on(PAIR, TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.add(a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_add_wide() -> Result[Int]
  a = try input_on(WIDE, TORCH)
  b = try input_on(WIDE, TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.add(a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_matmul() -> Result[Int]
  a = try input_on(LEFT, TORCH)
  b = try input_on(RIGHT, TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.matmul(a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_matmul_wide() -> Result[Int]
  a = try input_on(WIDE, TORCH)
  b = try input_on(WIDE, TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.matmul(a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_create() -> Result[Int]
  data = values(4096).map do |x: Float| Tensor.Values.Float(x) end
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.from_list(data, WIDE, Tensor.Dtype.F32, TORCH))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_exp() -> Result[Int]
  a = try input_on(WIDE, TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.exp(a))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_sum() -> Result[Int]
  a = try input_on(WIDE, TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Tensor.sum(a))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_step() -> Result[Int]
  x = try input_on(LEFT, TORCH)
  w = try input_on(RIGHT, TORCH)
  b = try input_on([LEFT[0], RIGHT[1]], TORCH)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    y = try Tensor.matmul(x, w)
    z = try Tensor.add(y, b)
    r = try Tensor.relu(z)
    free(r)
    free(z)
    free(y)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

## The element count of `shape`.
def shape_x(shape: List[Int]) -> Int
  total = 1
  for d in shape
    total = total * d
  end

  total
end
