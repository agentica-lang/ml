## What crossing the boundary costs, op by op: the same op, the same shape and
## the same input values on both backends — `tensor/`'s reference in Agentica
## lists, and libtorch in `torch/native/torch.c` over `extern module "torch"`.
##
## Every `torch` number includes the free of the tensor the op made
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

import Ref from "pkg/tensor/tensor"
import Torch from "../torch"

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
## intermediate.
def free(t: Torch.Tensor) -> Void
  case t.handle
    Torch.Handle.Native(h) -> Native.close(h)
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
  enabled = Torch.available()
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
  row("create [64, 64] (4096 values)", try reference_create(), try torch_create())
  row("exp [64, 64]", try reference_exp(), try torch_exp())
  row("sum [64, 64]", try reference_sum(), try torch_sum())
  row("linear+relu step, [4, 3]", try reference_step(), try torch_step())
  puts(format("%-30s %12s %12s", "", "ns/op", "ns/op"))
  ok()
end

def reference_add() -> Result[Int]
  a = try values_of(PAIR)
  b = try values_of(PAIR)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.add(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_add_wide() -> Result[Int]
  a = try values_of(WIDE)
  b = try values_of(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.add(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_matmul() -> Result[Int]
  a = try values_of(LEFT)
  b = try values_of(RIGHT)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.matmul(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_matmul_wide() -> Result[Int]
  a = try values_of(WIDE)
  b = try values_of(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.matmul(a, b)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_create() -> Result[Int]
  data = values(4096)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.from_list(to_values(data), WIDE, Ref.Dtype.F32, Ref.Backend.Reference)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_exp() -> Result[Int]
  a = try values_of(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.exp(a)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_sum() -> Result[Int]
  a = try values_of(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Ref.sum(a)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def reference_step() -> Result[Int]
  x = try values_of(LEFT)
  w = try values_of(RIGHT)
  b = try values_of([LEFT[0], RIGHT[1]])
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    y = try Ref.matmul(x, w)
    z = try Ref.add(y, b)
    _ = try Ref.relu(z)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_add() -> Result[Int]
  a = try torch_values(PAIR)
  b = try torch_values(PAIR)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.add(Torch.Device.Cpu, a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_add_wide() -> Result[Int]
  a = try torch_values(WIDE)
  b = try torch_values(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.add(Torch.Device.Cpu, a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_matmul() -> Result[Int]
  a = try torch_values(LEFT)
  b = try torch_values(RIGHT)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.matmul(Torch.Device.Cpu, a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_matmul_wide() -> Result[Int]
  a = try torch_values(WIDE)
  b = try torch_values(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.matmul(Torch.Device.Cpu, a, b))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_create() -> Result[Int]
  data = values(4096)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.from_list(Torch.Device.Cpu, to_values(data), WIDE, Torch.Dtype.F32))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_exp() -> Result[Int]
  a = try torch_values(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.exp(Torch.Device.Cpu, a))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_sum() -> Result[Int]
  a = try torch_values(WIDE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    free(try Torch.sum(Torch.Device.Cpu, a))
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

def torch_step() -> Result[Int]
  x = try torch_values(LEFT)
  w = try torch_values(RIGHT)
  b = try torch_values([LEFT[0], RIGHT[1]])
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    y = try Torch.matmul(Torch.Device.Cpu, x, w)
    z = try Torch.add(Torch.Device.Cpu, y, b)
    r = try Torch.relu(Torch.Device.Cpu, z)
    free(r)
    free(z)
    free(y)
  end

  ok((Time.monotonic_ns() - start) / ROUNDS)
end

## `data` as a reference `F32` tensor of `shape`.
def values_of(shape: List[Int]) -> Result[Ref.Tensor]
  Ref.from_list(to_values(values(shape_x(shape))), shape, Ref.Dtype.F32, Ref.Backend.Reference)
end

## `data` as a torch `F32` tensor of `shape`.
def torch_values(shape: List[Int]) -> Result[Torch.Tensor]
  Torch.from_list(Torch.Device.Cpu, to_values(values(shape_x(shape))), shape, Torch.Dtype.F32)
end

## One row of the table: nanoseconds per op on each backend.
def row(name: String, reference_ns: Int, torch_ns: Int) -> Void
  puts(format("%-30s %12d %12d", name, reference_ns, torch_ns))
end

## The element count of `shape`.
def shape_x(shape: List[Int]) -> Int
  total = 1
  for d in shape
    total = total * d
  end

  total
end

## Storage floats as dtype-tagged values, which is what `from_list` takes.
def to_values(data: List[Float]) -> List[Torch.Values] = data.map do |x: Float| Torch.Values.Float(x) end
