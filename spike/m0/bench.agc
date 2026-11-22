## What crossing the boundary costs, op by op.
##
## Two backends, the same ops, the same shapes, the same input values: the
## reference in Agentica lists and the native side in libtorch over the
## module. Every native number includes the free of the tensor the op made
## (`Native.close`), because a training step frees its intermediates too.
##
## From `spike/m0/`:
##
## ```sh
## eval "$(sh tools/build-torch.sh)"
## agentica bench.agc
## ```
##
## The numbers are the ones the README quotes; they are one machine and they
## are noise, not a benchmark suite: they exist to show what the boundary
## costs relative to the work it carries. The last row has no native column:
## the backward is Agentica code, the same on both sides, so a "native step"
## that ran the libtorch forward *and* the reference backward would time the
## crossing plus a duplicated step instead of one thing. The cost of putting
## libtorch's forward into a step is the difference between the columns of the
## row above it.

import Dataset from "dataset"
import Model from "model"
import TensorRef from "tensor_ref"
import Torch from "torch_spike"

## Rounds per measurement: enough that a clock read is noise, few enough that
## the run stays around a second at the reference backend's speed.
ROUNDS = 2000
## An op with a real tensor behind it: 64 by 64 floats across the boundary.
WIDE: List[Int] = [64, 64]
## The elementwise pair is [2, 2]: small enough that the fixed cost of a call
## is all that is left of the op.
PAIR: List[Int] = [2, 2]

# Deterministic values for a shape of `n` elements: no random source in the
# spike, so two runs of this file print the same table.
def values(n: Int) -> List[Float]
  out: List[Float] = []
  i = 0
  while i < n
    out = out.push(((i * 7 + 3) % 11).to_f / 11.0 - 0.4)
    i += 1
  end

  out
end

def main() -> Void
  case run()
    ok(_) ->
      puts("")
      puts("# ns/op, one call per round; the native side also frees what it made")
    error(e) ->
      puts("bench: #{e.message}")
      System.exit(1)
  end
end

def run() -> Result[Void]
  load_start = Time.monotonic_ns()
  enabled = Native.load("torchspike").ok?
  load_ns = Time.monotonic_ns() - load_start
  add_ns: Optional[Int] = none
  matmul_ns: Optional[Int] = none
  create_ns: Optional[Int] = none
  forward_ns: Optional[Int] = none
  if enabled
    add_ns = some(try timed_add())
    matmul_ns = some(try timed_matmul())
    create_ns = some(try timed_create())
    forward_ns = some(try timed_forward())
  end

  puts("# bench: #{ROUNDS} rounds per measurement")
  puts(format("torchspike library load: %.3f ms", load_ns.to_f / 1000000.0))
  puts(format("%-26s %12s %12s", "op (shape)", "reference", "torchspike"))
  report("add [2, 2]", try reference_add(), add_ns, "skipped")
  report("matmul [4, 3]x[3, 2]", try reference_matmul(), matmul_ns, "skipped")
  report("create [64, 64] (4096 values)", try reference_create(), create_ns, "skipped")
  report("f, demo shapes", try reference_forward(), forward_ns, "skipped")
  report("f + backward, demo shapes", try reference_step(), none, "n/a")
  ok()
end

# The reference side: one call of the op per round, in Agentica.
def reference_add() -> Result[Int]
  a = try TensorRef.from_list(values(4), PAIR)
  b = try TensorRef.from_list(values(4), PAIR)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try TensorRef.add(a, b)
  end

  ok(Time.monotonic_ns() - start)
end

def reference_matmul() -> Result[Int]
  a = try TensorRef.from_list(values(12), Dataset.X_SHAPE)
  b = try TensorRef.from_list(values(6), Dataset.W_SHAPE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try TensorRef.matmul(a, b)
  end

  ok(Time.monotonic_ns() - start)
end

def reference_create() -> Result[Int]
  wide = values(4096)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try TensorRef.from_list(wide, WIDE)
  end

  ok(Time.monotonic_ns() - start)
end

# The reference's forward pass, the same three ops `Torch.loss` runs natively.
def forward(x: TensorRef.Tensor, w: TensorRef.Tensor) -> Result[Float]
  y = try TensorRef.matmul(x, w)
  TensorRef.scalar(TensorRef.sum(TensorRef.relu(y)))
end

# `f` alone on the reference: the same function the native side runs.
def reference_forward() -> Result[Int]
  x = try TensorRef.from_list(values(12), Dataset.X_SHAPE)
  w = try TensorRef.from_list(values(6), Dataset.W_SHAPE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try forward(x, w)
  end

  ok(Time.monotonic_ns() - start)
end

# `f` and its hand-written backward, both on the reference.
def reference_step() -> Result[Int]
  x = try TensorRef.from_list(values(12), Dataset.X_SHAPE)
  w = try TensorRef.from_list(values(6), Dataset.W_SHAPE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Model.loss_and_grad(x, w)
  end

  ok(Time.monotonic_ns() - start)
end

# The native side: one call of the op per round through the module, plus the
# free of what it made.
def timed_add() -> Result[Int]
  Native.arena do
    a = try Torch.create(PAIR, values(4))
    b = try Torch.create(PAIR, values(4))
    start = Time.monotonic_ns()
    for _round in 0...ROUNDS
      c = try Torch.add(a, b)
      Native.close(c)
    end

    ok(Time.monotonic_ns() - start)
  end
end

def timed_matmul() -> Result[Int]
  Native.arena do
    a = try Torch.create(Dataset.X_SHAPE, values(12))
    b = try Torch.create(Dataset.W_SHAPE, values(6))
    start = Time.monotonic_ns()
    for _round in 0...ROUNDS
      c = try Torch.matmul(a, b)
      Native.close(c)
    end

    ok(Time.monotonic_ns() - start)
  end
end

def timed_create() -> Result[Int]
  Native.arena do
    wide = values(4096)
    start = Time.monotonic_ns()
    for _round in 0...ROUNDS
      t = try Torch.create(WIDE, wide)
      Native.close(t)
    end

    ok(Time.monotonic_ns() - start)
  end
end

# `f` alone on libtorch: the same function as `f` on the reference.
def timed_forward() -> Result[Int]
  x = try TensorRef.from_list(values(12), Dataset.X_SHAPE)
  w = try TensorRef.from_list(values(6), Dataset.W_SHAPE)
  start = Time.monotonic_ns()
  for _round in 0...ROUNDS
    _ = try Torch.loss(x, w)
  end

  ok(Time.monotonic_ns() - start)
end

def report(name: String, reference_ns: Int, native_ns: Optional[Int], missing: String) -> Void
  puts(format("%-26s %12d %12s", name, reference_ns / ROUNDS, cell(native_ns, missing)))
end

# A row without a native number says why instead of printing a zero: the
# module did not build ("skipped") or the row has no native side ("n/a").
def cell(ns: Optional[Int], missing: String) -> String
  case ns
    some(value) -> (value / ROUNDS).to_s
    none -> missing
  end
end
