## The M0 spike, as a program: one loss curve on the reference backend, the
## same curve on libtorch, and the gap between them.
##
## The function is f(w) = sum(relu(matmul(x, w))) over one fixed x and w, run
## for 20 plain SGD steps. Nothing here is random: the same input always gives
## the same numbers, on both backends, so this output is the evidence the
## README quotes.
##
## From `spike/m0/`, with libtorch found by `tools/build-torch.sh` (without it
## the native half is skipped and says so):
##
## ```sh
## eval "$(sh tools/build-torch.sh)"
## agentica demo.agc
## ```

import Dataset from "dataset"
import Model from "model"
import TensorRef from "tensor_ref"
import Torch from "torch_spike"

def main() -> Void
  case reference_curve()
    ok(reference) ->
      puts("# reference: f(w) = sum(relu(matmul(x, w))), #{Dataset.STEPS} steps, lr #{Dataset.LR}, F64")
      print_curve(reference)
      native_side(reference)
    error(e) -> fail(e)
  end
end

## The 20 steps on the reference backend: the loss and the hand-written
## backward are both Agentica code.
def reference_curve() -> Result[List[Float]]
  x = try TensorRef.from_list(Dataset.X_DATA, Dataset.X_SHAPE)
  w = try TensorRef.from_list(Dataset.W_DATA, Dataset.W_SHAPE)
  curve: List[Float] = []
  for _step in 0...Dataset.STEPS
    step = try Model.loss_and_grad(x, w)
    curve = curve.push(step.loss)
    w = try Model.sgd_step(w, step.grad, Dataset.LR)
  end

  ok(curve)
end

## The same 20 steps with the loss computed on libtorch. The gradient is the
## reference's hand-written one and the step happens in Agentica: a backend
## is forward-only (`tensor/DESIGN.md` §10), so `f` is the only thing that
## crosses. That also isolates what is measured: both curves step the same
## parameters, so the difference between them is the forward pass alone.
def native_curve() -> Result[List[Float]]
  x = try TensorRef.from_list(Dataset.X_DATA, Dataset.X_SHAPE)
  w = try TensorRef.from_list(Dataset.W_DATA, Dataset.W_SHAPE)
  curve: List[Float] = []
  for _step in 0...Dataset.STEPS
    loss = try Torch.loss(x, w)
    curve = curve.push(loss)
    step = try Model.loss_and_grad(x, w)
    w = try Model.sgd_step(w, step.grad, Dataset.LR)
  end

  ok(curve)
end

## Runs the native half when the module built, and says plainly when it did
## not: no libtorch on this machine is a skipped measurement, never a pass.
def native_side(reference: List[Float]) -> Void
  case Native.load("torchspike")
    ok(_) ->
      case native_curve()
        ok(native) -> compare(reference, native)
        error(e) -> fail(e)
      end

    error(e) ->
      puts("")
      puts("native side: SKIPPED, the torchspike module is not built or loadable")
      puts("  #{e.message}")
      puts("  (libtorch is a system dependency: see README, \"The native side\")")
  end
end

def compare(reference: List[Float], native: List[Float]) -> Void
  puts("")
  puts("# torchspike: the same function on libtorch CPU, float32")
  print_curve(native)
  gap = 0.0
  for i in 0...reference.size
    d = (reference[i] - native[i]).abs
    if d > gap
      gap = d
    end
  end

  puts("")
  puts(format("max |reference - torch| = %.9f, tolerance %.6f", gap, Dataset.TOLERANCE))
  if gap <= Dataset.TOLERANCE
    puts("loss curves: MATCH")
  else
    puts("loss curves: MISMATCH")
    System.exit(1)
  end
end

def print_curve(curve: List[Float]) -> Void
  curve.each_with_index do |loss, step| puts(format("step %2d  loss %.9f", step + 1, loss)) end
end

def fail(e: Error) -> Void
  puts("m0: #{e.message}")
  System.exit(1)
end
