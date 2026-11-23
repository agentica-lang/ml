## The `optional = true` contract of the native module: where libtorch is not
## installed the package still checks, formats and tests, and the `Torch` arm of
## the dispatch answers `error("backend not available")` — asserted here, never
## a silent skip. Where it is installed, the arm really computes.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_the_native_module_state_is_the_one_the_dispatch_answers() -> Result[Void]
  if !Tensor.available(Conform.TORCH)
    IO.warn(
      "SKIP: the native torch module is not built (no libtorch found); the conformance " +
      "checks assert the absent state instead. Run `eval \"$(sh tools/build-torch.sh)\"` in the " +
      "shell that runs agentica to compare against the reference (README, \"Local setup\").",
    )

    return Conform.absent()
  end

  try Test.assert(Tensor.available(Conform.TORCH), "the native torch module is not loaded")
  Test.ok?(Tensor.zeros([1], Tensor.Dtype.F32, Conform.TORCH))
end

def test_the_reference_backend_is_always_available() -> Result[Void]
  Test.assert(Tensor.available(Conform.REF), "the reference backend must always be available")
end
