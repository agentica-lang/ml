## The `optional = true` contract of `agentica.toml`: where libtorch is not
## installed the package still checks, formats and tests, and where it is, the
## ops really run. This file is the one that says which of the two happened,
## loudly, so a skipped conformance run never looks like a passing one.

import Support from "support"
import Torch from "../torch"

private

def test_the_native_module_state_is_the_one_the_ops_answer() -> Result[Void]
  if !Torch.available()
    puts(
      "SKIP: the native torch module is not built (no libtorch found); the conformance " +
      "checks above asserted the absent state instead. Run `eval \"$(sh tools/build-torch.sh)\"` " +
      "in the shell that runs agentica to run them (README, \"Local setup\").",
    )

    return Support.absent()
  end

  try Test.assert(Torch.available(), "the native torch module is not loaded")
  Test.ok?(Torch.zeros(Support.CPU, [1], Torch.Dtype.F32))
end

def test_every_op_names_the_missing_module_instead_of_halting() -> Result[Void]
  if Torch.available()
    IO.warn("the native torch module is loaded: the missing-module paths are not exercised here")
    return Test.ok?(Torch.zeros(Support.CPU, [1], Torch.Dtype.F32))
  end

  t = Torch.zeros(Support.CPU, [1], Torch.Dtype.F32)
  Test.error?(t)
end
