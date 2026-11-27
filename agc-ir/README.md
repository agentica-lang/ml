# agc-ir

A small Agentica library for building and printing MLIR. It has no dependency on
the tensor graph, StableHLO, PJRT, LLVM or a parser. The workspace's `tensor`
package is its first consumer.

`ir.agc` models SSA values, tensor types, attributes, operations and single-block
regions. `block` creates arguments, `append` adds an operation and returns its
typed results, and `function` prints a private MLIR function. The builder checks
operand visibility and the printer checks lexical scope again when embedding
regions. Operation-specific contracts belong in the caller's dialect layer.

The current API is intentionally limited to what the gated-delta lowering needs.
Run `agentica test` from this directory to test the package without any ML backend.
