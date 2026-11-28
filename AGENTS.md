# Agent instructions

## Backend support

- Prefer composing existing `nn/` and `tensor/` operations in one shared implementation. If a backend already implements and tests the primitives the feature uses, do not add a backend allowlist, lower feature limits, or require an experimental flag solely because that feature has not been tested end to end on that backend.
- Add a backend-specific restriction only for a concrete limitation, such as an unsupported operation, shape, or dtype, or a reproduced correctness failure. Keep the restriction as narrow as possible and document the evidence.
- Distinguish tested from untested backends in PRs and docs. Do not claim validation or performance measurements for runs that did not execute.
