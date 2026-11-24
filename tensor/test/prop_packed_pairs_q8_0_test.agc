## Every pair of ops (`prop_packed_support.agc`), from the input and from the weight itself, with ggml holding the
## weight as Q8_0: each answers within the derived bound of the reference or is refused with a message that fails
## closed, and what it refuses is pinned (nothing with F32). A file of its own: `agentica test` gives each file its
## step budget.

import Packed from "prop_packed_support"

def test_every_pair_of_ops_from_the_input_matches_the_reference_or_is_refused() -> Result[Void]
  Packed.pairs_for(Packed.kinds()[2], 0)
end

def test_every_pair_of_ops_from_the_weight_matches_the_reference_or_is_refused() -> Result[Void]
  Packed.pairs_for(Packed.kinds()[2], 1)
end
