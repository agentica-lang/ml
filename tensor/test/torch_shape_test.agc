## `reshape` through the dispatch, plus the volume rule that keeps a native
## reshape from reinterpreting memory the wrapper did not expect.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_reshape_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.reshape(try Conform.on_torch(data, [2, 3]), [3, 2])
  try Test.eq(Tensor.shape(ours), [3, 2])
  Conform.agrees(ours, try Tensor.reshape(try Conform.on_ref(data, [2, 3]), [3, 2]), "reshape")
end

def test_reshape_to_a_rank_zero_tensor_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ours = try Tensor.reshape(try Conform.on_torch([4.5], [1]), [])
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.reshape(try Conform.on_ref([4.5], [1]), []), "reshape to rank 0")
end

def test_reshape_to_a_bigger_rank_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  Conform.agrees(
    try Tensor.reshape(try Conform.on_torch(data, [4]), [2, 1, 2]),
    try Tensor.reshape(try Conform.on_ref(data, [4]), [2, 1, 2]),
    "reshape to rank 3",
  )
end

def test_reshape_with_a_wrong_volume_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  try Test.error?(Tensor.reshape(try Conform.on_torch(data, [3]), [2, 2]))
  Test.error?(Tensor.reshape(try Conform.on_ref(data, [3]), [2, 2]))
end

def test_reshape_with_a_negative_dimension_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0]
  try Test.error?(Tensor.reshape(try Conform.on_torch(data, [2]), [2, -1]))
  Test.error?(Tensor.reshape(try Conform.on_ref(data, [2]), [2, -1]))
end
