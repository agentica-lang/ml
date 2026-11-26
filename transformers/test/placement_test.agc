## Where `layers_on_device` puts the blocks, the table and the head: pure, so it runs on any machine.

import Placement from "../placement"
import Tensor from "pkg/tensor/tensor"

HOME = Tensor.Backend.Ggml(Tensor.Device.Cpu)
CARD = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
BLOCKS = 28

## On a Vulkan device the blocks from the `n`th on are on the CPU; with `none`, or on any other backend, they stay.
def test_the_blocks_from_n_on_are_on_the_cpu_under_a_vulkan_device() -> Result[Void]
  unset: Optional[Int] = none
  for i in [0, 13, BLOCKS - 1]
    try Test.eq(Placement.layer_backend(CARD, i, unset), CARD)
    try Test.eq(Placement.layer_backend(HOME, i, some(5)), HOME)
  end

  try Test.eq(Placement.layer_backend(CARD, 4, some(5)), CARD)
  try Test.eq(Placement.layer_backend(CARD, 5, some(5)), HOME)
  try Test.eq(Placement.layer_backend(CARD, 0, some(0)), HOME)
  ok()
end

## On a Vulkan device the table is on the CPU for an `n` below the blocks, and the head is there only for 0; everything
## else, and every other backend, stays where the model opens.
def test_the_placement_of_the_table_and_the_head() -> Result[Void]
  unset: Optional[Int] = none
  for n in [unset, some(BLOCKS), some(BLOCKS + 1)]
    for backend in [HOME, CARD]
      side = Placement.placement(backend, BLOCKS, n)
      try Test.eq(side.table, backend)
      try Test.eq(side.head, backend)
    end
  end

  for n in [0, 1, BLOCKS - 1]
    away = Placement.placement(CARD, BLOCKS, some(n))
    try Test.eq(away.table, HOME)
    head = CARD
    if n == 0
      head = HOME
    end

    try Test.eq(away.head, head)
    same = Placement.placement(HOME, BLOCKS, some(n))
    try Test.eq(same.table, HOME)
    try Test.eq(same.head, HOME)
  end

  ok()
end
