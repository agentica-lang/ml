## Where `layers_on_device` puts the blocks, the table and the head: pure, so it runs on any machine.

import Placement from "../placement"
import Tensor from "pkg/tensor/tensor"

HOME = Tensor.Backend.Ggml(Tensor.Device.Cpu)
CARD = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
BLOCKS = 28
MB = 1000000

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

## The keys and values of a cache are made where the block that makes them is; `Host` keeps them all in host memory
## (llama.cpp's `--no-kv-offload`), `Device` leaves them where they are made.
def test_the_cache_follows_its_block_or_stays_in_host_memory() -> Result[Void]
  for backend in [HOME, CARD, Tensor.Backend.Reference]
    try Test.eq(Placement.cache_backend(Placement.Kv.Device, backend), backend)
  end

  try Test.eq(Placement.cache_backend(Placement.Kv.Host, CARD), HOME)
  try Test.eq(Placement.cache_backend(Placement.Kv.Host, HOME), HOME)
  try Test.eq(Placement.cache_backend(Placement.Kv.Host, Tensor.Backend.Reference), Tensor.Backend.Reference)
  ok()
end

## The tools read `AGENTICA_ML_KV` (`host` or `device`) as a `Kv`.
def test_a_host_flag_gives_the_kv_of_the_tools() -> Result[Void]
  try Test.eq(Placement.kv_of(true), Placement.Kv.Host)
  Test.eq(Placement.kv_of(false), Placement.Kv.Device)
end

## What the card holds for `n` blocks of 100 MB each, in 5 blocks, with a table of 40 MB and a head of 30 MB (`fit` counts
## the head's own copy, as `layers_on_device` makes it, for an `n` between 1 and the blocks, and the table and the head
## once each when all of them are on the card).
def sizes(tied: Bool) -> Placement.Footprint
  Placement.Footprint(
    blocks: [100 * MB, 100 * MB, 100 * MB, 100 * MB, 100 * MB],
    table: 40 * MB,
    head: 30 * MB,
    tied: tied,
  )
end

## `fit` is the largest `n` whose layout is within the budget, and 0 when not even one block fits.
def test_fit_gives_the_most_blocks_that_the_budget_holds() -> Result[Void]
  untied = sizes(false)
  try Test.eq(Placement.fit(untied, 0), 0)
  try Test.eq(Placement.fit(untied, 129 * MB), 0)
  try Test.eq(Placement.fit(untied, 130 * MB), 1)
  try Test.eq(Placement.fit(untied, 329 * MB), 2)
  try Test.eq(Placement.fit(untied, 430 * MB), 4)
  try Test.eq(Placement.fit(untied, 569 * MB), 4)
  try Test.eq(Placement.fit(untied, 570 * MB), 5)
  try Test.eq(Placement.fit(untied, 10000 * MB), 5)
  ok()
end

## A tied head is the table when every block is on the card, so it costs nothing more there: 540 MB hold all five.
def test_a_tied_head_is_not_counted_twice_with_every_block_on_the_card() -> Result[Void]
  tied = sizes(true)
  try Test.eq(Placement.fit(tied, 539 * MB), 4)
  try Test.eq(Placement.fit(tied, 540 * MB), 5)
  try Test.eq(Placement.fit(tied, 430 * MB), 4)
  ok()
end

## The budget is what is free less the reserve, capped by the limit the user set; it is never below 0.
def test_the_budget_is_the_free_memory_less_the_reserve_up_to_the_cap() -> Result[Void]
  try Test.eq(Placement.budget(1000 * MB, 200 * MB, none), 800 * MB)
  try Test.eq(Placement.budget(1000 * MB, 200 * MB, some(500 * MB)), 500 * MB)
  try Test.eq(Placement.budget(1000 * MB, 200 * MB, some(900 * MB)), 800 * MB)
  try Test.eq(Placement.budget(100 * MB, 200 * MB, none), 0)
  ok()
end
