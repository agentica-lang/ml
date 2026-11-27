## Where `layers_on_device` puts the blocks, the table and the head: pure, so it runs on any machine.

import Placement from "../placement"
import Hub from "pkg/hub/hub"
import Tensor from "pkg/tensor/tensor"
import Write from "pkg/hub/safetensors_write"

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
  head = 30 * MB
  if tied
    head = 40 * MB
  end

  Placement.Footprint(
    blocks: [100 * MB, 100 * MB, 100 * MB, 100 * MB, 100 * MB],
    table: 40 * MB,
    head: head,
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
  try Test.eq(Placement.fit(tied, 440 * MB), 4)
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

def test_bf16_head_widening_counts_even_when_the_table_is_shared() -> Result[Void]
  tied = sizes(true).with(head_widen: 200 * MB)
  try Test.eq(Placement.fit(tied, 339 * MB), 0)
  try Test.eq(Placement.fit(tied, 340 * MB), 1)
  try Test.eq(Placement.fit(tied, 739 * MB), 4)
  Test.eq(Placement.fit(tied, 740 * MB), 5)
end

## A checkpoint of two blocks (the second twice the first), a table of 10 x 4 and a norm, with a head of its own when
## `tied` is false, and a block `10` that the second must not count.
def with_checkpoint(tied: Bool, body: Fn[Hub.Checkpoint, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-placement-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  zeros = ->(shape: List[Int]) { Tensor.zeros(shape, Tensor.Dtype.F32, Tensor.Backend.Reference) }
  tensors = {
    "model.embed_tokens.weight": try zeros.call([10, 4]),
    "model.norm.weight": try zeros.call([4]),
    "model.layers.0.up.weight": try zeros.call([3, 4]),
    "model.layers.0.norm.weight": try zeros.call([4]),
    "model.layers.1.up.weight": try zeros.call([6, 4]),
    "model.layers.1.norm.weight": try zeros.call([4]),
    "model.layers.10.up.weight": try zeros.call([100, 4]),
  }

  if not tied
    tensors = Map.put(tensors, "lm_head.weight", try zeros.call([10, 4]))
  end

  try Write.write("#{dir}/model.safetensors", tensors)
  result = body.call(try Hub.open("#{dir}/model.safetensors"))
  _ = Path.remove(dir, 5000)
  result
end

## The footprint of a checkpoint is its blocks by their prefix (a block `1` is not block `10`), the KV cache of each one added,
## the table, and the head with the final norm: of its own when the checkpoint has one, the size of the table when tied.
def test_the_footprint_counts_the_tensors_of_each_block_the_table_and_the_head() -> Result[Void]
  f32 = Tensor.Dtype.F32
  try with_checkpoint(false) do |c|
    got = try Placement.footprint(
      c,
      "model.layers.",
      2,
      "model.embed_tokens.weight",
      "lm_head.weight",
      "model.norm.weight",
      f32,
      1000,
    )

    try Test.eq(got.blocks, [(12 + 4) * 4 + 1000, (24 + 4) * 4 + 1000])
    try Test.eq(got.table, 10 * 4 * 4)
    try Test.eq(got.head, 10 * 4 * 4 + 4 * 4)
    try Test.eq(got.tied, false)
    ok()
  end

  with_checkpoint(true) do |c|
    got = try Placement.footprint(
      c,
      "model.layers.",
      2,
      "model.embed_tokens.weight",
      "lm_head.weight",
      "model.norm.weight",
      f32,
      0,
    )

    try Test.eq(got.blocks, [(12 + 4) * 4, (24 + 4) * 4])
    try Test.eq(got.head, 10 * 4 * 4 + 4 * 4)
    Test.eq(got.tied, true)
  end
end
