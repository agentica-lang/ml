## What this spike can prove on its own: the boundary does what it says on the
## CPU backend, and — when this machine has a Vulkan device — the Vulkan
## backend computes the same tensors.
##
## The assertions that need the iGPU are written so that "no Vulkan device" is
## a stated skip and not a silent pass: `Native.load` failing, or
## `backend_vulkan` returning an error, is what a machine without Vulkan
## answers, and the test says which of the two happened.
##
## ```sh
## agentica test
## ```

import Ggml from "../ggml_spike"

## The values `native/ggmlspike.c` fills every tensor with, `(i * 7 + 3) % 11`
## over 11 minus 0.4, in float32: the same pattern `tensor/tools/bench.agc`
## uses, so a readback can be checked against arithmetic instead of against
## another backend.
def pattern(i: Int) -> Float = ((i * 7 + 3) % 11).to_f / 11.0 - 0.4

def test_the_module_lists_ggml_devices_or_is_absent() -> Result[Void]
  return Test.assert(!ready(), "ggmlspike is not built: ggml is not built (see README)") if !ready()
  devices = try Ggml.devices()
  try Test.assert(devices.size >= 1, "ggml registered no device")
  try Test.assert(devices.any? do |d: String| d.start_with?("CPU") end, "no CPU device among #{devices.inspect}")
  ok()
end

def test_vulkan_is_an_igpu_and_not_a_cpu_fallback() -> Result[Void]
  Native.arena do
    case Ggml.backend_vulkan()
      ok(gpu) ->
        description = try Ggml.describe(gpu)
        try Test.assert(
          description.include?("(IGPU)") || description.include?("(GPU)"),
          "the vulkan backend describes itself as #{description}",
        )

        try Test.assert(description.include?("vulkan=1"), "the backend is not ggml's Vulkan one: #{description}")
      error(e) -> try Test.assert(true, "no Vulkan device on this machine: #{e.message}")
    end

    ok()
  end
end

def test_readback_returns_the_pattern() -> Result[Void]
  Native.arena do
    cpu = try Ggml.backend_cpu(1)
    t = try Ggml.create(cpu, [2, 3])
    values = try Ggml.readback(t)
    i = 0
    while i < values.size
      try Test.assert(
        (values[i] - pattern(i)).abs < 0.000001,
        "value #{i} is #{values[i]}, the pattern gives #{pattern(i)}",
      )

      i += 1
    end

    # ggml sizes a backend buffer by its own alignment, so the number is the
    # tensor's bytes rounded up: 16384 for [64, 64] of F32 is exact.
    wide = try Ggml.create(cpu, [64, 64])
    try Test.eq(Ggml.allocated(wide), ok(16384))
    try Test.assert(try Ggml.allocated(t) >= 24, "a [2, 3] tensor allocates fewer bytes than it holds")
    ok()
  end
end

def test_add_and_relu_agree_between_the_backends() -> Result[Void]
  Native.arena do
    cpu = try Ggml.backend_cpu(8)
    case Ggml.backend_vulkan()
      ok(gpu) ->
        a = try Ggml.create(cpu, [64, 64])
        b = try Ggml.create(cpu, [64, 64])
        ga = try Ggml.create(gpu, [64, 64])
        gb = try Ggml.create(gpu, [64, 64])
        cpu_sum = try Ggml.checksum(try Ggml.relu(try Ggml.add(a, b)))
        gpu_sum = try Ggml.checksum(try Ggml.relu(try Ggml.add(ga, gb)))
        try Test.assert(
          (cpu_sum - gpu_sum).abs < 0.000001,
          "relu(add) sums to #{cpu_sum} on the cpu and #{gpu_sum} on the gpu",
        )
      error(_) -> try Test.assert(true, "no Vulkan device on this machine: the cross-backend checks did not run")
    end

    ok()
  end
end

def test_mul_mat_agrees_within_the_arithmetic_the_device_uses() -> Result[Void]
  Native.arena do
    cpu = try Ggml.backend_cpu(8)
    case Ggml.backend_vulkan()
      ok(gpu) ->
        a = try Ggml.create(cpu, [64, 64])
        b = try Ggml.create(cpu, [64, 64])
        ga = try Ggml.create(gpu, [64, 64])
        gb = try Ggml.create(gpu, [64, 64])
        cpu_sum = try Ggml.checksum(try Ggml.mul_mat(a, b))
        gpu_sum = try Ggml.checksum(try Ggml.mul_mat(ga, gb))
        relative = (cpu_sum - gpu_sum).abs / cpu_sum.abs
        # ggml's Vulkan backend computes F32 mul_mat in fp16 on this device
        # (GGML_VK_DISABLE_F16=1 turns that off), so the two are not the same
        # arithmetic; 2e-3 is the measured gap of the fp16 path.
        try Test.assert(
          relative < 0.002,
          "mul_mat sums to #{cpu_sum} on the cpu and #{gpu_sum} on the gpu (#{relative} relative)",
        )
      error(_) -> try Test.assert(true, "no Vulkan device on this machine: the cross-backend checks did not run")
    end

    ok()
  end
end

def test_a_shape_mismatch_is_an_error_and_names_the_shapes() -> Result[Void]
  Native.arena do
    cpu = try Ggml.backend_cpu(1)
    wide = try Ggml.create(cpu, [2, 3])
    tall = try Ggml.create(cpu, [3, 2])
    case Ggml.add(wide, tall)
      ok(_) -> try Test.assert(false, "add [2, 3] + [3, 2] returned a tensor")
      error(e) ->
        try Test.assert(
          e.message.include?("[2, 3]") && e.message.include?("[3, 2]"),
          "the error does not name both shapes: #{e.message}",
        )
    end

    ok()
  end
end

def test_create_takes_a_rank_two_shape() -> Result[Void]
  Native.arena do
    cpu = try Ggml.backend_cpu(1)
    try Test.error?(Ggml.create(cpu, [3]))
    try Test.error?(Ggml.create(cpu, [0, 4]))
    ok()
  end
end

def test_an_empty_graph_runs_on_both_backends() -> Result[Void]
  Native.arena do
    cpu = try Ggml.backend_cpu(1)
    try Test.eq(Ggml.empty_graph(cpu).ok?, true)
    case Ggml.backend_vulkan()
      ok(gpu) -> try Test.eq(Ggml.empty_graph(gpu).ok?, true)
      error(_) -> try Test.assert(true, "no Vulkan device on this machine: the empty graph ran on the cpu only")
    end

    ok()
  end
end

private

def ready() -> Bool = Native.load("ggmlspike").ok?
