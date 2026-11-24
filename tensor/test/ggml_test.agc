## The ggml backend (`src/ggml.agc`, `tensor/DESIGN.md` §7): availability,
## creation and readback against the reference on ggml's CPU backend and on
## Vulkan device 0, the refusals, two tasks on one Vulkan device, and lifetimes.
##
## The module is `optional`: without it (no `tools/build-engine.sh` exports)
## the checks assert the absent state, and without a Vulkan device the Vulkan
## checks assert that one. The runner prints only PASS (note.md §5), so a
## skipped native path is invisible: CI sets `TENSOR_REQUIRE_GGML`, and then a
## missing module fails every check that needs it.

import Conform from "torch_support"
import Tensor from "../tensor"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

def floats(xs: List[Float]) -> List[Tensor.Values] = xs.map do |x: Float| Tensor.Values.Float(x) end

def ints(xs: List[Int]) -> List[Tensor.Values] = xs.map do |x: Int| Tensor.Values.Int(x) end

## True when the ggml module is built. With `TENSOR_REQUIRE_GGML` set to
## anything but "" a missing module is an error, never a pass.
def present() -> Result[Bool]
  return ok(true) if Tensor.available(CPU)
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

## Uploads and reads back `rounds` tensors of whole numbers from `base` on
## Vulkan device 0, checking each one.
def churn(base: Int, rounds: Int) -> Result[Bool]
  data = floats((0...1024).to_a.map { |i| (base + i).to_f })
  for _ in 0...rounds
    t = try Tensor.from_list(data, [32, 32], Tensor.Dtype.F32, VULKAN)
    return error("a Vulkan round trip from #{base} came back different") if try Tensor.to_list(t) != data
  end

  ok(true)
end

## The error `r` answers names `text`.
def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("computed a tensor where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

## The same creation on `backend` and on the reference agree in shape, dtype
## and values; then `from_list` and `to_list` round-trip every dtype ggml holds.
def round_trips(backend: Tensor.Backend) -> Result[Void]
  f32 = Tensor.Dtype.F32
  ref = Conform.REF
  try Conform.agrees(try Tensor.zeros([2, 3], f32, backend), try Tensor.zeros([2, 3], f32, ref), "zeros")
  try Conform.agrees(try Tensor.ones([3], f32, backend), try Tensor.ones([3], f32, ref), "ones")
  try Conform.agrees(try Tensor.empty([2], f32, backend), try Tensor.empty([2], f32, ref), "empty")
  v = Tensor.Values.Float(-1.5)
  try Conform.agrees(try Tensor.full([2, 2], v, f32, backend), try Tensor.full([2, 2], v, f32, ref), "full")
  try Conform.agrees(try Tensor.scalar_to_tensor(v, f32, backend), try Tensor.scalar_to_tensor(v, f32, ref), "scalar")
  down = try Tensor.arange(5, -4, -2, Tensor.Dtype.I32, backend)
  try Conform.agrees(down, try Tensor.arange(5, -4, -2, Tensor.Dtype.I32, ref), "arange")
  try Conform.agrees(
    try Tensor.like(down),
    try Tensor.like(try Tensor.arange(5, -4, -2, Tensor.Dtype.I32, ref)),
    "like",
  )

  data = floats([0.1, -2.5, 3.0e6, 0.0, 1.0e-3, 7.25])
  try Conform.agrees(
    try Tensor.from_list(data, [1, 2, 1, 3], f32, backend),
    try Conform.on_ref([0.1, -2.5, 3.0e6, 0.0, 1.0e-3, 7.25], [1, 2, 1, 3]),
    "from_list F32",
  )

  big = ints([2147483647, -2147483648, 0])
  try Test.eq(try Tensor.to_list(try Tensor.from_list(big, [3], Tensor.Dtype.I32, backend)), big)
  small = ints([-128, 127, 0])
  try Test.eq(try Tensor.to_list(try Tensor.from_list(small, [3], Tensor.Dtype.I8, backend)), small)
  bytes = ints([0, 255])
  try Test.eq(try Tensor.to_list(try Tensor.from_list(bytes, [2], Tensor.Dtype.U8, backend)), bytes)
  mask = [Tensor.Values.Bool(true), Tensor.Values.Bool(false)]
  try Test.eq(try Tensor.to_list(try Tensor.from_list(mask, [2], Tensor.Dtype.Bool, backend)), mask)
  empty = try Tensor.from_list([], [0, 3], f32, backend)
  try Test.eq(Tensor.shape(empty), [0, 3])
  Test.eq(try Tensor.to_list(empty), [])
end

def test_floats_uploads_on_the_cpu_backend_and_checks_the_count() -> Result[Void]
  return ok() if !try present()
  data = [1.5, -2.0, 3.0, 0.25]
  try Test.eq(try Tensor.to_list(try Tensor.floats(data, [2, 2], CPU)), floats(data))
  Test.error?(Tensor.floats(data, [3], CPU))
end

def test_the_cpu_module_state_is_the_one_the_dispatch_answers() -> Result[Void]
  if !try present()
    IO.warn(
      "SKIP: the native ggml module is not built; the checks assert the absent state. Run " +
      "`eval \"$(sh tools/build-engine.sh | grep '^export ')\"` first (README, \"The ggml backend\").",
    )

    try Test.assert(!Tensor.available(VULKAN), "Vulkan is available without the module")
    try refused(Tensor.zeros([2], Tensor.Dtype.F32, CPU), "backend not available")
    return refused(Tensor.zeros([2], Tensor.Dtype.F32, VULKAN), "backend not available")
  end

  round_trips(CPU)
end

def test_vulkan_device_zero_round_trips_or_says_why_not() -> Result[Void]
  if !Tensor.available(VULKAN)
    IO.warn(
      "SKIP: no Vulkan device 0 (no module, a CPU-only ggml build, or no device); the check asserts " +
      "that a Vulkan tensor is an error. Build with `tools/build-engine.sh vulkan` on a machine with one.",
    )

    return refused(Tensor.zeros([2], Tensor.Dtype.F32, VULKAN), "backend not available") if !try present()
    return refused(Tensor.zeros([2], Tensor.Dtype.F32, VULKAN), "zeros: no Vulkan device 0")
  end

  round_trips(VULKAN)
end

def test_two_tasks_upload_and_read_back_on_vulkan_at_once() -> Result[Void]
  return ok() if !Tensor.available(VULKAN)

  a = Task.spawn do churn(0, 10) end
  b = Task.spawn do churn(5000, 10) end
  try Task.await(a)
  try Task.await(b)
  ok()
end

def test_every_op_points_at_trace_and_bad_storage_is_refused() -> Result[Void]
  return ok() if !try present()

  t = try Tensor.ones([2], Tensor.Dtype.F32, CPU)
  try refused(Tensor.add(t, t), "add does not run eagerly on the ggml backend")
  m = try Tensor.ones([2, 2], Tensor.Dtype.F32, CPU)
  try refused(Tensor.matmul(m, m), "Trace")
  try refused(Tensor.softmax(t, 0), "Trace")
  try refused(Tensor.zeros([2], Tensor.Dtype.F64, CPU), "no F64 tensor")
  try refused(Tensor.arange(0, 3, 1, Tensor.Dtype.I64, CPU), "no I64 tensor")
  try refused(Tensor.zeros([1, 1, 1, 1, 1], Tensor.Dtype.F32, CPU), "rank 4 at most")
  try refused(Tensor.zeros([1048576, 1048576, 1048576], Tensor.Dtype.F32, CPU), "too large")
  try refused(Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Ggml(Tensor.Device.Vulkan(-1))), ">= 0")
  try refused(
    Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Ggml(Tensor.Device.Vulkan(64))),
    "no Vulkan device 64",
  )

  ref = try Conform.on_ref([1.0, 2.0], [2])
  try refused(Tensor.add(t, ref), "Ggml(Cpu) and Reference")
  Test.ok?(Tensor.stop_grad(t))
end

def test_to_device_copies_and_leaves_the_source_readable() -> Result[Void]
  return ok() if !try present()

  data = floats([1.5, -2.0, 3.25])
  f32 = try Tensor.from_list(data, [3, 1], Tensor.Dtype.F32, CPU)
  moved = try Tensor.to_device(f32, Tensor.Device.Cpu)
  try Test.eq(Tensor.shape(moved), [3, 1])
  try Test.eq(Tensor.dtype(moved), Tensor.Dtype.F32)
  try Test.eq(try Tensor.to_list(moved), data)
  try Test.eq(try Tensor.to_list(f32), data)
  labels = ints([1, -7, 16777216])
  i32 = try Tensor.from_list(labels, [3], Tensor.Dtype.I32, CPU)
  copied = try Tensor.to_device(i32, Tensor.Device.Cpu)
  try Test.eq(Tensor.dtype(copied), Tensor.Dtype.I32)
  try Test.eq(try Tensor.to_list(copied), labels)
  try Test.eq(try Tensor.to_list(i32), labels)
  refused(Tensor.to_device(f32, Tensor.Device.Vulkan(99)), "no Vulkan device")
end

def test_fp16_is_the_default_and_is_fixed_by_the_first_ggml_call() -> Result[Void]
  return ok() if !try present()

  _ = try Tensor.zeros([1], Tensor.Dtype.F32, CPU)
  case Tensor.vulkan_fp16(false)
    ok(_) -> return error("F32 was chosen after a ggml call fixed fp16")
    error(e) -> try Test.assert(e.message.include?("chose fp16 mul_mat"), "the error said \"#{e.message}\"")
  end

  Test.ok?(Tensor.vulkan_fp16(true))
end

def test_the_other_backends_refuse_a_vulkan_device() -> Result[Void]
  try refused(Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Torch(Tensor.Device.Vulkan(1))), "Vulkan(1)")
  ref = try Conform.on_ref([1.0], [1])
  refused(Tensor.to_device(ref, Tensor.Device.Vulkan(0)), "Cpu only")
end

def test_a_fill_past_the_list_cap_is_made_in_c() -> Result[Void]
  return ok() if !try present()

  t = try Tensor.full([1100, 1000], Tensor.Values.Float(0.5), Tensor.Dtype.F32, CPU)
  Test.eq(Tensor.shape(t), [1100, 1000])
end

def test_a_tensor_leaves_its_arena_and_closes_once() -> Result[Void]
  return ok() if !try present()

  kept = try Native.arena do Tensor.ones([2], Tensor.Dtype.F32, CPU) end

  try Test.assert(!Conform.native_closed(kept), "the arena closed its value")
  try Test.eq(try Tensor.to_list(kept), floats([1.0, 1.0]))
  case kept.handle
    Tensor.Handle.Ggml(h) -> Native.close(h)
    _ -> return error("a ggml tensor holds another handle")
  end

  Test.assert(Conform.native_closed(kept), "a closed ggml handle reads as open")
end
