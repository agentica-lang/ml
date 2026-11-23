## The ggml backend (`src/ggml.agc`, `tensor/DESIGN.md` §7): availability,
## creation and readback against the reference on ggml's CPU backend and on
## Vulkan device 0, the refusals, and the process-wide fp16 choice.
##
## The module is `optional`: without it (no `tools/build-engine.sh` exports)
## the checks assert the absent state, and without a Vulkan device the Vulkan
## checks assert that one, each saying so on stderr. The runner prints only
## PASS, so the reason is spelled out here too (note.md §5).

import Conform from "torch_support"
import Tensor from "../tensor"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

def floats(xs: List[Float]) -> List[Tensor.Values] = xs.map do |x: Float| Tensor.Values.Float(x) end

def ints(xs: List[Int]) -> List[Tensor.Values] = xs.map do |x: Int| Tensor.Values.Int(x) end

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

def test_the_cpu_module_state_is_the_one_the_dispatch_answers() -> Result[Void]
  if !Tensor.available(CPU)
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

    made = Tensor.zeros([2], Tensor.Dtype.F32, VULKAN)
    return Test.assert(!made.ok?, "a Vulkan tensor was made where available said no")
  end

  round_trips(VULKAN)
end

def test_the_fp16_choice_is_fixed_by_the_first_vulkan_tensor() -> Result[Void]
  if !Tensor.available(VULKAN)
    IO.warn("SKIP: no Vulkan device 0; the fp16 choice is only checked to be refused without the module")
    return Test.assert(Tensor.available(CPU) or !Tensor.vulkan_fp16(false).ok?, "vulkan_fp16 ran without the module")
  end

  _ = try Tensor.zeros([1], Tensor.Dtype.F32, VULKAN)
  try Test.eq(try System.env("GGML_VK_DISABLE_F16"), "1")
  try Test.ok?(Tensor.vulkan_fp16(false))
  case Tensor.vulkan_fp16(true)
    ok(_) -> error("fp16 was chosen after a Vulkan device opened with F32")
    error(e) -> Test.assert(e.message.include?("F32 mul_mat already"), "the error said \"#{e.message}\"")
  end
end

def test_every_op_points_at_trace_and_bad_storage_is_refused() -> Result[Void]
  return ok() if !Tensor.available(CPU)

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
  try refused(Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Ggml(Tensor.Device.Vulkan(64))), "Vulkan")
  ref = try Conform.on_ref([1.0, 2.0], [2])
  try refused(Tensor.add(t, ref), "Ggml(Cpu) and Reference")
  Test.ok?(Tensor.stop_grad(t))
end

def test_the_other_backends_refuse_a_vulkan_device() -> Result[Void]
  try refused(Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Torch(Tensor.Device.Vulkan(1))), "Vulkan(1)")
  ref = try Conform.on_ref([1.0], [1])
  refused(Tensor.to_device(ref, Tensor.Device.Vulkan(0)), "Cpu only")
end

def test_a_fill_past_the_list_cap_is_made_in_c() -> Result[Void]
  return ok() if !Tensor.available(CPU)

  t = try Tensor.full([1100, 1000], Tensor.Values.Float(0.5), Tensor.Dtype.F32, CPU)
  Test.eq(Tensor.shape(t), [1100, 1000])
end

def test_a_tensor_leaves_its_arena_and_closes_once() -> Result[Void]
  return ok() if !Tensor.available(CPU)

  kept = try Native.arena do Tensor.ones([2], Tensor.Dtype.F32, CPU) end

  try Test.assert(!Conform.native_closed(kept), "the arena closed its value")
  try Test.eq(try Tensor.to_list(kept), floats([1.0, 1.0]))
  case kept.handle
    Tensor.Handle.Ggml(h) -> Native.close(h)
    _ -> return error("a ggml tensor holds another handle")
  end

  Test.assert(Conform.native_closed(kept), "a closed ggml handle reads as open")
end
