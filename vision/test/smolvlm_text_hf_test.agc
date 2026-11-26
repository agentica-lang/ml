## The text model of SmolVLM against Hugging Face (`smolvlm_text_hf_golden.py`, `fixtures/smolvlm/text_golden.json`), on
## the real `HuggingFaceTB/SmolVLM-256M-Instruct` in `float32`, read by `Qwen3.open_idefics3`: the logits of every
## position of a prompt of 20 token ids (no image): their mean, the mean of their absolute values and one value of
## every `stride`th place of them, and the ten largest logits of the last position with their ids.
##
## It needs the checkpoint, which is not in the repository, so it runs only when `SMOLVLM_MODEL` names its folder
## (`config.json` and `model.safetensors`). `SMOLVLM_DEVICE=vulkan` runs on Vulkan, the CPU otherwise. Run from `vision/`:
## `SMOLVLM_MODEL=SmolVLM-256M-Instruct agentica test --max-steps 1000000000000 test/smolvlm_text_hf_test.agc`.

import Hf from "hf"
import Qwen3 from "pkg/transformers/qwen3"
import Tensor from "pkg/tensor/tensor"

## What passes for a logit or a mean: about 2.7 times the largest gap measured (4.7e-5 on the CPU, 1.1e-4 on Vulkan, for
## logits of about 20 at the top).
LOGIT_ABS = 0.0003

struct Golden
  ids: List[Int]
  shape: List[Int]
  stride: Int
  mean: Float
  abs_mean: Float
  samples: List[Float]
  top_ids: List[Int]
  top_values: List[Float]
end

private

def test_the_logits_of_the_prompt_are_hugging_faces() -> Result[Void]
  listed = System.env("SMOLVLM_MODEL")
  if !listed.ok? or try listed == ""
    puts("smolvlm_text_hf_test: skipped, SMOLVLM_MODEL names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("SMOLVLM_DEVICE")
  return error("SMOLVLM_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/smolvlm/text_golden.json"), Golden)
  puts(try check(try listed, want, backend))
  ok()
end

def check(dir: String, want: Golden, backend: Tensor.Backend) -> Result[String]
  m = try Qwen3.open_idefics3(dir, backend)
  out = try Qwen3.run(m, [want.ids])
  try Test.eq(Tensor.shape(out), [1, want.shape[0], want.shape[1]])
  values = try floats_of(out)
  Qwen3.close_model(m)
  vocab = want.shape[1]
  gap = 0.0
  for i in 0...want.samples.size
    gap = [gap, (values[i * want.stride] - want.samples[i]).abs].max.unwrap!
  end

  mean = values.sum / values.size.to_f
  abs_mean = values.map { |v| v.abs }.sum / values.size.to_f
  gap = [gap, (mean - want.mean).abs, (abs_mean - want.abs_mean).abs].max.unwrap!
  last = values.drop((want.shape[0] - 1) * vocab)
  for k in 0...want.top_ids.size
    gap = [gap, (last[want.top_ids[k]] - want.top_values[k]).abs].max.unwrap!
  end

  best = last.max.unwrap!
  try Test.eq(last.index(best).unwrap!, want.top_ids[0])
  try Test.assert(gap <= LOGIT_ABS, "a logit is #{gap} from Hugging Face's, more than #{LOGIT_ABS}")
  ok("#{want.samples.size} logits, the means and the top 10 within #{gap}")
end

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end
