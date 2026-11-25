# Models: weights, text generation, embeddings

## Get a checkpoint

The packages read local files and download nothing. A model is a folder. Fetch the files you need from the Hugging
Face Hub with `curl`, or with `hf download REPO --revision REV --local-dir DIR`. Pin a revision (a commit of the repo,
not `main`, which moves) and check the sha256 of the weights, which the Hub shows as the `X-Linked-ETag` of the file:

```sh
rev=c1899de289a04d12100db370d81485cdf75e47ca     # a commit of Qwen/Qwen3-0.6B; for another repo, the `sha` of https://huggingface.co/api/models/<repo>
mkdir -p Qwen3-0.6B && cd Qwen3-0.6B
for f in model.safetensors config.json tokenizer.json tokenizer_config.json; do
  curl -fL -O "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$rev/$f"
done

echo "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  model.safetensors" | sha256sum -c -
curl -sIL "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$rev/model.safetensors" | grep -i x-linked-etag   # for another model
```

| Model | Hub repo | Files | Opened by |
|---|---|---|---|
| Qwen3-0.6B (text) | `Qwen/Qwen3-0.6B` | `model.safetensors`, `config.json`, `tokenizer.json` (and `tokenizer_config.json`) | `Qwen3.open` |
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` | the same | `Qwen3.open` |
| DistilBERT sentence embeddings | `sentence-transformers/msmarco-distilbert-cos-v5` | `model.safetensors`, `config.json`, `tokenizer.json` | `DistilBert.open` |
| DistilBERT | `distilbert/distilbert-base-uncased` | the same | `DistilBert.open` |
| BERT | `google-bert/bert-base-uncased` | the same | `Bert.open` |
| ResNet-50, ResNet-18 (image classification) | `microsoft/resnet-50`, `microsoft/resnet-18` | `model.safetensors`, `config.json` | `ResNet.open` (the batch norms fold into the convolutions at load; the input is `[batch, 3, 224, 224]` after `vision`'s `Resize(256, Vision.Kernel.Cubic)`, `CenterCrop(224)` and `normalize` with the ImageNet mean and std) |
| ViT-B/16 (image classification) | `google/vit-base-patch16-224` | `model.safetensors`, `config.json` | `ViT.open` (the input is `[batch, 3, 224, 224]` after `vision`'s `ResizeTo(224, 224, Vision.Kernel.Linear)` and `normalize` with mean and std 0.5) |
| CLIP ViT-B/32 (image and text embeddings, zero-shot) | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (`openai/clip-vit-base-patch32` has no safetensors) | `model.safetensors`, `config.json`, `vocab.json`, `merges.txt` | `Clip.open` (its tokenizer is `Clip.open` of `tokenizer/clip`; the image input is `[batch, 3, 224, 224]` after `vision`'s `Resize(224, Vision.Kernel.Cubic)`, `CenterCrop(224)` and `normalize` with CLIP's mean and std; `Clip.probabilities` is the zero-shot softmax over prompts) |
| Whisper tiny | `openai/whisper-tiny` | the same | `Asr.open` (`speech.md`) |

Only `.safetensors` is read, never `pytorch_model.bin`. A sharded model has `model.safetensors.index.json` beside its
shards and opens by that file (`Hub.open`, `Asr.open`; `Bert.open` does not read sharded checkpoints and
`Qwen3.open` is untested on them). A `.gguf` file of a Qwen3 opens too, with `Qwen3.open("model.gguf", backend)`.

## Look inside a checkpoint

`Hub.open` reads only the header, so a checkpoint of several GB opens at once. This program writes a tiny file, opens it
and reads a tensor back:

```agentica
import Hub from "pkg/hub/hub"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  dir = "/tmp/ml-skill-hub-#{Random.int(0, 999999999)}"
  Dir.mkdir_private(dir, 5000).unwrap!
  path = "#{dir}/model.safetensors"
  w = Tensor.floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap!
  SafetensorsWrite.write(path, {"fc.weight": w}).unwrap!

  c = Hub.open(path).unwrap!
  puts(Hub.names(c))
  puts(Hub.shape(c, "fc.weight").unwrap!)
  back = Hub.tensor(c, "fc.weight", Tensor.Backend.Reference).unwrap!
  puts(Tensor.to_list(back).unwrap!)
  Path.remove(dir, 5000).unwrap!
end
```

```output
["fc.weight"]
[2, 3]
[Values.Float(1.0), Values.Float(2.0), Values.Float(3.0), Values.Float(4.0), Values.Float(5.0), Values.Float(6.0)]
```

Float weights of 16 bits (`F16`, `BF16`) come back as `F32`. `Hub.tensors(c, backend)` reads them all as a map by name.
Half precision weights (`weights: Dtype.F16`) are in `memory-and-errors.md`.

## Generate text with Qwen3

`Qwen3.open` loads the model, `Bpe` is its tokenizer and `Generate` runs the KV-cached decode. `Generate.generate`
returns the new ids only, and stops before an id of `stop` (`<|im_end|>` is 151645, end of text is 151643).

```agentica
import Bpe from "pkg/transformers/tokenizer/bpe"
import Generate from "pkg/transformers/generate"
import Qwen3 from "pkg/transformers/qwen3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  tok = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  m = Qwen3.open("Qwen3-0.6B", backend).unwrap!
  prompt = Bpe.encode(tok, "The capital of France is").unwrap!
  g = Generate.open(m).unwrap!
  made = Generate.generate(g, prompt, 16, [151645, 151643]).unwrap!
  puts(Bpe.decode(tok, made).unwrap!)
  Generate.close(g)
  Qwen3.close_model(m)
end
```

Greedy decoding is the default. For a chat, build the prompt with `Chat.apply` (the model's template), and sample:

```agentica
import Bpe from "pkg/transformers/tokenizer/bpe"
import Chat from "pkg/transformers/chat"
import Generate from "pkg/transformers/generate"
import Qwen3 from "pkg/transformers/qwen3"
import Sample from "pkg/transformers/sample"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  tok = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  m = Qwen3.open("Qwen3-0.6B", backend).unwrap!
  messages = [Chat.Message(role: "user", content: "What is the capital of France? Answer in one sentence.")]
  text = Chat.apply(m.config.model_type, messages, true, some(false)).unwrap!
  prompt = Bpe.encode(tok, text).unwrap!
  sampling = some(Sample.Sampling(temperature: 0.7, top_k: 40, top_p: 0.95, seed: 7))
  g = Generate.open(m).unwrap!
  made = Generate.generate(g, prompt, 64, [151645, 151643], sampling).unwrap!
  puts(Bpe.decode(tok, made).unwrap!)
  Generate.close(g)
  Qwen3.close_model(m)
end
```

- `Chat.apply(model_type, messages, add_generation_prompt, enable_thinking)`: the last argument is an `Optional[Bool]`;
  `some(false)` makes Qwen3 and Qwen3.5 answer with no `<think>` block (left at `none`, Qwen3 thinks and Qwen3.5 does not,
  as their templates do). `Chat.Message` roles are `system`, `user` and `assistant`.
- The same seed draws the same ids. `top_k: 0` (the default) draws among the 256 largest logits.
- The first call of a new prompt length compiles graphs, which takes a moment; later calls compile nothing.
- The prompt and the new ids must fit `max_position_embeddings` (40960). One prompt at a time: there is no batching.
- `Qwen3.run(m, [ids])` gives the logits of whole sequences, with no cache. It is for scoring, not for generating.
- `Qwen3.open(dir, backend, Tensor.Dtype.F16)` keeps half weights (`memory-and-errors.md`).

## Embed sentences with DistilBERT

`DistilBert.embed(m, encodings, normalize)` gives one vector per sentence, `[batch, 768]`. A sentence-transformers model
whose `modules.json` lists `Normalize` (as `msmarco-distilbert-cos-v5` does) needs `normalize: true`. This program
compares two sentences by the dot product of their unit vectors:

```agentica
import DistilBert from "pkg/transformers/distilbert"
import Lift from "pkg/transformers/lift"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "pkg/transformers/tokenizer/wordpiece"

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("an embedding holds Float values")
    end
  end

  ok(out)
end

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  dir = "msmarco-distilbert-cos-v5"
  tok = WordPiece.open("#{dir}/tokenizer.json").unwrap!
  m = DistilBert.open(dir, backend).unwrap!
  sentences = ["How big is London?", "London has 9 million people."]
  encodings = sentences.map { |s: String| WordPiece.encode(tok, s).unwrap! }
  vectors = DistilBert.embed(m, encodings, true).unwrap!
  values = floats(vectors).unwrap!
  width = Tensor.shape(vectors)[1]
  similarity = 0.0
  for i in 0...width
    similarity = similarity + values[i] * values[width + i]
  end

  puts(similarity)
  Lift.close([vectors])
  DistilBert.close_model(m)
end
```

- `WordPiece.encode(tok, text)` adds `[CLS]` and `[SEP]` and cuts at the file's limit. Sentences of different
  lengths are padded to the longest, and the padding is masked.
- `Bert.run(m, encodings)` gives `Output(hidden, pooled)` for BERT; `Bert.embed` gives the mean vectors. Close an
  `Output` with `Bert.close(out)`.
- A tensor that a model call returns belongs to you: read it, then close it (`Lift.close([t])`).
- The sequence limit is the model's (512 for BERT and DistilBERT).

Pages: `https://www.agentica-lang.com/ml/hub/` and `https://www.agentica-lang.com/ml/transformers/`.
