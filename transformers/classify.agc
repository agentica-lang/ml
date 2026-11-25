## What the image classifiers (`ResNet`, `ViT`) share: the best classes of a batch of logits.

import Tensor from "pkg/tensor/tensor"

## A class of an image: its index, its name and its logit.
struct Prediction
  id: Int
  label: String
  logit: Float
end

## The `k` best classes of each row of `logits` (`[batch, classes]`), best first, named by `id2label` (the id as text is
## the name of a class it does not list).
def top(
  id2label: Map[String, String],
  logits: Tensor.Tensor,
  k: Int,
) -> Result[List[List[Prediction]]]
  shape = Tensor.shape(logits)
  return error("top needs logits of [batch, classes], got #{shape}") if shape.size != 2 or k < 1 or k > shape[1]
  values: List[Float] = []
  for v in try Tensor.to_list(logits)
    case v
      Tensor.Values.Float(f) -> values = values.push(f)
      _ -> return error("the logits are floats")
    end
  end

  out: List[List[Prediction]] = []
  for row in 0...shape[0]
    ids = (0...shape[1]).to_a.sort_by { |i| -values[row * shape[1] + i] }.take(k)
    out = out.push(
      ids.map { |i|
        Prediction(id: i, label: id2label.get("#{i}").unwrap_or("#{i}"), logit: values[row * shape[1] + i])
      },
    )
  end

  ok(out)
end
