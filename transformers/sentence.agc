## The pieces a sentence-embedding model adds to an encoder (sentence-transformers): the mean of the token vectors,
## the L2 norm of an embedding, and the cosine loss of scored pairs. Each is `Tensor.*` calls, so it runs eagerly and
## inside a trace, where `Trace` takes its gradient.

import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

## The least count of tokens `mean_pooled` divides by, as sentence-transformers clamps it: a row with no token is
## zeros, not NaN.
MIN_COUNT = 0.000000001

## `hidden` `[batch, seq, width]` averaged over the tokens `mask` keeps (`[batch, seq]`, 1.0 for a token and 0.0 for
## padding), as `[batch, width]`. A row whose mask is all zeros gives zeros. The tokens are the last axis while they
## are summed, as ggml lowers only sums over a leading or trailing block of axes.
def mean_pooled(hidden: Tensor.Tensor, mask: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(hidden)
  if shape.size != 3 or Tensor.shape(mask) != [shape[0], shape[1]]
    return error(
      "mean_pooled needs [batch, seq, width] and a [batch, seq] mask, got #{shape} and #{Tensor.shape(mask)}",
    )
  end

  across = [shape[0], shape[2], shape[1]]
  row = try Tensor.reshape(mask, [shape[0], 1, shape[1]])
  masked = try Tensor.mul(try Tensor.permute(hidden, [0, 2, 1]), try Tensor.broadcast_to(row, across))
  total = try Tensor.sum(masked, some([2]))
  # `max(count, MIN_COUNT)` as `relu(count - MIN_COUNT) + MIN_COUNT`: ggml lowers no `clamp`.
  floor = try Tensor.full([shape[0], 1], Tensor.Values.Float(MIN_COUNT), Tensor.dtype(mask), Tensor.backend(mask))
  count = try Tensor.add(try Tensor.relu(try Tensor.sub(try Tensor.sum(mask, some([1]), true), floor)), floor)
  Tensor.div(total, try Tensor.broadcast_to(count, [shape[0], shape[2]]))
end

## Each row of `x` (`[batch, width]`) over its length, so that it has length 1: the `Normalize` module of a
## sentence-transformers model. `Nn.COSINE_EPS` is added to the squared length under the root, so a row of zeros stays
## zeros.
def normalized(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("normalized needs [batch, width], got #{shape}") if shape.size != 2
  squares = try Tensor.sum(try Tensor.mul(x, x), some([1]), true)
  floor = try Tensor.full([shape[0], 1], Tensor.Values.Float(Nn.COSINE_EPS), Tensor.dtype(x), Tensor.backend(x))
  Tensor.div(x, try Tensor.broadcast_to(try Tensor.sqrt(try Tensor.add(squares, floor)), shape))
end

## The mean of `(Nn.cosine_similarity(a, b) - scores)^2`, with `scores` of `[batch]`: the loss of a model trained on
## scored pairs of sentences (sentence-transformers' `CosineSimilarityLoss`). A rank-0 tensor.
def cosine_similarity_loss(a: Tensor.Tensor, b: Tensor.Tensor, scores: Tensor.Tensor) -> Result[Tensor.Tensor]
  cos = try Nn.cosine_similarity(a, b)
  if Tensor.shape(scores) != Tensor.shape(cos)
    return error("cosine_similarity_loss needs #{Tensor.shape(cos)} scores, got #{Tensor.shape(scores)}")
  end

  gap = try Tensor.sub(cos, scores)
  Tensor.mean(try Tensor.mul(gap, gap))
end
