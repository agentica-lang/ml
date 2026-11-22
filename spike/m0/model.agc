## The demo function, f(w) = sum(relu(matmul(x, w))), and its hand-written
## backward.
##
## There is no autograd here on purpose. `tensor/DESIGN.md` §10 asks for a
## traced graph of primitives and a `grad` computed in Agentica over it; the
## spike stands in for that graph with the calls written out in order, so what
## it proves and what it does not is visible in this file (see `README.md`).

import TensorRef from "tensor_ref"

## One backward pass: the loss at `w` and dL/dw.
struct LossAndGrad
  loss: Float
  grad: TensorRef.Tensor
end

## The loss and its gradient with respect to `w`, from one forward pass.
##
##   y  = matmul(x, w)                       [batch, out]
##   r  = relu(y)
##   L  = sum(r)                             the loss, a rank-0 tensor
##   dL/dr = 1 (sum)                         so the upstream gradient is ones
##   dL/dy = dL/dr * 1[y > 0]                relu's rule, elementwise
##   dL/dw = transpose(x) . dL/dy            matmul's rule, for `a`
##
## dL/dx is not computed: nothing in the spike asks for it, and a real `grad`
## must give both (see the README's gap list).
def loss_and_grad(x: TensorRef.Tensor, w: TensorRef.Tensor) -> Result[LossAndGrad]
  y = try TensorRef.matmul(x, w)
  r = TensorRef.relu(y)
  loss = try TensorRef.scalar(TensorRef.sum(r))
  upstream = try TensorRef.ones(r.shape)
  dy = try TensorRef.relu_backward(upstream, y)
  xt = try TensorRef.transpose(x)
  grad = try TensorRef.matmul(xt, dy)
  ok(LossAndGrad(loss: loss, grad: grad))
end

## One plain SGD step, w <- w + (-lr) * dL/dw, with no momentum and no state:
## the parameters are values, so a step is a new `Tensor`.
def sgd_step(w: TensorRef.Tensor, grad: TensorRef.Tensor, lr: Float) -> Result[TensorRef.Tensor]
  step = try TensorRef.from_list(List.repeat(-lr, grad.data.size), grad.shape)
  scaled = try TensorRef.mul(grad, step)
  TensorRef.add(w, scaled)
end
