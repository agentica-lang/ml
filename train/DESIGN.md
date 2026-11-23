# train — design

Losses, optimizers and a minimal training loop over `tensor/`'s
`Trace.value_and_grad` (`tensor/DESIGN.md` §10). Depends on `tensor/`; `nn/`
is a test dependency (`group = "test"`), used by the XOR test.

## API

```text
Loss.mse(prediction, target)                   mean of (p - t)^2
Loss.binary_cross_entropy(logits, targets)     mean of softplus(z) - t z
Loss.cross_entropy(logits, labels)             mean of -log softmax(z)[label]

Optim.Optimizer = Sgd(Sgd { lr, momentum = 0 }) | Adam(Adam { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 })
Optim.init(optimizer, params) -> Result[State]
Optim.step(state, params, grads) -> Result[Step { state, params }]

Train.Batch { inputs, targets }
Train.fit(loss, params, batches, optimizer, epochs, on_epoch?) -> Result[Fit { params, history }]
  loss: Fn[List[Tensor], Batch, Result[Tensor]]
  on_epoch: Optional[Fn[Epoch { epoch, loss }, Void]]
```

## Decisions

- **Parameters are a `List[Tensor]`.** That is what `Trace.value_and_grad`
  takes, so `fit` hands it over unchanged. The loss function rebuilds the `nn/`
  structs from the list (`Nn.Dense(w: ps[0], b: some(ps[1]))`), which costs a
  struct per step and no copy. Flattening any struct of layers on its own
  (JAX's pytrees) waits for a model with enough layers to need it.
- **The batch enters the trace as constants.** `fit` wraps `inputs` and
  `targets` in `Trace.constant`, so the loss function sees traced tensors
  everywhere and is written once, as it would be eagerly.
- **Optimizers are pure.** `step` takes the state, the parameters and the
  gradients and answers new ones; nothing mutates. Updates run eagerly on the
  parameters' backend, with scalars as `Tensor.full` of the parameter's shape
  (`tensor/` ops take exact shapes). Adam's bias correction needs `beta^t`;
  Agentica has no `Float` power, so `optim.agc` computes it by squaring.
  SGD's momentum is PyTorch's form (`v = m v + g`, `p -= lr v`).
- **Binary cross-entropy is built from ops that have backward rules.** It is
  `relu(z) + log(1 + exp(-|z|)) - t z`, stable for any `z`, with
  `|z| = 2 relu(z) - z`: `relu`'s derivative at `0` is `0`, and with this form
  the gradient at `z = 0` is still `sigmoid(0) - t` (a test pins it).
- **Softmax cross-entropy takes integer labels.** `logits` are
  `[batch, classes]` and `labels` are `[batch]` class ids. The log-softmax
  subtracts the row maximum through `Tensor.stop_grad`, so it is stable and
  its gradient is `(softmax - onehot) / batch`. The one-hot mask is built in
  the graph (`arange`, `eq`, `where`). A label outside `0...classes` is an
  error in an eager call. Traced labels, those `fit` lifts with
  `Trace.constant` included, are not checked, and such a label adds `0`.
- **Embedding gradients cost `[rows, k]`.** The backward of `gather` (an
  embedding lookup) is a one-hot `matmul`: four `[rows, k]` intermediates,
  about 800 MB each for a 50k vocabulary over 4096 tokens at `F32`, plus
  `O(rows k width)`. A scatter-add primitive would fix it (ml#25).
- **One native arena per step.** `fit` runs each step in `Native.arena`, so
  the intermediates of the trace's evaluation die at the step's end on
  libtorch and only the new state, parameters and loss leave it
  (`tensor/DESIGN.md` §8). The previous parameters and state were made
  before the arena, so it does not free them: they wait for the GC.
- **Reporting is a callback and a history.** `fit` answers the mean loss of
  every epoch and calls `on_epoch` after each; a caller logs from there
  (`std/logger` or `puts`). `fit` itself prints nothing.
- **Left out until a client needs them:** shuffling, evaluation sets, early
  stopping, learning-rate schedules, weight decay, gradient clipping, and
  resuming from a saved `State`.

## Cost

`fit` traces the loss again on every step (`tensor/DESIGN.md` §10.1). On the
reference backend an epoch of the XOR test (a 2-8-1 relu MLP, four rows)
costs about 75,000 VM steps, most of it the evaluation of the forward and
gradient graphs; a test file's step budget (1M steps per test, plus 1M) has to
hold the reference run and the libtorch run, so the test trains for 20. On libtorch an epoch costs about 45,000.

## The repository is a workspace

`train/` depends on `nn/`, which depends on `tensor/`. Only a root project may
write a `path` dependency (`PKG-ROOT-ONLY`), so the repository root has an
`agentica.toml` with `[workspace]`, and members name each other by the
repository and a `subdir`: a member reads another from its folder, with no
fetch, and the `ref` it writes is not used inside the workspace. A
project outside the workspace fetches a member at the `ref` written, so:

- `train/` cannot be used from outside until its `ref` to `tensor/` is
  bumped to a commit of `main` that has `train/`'s needs (after this PR
  merges, the `ref` still names the commit before it).
- A PR that changes several members leaves outside users on the old pin
  until a follow-up bumps the `ref` (or a release tag replaces it).
