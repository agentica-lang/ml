# train — design

Losses, optimizers, a data loader and a minimal training loop over
`tensor/`'s tracing (`tensor/DESIGN.md` §10). Depends on
`tensor/`, `hub/` (a dataset from a safetensors file) and `nn/` (the start of
a LoRA adapter, and the XOR test); `prop` is a test dependency
(`group = "test"`).

## API

```text
Loss.mse(prediction, target)                   mean of (p - t)^2
Loss.binary_cross_entropy(logits, targets)     mean of softplus(z) - t z
Loss.cross_entropy(logits, labels)             mean of -log softmax(z)[label]

Optim.Optimizer = Sgd(Sgd { lr, momentum = 0, schedule = Constant }) | Adam(Adam { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, schedule = Constant })
                | AdamW(AdamW { Adam's fields, weight_decay = 0.01, decay: List[Bool] = [] })
Optim.Schedule = Constant | WarmupLinear(warmup, total) | WarmupCosine(warmup, total)
Optim.init(optimizer, params) -> Result[State]
Optim.step(state, params, grads, scalars = []) -> Result[Step { state, params }]   scalars: tensors of step_inputs, or none
Optim.learning_rate(state) -> Float             lr times the schedule at the step count
Optim.step_inputs(state) -> List[List[Float]]   the inputs of the next step: SGD's rate, or Adam's 7 numbers (AdamW has two rows)

Data.dataset(inputs, targets) -> Result[Dataset]            [N, ...] and [N, ...]
Data.load(path, inputs, targets, backend) -> Result[Dataset]  two named tensors of a safetensors file
Data.Loader { dataset, batch_size, shuffle = true, seed = 0, drop_last = false, prefetch = 2 }
Data.size(dataset) / Data.count(loader) / Data.order(loader, epoch) / Data.batch(loader, rows, i)
Data.feed(loader, epochs) -> Result[Feed]; Data.epoch(feed, e); Data.take(feed, i); Data.stop(feed)
Data.release(batch)                                          frees its native tensors
Data.free(tensor)                                            frees one tensor's native memory now
Data.freed?(batch) -> Bool                                   a tensor of the batch was freed
Data.Batch { inputs, targets, index? }

Metrics.open(root, name, sinks = Both) -> Result[Run]   ROOT/NAME/metrics.jsonl, events.out.tfevents.*
Metrics.scalar(run, name, step, value) -> Result[Void]
Metrics.close(run) -> Result[Void]

Train.fit(loss, params, data: Data.Loader, optimizer, epochs, on_epoch?, metrics?) -> Result[Fit { params, history }]
Train.fit_batches(loss, params, batch: Fn[Int, Int, Result[Batch]], count, optimizer, epochs, on_epoch?, metrics?) -> Result[Fit]
  loss: Fn[List[Tensor], Data.Batch, Result[Tensor]]
  on_epoch: Optional[Fn[Epoch { epoch, loss }, Void]]
Train.stablehlo(loss, params, batch: Data.Batch, optimizer) -> Result[Trace.Module]

Params.split(weights: Map[String, Tensor], trainable: Fn[String, Bool]) -> Result[Split { names, trainable, frozen }]
Params.join(split, params) -> Result[Map[String, Tensor]]     every weight; the frozen ones as trace constants in a loss
Params.named(split, params) -> Result[Map[String, Tensor]]    the trainable ones alone
Params.update(weights, changed) -> Result[Map[String, Tensor]]

Lora.Config { targets: List[String], r = 8, alpha = 16.0, dropout = 0.0 }
Lora.init(weights, config, seed, backend) -> Result[Adapter { config, names, a, b }]
Lora.params(adapter) -> Result[List[Tensor]]                  A then B of each adapted weight: what fit trains
Lora.with_params(adapter, params) -> Result[Adapter]
Lora.applied(weights, adapter) -> Result[Map[String, Tensor]] every weight, the adapted ones as W + (alpha / r) B A
Lora.merged(weights, adapter) -> Result[Map[String, Tensor]]  the same sums, computed once on the backend
Lora.save(dir, adapter, embeddings = []) -> Result[Void]      PEFT's adapter_config.json and adapter_model.safetensors
Lora.open(dir, backend) -> Result[Adapter]
```

## Decisions

- **Parameters are a `List[Tensor]`.** That is what `Trace` takes, so `fit`
  hands it over unchanged. The loss function rebuilds the `nn/`
  structs from the list (`Nn.Dense(w: ps[0], b: some(ps[1]))`), which costs a
  struct per step and no copy. Flattening any struct of layers on its own
  (JAX's pytrees) waits for a model with enough layers to need it.
- **Freezing is a split outside `fit`.** `fit` already gives a gradient and
  optimizer state only to the tensors of `params`, so training part of a model
  needs no second `fit`: `Params.split` picks the trainable weights of a
  `Map[String, Tensor]` by a predicate on the name and hands `fit` their list,
  and `Params.join` inside the loss gives the model every weight back, the
  frozen ones as `Trace.constant`. The frozen tensors are then leaves the
  graph only reads: no gradient is built for them and the step has no inputs,
  moments or outputs for them. A model says its names itself
  (`Qwen3.weights`, `Qwen3.with_weights` in `transformers/`). The trainable
  part is saved as a safetensors file by `hub/`
  (`SafetensorsWrite.write(path, Params.named(split, fit.params))`) and put
  back with `Params.update`, which refuses a name the model lacks and another
  shape.
- **LoRA is a sum of weights, not a second path in the layer.** PEFT wraps
  each `Linear` and adds `(alpha / r) · B (A x)` to its output. Here
  `Lora.applied` builds `W + (alpha / r) · B A` and the model's own forward
  runs on it (`with_weights`), which is the same function of `x`. So no layer
  and no model knows about adapters, every model with `weights` and
  `with_weights` takes one, and one function is the training forward (inside
  a loss, where `W` is a `Trace.constant`) and the merge for inference:
  `Lora.merged` runs the sums of `applied` as one traced graph on the
  adapter's backend and answers the weights with the adapted ones replaced.
  A traced graph because ggml computes nothing else; so there is no
  "unmerged" run to keep apart, a model made from `merged` costs what the
  model costs. The price: the backward pass forms the `[out, in]`
  gradient of the sum before it reaches `A` and `B`, which PEFT's path does
  not, and there is no `x` to drop, so `lora_dropout` is not trained
  (`Lora.params` refuses a dropout above 0; ml#209). Whether the adapter
  moves into `Nn.dense`, as PEFT has it, is open (ml#236: the DistilBERT
  fine-tune takes 1.35 times PEFT's time per step, `benchmarks/README.md`). The scale multiplies `B`, the small matrix, before the
  product, and its `r` is the first dimension of `A`, so an adapter read from
  a file has its own. `train` imports `nn` for the start of `A`
  (`Init.uniform`); it already reached it through `hub`. The adapter is
  `F32` and so are the weights it is applied to (`init`, `applied` and
  `merged` refuse `F16` and `BF16` ones: open the model with `weights: F32`).
  `merged` adds 256 weights in a graph: a ggml graph answers at most 512
  tensors, and a model has more linear layers than that only when it is
  large, where fewer sums at once also hold less memory.
- **An adapter is kept in PEFT's files.** `Lora.save` writes
  `adapter_config.json` and `adapter_model.safetensors` as
  `PeftModel.save_pretrained` does: the weight `x.weight` has
  `base_model.model.x.lora_A.weight` `[r, in]` and `...lora_B.weight`
  `[out, r]`, `F32`, with the `format: pt` metadata. The config holds the keys
  PEFT needs and the ones that change the math; PEFT fills the rest with its
  defaults. `Lora.open` takes the adapted weights from the names of the
  tensors, not from `target_modules`, and refuses what `applied` does not
  compute, each by name: another `peft_type`; DoRA, rsLoRA, aLoRA, QALoRA and
  BD-LoRA (`use_dora`, `use_rslora`, `alora_invocation_tokens`, `use_qalora`,
  `use_bdlora`) and the variants with a `*_config` (Arrow, KaSA,
  MonteCLoRA, VeLoRA); `fan_in_fan_out`; a trained bias (`bias`, `lora_bias`);
  `modules_to_save`, `target_parameters` and `trainable_token_indices`;
  `layer_replication`; a rank or alpha pattern; `target_modules` as a regular
  expression; any tensor that is not a float `lora_A` or `lora_B` of a
  layer; and the adapter of an `Embedding` (`lora_embedding_A` and `B`, the
  transposed layout). The list is the keys of PEFT 0.21 that change the
  math; a key set to null, false or empty is not set. A `lora_dropout` is
  read and written back: it runs, and `params` refuses to train it. A
  half-float adapter is widened to `F32`. `base_model_name_or_path` is not
  kept: an adapter does not know its model, and `save` writes null.
  The other way has one gap, closed by the caller: an adapter here does not
  know which of its weights is an `Embedding` in PyTorch, so `save` takes
  the names of the model's embedding tables (`embeddings`) and refuses an
  adapter of one; without them it writes every weight as a linear layer's.
  `save` also refuses a rank that is not `config.r` and a weight that no
  target takes, since PEFT would not load them, and gives each file its name
  when it is whole (`Path.replace`, so a second save into a folder replaces
  the first). The test reads files that PEFT wrote (checked in in
  `test/fixtures/lora_peft/`, 1.7 kB: `.gitattributes` notes the
  `.safetensors` exception), and `test/lora_golden.py check` loads what `save` wrote with PEFT
  (manual).
- **AdamW's decay is a flag per parameter, set by the caller.** PyTorch has
  parameter groups; here `decay: List[Bool]` follows the list of parameters,
  and a caller builds it from the names of a `Params.split`. The rule for a
  fine-tune (and for the DistilBERT run of the benchmarks and its PyTorch
  peer) is Hugging Face `Trainer`'s: no decay on a bias and on a layer-norm
  weight, which in DistilBERT are `embeddings.LayerNorm.*`,
  `sa_layer_norm.*` and `output_layer_norm.*`. It is not
  sentence-transformers' older `no_decay` list (`bias`, `LayerNorm.bias`,
  `LayerNorm.weight`), which misses DistilBERT's norm names.
- **The batch enters the trace as inputs.** `fit` compiles the step once for
  each batch shape (`Trace.compile`: the parameters, the optimizer state, the
  step's scalars, then `inputs` and `targets` are its arguments) and runs it
  on every batch of that shape, so the loss function sees traced tensors
  everywhere and is written once, as it would be eagerly, and a step pays no
  tracing or planning. A smaller last batch is a second compiled step; the
  compiled steps are closed when `fit` returns. The native constants of
  `loss` (frozen weights, masks) are built when it is traced, so they live for
  the whole fit, once for each shape, and not for one step.
- **Optimizers are pure.** `step` takes the state, the parameters and the
  gradients and answers new ones; nothing mutates. Updates are `Tensor.*`
  calls, with scalars as `Tensor.full` of the parameter's shape (`tensor/` ops
  take exact shapes), so they run eagerly or traced. Adam's bias correction
  needs `beta^t`; Agentica has no `Float` power, so `optim.agc` computes it by
  squaring. SGD's momentum is PyTorch's form (`v = m v + g`, `p -= lr v`).
- **A step is one traced evaluation.** `fit` traces the loss, its gradients
  (`Trace.gradients`) and `Optim.step` over the parameters and the optimizer
  state, and compiles them (`Trace.compile`) once for each batch shape: on
  `Ggml` one graph and on `Xla` one program, a last partial batch being a
  second one. The step count
  stays on the host, so the learning rate and Adam's bias correction are
  scalar inputs, set for every step, and a schedule is not a new graph. A ggml graph has at most 512
  outputs, the loss and the new tensors, so a step on `Ggml` takes at most 170
  parameters with Adam, 255 with momentum SGD and 511 with plain SGD.
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
- **A learning rate is a schedule of the step count.** `lr` is the peak and a
  `Schedule` gives the factor at step `t` (from 0): `WarmupLinear(w, n)` rises
  from 0 to 1 over `w` steps and falls straight to 0 at `n`; `WarmupCosine` falls
  along a half cosine. They are Hugging Face's `Trainer` schedules, the ones a
  sentence-transformers fine-tune uses, up to `n`; after it the factor stays 0,
  where Hugging Face's cosine rises again. `fit` starts every run at step 0, so
  a resumed run starts its warmup again. One
  way: the schedule is a field of the optimizer, not a hook of `fit`, and
  `Optim.learning_rate(state)` is what `Metrics` logs. There is no one-cycle
  schedule.
- **A step is one program of every step.** `Train.stablehlo` traces the same
  step as `fit` (`traced_step`) and writes it as StableHLO
  (`tensor/DESIGN.md` §12). Adam's bias corrections depend on the step count, so
  they are inputs after the moments and not constants of one step: the
  `Optim.step_inputs` (SGD's rate as a rank-0 input, Adam's 7 numbers in one
  rank-1 input), passed as the `scalars` of `Optim.step`, as `fit` passes them;
  so a learning rate that changes at every step (a schedule, which starts at 0)
  is not a new graph. An `adam_step` node has no StableHLO: `Trace.stablehlo`
  writes it as the composition of ops that `Tensor.adam_step` is on every other
  backend, one formula. The goldens in `test/fixtures/stablehlo/`
  are three chained runs on IREE, which `fit` reproduces at steps 1 and 3.
- **Adam's moments are updated where they lie.** A step on ggml runs Adam's
  update as `Tensor.adam_step`, and in the compiled step of `fit` its first and
  second moments are inputs that nothing else reads, so the lowering updates
  them in place (on XLA the outputs are aliased to the moments): the step's results for them are the very tensors it was given
  (`Trace.donated`), and `run` does not free the old moment that is the new one
  (`Stepped.kept`). The parameters are never written (the loss reads them), so
  the caller's weights need no copy on the first step; the moments are the
  run's own zeros. The first moment is not checked for finite values where it is
  written (a non-finite one reaches the new parameter through the update, which
  is checked); the second is, since an infinite one leaves the parameter finite
  and frozen.
- **One native arena per step.** `fit` runs each step in `Native.arena`, so
  the intermediates of the trace's evaluation die at the step's end on
  libtorch and only the new state, parameters and loss leave it
  (`tensor/DESIGN.md` §8). The previous parameters and state were made
  before the arena, so it does not free them: `fit` closes them itself once
  the step has replaced them, and the last moments when it returns. The GC
  does not count native memory, so waiting for it let a run of a large model
  hold several copies of its weights. The caller's `params` are never
  closed, and neither are the trained ones of the result.
  `tools/fit_memory.agc STEPS` under `/usr/bin/time -v` shows it: the peak
  of a fit of one 64 MB parameter does not grow with the step count.
- **`fit_batches` takes a function, not a second kind of loader.** A loader
  cuts two tensors of one shape; sentences padded to the longest of each
  batch have a shape per batch, which no tensor holds. `fit_batches(loss,
  params, batch, count, ...)` asks `batch(epoch, i)` for each batch and is
  otherwise `fit`, which runs on the same loop with the loader behind such a
  function. The step is compiled once per batch shape either way. A batch
  is two tensors; what else a loss needs it derives (a padding mask is `ids
  != 0`).
- **Metrics live in `train/`.** Only `fit` reports through them, and the logger
  needs no native code and no `tensor/` type, so a package of its own would be
  a `agentica.toml`, a lock, a version pin and a CI job for one consumer. It is
  three small modules (`metrics.agc`, `events.agc`, `crc32c.agc`), and a
  package can take them out when a second consumer appears.
- **One call, every sink, scalars only.** `Metrics.scalar(run, name, step,
  value)` writes to each sink of the run and flushes, so TensorBoard and
  `tail -f` see a value as it happens. Histograms, text and images are not
  there; the series name is free text (`train/loss`, `/` groups in TensorBoard).
  A value that is not finite or beyond float32 (the event file holds a float32)
  and a negative step are errors: JSON cannot hold `NaN`, so the two sinks
  agree on what a run holds. `fit` then stops, naming the epoch and the batch,
  on a loss that is `NaN`, which a diverged run is better for.
- **JSONL line.** `{"run", "name", "step", "value", "time"}`, `time` the Unix
  time in seconds as a float. A run folder that exists is appended to.
- **TensorBoard file, by hand.** The file holds the records of the TFRecord
  framing: a `u64` length, the masked CRC-32C of those 8 bytes, the data, and the
  masked CRC-32C of the data, little-endian (the mask is the CRC rotated right
  by 15 bits, plus `0xA282EAD8`). The first record is an `Event` with
  `file_version = "brain.Event:2"`, the others an `Event` with `wall_time`
  (double), `step` and a `Summary` with one `Value { tag, simple_value }`
  (a float32). The protobuf is a dozen lines (a varint, a key, a length) and
  the checksum a table of 256 entries; the file is named
  `events.out.tfevents.<seconds>.agentica.<pid>.<microseconds>`, which is what
  TensorBoard looks for. No protobuf or TensorFlow dependency, and no C: neither
  `std` nor `packages` has a CRC-32C (`Zlib.crc32` is the other polynomial).
  `test/tfevents_peer.py` reads a file with the standard library only and
  recomputes every CRC, so the test does not check the encoder with itself.
- **`fit` reports per step.** With `metrics`, each step writes `train/loss`,
  `train/lr` and `train/step_ms` (from a monotonic clock) at the optimizer's step count (from 1).
  `fit` does not open or close the run.
- **Reporting is a callback and a history.** `fit` answers the mean loss of
  every epoch and calls `on_epoch` after each; a caller logs from there
  (`std/logger` or `puts`). `fit` itself prints nothing.
- **A batch is one op per tensor.** The epoch's order is a Fisher-Yates
  permutation drawn in Agentica with the Park-Miller generator of `nn/init.agc`,
  from the seed and the epoch, so a run repeats and epochs differ (`std`
  `Random` has no seedable generator of its own). Seeds below `2^20` and
  epochs below 2048 give distinct starting states; larger ones wrap, `s` and
  `-s` are one seed, and a dataset holds fewer than `2^31 - 2` rows. The
  shuffle runs in VM steps, `O(N)` per epoch, and counts against
  `--max-steps`. A batch is one `from_list` of its `I32` row ids (in order
  when not shuffled) and one `gather` per tensor, both in one
  `Trace.eval_many`, so a backend that computes only whole graphs (`Ggml`)
  makes batches too, with one compiled graph per batch shape. On `Ggml` a
  dataset holds at most 2^24 rows, the row ids F32 counts exactly. The last
  partial batch is kept unless `drop_last`.
- **Prefetch is a task and a bounded channel.** With `prefetch = K > 0`,
  `Data.feed` starts a task that builds the batches of every epoch in order and
  sends them through a channel of capacity `K`, so it stays at most `K` batches
  ahead and builds the next while `fit` steps. A failed batch goes through the
  channel as an error in its place; a crash of the task reaches the reader
  through a `Task.monitor` notice into the same channel. A feed is a
  sequential stream: under prefetch `take` gives the next batch and does not
  read its index. On ggml and XLA `fit` does not use it when the dataset lives on the
  parameters' backend: a batch is its row ids (made in the order `Data.order`
  gives, so the same batches), the step takes the dataset's tensors as
  constants and gathers in its own graph, which saves a graph run and the
  readback and re-upload of the batch (on XLA, a compile per batch shape);
  `Data.feed` and `fit_batches` keep the prefetch.
- **The prefetch task lives until `stop`.** A native tensor belongs to the
  task that made it and is freed when that task ends, and a tensor made
  directly from a freed one can no longer be used (the VM keeps one level of
  parents open). Torch tensors cross tasks, so the task builds them, but it
  then waits for `Data.stop`, which `fit` calls on every path: it closes the
  channel, cancels the task and awaits it. `fit` frees each batch after its
  step (`Data.release`: inputs, targets and the gathered row ids); nothing it
  keeps is made directly from a batch. A batch whose tensors were freed
  before it was taken (the task ended without `stop`) is an error of `take`.

## Cost

`fit` traces the step once for each batch shape; on the reference backend and
libtorch the compiled step is the traced graph, evaluated node by node on
every run (`Trace.compile`). An epoch of the XOR test (a 2-8-1 relu MLP, four
rows) costs about 100,000 VM steps on the reference, most of it the
evaluation of the forward, gradient and update graphs, so the test trains for
20 epochs. On libtorch an epoch costs about 70,000.

Prefetch hides the cost of building a batch, not more.

## The repository is a workspace

`train/` depends on `hub/`, `nn/` and `tensor/`, and `hub/` on `nn/`, which
depends on `tensor/`. Only a root project may
write a `path` dependency (`PKG-ROOT-ONLY`), so the repository root has an
`agentica.toml` with `[workspace]`, and members name each other by the
repository and a `subdir`: a member reads another from its folder, with no
fetch, and the `tag` it writes is not used inside the workspace. A project
outside the workspace fetches a member at the tag of the release written, so
the members are released together and a release PR bumps those tags.
