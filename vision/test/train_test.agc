## The data path: a folder per class of images on disk becomes a dataset for the loader of `train/`, and a small MLP
## learns it. The images are made here: 4 by 2 greys, bright on the left in one class and on the right in the other,
## with a different noise in each. `train` and `nn` are test dependencies.

import Data from "pkg/train/data"
import Image from "pkg/image/image"
import Loss from "pkg/train/loss"
import Nn from "pkg/nn/nn"
import Optim from "pkg/train/optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"
import Vision from "../vision"

PER_CLASS = 4
EPOCHS = 25

private

## The pixels of the image `i` of a class: 200 on the bright half and 30 on the other, plus a noise below 24 that
## changes with the image and the place.
def grey(bright_left: Bool, i: Int) -> List[Int]
  out: List[Int] = []
  for y in 0...2
    for x in 0...4
      base = 30
      if (x < 2) == bright_left
        base = 200
      end

      out = out.push(base + (i * 7 + x * 5 + y * 3) % 24)
    end
  end

  out
end

def make_dataset(root: String) -> Result[Void]
  for name in ["left", "right"]
    try Dir.mkdir("#{root}/#{name}", 5000)
    for i in 0...PER_CLASS
      pixels = grey(name == "left", i)
      try Image.save(try S.u8(pixels, 4, 2, 1), "#{root}/#{name}/#{i}.png")
    end
  end

  ok()
end

def model(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  up = Nn.Dense(w: ps[0], b: some(ps[1]))
  down = Nn.Dense(w: ps[2], b: some(ps[3]))
  Nn.mlp(Nn.Mlp(up: up, down: down, activation: Nn.Activation.Relu), x)
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.cross_entropy(try model(ps, b.inputs), b.targets)
end

## The images as rows of 24 values, which is what the MLP takes.
def flat(inputs: Tensor.Tensor) -> Result[Tensor.Tensor]
  n = Tensor.shape(inputs)[0]
  Trace.eval([inputs]) do |xs| Tensor.reshape(xs[0], [n, 24]) end
end

## The class that the logits' gap says: 1 when its logit is above the other.
def class_of(margin: Float) -> Int
  return 1 if margin > 0.0
  0
end

def train_on(root: String, backend: Tensor.Backend) -> Result[Void]
  f = try Vision.folder(root)
  s = try Vision.samples(f, (0...f.paths.size).to_a, [], 0, backend)
  inputs = try flat(s.inputs)
  loader = Data.Loader(dataset: try Data.dataset(inputs, s.targets), batch_size: f.paths.size, shuffle: false)
  up = try Nn.init_dense(24, 4, 1, backend)
  down = try Nn.init_dense(4, 2, 2, backend)
  params = [up.w, up.b.unwrap!, down.w, down.b.unwrap!]
  fit = try Train.fit(&loss, params, loader, Optim.Optimizer.Adam(Optim.Adam(lr: 0.05)), EPOCHS)
  try Test.assert(
    fit.history[EPOCHS - 1] < 0.1 * fit.history[0],
    "the loss went #{fit.history[0]} to #{fit.history[EPOCHS - 1]}",
  )

  args = fit.params + [inputs]
  logits = try Trace.eval(args, ->(xs: List[Tensor.Tensor]) { model(xs.take(4), xs[4]) })
  n = f.paths.size
  margin = try Trace.eval([logits]) do |xs|
    Tensor.sub(try Tensor.slice(xs[0], [0, 1], [n, 2]), try Tensor.slice(xs[0], [0, 0], [n, 1]))
  end

  picked = (try S.floats_of(margin)).map { |m| class_of(m) }
  Test.eq(picked, try S.ints_of(s.targets))
end

def test_a_folder_of_images_trains_an_mlp() -> Result[Void]
  S.with_dir do |root|
    try make_dataset(root)
    for backend in try S.backends()
      try train_on(root, backend)
    end

    ok()
  end
end

## The images of the rows of batch `i`, two at a time, decoded where this runs and made a batch of rows of 24 values.
def decoded_batch(f: Vision.Folder, backend: Tensor.Backend, i: Int) -> Result[Data.Batch]
  s = try Vision.samples(f, [2 * i, 2 * i + 1], [], 0, backend)
  ok(Data.Batch(inputs: try flat(s.inputs), targets: s.targets))
end

def fit_decoded(f: Vision.Folder, backend: Tensor.Backend, prefetch: Int) -> Result[Train.Fit]
  up = try Nn.init_dense(24, 4, 1, backend)
  down = try Nn.init_dense(4, 2, 2, backend)
  params = [up.w, up.b.unwrap!, down.w, down.b.unwrap!]
  decode = ->(_e: Int, i: Int) { decoded_batch(f, backend, i) }
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.05))
  Train.fit_batches(&loss, params, decode, f.paths.size / 2, adam, 5, prefetch: prefetch)
end

## Decoding the files on the prefetch task of `fit_batches` trains the same as decoding them in the step.
def test_images_decoded_on_the_prefetch_task_train_the_same() -> Result[Void]
  S.with_dir do |root|
    try make_dataset(root)
    f = try Vision.folder(root)
    for backend in try S.backends()
      want = try fit_decoded(f, backend, 0)
      got = try fit_decoded(f, backend, 2)
      try Test.eq(got.history, want.history)
      try Test.assert(got.history[4] < got.history[0], "the loss did not fall on #{backend}")
    end

    ok()
  end
end
