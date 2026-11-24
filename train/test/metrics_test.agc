## Metrics (`metrics.agc`): the CRC-32C, the JSONL sink read back, the TensorBoard event file read by a peer that shares
## no code with `events.agc` (`tfevents_peer.py`: it recomputes every CRC and decodes the protobuf by hand), and `fit`
## reporting through a run. Run from `train/`.

import Crc32c from "../crc32c"
import Data from "../data"
import Loss from "../loss"
import Metrics from "../metrics"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

## A step beyond 2^31, which a varint writes in five bytes.
STEP = 3000000000

private

def test_the_crc32c_of_the_standard_vectors() -> Result[Void]
  try Test.eq(Crc32c.checksum("123456789".bytes), 0xE3069283)
  try Test.eq(Crc32c.checksum(try Bytes.from_list((0...32).to_a.map { |_| 0 })), 0x8A9136AA)
  try Test.eq(Crc32c.checksum(try Bytes.from_list((0...32).to_a.map { |_| 255 })), 0x62A8AB43)
  Test.eq(Crc32c.checksum("".bytes), 0)
end

def test_the_jsonl_sink_round_trips() -> Result[Void]
  with_dir do |root|
    run = try Metrics.open(root, "a", Metrics.Sinks.Jsonl)
    try Metrics.scalar(run, "loss", 1, 0.5)
    try Metrics.scalar(run, "lr", STEP, 0.00001)
    try Metrics.close(run)
    lines = try jsonl_lines("#{root}/a/metrics.jsonl")
    first = try JSON.parse(lines[0])
    try Test.eq(try JSON.string(try JSON.get(first, "run")), "a")
    try Test.eq(try JSON.string(try JSON.get(first, "name")), "loss")
    try Test.eq(try JSON.int(try JSON.get(first, "step")), 1)
    try Test.eq(try JSON.float(try JSON.get(first, "value")), 0.5)
    second = try JSON.parse(lines[1])
    try Test.eq(try JSON.int(try JSON.get(second, "step")), STEP)
    try Test.eq(try JSON.float(try JSON.get(second, "value")), 0.00001)
    try Test.assert(
      try JSON.float(try JSON.get(second, "time")) >= try JSON.float(try JSON.get(first, "time")),
      "time",
    )

    for name in try Dir.list("#{root}/a")
      try Test.assert(!name.start_with?("events.out.tfevents."), "an event file without the sink: #{name}")
    end

    ok()
  end
end

def test_a_peer_reads_the_event_file() -> Result[Void]
  return ok() if !try python_ready()
  with_dir do |root|
    before = Time.now().to_us.to_f / 1000000.0
    run = try Metrics.open(root, "b", Metrics.Sinks.TensorBoard)
    try Metrics.scalar(run, "loss", 0, 0.1)
    try Metrics.scalar(run, "train/lr", STEP, -2.5)
    try Metrics.scalar(run, "loss", 7, 1000000.0)
    try Metrics.close(run)
    after = Time.now().to_us.to_f / 1000000.0
    records = try JSON.list(try JSON.parse(try peer(try the_event_file("#{root}/b"))))
    try Test.eq(records.size, 4)
    try Test.eq(try JSON.string(try JSON.get(records[0], "file_version")), "brain.Event:2")
    tags = ["loss", "train/lr", "loss"]
    steps = [0, STEP, 7]
    # What a float32 holds of 0.1, -2.5 and 1000000.0.
    values = [0.10000000149011612, -2.5, 1000000.0]
    for i in 0...3
      r = records[i + 1]
      try Test.eq(try JSON.string(try JSON.get(r, "tag")), tags[i])
      try Test.eq(try JSON.int(try JSON.get(r, "step")), steps[i])
      try Test.eq(try JSON.float(try JSON.get(r, "value")), values[i])
      time = try JSON.float(try JSON.get(r, "wall_time"))
      try Test.assert(time >= before - 0.001 and time <= after + 0.001, "wall_time #{time}")
    end

    ok()
  end
end

## A bad name, step or value is an error, and a second `open` of a run adds to its files.
def test_the_errors_and_an_open_run_that_exists() -> Result[Void]
  with_dir do |root|
    for name in ["", ".", "..", "a/b", "a\\b"]
      try Test.assert(!Metrics.open(root, name).ok?, "the run name #{name} is refused")
    end

    run = try Metrics.open(root, "c")
    nan = Float.from_bits(0x7FF8000000000000)
    infinity = Float.from_bits(0x7FF0000000000000)
    for value in [nan, infinity, -infinity, 1.0e39]
      try Test.assert(!Metrics.scalar(run, "loss", 1, value).ok?, "the value #{value} is refused")
    end

    try Test.assert(!Metrics.scalar(run, "loss", -1, 1.0).ok?, "a negative step is refused")
    try Metrics.scalar(run, "loss", 1, 3.4028234663852886e38)
    try Metrics.close(run)
    again = try Metrics.open(root, "c")
    try Metrics.scalar(again, "loss", 2, 2.0)
    try Metrics.close(again)
    try Test.eq((try jsonl_lines("#{root}/c/metrics.jsonl")).size, 2)
    files = (try Dir.list("#{root}/c")).select { |n| n.start_with?("events.out.tfevents.") }
    Test.eq(files.size, 2)
  end
end

## A loss `fit` cannot report stops it with the epoch and the batch.
def test_fit_stops_on_a_value_the_run_refuses() -> Result[Void]
  with_dir do |root|
    run = try Metrics.open(root, "big")
    big = try Tensor.full([], Tensor.Values.Float(1.0e41), Tensor.Dtype.F32, S.REF)
    inputs = try S.tensor([0.0, 1.0], [2, 1], Tensor.Dtype.F32)
    loader = Data.Loader(dataset: try Data.dataset(inputs, inputs), batch_size: 2, shuffle: false)
    params = [try S.tensor([1.0], [1, 1], Tensor.Dtype.F32), try S.tensor([0.5], [1], Tensor.Dtype.F32)]
    sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1))
    loss = ->(ps: List[Tensor.Tensor], b: Data.Batch) { scaled(ps, b, big) }
    r = Train.fit(loss, params, loader, sgd, 1, none, some(run))
    try Metrics.close(run)
    case r
      ok(_) -> error("fit went on with a loss beyond float32")
      error(e) -> Test.assert(e.message.start_with?("epoch 1, batch 0: "), e.message)
    end
  end
end

def test_fit_reports_every_step() -> Result[Void]
  return ok() if !try python_ready()
  with_dir do |root|
    run = try Metrics.open(root, "fit")
    inputs = try S.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2], Tensor.Dtype.F32)
    targets = try S.tensor([0.0, 1.0, 1.0, 0.0], [4, 1], Tensor.Dtype.F32)
    params = [try S.tensor([0.5, -0.5], [2, 1], Tensor.Dtype.F32), try S.tensor([0.1], [1], Tensor.Dtype.F32)]
    loader = Data.Loader(dataset: try Data.dataset(inputs, targets), batch_size: 2, shuffle: false)
    adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.05))
    fit = try Train.fit(&regression, params, loader, adam, 2, none, some(run))
    try Metrics.close(run)
    lines = try jsonl_lines("#{root}/fit/metrics.jsonl")
    names: List[String] = []
    for l in lines
      names = names.push(try JSON.string(try JSON.get(try JSON.parse(l), "name")))
    end

    try Test.eq(names, (0...4).to_a.flat_map { |_| ["train/loss", "train/lr", "train/step_ms"] })
    last = try JSON.parse(lines[10])
    try Test.eq(try JSON.int(try JSON.get(last, "step")), 4)
    try Test.eq(try JSON.float(try JSON.get(last, "value")), 0.05)
    first_loss = try JSON.float(try JSON.get(try JSON.parse(lines[0]), "value"))
    try Test.assert(first_loss > 0.0 and fit.history.size == 2, "the loss of the first step is #{first_loss}")
    events = try JSON.list(try JSON.parse(try peer(try the_event_file("#{root}/fit"))))
    Test.eq(events.size, 13)
  end
end

def regression(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.mse(try Tensor.linear(b.inputs, ps[0], some(ps[1])), b.targets)
end

## The squared error times `big`.
def scaled(ps: List[Tensor.Tensor], b: Data.Batch, big: Tensor.Tensor) -> Result[Tensor.Tensor]
  Tensor.mul(try regression(ps, b), Trace.constant(big))
end

def jsonl_lines(path: String) -> Result[List[String]]
  text = try File.read_text(path)
  ok(text.split("\n").select { |l| !l.empty? })
end

def the_event_file(dir: String) -> Result[String]
  for name in try Dir.list(dir)
    return Path.join([dir, name]) if name.start_with?("events.out.tfevents.")
  end

  error("no event file in #{dir}")
end

# Whether the peer can run. A machine without python3 skips the peer's tests; CI sets `CI` and must have it.
def python_ready() -> Result[Bool]
  p = Process.spawn_captured("python3", ["-I", "-c", "pass"])
  if p.ok?
    try Process.wait(p.unwrap!, 60000)
    return ok(true)
  end

  flag = System.env("CI")
  return error("python3 is required in CI: #{p.message}") if flag.ok? and try flag != ""
  ok(false)
end

def peer(path: String) -> Result[String]
  p = try Process.spawn_captured("python3", ["-I", "test/tfevents_peer.py", path])
  out = try Process.output(p, 60000)
  code = try Process.wait(p, 60000)
  return error("tfevents_peer.py exited with #{code}: #{out.stderr}") if code != 0
  ok(out.stdout)
end

def with_dir(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-metrics-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end
