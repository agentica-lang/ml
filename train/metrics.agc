## Training metrics: scalars written, as they happen, to a run folder (`train/DESIGN.md`, "Metrics").
##
## `open` makes `ROOT/NAME/` with one file per sink: `metrics.jsonl` (a JSON object per line) and a TensorBoard event
## file `events.out.tfevents.*` (`tensorboard --logdir ROOT` shows every run under it). `scalar` writes to every sink
## the run has and flushes, so a viewer sees the value at once.

import Events from "events"

## The largest float32; a value beyond it would round to infinity in the event file.
FLOAT32_MAX = 3.4028234663852886e38

## One line of `metrics.jsonl`: `time` is the Unix time in seconds.
struct Line
  run: String
  name: String
  step: Int
  value: Float
  time: Float
end

## The files of a run: `metrics.jsonl`, the TensorBoard event file, or both.
enum Sinks
  Both
  Jsonl
  TensorBoard
end

## An open run. Close it with `close`.
struct Run
  name: String
  jsonl: Optional[File.Handle]
  events: Optional[File.Handle]
end

## The run `name` under `root`, with the files of `sinks`. The name is one folder name. The folder is made if it is
## missing, and a run that exists gets its `metrics.jsonl` appended to and a new event file.
def open(root: String, name: String, sinks: Sinks = Sinks.Both) -> Result[Run]
  return error("a run name is one folder name, got #{name}") if !folder_name?(name)
  dir = try Path.join([root, name])
  try Dir.mkdir(dir)
  jsonl: Optional[File.Handle] = none
  events: Optional[File.Handle] = none
  if sinks == Sinks.Both or sinks == Sinks.Jsonl
    jsonl = some(try File.open(try Path.join([dir, "metrics.jsonl"]), "append"))
  end

  if sinks == Sinks.Both or sinks == Sinks.TensorBoard
    started = start_events(dir)
    if !started.ok?
      _ = close_file(jsonl)
      return error(started.message)
    end

    events = some(started.unwrap!)
  end

  ok(Run(name: name, jsonl: jsonl, events: events))
end

## The `value` of the series `name` at `step`, which is 0 or more. The value is finite and within float32, which is
## what an event file holds.
def scalar(run: Run, name: String, step: Int, value: Float) -> Result[Void]
  return error("a step is 0 or more, got #{step}") if step < 0
  if value - value != 0.0 or value.abs > FLOAT32_MAX
    return error("a metric value is finite and within float32, got #{value} for #{name}")
  end

  time = Time.now().to_us.to_f / 1000000.0
  case run.jsonl
    some(f) ->
      try File.write(f, "#{JSON.encode(Line(run: run.name, name: name, step: step, value: value, time: time))}\n")
      try File.flush(f)
    none -> nil
  end

  case run.events
    some(f) ->
      try write_record(f, Events.scalar(time, step, name, value))
      try File.flush(f)
    none -> nil
  end

  ok()
end

## Closes the files of `run`, both even if one fails to close.
def close(run: Run) -> Result[Void]
  jsonl = close_file(run.jsonl)
  events = close_file(run.events)
  try jsonl
  events
end

private

def folder_name?(name: String) -> Bool
  return false if name.empty? or name == "." or name == ".."
  not (name.include?("/") or name.include?("\\"))
end

def close_file(f: Optional[File.Handle]) -> Result[Void]
  case f
    some(h) -> File.close(h)
    none -> ok()
  end
end

# The event file of a run: named by the second it starts, the process and the microseconds, so two runs never
# share one.
def start_events(dir: String) -> Result[File.Handle]
  us = Time.now().to_us
  stamp = "#{us / 1000000}.agentica.#{System.pid()}.#{format("%06d", us % 1000000)}"
  f = try File.open(try Path.join([dir, "events.out.tfevents.#{stamp}"]), "write")
  try write_record(f, Events.file_version(us.to_f / 1000000.0))
  try File.flush(f)
  ok(f)
end

def write_record(f: File.Handle, data: Bytes) -> Result[Void] = File.write_bytes(f, try Events.record(data))
