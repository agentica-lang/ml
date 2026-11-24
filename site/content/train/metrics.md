---
title: metrics
description: Scalars written to a run folder as JSON lines and as a TensorBoard event file, and how Train.fit reports through them.
weight: 2
---

`Metrics` records the numbers of a training run while it runs, so you can watch them live and compare runs. One call writes a value to every sink of the run, and each sink is a file in the run folder.

## Example

```agentica
import Metrics from "pkg/train/metrics"

def main() -> Void
  run = Metrics.open("runs", "demo").unwrap!
  for step in 1..3
    Metrics.scalar(run, "loss", step, 1.0 / step.to_f).unwrap!
  end

  Metrics.close(run).unwrap!
end
```

This makes `runs/demo/metrics.jsonl` and `runs/demo/events.out.tfevents.*`. View every run under `runs` with `tensorboard --logdir runs`, or read the JSON lines with anything that reads JSON.

- `Metrics.open(root, name, sinks)` makes the folder `root/name` if it is missing and opens the files of `sinks`: `Metrics.Sinks.Jsonl`, `Metrics.Sinks.TensorBoard` or `Metrics.Sinks.Both` (the default). A run folder that exists gets lines appended and a new event file.
- `Metrics.scalar(run, name, step, value)` writes the value of the series `name` at `step`, which is 0 or more, with the time. A `/` in the name groups series in TensorBoard (`train/loss`). It flushes, so a viewer sees the value at once. A value that is not finite is an `error`.
- `Metrics.close(run)` closes the files.

A line of `metrics.jsonl` is `{"run": "demo", "name": "loss", "step": 1, "value": 1.0, "time": 1795561234.5}`, where `time` is the Unix time in seconds.

## Training

`Train.fit(..., on_epoch, metrics)` takes an open run. Each step writes `train/loss`, `train/lr` and `train/step_ms` at the optimizer's step count, which starts at 1. `fit` does not open or close the run, and a loss that is not finite stops it with an `error`.

```agentica
run = Metrics.open("runs", "xor").unwrap!
fit = Train.fit(&loss, params, loader, adam, 20, none, some(run))
Metrics.close(run).unwrap!
```

## Limits

- Scalars only: no histograms, text or images.
- The TensorBoard file is written in Agentica, with no protobuf or TensorFlow. It stores each value as a 32-bit float.
- The step time includes the first compile on `Ggml`.
