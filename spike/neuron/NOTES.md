# ml#80 part 2: Neuron spike notes

Setup: `python3 -m venv .venv && .venv/bin/pip install --only-binary :all: ansible==11.1.0 ansible-core==2.18.1`.
Run: `.venv/bin/ansible-playbook -i inventory.ini site.yml [--tags probe|test|bench|extra|gspmd]`
(sandbox off). Setup always runs; logs land in `results/`. `TREE=` points at an
ml tree (default /tmp/ml-neuron-tree, made from origin/main +
plugin-path.patch when missing).

## Status
Done (2026-11-25, about 35 min of instance time): probe, tests, training
bench, extras, GSPMD. Results are in ml#80, "Part 2 results".

## What works
- `Backend.Xla(Device.Cpu)` on Neuron's PJRT plugin with no C change. The only
  change: `plugin_path()` honors `AGENTICA_ML_XLA_PLUGIN` (plugin-path.patch).
- tensor/test/xla_test.agc: 13/13 pass on the NeuronCore (TOL 1e-6, fp32).
- Facts:
  - plugin: /opt/aws_neuronx_venv_jax_0_10/lib/python3.12/site-packages/libneuronxla/libneuronpjrt.so
    (libneuronxla 3.0.12293.0, PJRT C API 0.104; xla.c asks 0.81+, fine).
    It links only /opt/aws/neuron/lib/libnrt.so.1 (2.35.33.0), not Python.
  - compile = `neuronx-cc compile` subprocess (2.28.7631.0): the venv's bin
    must be on PATH, or the compile fails with "neuronx-cc: not found".
  - disk NEFF cache: /var/tmp/neuron-compile-cache/neuronxcc-<ver>/MODULE_<hash>+<flags>/
    (model.neff, the HLO, compile_flags.json `target=trn1`); 3.9 MB for the
    whole spike.
  - auto-cast: none by default (fp32). bf16: `NEURON_CC_FLAGS="--auto-cast matmult"`
    (flags are part of the cache key).
  - devices: 2 (`NC_v2`, NeuronCore id 0 and 1), MemoryStats bytes_limit 16 GiB
    each. Driver 2.31.8.0. The AMI is Deep Learning AMI Neuron, Ubuntu 24.04.
  - the plugin logs "Use PJRT C-API 0.104 as client did not specify"
    (`NEURON_INTERNAL_PJRT_C_API_VERSION` sets it) and sets
    `NEURON_RT_ROOT_COMM_ID` by itself.
  - Pitfall: `PJRT_Buffer_ToHostBuffer` with dst=NULL (the size query)
    corrupts the heap in this plugin. xla.c does not use it.

## Numbers
Probe (results/probe-*.log):
| | neuron | cpu xla |
|---|---|---|
| plugin open (first use) | 5.0-5.5 s | 15 ms |
| compile add[4], cold / cached | 1.58 s / 0.21 s | 22 ms |
| compile MLP fwd per batch, cold / cached | 1.6-1.7 s / 30-33 ms | 27 ms |
| run add[4] (overhead per run) | 0.22 ms | 0.015 ms |
| run+read add[4] | 0.32 ms | 0.075 ms |
| MLP fwd 784-512-512-10, b64/256/1024 | 0.28/0.33/0.41-0.66 ms | 0.59/1.75/6.47 ms |

Train step, median ms at b64/256/1024 (results/bench-*.jsonl, extra-*.jsonl):
| | sgd | adam |
|---|---|---|
| neuron compiled | 0.69/0.83/1.64 (cached rerun 0.58/0.62/0.82) | 0.99/1.10/1.42 (0.77/0.80/0.98) |
| neuron fit | 1.00/1.27/2.28 | 1.27/1.43/2.22 |
| neuron bf16 compiled | 0.58/0.63/0.74 | 0.86/0.88/0.93 |
| cpu xla compiled | 2.78/5.91/18.66 | 2.99/6.26/18.73 |
| cpu xla fit | 2.32/5.49/18.44 | 2.34/6.05/18.85 |
| pytorch 2.14 eager cpu (4 thr) | 2.21/6.77/25.60 | 2.99/7.59/26.58 |
First step (compile) on Neuron: 2.7-3.2 s with SGD, 4.4-4.7 s with Adam cold,
0.23-0.26 s cached (CPU XLA: 0.19-0.43 s). The losses match the CPU to about
1e-6.

GSPMD (results/gspmd-*.log, jax-gspmd.log): the plugin accepts it as it is.
- Shardy text (`sdy.mesh`, `sdy.sharding`) plus options {num_partitions 2,
  use_spmd_partitioning, use_shardy_partitioner (19), device_assignment}:
  works. Classic `mhlo.sharding` with use_spmd_partitioning and
  device_assignment (no Shardy) works too.
- Without device_assignment the compile fails ("PartitionId: error
  condition"). Sdy text without the Shardy flag fails with the RET_CHECK
  "xla.sdy.* custom call".
- Column and row splits match one device (gap up to 7.5e-7). Cold compile
  is 1.3-1.6 s on two cores, against 1.6-2.0 s on one.
- Run at [1024,4096]x[4096,4096]: 2.5 ms on one core; 1.8-4.5 ms for the
  column split and 2.1-4.7 ms for the row split (noisy). There is no clear
  gain at this size. Smaller shapes: two cores 0.2-0.5 ms against 0.18-0.33 ms.
