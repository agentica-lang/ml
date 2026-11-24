"""Splits the stderr of a run with GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 by the MARK lines of
transformers/tools/profile_qwen.agc: for the `prefill` and `decode` sections, the GPU time per kernel, per graph.
    python3 -I vk_perf.py STDERR.txt"""
import re, sys, collections

section = None
graphs = {"prefill": [], "decode": []}
cur = None
for line in open(sys.argv[1], errors="replace"):
    line = line.rstrip("\n")
    m = re.match(r"MARK (\w+)", line)
    if m:
        section = m.group(1)
        cur = None
        continue
    if line.startswith("Vulkan Timings"):
        cur = collections.OrderedDict()
        if section in graphs:
            graphs[section].append(cur)
        continue
    if cur is None:
        continue
    m = re.match(r"(.*): (\d+) x ([\d.]+) us = ([\d.]+) us", line)
    if m:
        cur[m.group(1)] = (int(m.group(2)), float(m.group(4)))
for name, gs in graphs.items():
    if not gs:
        continue
    tot = collections.defaultdict(lambda: [0, 0.0])
    for g in gs:
        for k, (n, us) in g.items():
            tot[k][0] += n
            tot[k][1] += us
    total = sum(v[1] for v in tot.values())
    print(f"== {name}: {len(gs)} graphs, {total / len(gs) / 1000:.2f} ms of kernels per graph")
    for k, (n, us) in sorted(tot.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"  {k[:60]:60} {n / len(gs):7.1f} calls  {us / len(gs) / 1000:8.3f} ms  {100 * us / total:5.1f}%")
