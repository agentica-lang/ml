## The FFI boundary of the ggml-graph spike: a thin C layer that builds ONE
## ggml compute graph from the nodes `lower.agc` sends, allocates it once, and
## runs it once per training step on ggml's CPU or Vulkan backend.
##
## This file is the one source of the signatures. From this folder, after a
## change to the block below:
##
## ```sh
## agentica deps/native-gen/cli.agc ggml_graph.agc --output native/ggmlgraph_native.h
## agentica deps/native-gen/cli.agc ggml_graph.agc --output native/ggmlgraph_native.h --check
## ```
##
## A program is built node by node in the order `lower.agc` sends (a
## topological order), then `compile` makes the cgraph, allocates the leaves
## in one persistent backend buffer and the intermediates with one graph
## allocator, and checks that the backend supports every node. After that the
## graph is never rebuilt: `run` uploads the step's inputs and computes.

extern module "ggmlgraph"
  ## An open backend: ggml's CPU backend with a thread count, or the iGPU
  ## through ggml's Vulkan backend. The state owns the `ggml_backend_t`.
  resource Backend
  ## One compiled training step: its ggml contexts, graph, buffers and the
  ## host batches of a benchmark. It borrows the `ggml_backend_t` of the
  ## `Backend` it was made on, which must stay open while it is used.
  resource Program

  ## The CPU backend with `threads` threads.
  compute def backend_cpu(threads: Int) -> Result[Backend]
  ## Vulkan device 0, failing unless it is Vulkan and not a CPU. `mul_mat` is
  ## F32 unless `fp16` lets ggml convert its inputs to fp16 (its default on
  ## this iGPU). ggml reads the choice when it first opens the device, so the
  ## first call fixes it for the process and a later call that asks for the
  ## other one is an error.
  compute def backend_vulkan(fp16: Bool) -> Result[Backend]
  ## `"name — description (type)"` of the backend's device.
  compute def describe(backend: Backend) -> Result[String]

  ## An empty program with room for `capacity` nodes.
  compute def program(backend: Backend, capacity: Int) -> Result[Program]
  ## Appends one node and answers its index: `op` is a code of
  ## `native/ggmlgraph.c`, `args` earlier indices, `shape` the row-major shape
  ## the trace recorded, `ints` the op's integer parameters (a leaf's slot, a
  ## reduction's axes, arange's start, stop and step) and `value` a fill's
  ## value.
  compute def node(p: Program, op: Int, args: List[Int], shape: List[Int], ints: List[Int], value: Float) -> Result[Int]
  ## Points the leaf `node` of `p` at the leaf `source` of `other`, so two
  ## programs share one buffer (the two-graph optimizer). `other` must stay
  ## open while `p` is used: `Lower.Built` keeps it.
  compute def share(p: Program, node: Int, other: Program, source: Int) -> Result[Void]
  ## Builds the graph of `outputs` and of the writebacks (pairs of a state
  ## leaf and the node whose value replaces it after each run), allocates it
  ## and checks every node against the backend.
  compute def compile(p: Program, outputs: List[Int], writebacks: List[Int]) -> Result[Void]
  ## Uploads row-major `values` into the leaf `node`.
  compute def set(p: Program, node: Int, values: List[Float]) -> Result[Void]
  ## Fills the leaf `node` with values uniform in `[-scale, scale)` from
  ## `seed`, made in C: a parameter of the benchmark's size costs no list.
  compute def fill(p: Program, node: Int, seed: Int, scale: Float) -> Result[Void]
  ## `max |a - b| / max |b|` between node `node` of `p` and node `other_node`
  ## of `other`, both read back; an error on any non-finite value.
  compute def gap(p: Program, node: Int, other: Program, other_node: Int) -> Result[Float]
  ## `count` host batches of synthetic data for the leaves `x` (`[b, d]`
  ## floats) and `y` (`[b]` labels below `classes`), made in C from `seed`.
  compute def batches(p: Program, x: Int, y: Int, classes: Int, count: Int, seed: Int) -> Result[Void]
  ## One step: uploads host batch `batch` (when `batches` made any), runs
  ## the graph, waits, and answers the first output's first value (the loss).
  compute def run(p: Program, batch: Int) -> Result[Float]
  ## The values of node `node` after the last run: an output, a state or an
  ## input.
  compute def read(p: Program, node: Int) -> Result[List[Float]]
  ## What the program costs: graph nodes, leaf bytes, compute buffer bytes,
  ## compile ns, runs, upload ns, compute ns (totals over the runs).
  compute def stats(p: Program) -> Result[List[Int]]
end
