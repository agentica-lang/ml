## LoRA adapters (`Lora` of `train`) on the models, against the PyTorch + PEFT peer `lora_golden.py`: its adapters
## for the tiny Qwen3 and DistilBERT of the standard-library peers are in `fixtures/lora_qwen/` and
## `fixtures/lora_distilbert/`, as PEFT wrote them, and its outputs in `fixtures/lora_golden.json`. A model with the
## adapter merged gives PEFT's output, and so does the adapter applied inside a trace; a fit of a new adapter lowers
## the loss while the model keeps its bits, and what it trained is saved for PEFT and read back. Run from
## `transformers/`.

import Data from "pkg/train/data"
import DistilBert from "../distilbert"
import Lift from "../lift"
import Lora from "pkg/train/lora"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Qwen3 from "../qwen3"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"
import WordPiece from "../tokenizer/wordpiece"

## The reference computes in doubles, the peer in `F32`, and the adapter moves the outputs by about 2.
ABS = 0.0001
REL = 0.001

struct QwenGolden
  ids: List[List[Int]]
  logits: List[Float]
  merged: List[Float]
  moved: Float
end

struct DistilGolden
  ids: List[List[Int]]
  hidden: List[Float]
  moved: Float
end

struct Golden
  qwen: QwenGolden
  distilbert: DistilGolden
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/lora_golden.json"), Golden)

def with_qwen(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def backends() -> Result[List[Tensor.Backend]] = ok([S.REF].concat(try S.ggml_backends()))

def bits(t: Tensor.Tensor) -> Result[List[String]] = ok((try S.floats_of(t)).map { |x: Float| "#{Float.to_bits(x)}" })

## Token ids as an `I32` tensor `[rows, columns]`.
def tokens(rows: List[List[Int]], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  values = rows.flat_map { |row: List[Int]| row.map { |id: Int| Tensor.Values.Int(id) } }
  Tensor.from_list(values, [rows.size, rows[0].size], Tensor.Dtype.I32, backend)
end

## The logits of `ids` with `ps` as the adapter's tensors, the adapter applied inside the trace.
def unmerged(
  m: Qwen3.Qwen3,
  adapter: Lora.Adapter,
  ps: List[Tensor.Tensor],
  ids: Tensor.Tensor,
) -> Result[Tensor.Tensor]
  with = try Lora.applied(Qwen3.weights(m), try Lora.with_params(adapter, ps))
  Qwen3.logits(try Qwen3.with_weights(m, with), Trace.constant(ids))
end

def test_a_qwen3_with_a_peft_adapter_gives_the_logits_of_peft() -> Result[Void]
  g = try golden()
  try Test.assert(g.qwen.moved > 1.0, "the adapter of the peer moves the logits")
  with_qwen do |dir|
    for backend in try backends()
      m = try Qwen3.open("#{dir}/untied", backend)
      adapter = try Lora.open("test/fixtures/lora_qwen", backend)
      try Test.eq(adapter.names.size, 4)
      try Test.eq(adapter.names[0], "model.layers.0.self_attn.q_proj.weight")
      merged = try Qwen3.with_weights(m, try Lora.merged(Qwen3.weights(m), adapter))
      got = try S.floats_of(try Qwen3.run(merged, g.qwen.ids))
      try S.expect_close(got, g.qwen.logits, ABS, REL, "merged on #{backend}")
      try S.expect_close(got, g.qwen.merged, ABS, REL, "merged against PEFT's merge on #{backend}")
      ids = try tokens(g.qwen.ids, backend)
      traced = try Trace.eval(try Lora.params(adapter), ->(ps: List[Tensor.Tensor]) { unmerged(m, adapter, ps, ids) })

      try S.expect_close(try S.floats_of(traced), g.qwen.logits, ABS, REL, "unmerged on #{backend}")
      try S.expect_close(try S.floats_of(traced), got, ABS, REL, "unmerged against merged on #{backend}")
    end

    ok()
  end
end

def test_a_distilbert_with_a_peft_adapter_gives_the_hidden_state_of_peft() -> Result[Void]
  g = try golden()
  try Test.assert(g.distilbert.moved > 1.0, "the adapter of the peer moves the hidden state")
  S.with_fixtures("test/distilbert_peer.py", "agentica-distilbert-") do |dir|
    encodings = g.distilbert.ids.map do |ids: List[Int]|
      WordPiece.Encoding(ids: ids, type_ids: List.repeat(0, ids.size))
    end

    for folder in ["tiny", "prefixed"]
      for backend in try backends()
        m = try DistilBert.open("#{dir}/#{folder}", backend)
        adapter = try Lora.open("test/fixtures/lora_distilbert", backend)
        try Test.eq(adapter.names.size, 4)
        if folder == "prefixed"
          adapter = adapter.with(names: adapter.names.map { |name: String| "distilbert.#{name}" })
        end

        merged = try DistilBert.with_weights(m, try Lora.merged(DistilBert.weights(m), adapter))
        hidden = try DistilBert.run(merged, encodings)
        try S.expect_close(try S.floats_of(hidden), g.distilbert.hidden, ABS, REL, "#{folder} on #{backend}")
        Lift.close([hidden])
      end
    end

    ok()
  end
end

## The cross-entropy of the next token at every position, with `ps` as the adapter's tensors.
def loss(m: Qwen3.Qwen3, adapter: Lora.Adapter, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  with = try Lora.applied(Qwen3.weights(m), try Lora.with_params(adapter, ps))
  logits = try Qwen3.logits(try Qwen3.with_weights(m, with), b.inputs)
  Loss.cross_entropy(try Tensor.reshape(logits, [10, 13]), try Tensor.reshape(b.targets, [10]))
end

def trains_an_adapter(dir: String, out: String, backend: Tensor.Backend) -> Result[Void]
  g = try golden()
  m = try Qwen3.open("#{dir}/untied", backend)
  weights = Qwen3.weights(m)
  before: Map[String, List[String]] = {}
  for name in weights.keys
    before = before.put(name, try bits(weights[name]))
  end

  adapter = try Lora.init(weights, Lora.Config(targets: ["q_proj", "v_proj"], r: 2, alpha: 4.0), 5, backend)
  rows = try Data.dataset(
    try tokens(g.qwen.ids.map { |row: List[Int]| row.take(5) }, backend),
    try tokens(g.qwen.ids.map { |row: List[Int]| row.drop(1) }, backend),
  )

  fit = try Train.fit(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(m, adapter, ps, b) },
    try Lora.params(adapter),
    Data.Loader(dataset: rows, batch_size: 2, shuffle: false),
    Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.05, weight_decay: 0.0)),
    4,
  )

  try Test.assert(fit.history[3] < fit.history[0], "the loss went from #{fit.history[0]} to #{fit.history[3]}")
  for name in weights.keys
    try Test.eq(try bits(weights[name]), before[name])
  end

  trained = try Lora.with_params(adapter, fit.params)
  try Lora.save(out, trained, Qwen3.embeddings(m))
  back = try Lora.open(out, backend)
  try Test.eq(back.names, trained.names)
  want = try Qwen3.run(try Qwen3.with_weights(m, try Lora.merged(weights, trained)), g.qwen.ids)
  got = try Qwen3.run(try Qwen3.with_weights(m, try Lora.merged(weights, back)), g.qwen.ids)
  try S.expect_close(
    try S.floats_of(got),
    try S.floats_of(want),
    0.000001,
    0.000001,
    "the saved adapter on #{backend}",
  )

  plain = try Qwen3.run(m, g.qwen.ids)
  Test.assert(try S.floats_of(plain) != try S.floats_of(want), "the trained adapter changes the logits")
end

def test_a_fit_of_an_adapter_lowers_the_loss_and_is_saved_for_peft() -> Result[Void]
  with_qwen do |dir|
    for backend in try backends()
      try trains_an_adapter(dir, "#{dir}/adapter-#{(try backends()).index(backend).unwrap_or(0)}", backend)
    end

    ok()
  end
end

def test_an_adapter_of_an_embedding_table_is_not_saved() -> Result[Void]
  with_qwen do |dir|
    m = try Qwen3.open("#{dir}/untied", S.REF)
    try Test.eq(Qwen3.embeddings(m), ["model.embed_tokens.weight"])
    adapter = try Lora.init(Qwen3.weights(m), Lora.Config(targets: ["embed_tokens"], r: 2), 1, S.REF)
    try S.expect_error(Lora.save("#{dir}/table", adapter, Qwen3.embeddings(m)), "embedding table")
    ok()
  end
end

def test_the_embedding_tables_of_a_distilbert_have_the_names_of_its_weights() -> Result[Void]
  S.with_fixtures("test/distilbert_peer.py", "agentica-distilbert-") do |dir|
    for folder in ["tiny", "prefixed"]
      m = try DistilBert.open("#{dir}/#{folder}", S.REF)
      tables = DistilBert.embeddings(m)
      try Test.eq(tables.size, 2)
      for name in tables
        try Test.assert(DistilBert.weights(m).key?(name), "#{name} is a weight of #{folder}")
      end
    end

    ok()
  end
end
