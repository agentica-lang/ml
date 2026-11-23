## The `asr/` package: Whisper's model forward over `tensor/`, `nn/` and
## `hub/`, loaded from a Hugging Face checkpoint (`asr/DESIGN.md`).
##
## `encode` turns a log-mel batch into the encoder output; `decode` turns
## token ids and that output into logits; `listen` and `step` decode one token
## at a time with a KV cache; `transcribe` runs the log-mel frontend
## (`mel.agc`), the encoder and greedy decoding (`greedy.agc`) on up to 30 s of
## audio. Inference only. On `Ggml` each stage and each step is one compiled
## graph of the model's tensors as trace constants.

import Greedy from "greedy"
import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Mel from "mel"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "tokenizer"
import Trace from "pkg/tensor/trace"

## Whisper's layer norms use PyTorch's default.
EPS = 0.00001

## The prefix of every tensor `WhisperForConditionalGeneration` stores.
PREFIX = "model."

## The fields of Hugging Face's `config.json` the forward needs.
struct Config
  d_model: Int
  encoder_layers: Int
  decoder_layers: Int
  encoder_attention_heads: Int
  decoder_attention_heads: Int
  num_mel_bins: Int
  vocab_size: Int
  max_source_positions: Int
  max_target_positions: Int
  ## The ids greedy decoding never emits, and those it does not emit first;
  ## `Greedy.rules` has the defaults.
  suppress_tokens: Optional[List[Int]] = none
  begin_suppress_tokens: Optional[List[Int]] = none
  ## `gelu` (the exact form, as Whisper's checkpoints say) or `gelu_new` (the
  ## tanh approximation).
  activation_function: String = "gelu"
end

struct Model
  config: Config
  ## The GELU of `config.activation_function`, in the conv stem and the MLPs.
  activation: Nn.Activation
  conv1: Nn.Conv1d
  conv2: Nn.Conv1d
  encoder_positions: Nn.Embedding
  encoder: List[Nn.EncoderBlock]
  encoder_norm: Nn.LayerNorm
  tokens: Nn.Embedding
  decoder_positions: Nn.Embedding
  decoder: List[Nn.DecoderBlock]
  decoder_norm: Nn.LayerNorm
  ## The output projection, tied to `tokens`: its table transposed, no bias.
  logits: Nn.Dense
end

## One attention's keys and values, `[1, keys, d_model]` each: `keys` is the
## tokens so far, or on `Ggml` the cache's padded capacity.
struct Kv
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## What `step` keeps between calls: per decoder block the projected encoder
## output, made once by `listen`, and the self-attention keys and values of
## the `length` tokens so far. On `Ggml` those are padded with zero rows to a
## capacity (`capacity`), so a step's graph changes shape only when the
## capacity grows.
struct Cache
  cross: List[Heard]
  own: List[Kv]
  length: Int
end

## One block's cross-attention keys and values, split by head: `keys[h]` is
## `[head_width, frames]` (transposed), `values[h]` is `[frames, head_width]`.
struct Heard
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
end

## One step's forward: the logits of the last token, and per decoder block
## the self-attention keys and values of the new tokens (`keys`, `values`)
## and of all of them (`own`).
struct Forward
  logits: Tensor.Tensor
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
  own: List[Kv]
end

## Where a step writes into a padded cache: `positions` are the new tokens'
## positions (`I32`, `[n]`) and `capacity` the rows the cache has after it.
struct Slots
  positions: Tensor.Tensor
  capacity: Int
end

## What one step's arena hands back: the last logits, and per block the
## self-attention keys and values of all the tokens so far.
struct Grown
  logits: List[Float]
  own: List[Kv]
end

struct Stepped
  ## The logits of the last token fed, one per vocabulary id.
  logits: List[Float]
  cache: Cache
end

## What `transcribe` does.
struct Options
  ## The language code, such as `es`; none detects it.
  language: Optional[String] = none
  ## Translate to English instead of transcribing.
  translate: Bool = false
  ## Timestamp tokens, and the segments they give.
  timestamps: Bool = true
  ## At most this many tokens after the prompt, at least 1; none decodes up to
  ## `max_target_positions`, as Hugging Face's `max_length` (448 for
  ## Whisper) does. OpenAI's `sample_len` is 224.
  max_tokens: Optional[Int] = none
end

struct Transcript
  text: String
  language: String
  ## The softmax of `language` over the language tokens, one decoder step
  ## after `<|startoftranscript|>`; 1.0 when `Options.language` named it.
  language_probability: Float
  ## The ids emitted after the prompt, without `<|endoftext|>`.
  tokens: List[Int]
  segments: List[Greedy.Segment]
  ## OpenAI's `avg_logprob`: the log-probabilities of the emitted tokens and
  ## `<|endoftext|>` under the rules, summed, over `tokens.size + 1`.
  avg_logprob: Float
  ## OpenAI's `no_speech_prob`: the softmax of the logits after
  ## `<|startoftranscript|>` at `<|nospeech|>` (0.0 without that token).
  no_speech_prob: Float
end

## Hugging Face's `config.json` at `path`.
def config(path: String) -> Result[Config] = within(path, JSON.decode(try File.read_text(path), Config))

## The checkpoint at `path` (a `.safetensors` file or the `.index.json` of a
## sharded one) with the `config.json` of its folder.
def open(path: String, backend: Tensor.Backend) -> Result[Model]
  parts = path.split("/")
  load(try Hub.open(path), try config(parts.take(parts.size - 1).push("config.json").join("/")), backend)
end

## The model from an open checkpoint, its shapes checked against `config`.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[Model]
  e = "#{PREFIX}encoder."
  d = "#{PREFIX}decoder."
  width = config.d_model
  activation = try activation_of(config.activation_function)
  try expect_heads("encoder", width, config.encoder_attention_heads)
  try expect_heads("decoder", width, config.decoder_attention_heads)
  conv1 = try Load.conv1d(c, "#{e}conv1", 1, 1, backend)
  try expect_shape("#{e}conv1.weight", conv1.weight, [width, config.num_mel_bins, 3])
  conv2 = try Load.conv1d(c, "#{e}conv2", 2, 1, backend)
  try expect_shape("#{e}conv2.weight", conv2.weight, [width, width, 3])
  encoder_positions = try Load.embedding(c, "#{e}embed_positions.weight", backend)
  try expect_shape("#{e}embed_positions.weight", encoder_positions.table, [config.max_source_positions, width])
  tokens = try Load.embedding(c, "#{d}embed_tokens.weight", backend)
  try expect_shape("#{d}embed_tokens.weight", tokens.table, [config.vocab_size, width])
  decoder_positions = try Load.embedding(c, "#{d}embed_positions.weight", backend)
  try expect_shape("#{d}embed_positions.weight", decoder_positions.table, [config.max_target_positions, width])

  encoder: List[Nn.EncoderBlock] = []
  for i in 0...config.encoder_layers
    names = encoder_names("#{e}layers.#{i}.")
    encoder = encoder.push(try Load.encoder_block(c, names, config.encoder_attention_heads, activation, EPS, backend))
  end

  decoder: List[Nn.DecoderBlock] = []
  for i in 0...config.decoder_layers
    names = decoder_names("#{d}layers.#{i}.")
    decoder = decoder.push(try Load.decoder_block(c, names, config.decoder_attention_heads, activation, EPS, backend))
  end

  ok(
    Model(
      config: config,
      activation: activation,
      conv1: conv1,
      conv2: conv2,
      encoder_positions: encoder_positions,
      encoder: encoder,
      encoder_norm: try Load.layer_norm(c, "#{e}layer_norm", EPS, backend),
      tokens: tokens,
      decoder_positions: decoder_positions,
      decoder: decoder,
      decoder_norm: try Load.layer_norm(c, "#{d}layer_norm", EPS, backend),
      logits: Nn.Dense(w: try Load.transposed(tokens.table), b: none),
    ),
  )
end

## The encoder output `[batch, ceil(frames / 2), d_model]` of a log-mel batch
## `[batch, num_mel_bins, frames]`: two gelu convolutions (the second halves
## the frames), the positional table, the blocks and a final layer norm.
## `ceil(frames / 2)` may not pass `max_source_positions`.
def encode(m: Model, mel: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(mel)
  if shape.size != 3 or shape[1] != m.config.num_mel_bins
    return error("encode needs [batch, #{m.config.num_mel_bins}, frames], got #{shape}")
  end

  # The previous stage's tensor is closed, so a block's intermediates die
  # with it (`staged`).
  h = try staged(m, mel, &stem)
  for i in 0...m.encoder.size
    fresh = try staged(m, h, ->(model: Model, x: Tensor.Tensor) { Nn.encoder_block(model.encoder[i], x) })
    close([h])
    h = fresh
  end

  audio = try staged(m, h, ->(model: Model, x: Tensor.Tensor) { Nn.layer_norm(model.encoder_norm, x) })
  close([h])
  ok(audio)
end

## The logits `[batch, t, vocab_size]` of the integer token ids `[batch, t]`
## over `audio`, the output of `encode`. Every position is recomputed: no KV
## cache. `t` may not pass `max_target_positions`.
def decode(m: Model, tokens: Tensor.Tensor, audio: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(tokens)
  return error("decode needs token ids [batch, t], got #{shape}") if shape.size != 2
  heard = Tensor.shape(audio)
  if heard.size != 3 or heard[0] != shape[0] or heard[2] != m.config.d_model
    return error("decode needs audio [#{shape[0]}, frames, #{m.config.d_model}] from encode, got #{heard}")
  end

  rows = try Nn.embedding(m.tokens, try Tensor.reshape(tokens, [shape[0] * shape[1]]))
  h = try Nn.positional(m.decoder_positions, try Tensor.reshape(rows, [shape[0], shape[1], m.config.d_model]))
  for block in m.decoder
    h = try Nn.decoder_block(block, h, audio)
  end

  Nn.dense(m.logits, try Nn.layer_norm(m.decoder_norm, h))
end

## The empty cache over `audio`, the `[1, frames, d_model]` output of
## `encode`: the cross-attention keys and values of every decoder block.
def listen(m: Model, audio: Tensor.Tensor) -> Result[Cache]
  heard = Tensor.shape(audio)
  if heard.size != 3 or heard[0] != 1 or heard[2] != m.config.d_model
    return error("listen needs audio [1, frames, #{m.config.d_model}] from encode, got #{heard}")
  end

  # One graph per block keeps each under ggml's 256 outputs: 2 per head.
  cross: List[Heard] = []
  for i in 0...m.decoder.size
    heads = try run(m, [audio], ->(model: Model, xs: List[Tensor.Tensor]) { split_heads(model.decoder[i], xs[0]) })
    n = m.decoder[i].cross_attention.heads
    cross = cross.push(Heard(keys: heads.take(n), values: heads.drop(n)))
  end

  ok(Cache(cross: cross, own: [], length: 0))
end

## `ids` fed after the tokens of `cache`: the logits of the last one, as
## `decode` gives them, and the cache that holds them all. Only an empty
## cache takes several ids at once (a prompt). The new cache's self-attention
## keys and values are new tensors: on libtorch the caller closes the old
## ones once it has no use for them (`transcribe` does).
def step(m: Model, cache: Cache, ids: List[Int]) -> Result[Stepped]
  n = ids.size
  return error("step needs ids") if n == 0
  return error("step takes one id after the first call, got #{n}") if n > 1 and cache.length > 0
  if cache.length + n > m.config.max_target_positions
    return error("step needs at most #{m.config.max_target_positions} tokens, got #{cache.length + n}")
  end

  backend = Tensor.backend(m.tokens.table)
  values = ids.map { |i| Tensor.Values.Int(i) }
  if compiled(backend)
    at = try Tensor.from_list(values, [n], Tensor.Dtype.I32, backend)
    where = (cache.length...(cache.length + n)).map { |i| Tensor.Values.Int(i) }
    positions = try Tensor.from_list(where, [n], Tensor.Dtype.I32, backend)
    room = capacity(cache.length + n, m.config.max_target_positions)
    got = Trace.eval_many(
      [at, positions],
      ->(xs: List[Tensor.Tensor]) { stepped_graph(traced(m), traced_cache(cache), xs, room) },
    )

    close([at, positions])
    outs = try got
    layers = m.decoder.size
    own = (0...layers).map { |i| Kv(key: outs[1 + i], value: outs[1 + layers + i]) }
    logits = try floats(outs[0])
    close([outs[0]])
    return ok(Stepped(logits: logits, cache: Cache(cross: cache.cross, own: own, length: cache.length + n)))
  end

  # The arena keeps the new keys and values alone (the libtorch ops are
  # `AN_FRESH`), so the step's intermediates die with it.
  grown: Result[Grown] = Native.arena do
    f = try forward(m, cache, try Tensor.from_list(values, [n], Tensor.Dtype.I32, backend))
    ok(Grown(logits: try floats(f.logits), own: f.own))
  end

  g = try grown
  ok(Stepped(logits: g.logits, cache: Cache(cross: cache.cross, own: g.own, length: cache.length + n)))
end

## The text of `pcm` (16 kHz mono samples, about -1..1, at most one window:
## 30 s for Whisper): the log-mel, the encoder and greedy decoding at
## temperature 0 with OpenAI's rules (`Greedy`). Without a language in
## `options`, one decoder step after `<|startoftranscript|>` picks the likeliest
## language token.
def transcribe(
  m: Model,
  tok: Tokenizer.Tokenizer,
  pcm: List[Float],
  options: Options = Options(),
) -> Result[Transcript]
  frames = 2 * m.config.max_source_positions
  window = frames * HOP
  return error("transcribe needs samples, got none") if pcm.empty?
  if pcm.size > window
    return error("transcribe takes at most #{window} samples (#{window / SAMPLE_RATE} s), got #{pcm.size}")
  end

  if tok.vocab_size != m.config.vocab_size
    return error("the tokenizer has #{tok.vocab_size} ids and the model #{m.config.vocab_size}")
  end

  case options.max_tokens
    some(n) -> return error("transcribe needs max_tokens >= 1, got #{n}") if n < 1
    none -> nil
  end

  # A named language and the job are checked before the audio costs anything.
  _ = try Tokenizer.prompt(tok, options.language, job(options), options.timestamps)
  # On libtorch the call runs in a task of its own: when it ends, the task
  # closes what it opened, the encoder output and the cache included.
  call = Task.spawn do decoded(m, tok, pcm, options, frames) end
  Task.await(call)
end

private

## 16 kHz audio, and one log-mel frame every 160 samples (`mel.agc`).
SAMPLE_RATE = 16000
HOP = 160

## `transcribe` once its arguments are checked. Each step's self-attention
## keys and values are closed once the next step replaced them, so a call
## holds one cache at a time.
def decoded(m: Model, tok: Tokenizer.Tokenizer, pcm: List[Float], options: Options, frames: Int) -> Result[Transcript]
  features = try Mel.log_mel(pcm, m.config.num_mel_bins, frames)
  values = features.map { |x| Tensor.Values.Float(x) }
  mel = try Tensor.from_list(
    values,
    [1, m.config.num_mel_bins, frames],
    Tensor.Dtype.F32,
    Tensor.backend(m.tokens.table),
  )

  cache = try listen(m, try encode(m, mel))
  language = options.language.unwrap_or("en")
  probability = 1.0
  no_speech = 0.0
  detect = options.language == none and tok.multilingual
  # One step after `sot` alone, only when something reads its logits.
  if detect or tok.no_speech != none
    first = try step(m, cache, [tok.sot])
    close(kept(first.cache))
    case tok.no_speech
      some(id) -> no_speech = Greedy.probability(first.logits, id)
      none -> nil
    end

    if detect
      scores = tok.languages.map { |l| first.logits[l.id] }
      best = Greedy.pick(scores)
      language = tok.languages[best].code
      probability = Greedy.probability(scores, best)
    end
  end

  prompt = try Tokenizer.prompt(tok, some(language), job(options), options.timestamps)
  rules = Greedy.rules(tok, m.config.suppress_tokens, m.config.begin_suppress_tokens, options.timestamps)
  limit = m.config.max_target_positions - prompt.size
  limit = [limit, options.max_tokens.unwrap_or(limit)].min.unwrap!
  stepped = try step(m, cache, prompt)
  emitted: List[Int] = []
  logprob = 0.0
  while emitted.size < limit
    allowed = Greedy.allowed(rules, stepped.logits, emitted)
    id = Greedy.pick(allowed)
    logprob += Greedy.log_probability(allowed, id)
    break if id == tok.eot
    emitted = emitted.push(id)
    break if emitted.size == limit
    next = try step(m, stepped.cache, [id])
    close(kept(stepped.cache))
    stepped = next
  end

  close(kept(stepped.cache))
  average = logprob / (emitted.size + 1).to_f
  duration = pcm.size.to_f / SAMPLE_RATE.to_f
  ok(
    Transcript(
      text: try Tokenizer.decode(tok, emitted),
      language: language,
      language_probability: probability,
      tokens: emitted,
      segments: try Greedy.segments(tok, emitted, duration, average),
      avg_logprob: average,
      no_speech_prob: no_speech,
    ),
  )
end

def job(options: Options) -> Tokenizer.Job
  return Tokenizer.Job.Translate if options.translate
  Tokenizer.Job.Transcribe
end

## The self-attention keys and values `cache` holds.
def kept(cache: Cache) -> List[Tensor.Tensor] = cache.own.flat_map { |kv| [kv.key, kv.value] }

## Closes the libtorch and ggml tensors of `ts` now, not at the end of an
## arena; the reference has nothing to close.
def close(ts: List[Tensor.Tensor]) -> Void
  for t in ts
    case t.handle
      Tensor.Handle.Native(h) -> Native.close(h)
      Tensor.Handle.Ggml(h) -> Native.close(h)
      _ -> nil
    end
  end
end

## Whether `backend` computes only compiled graphs (`Ggml`).
def compiled(backend: Tensor.Backend) -> Bool
  case backend
    Tensor.Backend.Ggml(_) -> true
    _ -> false
  end
end

## `f` of the model and `args`; on `Ggml` one compiled graph of the traced
## model.
def run(
  m: Model,
  args: List[Tensor.Tensor],
  f: Fn[Model, List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[List[Tensor.Tensor]]
  if compiled(Tensor.backend(args[0]))
    return Trace.eval_many(args, ->(xs: List[Tensor.Tensor]) { f.call(traced(m), xs) })
  end

  f.call(m, args)
end

## `f` of the model and `x`. On `Ggml` it is one compiled graph; elsewhere it
## runs in an arena that keeps the result alone (the libtorch ops are
## `AN_FRESH`), so its intermediates die with the arena.
def staged(m: Model, x: Tensor.Tensor, f: Fn[Model, Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Tensor.Tensor]
  return Trace.eval([x], ->(xs: List[Tensor.Tensor]) { f.call(traced(m), xs[0]) }) if compiled(Tensor.backend(x))
  Native.arena do f.call(m, x) end
end

## The conv stem of the encoder over a log-mel batch, with the positions
## added: two gelu convolutions, the second halving the frames.
def stem(m: Model, mel: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(mel)
  items: List[Tensor.Tensor] = []
  for b in 0...shape[0]
    one = try Tensor.reshape(try Tensor.slice(mel, [b, 0, 0], [b + 1, shape[1], shape[2]]), [shape[1], shape[2]])
    x = try Nn.activate(m.activation, try Nn.conv1d(m.conv1, one))
    x = try Nn.activate(m.activation, try Nn.conv1d(m.conv2, x))
    items = items.push(try Tensor.transpose(x))
  end

  Nn.positional(m.encoder_positions, try Tensor.stack(items, 0))
end

## The cross-attention keys of each head of `block` over the encoder output
## `audio` (transposed), then the values of each head.
def split_heads(block: Nn.DecoderBlock, audio: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  frames = Tensor.shape(audio)[1]
  d = Tensor.shape(audio)[2]
  a = block.cross_attention
  k = try Tensor.reshape(try Nn.dense(a.key, audio), [frames, d])
  v = try Tensor.reshape(try Nn.dense(a.value, audio), [frames, d])
  width = d / a.heads
  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  for h in 0...a.heads
    keys = keys.push(try Tensor.transpose(try Tensor.slice(k, [0, h * width], [frames, (h + 1) * width])))
    values = values.push(try Tensor.slice(v, [0, h * width], [frames, (h + 1) * width]))
  end

  ok(keys + values)
end

## The forward of the integer ids `ids` (`[n]`) after the tokens of `cache`.
## With `slots`, the cache is padded (`Ggml`): the new rows are written at
## their positions, and the mask hides the later positions and the padding.
def forward(m: Model, cache: Cache, ids: Tensor.Tensor, slots: Optional[Slots] = none) -> Result[Forward]
  n = Tensor.shape(ids)[0]
  d = m.config.d_model
  rows = try Nn.embedding(m.tokens, ids)
  positions = case slots
    some(s) -> try Nn.embedding(m.decoder_positions, s.positions)
    none -> try Tensor.slice(m.decoder_positions.table, [cache.length, 0], [cache.length + n, d])
  end

  h = try Tensor.reshape(try Tensor.add(rows, positions), [1, n, d])
  mask: Optional[Tensor.Tensor] = none
  spread: Optional[Tensor.Tensor] = none
  case slots
    some(s) ->
      mask = some(try padded_mask(s, Tensor.dtype(h)))
      spread = some(try spread_of(s, Tensor.dtype(h)))
    none ->
      if n > 1
        mask = some(try Nn.causal_mask(n, Tensor.dtype(h), Tensor.backend(h)))
      end
  end

  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  own: List[Kv] = []
  for i in 0...m.decoder.size
    block = m.decoder[i]
    a = block.self_attention
    x = try Nn.layer_norm(block.norm1, h)
    k = try Nn.dense(a.key, x)
    v = try Nn.dense(a.value, x)
    keys = keys.push(k)
    values = values.push(v)
    case spread
      some(p) ->
        mine: Optional[Kv] = none
        if i < cache.own.size
          mine = some(cache.own[i])
        end

        k = try written(mine.map { |kv| kv.key }, k, p)
        v = try written(mine.map { |kv| kv.value }, v, p)
      none ->
        if cache.length > 0
          k = try Tensor.concat([cache.own[i].key, k], 1)
          v = try Tensor.concat([cache.own[i].value, v], 1)
        end
    end

    own = own.push(Kv(key: k, value: v))
    h = try Tensor.add(
      h,
      try Nn.dense(a.output, try Tensor.multi_head_attention(try Nn.dense(a.query, x), k, v, a.heads, mask)),
    )

    c = block.cross_attention
    x = try Nn.layer_norm(block.norm2, h)
    heard = try attend(try Tensor.reshape(try Nn.dense(c.query, x), [n, d]), cache.cross[i])
    h = try Tensor.add(h, try Nn.dense(c.output, try Tensor.reshape(heard, [1, n, d])))
    h = try Tensor.add(h, try Nn.mlp(block.mlp, try Nn.layer_norm(block.norm3, h)))
  end

  last = try Tensor.slice(h, [0, n - 1, 0], [1, n, d])
  logits = try Nn.dense(m.logits, try Nn.layer_norm(m.decoder_norm, last))
  ok(Forward(logits: logits, keys: keys, values: values, own: own))
end

## A step as one graph: the logits, then the keys of every block, then the
## values.
def stepped_graph(m: Model, cache: Cache, xs: List[Tensor.Tensor], room: Int) -> Result[List[Tensor.Tensor]]
  f = try forward(m, cache, xs[0], some(Slots(positions: xs[1], capacity: room)))
  ok([f.logits] + f.own.map { |kv| kv.key } + f.own.map { |kv| kv.value })
end

## The rows a padded cache has for `needed` tokens: 8, 16, 32 and so on, at
## most `limit`.
def capacity(needed: Int, limit: Int) -> Int
  rows = 8
  while rows < needed
    rows *= 2
  end

  [rows, limit].min.unwrap!
end

## `[n, capacity]`: `MASKED` where the key's position is after the query's
## (later tokens and the padding), 0 elsewhere.
def padded_mask(s: Slots, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(s.positions)[0]
  backend = Tensor.backend(s.positions)
  grid = [n, s.capacity]
  queries = try Tensor.broadcast_to(try Tensor.reshape(s.positions, [n, 1]), grid)
  keys = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, s.capacity, 1, Tensor.Dtype.I32, backend), [1, s.capacity]),
    grid,
  )

  blocked = try Tensor.full(grid, Tensor.Values.Float(Nn.MASKED), dtype, backend)
  Tensor.where(try Tensor.lt(queries, keys), blocked, try Tensor.zeros(grid, dtype, backend))
end

## `[capacity, n]`: 1 at row `positions[j]` of column `j`, 0 elsewhere.
def spread_of(s: Slots, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(s.positions)[0]
  grid = [s.capacity, n]
  rows = try Tensor.broadcast_to(
    try Tensor.reshape(
      try Tensor.arange(0, s.capacity, 1, Tensor.Dtype.I32, Tensor.backend(s.positions)),
      [s.capacity, 1],
    ),
    grid,
  )

  at = try Tensor.broadcast_to(try Tensor.reshape(s.positions, [1, n]), grid)
  Tensor.to_dtype(try Tensor.eq(rows, at), dtype)
end

## `fresh` (`[1, n, d]`) written into `old` (`[1, rows, d]`, zero past the
## cache's length; none when empty) at the rows `spread` (`[capacity, n]`)
## names, padded with zeros to `capacity` rows. A product with a 0/1 matrix
## and a sum with zeros, so exact in F32.
def written(old: Optional[Tensor.Tensor], fresh: Tensor.Tensor, spread: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(fresh)
  room = Tensor.shape(spread)[0]
  placed = try Tensor.reshape(
    try Tensor.matmul(spread, try Tensor.reshape(fresh, [shape[1], shape[2]])),
    [1, room, shape[2]],
  )

  case old
    none -> ok(placed)
    some(o) ->
      rows = Tensor.shape(o)[1]
      if rows < room
        gap = try Tensor.zeros([1, room - rows, shape[2]], Tensor.dtype(o), Tensor.backend(o))
        o = try Tensor.concat([o, gap], 1)
      end

      Tensor.add(o, placed)
  end
end

## `m` with every tensor a constant of the current trace.
def traced(m: Model) -> Model
  m.with(
    conv1: m.conv1.with(weight: Trace.constant(m.conv1.weight), bias: lifted(m.conv1.bias)),
    conv2: m.conv2.with(weight: Trace.constant(m.conv2.weight), bias: lifted(m.conv2.bias)),
    encoder_positions: Nn.Embedding(table: Trace.constant(m.encoder_positions.table)),
    encoder: m.encoder.map { |b| traced_encoder(b) },
    encoder_norm: traced_norm(m.encoder_norm),
    tokens: Nn.Embedding(table: Trace.constant(m.tokens.table)),
    decoder_positions: Nn.Embedding(table: Trace.constant(m.decoder_positions.table)),
    decoder: m.decoder.map { |b| traced_decoder(b) },
    decoder_norm: traced_norm(m.decoder_norm),
    logits: traced_dense(m.logits),
  )
end

def traced_encoder(b: Nn.EncoderBlock) -> Nn.EncoderBlock
  Nn.EncoderBlock(
    norm1: traced_norm(b.norm1),
    attention: traced_attention(b.attention),
    norm2: traced_norm(b.norm2),
    mlp: traced_mlp(b.mlp),
  )
end

def traced_decoder(b: Nn.DecoderBlock) -> Nn.DecoderBlock
  Nn.DecoderBlock(
    norm1: traced_norm(b.norm1),
    self_attention: traced_attention(b.self_attention),
    norm2: traced_norm(b.norm2),
    cross_attention: traced_attention(b.cross_attention),
    norm3: traced_norm(b.norm3),
    mlp: traced_mlp(b.mlp),
  )
end

def traced_attention(a: Nn.Attention) -> Nn.Attention
  a.with(
    query: traced_dense(a.query),
    key: traced_dense(a.key),
    value: traced_dense(a.value),
    output: traced_dense(a.output),
  )
end

def traced_mlp(p: Nn.Mlp) -> Nn.Mlp = p.with(up: traced_dense(p.up), down: traced_dense(p.down))

def traced_dense(p: Nn.Dense) -> Nn.Dense = Nn.Dense(w: Trace.constant(p.w), b: lifted(p.b))

def traced_norm(p: Nn.LayerNorm) -> Nn.LayerNorm = p.with(gamma: Trace.constant(p.gamma), beta: Trace.constant(p.beta))

def lifted(t: Optional[Tensor.Tensor]) -> Optional[Tensor.Tensor] = t.map { |x| Trace.constant(x) }

## `c` with every tensor a constant of the current trace.
def traced_cache(c: Cache) -> Cache
  c.with(
    cross: c.cross.map { |h|
      Heard(keys: h.keys.map { |t| Trace.constant(t) }, values: h.values.map { |t| Trace.constant(t) })
    },
    own: c.own.map { |kv| Kv(key: Trace.constant(kv.key), value: Trace.constant(kv.value)) },
  )
end

## Attention of the queries `[n, d_model]` over one block's cached heads, as
## `Tensor.multi_head_attention` computes it, without slicing the keys and
## values again at every step.
def attend(queries: Tensor.Tensor, heard: Heard) -> Result[Tensor.Tensor]
  n = Tensor.shape(queries)[0]
  width = Tensor.shape(heard.values[0])[1]
  scale = try Tensor.full(
    [n, width],
    Tensor.Values.Float(1.0 / Math.sqrt(width.to_f)),
    Tensor.dtype(queries),
    Tensor.backend(queries),
  )

  parts: List[Tensor.Tensor] = []
  for h in 0...heard.keys.size
    q = try Tensor.mul(try Tensor.slice(queries, [0, h * width], [n, (h + 1) * width]), scale)
    parts = parts.push(try Tensor.matmul(try Tensor.softmax(try Tensor.matmul(q, heard.keys[h]), 1), heard.values[h]))
  end

  Tensor.concat(parts, 1)
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("expected Float values, got #{v}")
    end
  end

  ok(out)
end

## Whisper's names for encoder block `p` (`model.encoder.layers.N.`).
def encoder_names(p: String) -> Load.BlockNames
  Load.BlockNames(
    norm1: "#{p}self_attn_layer_norm",
    query: "#{p}self_attn.q_proj",
    key: "#{p}self_attn.k_proj",
    value: "#{p}self_attn.v_proj",
    output: "#{p}self_attn.out_proj",
    norm2: "#{p}final_layer_norm",
    up: "#{p}fc1",
    down: "#{p}fc2",
  )
end

## Whisper's names for decoder block `p` (`model.decoder.layers.N.`).
def decoder_names(p: String) -> Load.DecoderBlockNames
  Load.DecoderBlockNames(
    norm1: "#{p}self_attn_layer_norm",
    query: "#{p}self_attn.q_proj",
    key: "#{p}self_attn.k_proj",
    value: "#{p}self_attn.v_proj",
    output: "#{p}self_attn.out_proj",
    norm2: "#{p}encoder_attn_layer_norm",
    cross_query: "#{p}encoder_attn.q_proj",
    cross_key: "#{p}encoder_attn.k_proj",
    cross_value: "#{p}encoder_attn.v_proj",
    cross_output: "#{p}encoder_attn.out_proj",
    norm3: "#{p}final_layer_norm",
    up: "#{p}fc1",
    down: "#{p}fc2",
  )
end

## The activation `config.json` names.
def activation_of(name: String) -> Result[Nn.Activation]
  return ok(Nn.Activation.GeluErf) if name == "gelu"
  return ok(Nn.Activation.Gelu) if name == "gelu_new"
  error("the activation #{name} is not gelu or gelu_new")
end

def expect_heads(stack: String, width: Int, heads: Int) -> Result[Void]
  if heads < 1 or width % heads != 0
    return error("the #{stack} needs heads >= 1 dividing d_model #{width}, got #{heads}")
  end

  ok()
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: the config needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
end
