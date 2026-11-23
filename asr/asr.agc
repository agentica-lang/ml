## The `asr/` package: Whisper's model forward over `tensor/`, `nn/` and
## `hub/`, loaded from a Hugging Face checkpoint (`asr/DESIGN.md`).
##
## `encode` turns a log-mel batch into the encoder output; `decode` turns
## token ids and that output into logits; `listen` and `step` decode one token
## at a time with a KV cache; `transcribe` runs the log-mel frontend
## (`mel.agc`), the encoder and greedy decoding (`greedy.agc`) on up to 30 s of
## audio. Inference only.

import Greedy from "greedy"
import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Mel from "mel"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "tokenizer"

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
end

struct Model
  config: Config
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

## One attention's keys and values, `[1, keys, d_model]` each.
struct Kv
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## What `step` keeps between calls: per decoder block the projected encoder
## output, made once by `listen`, and the self-attention keys and values of
## the `length` tokens so far.
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

## What one step's arena hands back: the last logits, and per block the new
## rows of the self-attention keys and values, `[n, d_model]` row-major.
struct Grown
  logits: List[Float]
  keys: List[List[Float]]
  values: List[List[Float]]
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
    encoder = encoder.push(
      try Load.encoder_block(c, names, config.encoder_attention_heads, Nn.Activation.Gelu, EPS, backend),
    )
  end

  decoder: List[Nn.DecoderBlock] = []
  for i in 0...config.decoder_layers
    names = decoder_names("#{d}layers.#{i}.")
    decoder = decoder.push(
      try Load.decoder_block(c, names, config.decoder_attention_heads, Nn.Activation.Gelu, EPS, backend),
    )
  end

  ok(
    Model(
      config: config,
      conv1: conv1,
      conv2: conv2,
      encoder_positions: encoder_positions,
      encoder: encoder,
      encoder_norm: try Load.layer_norm(c, "#{e}layer_norm", EPS, backend),
      tokens: tokens,
      decoder_positions: decoder_positions,
      decoder: decoder,
      decoder_norm: try Load.layer_norm(c, "#{d}layer_norm", EPS, backend),
      logits: Nn.Dense(w: try Tensor.transpose(tokens.table), b: none),
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

  # Each stage hands only a list out of its arena (as `step` does) and the
  # previous stage's tensor is closed, so a block's intermediates die with it.
  out = [shape[0], (shape[2] + 1) / 2, m.config.d_model]
  start: Result[List[Tensor.Values]] = Native.arena do
    items: List[Tensor.Tensor] = []
    for b in 0...shape[0]
      one = try Tensor.reshape(try Tensor.slice(mel, [b, 0, 0], [b + 1, shape[1], shape[2]]), [shape[1], shape[2]])
      x = try Nn.activate(Nn.Activation.Gelu, try Nn.conv1d(m.conv1, one))
      x = try Nn.activate(Nn.Activation.Gelu, try Nn.conv1d(m.conv2, x))
      items = items.push(try Tensor.transpose(x))
    end

    Tensor.to_list(try Nn.positional(m.encoder_positions, try Tensor.stack(items, 0)))
  end

  h = try rebuilt(start, out, m.encoder_positions.table)
  for block in m.encoder
    done: Result[List[Tensor.Values]] = Native.arena do Tensor.to_list(try Nn.encoder_block(block, h)) end

    fresh = try rebuilt(done, out, h)
    close([h])
    h = fresh
  end

  normed: Result[List[Tensor.Values]] = Native.arena do Tensor.to_list(try Nn.layer_norm(m.encoder_norm, h)) end

  audio = try rebuilt(normed, out, h)
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

  frames = heard[1]
  d = m.config.d_model
  cross: List[Heard] = []
  for block in m.decoder
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

    cross = cross.push(Heard(keys: keys, values: values))
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

  d = m.config.d_model
  backend = Tensor.backend(m.tokens.table)
  # The arena hands back only lists: an arena keeps the resource its value
  # holds and everything that resource comes from, all the way up, so a
  # returned tensor would keep the step's intermediates.
  grown: Result[Grown] = Native.arena do
    rows = try Nn.embedding(
      m.tokens,
      try Tensor.from_list(ids.map { |i| Tensor.Values.Int(i) }, [n], Tensor.Dtype.I64, backend),
    )

    positions = try Tensor.slice(m.decoder_positions.table, [cache.length, 0], [cache.length + n, d])
    h = try Tensor.reshape(try Tensor.add(rows, positions), [1, n, d])
    mask: Optional[Tensor.Tensor] = none
    if n > 1
      mask = some(try Nn.causal_mask(n, Tensor.dtype(h), backend))
    end

    keys: List[List[Float]] = []
    values: List[List[Float]] = []
    for i in 0...m.decoder.size
      block = m.decoder[i]
      a = block.self_attention
      x = try Nn.layer_norm(block.norm1, h)
      k = try Nn.dense(a.key, x)
      v = try Nn.dense(a.value, x)
      keys = keys.push(try floats(k))
      values = values.push(try floats(v))
      if cache.length > 0
        k = try Tensor.concat([cache.own[i].key, k], 1)
        v = try Tensor.concat([cache.own[i].value, v], 1)
      end

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
    ok(Grown(logits: try floats(logits), keys: keys, values: values))
  end

  g = try grown
  dtype = Tensor.dtype(m.tokens.table)
  own: List[Kv] = []
  for i in 0...m.decoder.size
    k = try Tensor.from_list(g.keys[i].map { |x| Tensor.Values.Float(x) }, [1, n, d], dtype, backend)
    v = try Tensor.from_list(g.values[i].map { |x| Tensor.Values.Float(x) }, [1, n, d], dtype, backend)
    if cache.length > 0
      # The arena concatenated the old cache with these rows too, for its own
      # attention, but could not hand that tensor out; it is built again here.
      fresh = [k, v]
      k = try Tensor.concat([cache.own[i].key, k], 1)
      v = try Tensor.concat([cache.own[i].value, v], 1)
      close(fresh)
    end

    own = own.push(Kv(key: k, value: v))
  end

  ok(Stepped(logits: g.logits, cache: Cache(cross: cache.cross, own: own, length: cache.length + n)))
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

## Closes the libtorch tensors of `ts` now, not at the end of an arena; the
## other backends have nothing to close.
def close(ts: List[Tensor.Tensor]) -> Void
  for t in ts
    case t.handle
      Tensor.Handle.Native(h) -> Native.close(h)
      _ -> nil
    end
  end
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

## `values`, what an arena handed out, as a tensor of `shape` with the dtype
## and backend of `like`.
def rebuilt(values: Result[List[Tensor.Values]], shape: List[Int], like: Tensor.Tensor) -> Result[Tensor.Tensor]
  Tensor.from_list(try values, shape, Tensor.dtype(like), Tensor.backend(like))
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
