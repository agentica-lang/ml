## The rules of Whisper's greedy decoding over one step's logits, as OpenAI's
## and Hugging Face's logits filters run them (`asr/DESIGN.md`): the blank at
## the start, the suppressed tokens and the timestamp rules; then the pick and
## the segments of the emitted tokens. Plain lists, no tensors.

import Tokenizer from "tokenizer"

## The score of a token the rules forbid: finite, so the sums stay finite,
## and far below any logit.
FORBIDDEN = -1.0e30

## What the rules need to know of the vocabulary and the request.
struct Rules
  ## Never emitted (the config's `suppress_tokens`, or the special tokens).
  suppress: List[Int]
  ## Not emitted as the first token (the blank and `eot`).
  begin_suppress: List[Int]
  eot: Int
  no_timestamps: Int
  timestamp_begin: Int
  ## False when the prompt ends with `<|notimestamps|>`: no timestamp rules.
  timestamps: Bool
  ## The first timestamp is at most `timestamp_begin + max_initial`.
  max_initial: Int
end

## One span of the transcript, in seconds.
struct Segment
  start: Float
  stop: Float
  text: String
  ## Its ids, with the timestamps that open and close it, as OpenAI's.
  tokens: List[Int]
  ## OpenAI's `avg_logprob` of the window the segment is in.
  avg_logprob: Float
end

## The rules for `tok`, with the token ids `suppress` and `begin_suppress`
## from the config when it has them. Without `suppress`, only the special
## tokens are suppressed (`sot` and those from `<|translate|>` to just before
## `<|notimestamps|>`), not OpenAI's list of non-speech symbols, which needs
## the encoder of the tokenizer. Without `begin_suppress`, the blank (`Ġ`) and
## `eot` are not emitted first, as OpenAI does.
def rules(
  tok: Tokenizer.Tokenizer,
  suppress: Optional[List[Int]],
  begin_suppress: Optional[List[Int]],
  timestamps: Bool,
) -> Rules
  blank: List[Int] = []
  case tok.pieces.index("Ġ")
    some(i) -> blank = [i]
    none -> blank = []
  end

  Rules(
    suppress: suppress.unwrap_or([tok.sot] + (tok.translate...tok.no_timestamps).to_a),
    begin_suppress: begin_suppress.unwrap_or(blank.push(tok.eot)),
    eot: tok.eot,
    no_timestamps: tok.no_timestamps,
    timestamp_begin: tok.timestamp_begin,
    timestamps: timestamps,
    max_initial: MAX_INITIAL,
  )
end

## `logits` (one per vocabulary id) with the forbidden tokens at `FORBIDDEN`,
## given the tokens `emitted` after the prompt so far.
def allowed(r: Rules, logits: List[Float], emitted: List[Int]) -> List[Float]
  out = logits
  if emitted.empty?
    out = forbid(out, r.begin_suppress)
  end

  out = forbid(out, r.suppress)
  return out if not r.timestamps
  timestamp_rules(r, out, emitted)
end

## The id of the largest logit; the first one on a tie.
def pick(logits: List[Float]) -> Int
  best = 0
  for i in 1...logits.size
    if logits[i] > logits[best]
      best = i
    end
  end

  best
end

## The segments of the emitted tokens of one window of `duration` seconds: a
## timestamp opens a segment and the next one closes it. Text after an opening
## timestamp with no closing one starts at that timestamp, as Hugging Face
## does (OpenAI starts it at 0), and ends at `duration`; text before any
## timestamp starts at 0, and with no timestamps at all the whole text is one
## segment from 0. A segment with no text is left out. Each segment carries
## `avg_logprob`, the window's, as OpenAI's do.
def segments(
  tok: Tokenizer.Tokenizer,
  emitted: List[Int],
  duration: Float,
  avg_logprob: Float,
) -> Result[List[Segment]]
  out: List[Segment] = []
  opened: List[Int] = []
  text: List[Int] = []
  for id in emitted
    case Tokenizer.seconds(tok, id)
      some(at) ->
        out = try close(tok, out, opened, text, [id], at, avg_logprob)
        if opened.empty?
          opened = [id]
        else
          opened = []
        end

        text = []
      none -> text = text.push(id)
    end
  end

  ok(try close(tok, out, opened, text, [], duration, avg_logprob))
end

## The softmax of `logits` at `i`, in doubles.
def probability(logits: List[Float], i: Int) -> Float = Math.exp(log_probability(logits, i))

## The log-softmax of `logits` at `i`, in doubles.
def log_probability(logits: List[Float], i: Int) -> Float = logits[i] - log_sum_exp(logits)

private

## OpenAI's `max_initial_timestamp` of 1 s.
MAX_INITIAL = 50

## OpenAI's `ApplyTimestampRules`: timestamps come in pairs (except right
## before `eot`), never decrease, the first one is at most `max_initial`, and
## when the timestamps together are likelier than the best text token, a
## timestamp must come.
def timestamp_rules(r: Rules, logits: List[Float], emitted: List[Int]) -> List[Float]
  out = logits.set(r.no_timestamps, FORBIDDEN)
  n = emitted.size
  last = n >= 1 and emitted[n - 1] >= r.timestamp_begin
  penultimate = n < 2 or emitted[n - 2] >= r.timestamp_begin
  if last and penultimate
    out = forbid_range(out, r.timestamp_begin, out.size)
  elsif last
    out = forbid_range(out, 0, r.eot)
  end

  stamps = emitted.select { |id| id >= r.timestamp_begin }
  if not stamps.empty?
    # A pair may close on the same time it opened; a new one starts later.
    floor = stamps.last
    if not last or penultimate
      floor += 1
    end

    out = forbid_range(out, r.timestamp_begin, floor)
  end

  if n == 0
    out = forbid_range(out, 0, r.timestamp_begin)
    out = forbid_range(out, r.timestamp_begin + r.max_initial + 1, out.size)
  end

  text = out.take(r.timestamp_begin).max.unwrap_or(FORBIDDEN)
  return forbid_range(out, 0, r.timestamp_begin) if log_sum_exp(out.drop(r.timestamp_begin)) > text
  out
end

## `log(sum(exp(xs)))`, shifted by the maximum.
def log_sum_exp(xs: List[Float]) -> Float
  top = xs.max.unwrap_or(FORBIDDEN)
  sum = 0.0
  for x in xs
    sum += Math.exp(x - top)
  end

  top + Math.log(sum)
end

def forbid(logits: List[Float], ids: List[Int]) -> List[Float]
  out = logits
  for id in ids
    if id >= 0 and id < out.size
      out = out.set(id, FORBIDDEN)
    end
  end

  out
end

## `logits` with the ids `from...to` forbidden.
def forbid_range(logits: List[Float], from: Int, to: Int) -> List[Float]
  return logits if from >= to
  logits.take(from) + List.repeat(FORBIDDEN, [to, logits.size].min.unwrap! - from) + logits.drop(to)
end

## `out` with the segment of `text` between the timestamp `opened` (none:
## 0) and `stop`, unless `text` is empty.
def close(
  tok: Tokenizer.Tokenizer,
  out: List[Segment],
  opened: List[Int],
  text: List[Int],
  closing: List[Int],
  stop: Float,
  avg_logprob: Float,
) -> Result[List[Segment]]
  return ok(out) if text.empty?
  start = 0.0
  for id in opened
    start = Tokenizer.seconds(tok, id).unwrap_or(0.0)
  end

  segment = Segment(
    start: start,
    stop: stop,
    text: try Tokenizer.decode(tok, text),
    tokens: opened + text + closing,
    avg_logprob: avg_logprob,
  )

  ok(out.push(segment))
end
