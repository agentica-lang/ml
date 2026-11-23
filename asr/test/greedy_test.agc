## The rules of greedy decoding on hand-made logits, and the segments over
## `fixtures/tokenizer.json`. Run from `asr/`.

import Greedy from "../greedy"
import Tokenizer from "../tokenizer"

FIXTURE = "test/fixtures/tokenizer.json"
CLOSE = 0.000000001

## A vocabulary of 16: text 0..4, `eot` 5, specials 6..9 (`<|notimestamps|>`
## is 9), timestamps 10..15; 7 is suppressed, 3 and `eot` not emitted first,
## and the first timestamp is at most 12.
RULES = Greedy.Rules(
  suppress: [7],
  begin_suppress: [3, 5],
  eot: 5,
  no_timestamps: 9,
  timestamp_begin: 10,
  timestamps: true,
  max_initial: 2,
)
FLAT = List.repeat(0.0, 16)

private

## The ids `allowed` leaves open.
def open_ids(r: Greedy.Rules, logits: List[Float], emitted: List[Int]) -> List[Int]
  out = Greedy.allowed(r, logits, emitted)
  (0...out.size).to_a.select { |i| out[i] > Greedy.FORBIDDEN }
end

def test_the_first_token_is_a_timestamp_up_to_the_initial_maximum() -> Result[Void]
  Test.eq(open_ids(RULES, FLAT, []), [10, 11, 12])
end

def test_a_lone_first_timestamp_is_followed_by_text() -> Result[Void]
  Test.eq(open_ids(RULES, FLAT, [10]), [0, 1, 2, 3, 4, 5, 6, 8])
end

def test_timestamps_never_decrease_and_a_new_pair_starts_later() -> Result[Void]
  text = FLAT.set(1, 5.0)
  try Test.eq(open_ids(RULES, text, [10, 1]), [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15])
  try Test.eq(open_ids(RULES, FLAT, [10, 1, 13, 13]), [0, 1, 2, 3, 4, 5, 6, 8])
  Test.eq(open_ids(RULES, text, [10, 1, 13, 13, 2, 14]), [14, 15])
end

## After text and one timestamp only `eot`, the specials and a closing
## timestamp at the same time or later may come.
def test_a_pair_closes_on_a_timestamp_or_eot() -> Result[Void]
  Test.eq(open_ids(RULES, FLAT.set(5, 5.0), [10, 1, 13]), [5, 6, 8, 13, 14, 15])
end

## The five open timestamps together (1 + log 5) beat the best text token.
def test_likelier_timestamps_forbid_the_text() -> Result[Void]
  logits = FLAT.take(10).set(1, 1.0) + List.repeat(1.0, 6)
  Test.eq(open_ids(RULES, logits, [10, 1]), [11, 12, 13, 14, 15])
end

## The four numbers a step graph reads (`Asr.scored`): the first timestamp that may come, no timestamp, no text before
## `eot`, the first token.
def test_state_reads_the_end_of_the_tokens() -> Result[Void]
  try Test.eq(Greedy.state(RULES, []), [10.0, 0.0, 0.0, 1.0])
  try Test.eq(Greedy.state(RULES, [10]), [11.0, 1.0, 0.0, 0.0])
  try Test.eq(Greedy.state(RULES, [10, 1]), [11.0, 0.0, 0.0, 0.0])
  try Test.eq(Greedy.state(RULES, [10, 1, 13]), [13.0, 0.0, 1.0, 0.0])
  try Test.eq(Greedy.state(RULES, [10, 1, 13, 13]), [14.0, 1.0, 0.0, 0.0])
  Test.eq(Greedy.state(RULES, [1, 2]), [10.0, 0.0, 0.0, 0.0])
end

## The mask forbids the suppressed ids, the first token's ones for the first token, and `<|notimestamps|>` when there are
## timestamp rules.
def test_the_mask_forbids_the_ids_that_never_come() -> Result[Void]
  forbidden = ->(r: Greedy.Rules, first: Bool) {
    mask = Greedy.mask(r, 16, first)
    (0...16).to_a.select { |i| mask[i] == Greedy.FORBIDDEN }
  }

  try Test.eq(forbidden.call(RULES, true), [3, 5, 7, 9])
  try Test.eq(forbidden.call(RULES, false), [7, 9])
  Test.eq(forbidden.call(RULES.with(timestamps: false), false), [7])
end

def test_without_timestamps_only_the_suppression_applies() -> Result[Void]
  plain = RULES.with(timestamps: false)
  try Test.eq(open_ids(plain, FLAT, []), [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15])
  Test.eq(open_ids(plain, FLAT, [10]), [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
end

def test_pick_takes_the_first_largest_and_probability_is_a_softmax() -> Result[Void]
  try Test.eq(Greedy.pick([1.0, 3.0, 3.0, -2.0]), 1)
  try Test.assert((Greedy.probability([0.0, 0.0, 0.0, 0.0], 2) - 0.25).abs < CLOSE, "uniform")
  Test.assert((Greedy.probability([1000.0, 1000.0], 0) - 0.5).abs < CLOSE, "large logits")
end

def test_the_default_rules_suppress_the_specials_and_the_blank() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  r = Greedy.rules(tok, none, none, true)
  try Test.eq([r.suppress, r.begin_suppress], [[22, 26, 27, 28, 29, 30], [5, 21]])
  Test.eq(Greedy.rules(tok, some([1]), some([2]), false).suppress, [1])
end

## `<|0.00|>` is 32, `<|0.02|>` 33, `<|0.06|>` 35.
def test_timestamps_give_segments() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  pairs = try Greedy.segments(tok, [32, 4, 33, 33, 6, 16, 35], 1.5, -0.25)
  try Test.eq(
    pairs,
    [
      Greedy.Segment(start: 0.0, stop: 0.02, text: "Hola", tokens: [32, 4, 33], avg_logprob: -0.25),
      Greedy.Segment(start: 0.02, stop: 0.06, text: " mundo.", tokens: [33, 6, 16, 35], avg_logprob: -0.25),
    ],
  )

  try Test.eq(segment(tok, [4, 6]), some([0.0, 1.5]))
  try Test.eq(segment(tok, [33, 4]), some([0.02, 1.5]))
  try Test.eq(segment(tok, [4, 33]), some([0.0, 0.02]))
  try Test.eq(segment(tok, [32, 4, 33]), some([0.0, 0.02]))
  Test.eq(try Greedy.segments(tok, [], 1.5, 0.0), [])
end

## The start and stop of the one segment of `ids` with the text `Hola`, if
## that is what they give.
def segment(tok: Tokenizer.Tokenizer, ids: List[Int]) -> Optional[List[Float]]
  case Greedy.segments(tok, ids, 1.5, 0.0)
    ok(found) ->
      return none if found.size != 1 or not found[0].text.start_with?("Hola") or found[0].tokens != ids
      some([found[0].start, found[0].stop])
    error(_) -> none
  end
end
