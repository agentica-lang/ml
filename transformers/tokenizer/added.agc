## The added tokens of a `tokenizer.json` (`[CLS]`, `<|im_start|>`): text that
## the tokenizer cuts out of its input, longest first, before the rest of its
## pipeline reads what is left. Only tokens that match the raw text are kept
## (`normalized` false, no `lstrip`, `rstrip` or `single_word`), which is what
## BERT and Qwen declare.

import Unicode from "unicode"

## An entry of `added_tokens`, as the file spells it.
struct Entry
  id: Int
  content: String
  special: Bool
  normalized: Bool
  lstrip: Bool
  rstrip: Bool
  single_word: Bool
end

struct Added
  id: Int
  content: String
  special: Bool
  chars: List[Int]
end

## What `truncation` or `padding` holds when the file sets one.
struct Setting
  strategy: Optional[String]
  max_length: Optional[Int]
  direction: Optional[String]
  stride: Optional[Int]
  pad_to_multiple_of: Optional[Int]
end

enum Part
  ## A stretch of text between added tokens, as code points.
  Text(List[Int])
  Token(Int)
end

## An error when the file truncates or pads: the tokenizers here do neither.
def refuse_limits(truncation: Optional[Setting], padding: Optional[Setting]) -> Result[Void]
  return error("the file sets truncation, which is not supported") if truncation.some?
  return error("the file sets padding, which is not supported") if padding.some?
  ok()
end

## The most tokens an encoding has when the file truncates (`max_length`, the special tokens counted), none when it
## does not. Only the plain form is read: cut at the right, `LongestFirst`, no stride. `padding` to the longest of a
## batch is accepted and changes nothing, since one text is its own batch. Any other strategy is an error (a fixed
## length is an object in the file, so it fails when the file is decoded), and so is `pad_to_multiple_of` above 1,
## which pads one text too.
def limit(truncation: Optional[Setting], padding: Optional[Setting]) -> Result[Optional[Int]]
  case padding
    some(p) ->
      if p.strategy != some("BatchLongest")
        return error("the padding strategy #{p.strategy.unwrap_or("")} is not supported, only BatchLongest")
      end

      if p.pad_to_multiple_of.unwrap_or(0) > 1
        return error("padding to a multiple of #{p.pad_to_multiple_of.unwrap_or(0)} is not supported")
      end

    none -> nil
  end

  case truncation
    some(t) ->
      if t.strategy.unwrap_or("LongestFirst") != "LongestFirst"
        return error("the truncation strategy #{t.strategy.unwrap_or("")} is not supported, only LongestFirst")
      end

      return error("truncation at the left is not supported") if t.direction.unwrap_or("Right") != "Right"
      return error("truncation with a stride is not supported") if t.stride.unwrap_or(0) != 0
      max = try t.max_length.to_result("the truncation has no max_length")
      ok(some(max))
    none -> ok(none)
  end
end

## The added token `content` of `id`.
def of(id: Int, content: String, special: Bool) -> Added
  Added(id: id, content: content, special: special, chars: content.split("").map { |c| c.code })
end

## The added tokens that `entries` declare.
def parse(entries: List[Entry]) -> Result[List[Added]]
  out: List[Added] = []
  for e in entries
    if e.normalized or e.lstrip or e.rstrip or e.single_word
      return error("the added token #{e.content} is matched on normalized text or by words, which is not supported")
    end

    return error("the added token #{e.id} has no text") if e.content.empty?
    return error("the added token #{e.content} has the id #{e.id}") if e.id < 0
    out = out.push(of(e.id, e.content, e.special))
  end

  ok(out)
end

## `chars` cut at the added tokens: at each place the longest token that
## starts there wins.
def split(added: List[Added], chars: List[Int]) -> List[Part]
  out: List[Part] = []
  start = 0
  at = 0
  while at < chars.size
    best = longest(added, chars, at)
    if best < 0
      at += 1
    else
      if at > start
        out = out.push(Part.Text(Unicode.span(chars, start, at)))
      end

      out = out.push(Part.Token(added[best].id))
      at += added[best].chars.size
      start = at
    end
  end

  if start < chars.size
    out = out.push(Part.Text(Unicode.span(chars, start, chars.size)))
  end

  out
end

private

## The index of the longest added token that `chars` has at `at`, or -1.
def longest(added: List[Added], chars: List[Int], at: Int) -> Int
  best = -1
  size = 0
  for i in 0...added.size
    t = added[i].chars
    if t.size > size and t[0] == chars[at] and at + t.size <= chars.size and matches(t, chars, at)
      best = i
      size = t.size
    end
  end

  best
end

def matches(token: List[Int], chars: List[Int], at: Int) -> Bool
  for i in 0...token.size
    return false if token[i] != chars[at + i]
  end

  true
end
