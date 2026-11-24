## The added tokens of a `tokenizer.json` (`[CLS]`, `<|im_start|>`): text that
## the tokenizer cuts out of its input, longest first, before the rest of its
## pipeline reads what is left. Only tokens that match the raw text are kept
## (`normalized` false, no `lstrip`, `rstrip` or `single_word`), which is what
## BERT and Qwen declare.

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

enum Part
  ## A stretch of text between added tokens, as code points.
  Text(List[Int])
  Token(Int)
end

## The added tokens that `entries` declare.
def parse(entries: List[Entry]) -> Result[List[Added]]
  out: List[Added] = []
  for e in entries
    if e.normalized or e.lstrip or e.rstrip or e.single_word
      return error("the added token #{e.content} is matched on normalized text or by words, which is not supported")
    end

    return error("the added token #{e.id} has no text") if e.content.empty?
    out = out.push(
      Added(id: e.id, content: e.content, special: e.special, chars: e.content.split("").map { |c| c.code }),
    )
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
        out = out.push(Part.Text(chars.drop(start).take(at - start)))
      end

      out = out.push(Part.Token(added[best].id))
      at += added[best].chars.size
      start = at
    end
  end

  if start < chars.size
    out = out.push(Part.Text(chars.drop(start)))
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
