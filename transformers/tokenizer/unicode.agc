## What the tokenizers need of Unicode, over lists of code points: the
## properties that Hugging Face's pre-tokenizers read (letter, number,
## nonspacing mark, punctuation, other, white space), lowercase, and the
## normalization forms NFD and NFC. The tables are `unicode_data.agc`'s.

import Data from "unicode_data"

## The code points of `text`.
def codepoints(text: String) -> List[Int] = text.split("").map { |c| c.code }

## The text of `cps`.
def text_of(cps: List[Int]) -> String = cps.map { |c| String.from_code(c) }.join("")

def letter?(cp: Int) -> Bool = in_ranges(Data.LETTER, cp)

def number?(cp: Int) -> Bool = in_ranges(Data.NUMBER, cp)

## A nonspacing mark (Mn): what removing accents removes.
def mark?(cp: Int) -> Bool = in_ranges(Data.MARK_NONSPACING, cp)

## Any mark (Mn, Mc or Me): `\p{M}`.
def any_mark?(cp: Int) -> Bool = in_ranges(Data.MARK, cp)

def punctuation?(cp: Int) -> Bool = in_ranges(Data.PUNCTUATION, cp)

## Control, format, unassigned and private use (the C categories).
def other?(cp: Int) -> Bool = in_ranges(Data.OTHER, cp)

## White_Space: what `\s` is in Oniguruma and `char::is_whitespace` in Rust.
def space?(cp: Int) -> Bool = in_ranges(Data.SPACE, cp)

## `cps` in lowercase, one character at a time (no context: a capital sigma
## is always the small sigma, as Hugging Face's does it).
def lower(cps: List[Int]) -> List[Int]
  out: List[Int] = []
  for cp in cps
    if cp == 0x130
      out = out.push(0x69).push(0x307)
      next
    end

    at = find(Data.LOWER_FROM, cp)
    if at < 0
      out = out.push(cp)
    else
      out = out.push(Data.LOWER_TO[at])
    end
  end

  out
end

## `xs` after `out`, one push at a time: `+` copies both lists when `xs` has
## more than 32 items.
def append(out: List[Int], xs: List[Int]) -> List[Int]
  joined = out
  for x in xs
    joined = joined.push(x)
  end

  joined
end

## The characters of `cps` from `from` up to `to`.
def span(cps: List[Int], from: Int, to: Int) -> List[Int]
  out: List[Int] = []
  for i in from...to
    out = out.push(cps[i])
  end

  out
end

## The canonical decomposition of `cps` with the combining marks in canonical
## order (NFD). Text below U+00C0 has no decomposition and no mark, and comes
## back as it is.
def nfd(cps: List[Int]) -> List[Int]
  return cps if plain?(cps)
  out: List[Int] = []
  for cp in cps
    out = append(out, decomposed(cp))
  end

  ordered(out)
end

## The canonical composition of `cps` (NFC): NFD, then each character joins
## the last starter when nothing of its class or a higher one stands between
## and the pair is a primary composite (Hangul syllables included).
def nfc(cps: List[Int]) -> List[Int]
  return cps if plain?(cps)
  out: List[Int] = []
  starter = -1
  for cp in nfd(cps)
    cc = combining(cp)
    if starter >= 0
      blocked = out.size - 1 != starter and combining(out[out.size - 1]) >= cc
      if not blocked
        joined = compose(out[starter], cp)
        if joined >= 0
          out = out.set(starter, joined)
          next
        end
      end
    end

    out = out.push(cp)
    if cc == 0
      starter = out.size - 1
    end
  end

  out
end

private

## Hangul's arithmetic (Unicode 3.12).
S_BASE = 0xAC00
L_BASE = 0x1100
V_BASE = 0x1161
T_BASE = 0x11A7
L_COUNT = 19
V_COUNT = 21
T_COUNT = 28
N_COUNT = 588
S_COUNT = 11172

## Whether every character is below U+00C0: nothing there decomposes, combines
## or is combined into.
def plain?(cps: List[Int]) -> Bool
  for c in cps
    return false if c >= 0xC0
  end

  true
end

## Whether `cp` is in one of the ranges `[lo, hi, lo, hi, ...]` (sorted).
def in_ranges(table: List[Int], cp: Int) -> Bool
  lo = 0
  hi = table.size / 2
  while lo < hi
    mid = (lo + hi) / 2
    if cp < table[2 * mid]
      hi = mid
    elsif cp > table[2 * mid + 1]
      lo = mid + 1
    else
      return true
    end
  end

  false
end

## The index of `key` in the sorted `keys`, or -1.
def find(keys: List[Int], key: Int) -> Int
  lo = 0
  hi = keys.size
  while lo < hi
    mid = (lo + hi) / 2
    if keys[mid] < key
      lo = mid + 1
    elsif keys[mid] > key
      hi = mid
    else
      return mid
    end
  end

  -1
end

## The full canonical decomposition of one character.
def decomposed(cp: Int) -> List[Int]
  if cp >= S_BASE and cp < S_BASE + S_COUNT
    index = cp - S_BASE
    parts = [L_BASE + index / N_COUNT, V_BASE + (index % N_COUNT) / T_COUNT]
    if index % T_COUNT != 0
      parts = parts.push(T_BASE + index % T_COUNT)
    end

    return parts
  end

  at = find(Data.DECOMPOSED, cp)
  return [cp] if at < 0
  start = Data.DECOMPOSED_AT[at]
  Data.DECOMPOSITION.drop(start + 1).take(Data.DECOMPOSITION[start])
end

## The canonical combining class of `cp`.
def combining(cp: Int) -> Int
  lo = 0
  hi = Data.CLASS_FROM.size
  while lo < hi
    mid = (lo + hi) / 2
    if cp < Data.CLASS_FROM[mid]
      hi = mid
    elsif cp > Data.CLASS_TO[mid]
      lo = mid + 1
    else
      return Data.CLASS_OF[mid]
    end
  end

  0
end

## `cps` with each run of characters of a class above 0 sorted by class, the
## order of equal classes kept.
def ordered(cps: List[Int]) -> List[Int]
  out: List[Int] = []
  run: List[Int] = []
  for cp in cps
    if combining(cp) == 0
      out = append(out, run.sort_by { |c| combining(c) }).push(cp)
      run = []
    else
      run = run.push(cp)
    end
  end

  append(out, run.sort_by { |c| combining(c) })
end

## The primary composite of `a` followed by `b`, or -1.
def compose(a: Int, b: Int) -> Int
  if a >= L_BASE and a < L_BASE + L_COUNT and b >= V_BASE and b < V_BASE + V_COUNT
    return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT
  end

  if a >= S_BASE and a < S_BASE + S_COUNT and (a - S_BASE) % T_COUNT == 0 and b > T_BASE and b < T_BASE + T_COUNT
    return a + (b - T_BASE)
  end

  at = find(Data.COMPOSE_PAIR, a * 2097152 + b)
  return -1 if at < 0
  Data.COMPOSE_TO[at]
end
