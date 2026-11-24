## `tokenizer/unicode.agc` against Python's `unicodedata` (`unicode_peer.py`).
## Run from `transformers/`.

import Unicode from "../tokenizer/unicode"

## The characters Unicode 15.1 added (CJK extension I, the ideographic
## description characters U+2FFC..U+2FFF, U+31EF): left out of the samples, so
## that the peer agrees on a Python of Unicode 15.0 (3.12, which CI may run)
## as on 3.13's 15.1. Nothing else of 15.1 changes what is checked.
def added_in_15_1?(c: Int) -> Bool = (c >= 0x2EBF0 and c <= 0x2EE5D) or (c >= 0x2FFC and c <= 0x2FFF) or c == 0x31EF

## Precomposed Latin, Greek and Cyrillic, the combining marks on a base, marks
## out of order, Hangul syllables and jamo, compatibility characters (which
## NFD and NFC keep), the singletons that decompose to a letter, and one
## character of each property.
def samples() -> List[List[Int]]
  out: List[List[Int]] = []
  out = out.push((0xC0..0x24F).to_a)
  out = out.push((0x370..0x3FF).to_a)
  out = out.push((0x400..0x4FF).to_a)
  out = out.push((0x1E00..0x1FFF).to_a)
  out = out.push((0xAC00...0xD7A4).select { |c| c % 53 == 0 })
  out = out.push([0x1100, 0x1161, 0x11A8, 0x1112, 0x1161, 0x11AB, 0x1100, 0x1161])
  out = out.push([0x61, 0x323, 0x307, 0x61, 0x307, 0x323, 0x61, 0x31B, 0x323, 0x301, 0x65, 0x301, 0x323])
  out = out.push([0x212B, 0x2126, 0x1E9B, 0x323, 0xFB01, 0x2460, 0x3000, 0xFF21, 0x340, 0x344, 0x958, 0x1D15E])
  out = out.push([0x130, 0x3A3, 0x1C5, 0xDF, 0x1E9E, 0x41, 0x5A, 0x1F88, 0x2160, 0x24B6, 0x10400, 0x1C4])
  out = out.push([
    0x41, 0x30, 0x2E, 0x20, 0x0A, 0xAD, 0x200B, 0x300, 0x2028, 0x85, 0xA0, 0xE000, 0x378, 0x10FFFF, 0x20000, 0x1F600,
  ])

  out = out.push([0x1C, 0x7F, 0xFEFF, 0x2003, 0x3007, 0xB2, 0xBD, 0x660, 0xFF5E, 0x2212, 0x24, 0x60, 0x5E, 0xAA])
  out
end

def test_normalization_and_properties_match_unicodedata() -> Result[Void]
  cases = samples()
  try Test.assert(not cases.any? { |cps| cps.any? { |c| added_in_15_1?(c) } }, "a sample has a 15.1 character")
  want = try JSON.decode(try peer(cases), List[Want])
  try Test.eq(want.size, cases.size)
  for i in 0...cases.size
    cps = cases[i]
    try Test.eq(Unicode.nfd(cps), want[i].nfd)
    try Test.eq(Unicode.nfc(cps), want[i].nfc)
    try Test.eq(Unicode.lower(cps), want[i].lower)
    try Test.eq(kinds(cps), want[i].kinds)
  end

  Test.eq(Unicode.text_of(Unicode.codepoints("añ😀")), "añ😀")
end

private

struct Want
  nfd: List[Int]
  nfc: List[Int]
  lower: List[Int]
  kinds: String
end

def kinds(cps: List[Int]) -> String = cps.map { |c| kind(c) }.join("")

def kind(c: Int) -> String
  return "L" if Unicode.letter?(c)
  return "N" if Unicode.number?(c)
  return "M" if Unicode.mark?(c)
  return "P" if Unicode.punctuation?(c)
  return "S" if Unicode.space?(c)
  return "C" if Unicode.other?(c)
  "."
end

def peer(cases: List[List[Int]]) -> Result[String]
  p = try Process.spawn_captured("python3", ["-I", "test/unicode_peer.py", JSON.encode(cases)])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("unicode_peer.py exited with #{code}: #{out.stderr}") if code != 0
  ok(out.stdout)
end
