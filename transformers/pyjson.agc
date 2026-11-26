## A JSON value as the text that Python's `json.dumps(value, ensure_ascii=False)` writes (separators `, ` and `: `, no
## escaping of non-ASCII), which is what the `tojson` filter of Hugging Face chat templates gives. `JSON.encode` writes
## it compact.

## The text of `value`.
def dumps(value: JSON.Value) -> String
  case JSON.node(value)
    JSON.Node.Null -> "null"
    JSON.Node.Bool(b) -> bool_text(b)
    JSON.Node.Int(n) -> "#{n}"
    JSON.Node.BigInt(text) -> text
    JSON.Node.Float(f) -> float_text(f)
    JSON.Node.String(s) -> quoted(s)
    JSON.Node.Array(items) -> "[" + items.map { |item| dumps(item) }.join(", ") + "]"
    JSON.Node.Object(fields) -> "{" + fields.keys.map { |k| quoted(k) + ": " + dumps(fields[k]) }.join(", ") + "}"
  end
end

private

## `true` or `false`.
def bool_text(b: Bool) -> String
  return "true" if b
  "false"
end

## Python's `repr` of a float: the shortest digits that read back, in the decimal form for exponents from -4 to 15 and
## as `1e+16` or `2.5e-07` outside it.
def float_text(f: Float) -> String
  text = f.to_s
  sign = ""
  if text.start_with?("-")
    sign = "-"
    text = text.slice(1, text.size - 1)
  end

  parts = text.split("e")
  exponent = 0
  if parts.size == 2
    written = parts[1]
    if written.start_with?("+")
      written = written.slice(1, written.size - 1)
    end

    exponent = written.to_i.unwrap_or(0)
  end

  whole = parts[0].split(".")
  digits = whole[0]
  if whole.size == 2
    digits = digits + whole[1]
  end

  point = whole[0].size + exponent
  lead = 0
  while lead < digits.size and digits.slice(lead, 1) == "0"
    lead += 1
  end

  digits = digits.slice(lead, digits.size - lead)
  point -= lead
  while digits.size > 0 and digits.slice(digits.size - 1, 1) == "0"
    digits = digits.slice(0, digits.size - 1)
  end

  return sign + "0.0" if digits.empty?
  magnitude = point - 1
  if magnitude < -4 or magnitude >= 16
    rest = ""
    if digits.size > 1
      rest = "." + digits.slice(1, digits.size - 1)
    end

    return sign + digits.slice(0, 1) + rest + "e" + exponent_text(magnitude)
  end

  return sign + "0." + "0" * (0 - point) + digits if point <= 0
  return sign + digits + "0" * (point - digits.size) + ".0" if point >= digits.size
  sign + digits.slice(0, point) + "." + digits.slice(point, digits.size - point)
end

def exponent_text(n: Int) -> String
  sign = "+"
  if n < 0
    sign = "-"
  end

  size = [n, 0 - n].max.unwrap!
  return sign + "0#{size}" if size < 10
  sign + "#{size}"
end

## `s` in double quotes: the quote, the backslash and the control characters escaped.
def quoted(s: String) -> String = "\"" + s.split("").map { |c| escaped_char(c) }.join("") + "\""

def escaped_char(c: String) -> String
  code = c.code
  case code
    34 -> "\\\""
    92 -> "\\\\"
    8 -> "\\b"
    9 -> "\\t"
    10 -> "\\n"
    12 -> "\\f"
    13 -> "\\r"
    _ -> escaped(c, code)
  end
end

def escaped(c: String, code: Int) -> String
  return format("\\u%04x", code) if code < 32
  c
end
