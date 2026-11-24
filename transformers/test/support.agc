## Helpers of the tokenizer tests (not a test file itself).

## `text` with the `n`th (from 0) occurrence of `needle` replaced by `with`;
## an error when there is no such occurrence, so a test never "refuses" a file
## it did not change.
def edited(text: String, needle: String, with: String, n: Int) -> Result[String]
  parts = text.split(needle)
  return error("no occurrence #{n} of #{needle}") if parts.size <= n + 1
  out = parts[0]
  for i in 1...parts.size
    if i - 1 == n
      out = out + with + parts[i]
    else
      out = out + needle + parts[i]
    end
  end

  ok(out)
end

## Every occurrence of `needle` replaced, an error when there is none.
def replaced(text: String, needle: String, with: String) -> Result[String]
  return error("no #{needle} in the file") if not text.include?(needle)
  ok(text.replace(needle, with))
end
