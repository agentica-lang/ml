## Reading a `tokenizer.json`.

## The whole file at `path`, read in chunks (`File.read_text` stops at 1 MiB).
def read_text(path: String) -> Result[String]
  bytes: Result[Bytes] = File.open(path) do |f|
    parts: List[Bytes] = []
    while true
      chunk = try File.read_bytes(f, 1048576)
      break if chunk.empty?
      parts = parts.push(chunk)
    end

    ok(Bytes.concat(parts))
  end

  String.from_bytes(try bytes)
end

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
end
