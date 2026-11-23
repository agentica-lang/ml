## Reading a byte range of a file in Agentica, for the reference backend and for
## the native backends when `TENSOR_LOAD=agentica` (`tensor/DESIGN.md` §4).

## Whether `Tensor.from_file` reads the ranges for the native backends:
## `TENSOR_LOAD=agentica` in the environment. Anything else, or nothing, leaves
## the native `pread`.
def agentica_read?() -> Bool = System.env("TENSOR_LOAD").unwrap_or("") == "agentica"

## `length` bytes at `offset` of `path`: an `error` for a file that is too short.
def read_range(path: String, offset: Int, length: Int) -> Result[Bytes]
  return error("from_file: the offset is #{offset}, not 0 or more") if offset < 0
  size = (try File.stat(path)).size_bytes
  if offset + length > size
    return error("#{path} has #{size} bytes: the range of #{length} bytes at #{offset} ends past them")
  end

  File.open(path) do |f|
    try File.seek(f, offset)
    parts: List[Bytes] = []
    left = length
    while left > 0
      chunk = try File.read_bytes(f, [left, CHUNK].min.unwrap!)
      return error("#{path} ended while it was read") if chunk.empty?
      parts = parts.push(chunk)
      left = left - chunk.size
    end

    ok(Bytes.concat(parts))
  end
end

private

## The bytes read at a time.
CHUNK = 1048576
