## A table read from a CSV or a Parquet file (`table.agc`) and the dataset `Data.from_table` makes of its columns: the
## fixtures `fixtures/table/scores.csv` and `scores.parquet` (written by `make_fixtures.py`, next to the values they must
## read as in `scores.json`) have float, integer, text (with a comma and quotes in it) and incomplete columns of 6 rows;
## the Parquet file has also booleans and keeps the integers and floats in their widths.

import Data from "../data"
import S from "support"
import Table from "../table"
import Tensor from "pkg/tensor/tensor"

FILE = "test/fixtures/table/scores.csv"
PARQUET = "test/fixtures/table/scores.parquet"
BIG_PARQUET = "test/fixtures/table/big.parquet"
BIG_ROWS = 2000
BIG_WIDTH = 1100

private

def expected() -> Result[JSON.Value] = JSON.parse(try File.read_text("test/fixtures/table/scores.json"))

def numbers(name: String) -> Result[List[Float]]
  values: List[Float] = []
  for v in try JSON.list(try JSON.get(try expected(), name))
    values = values.push(try JSON.float(v))
  end

  ok(values)
end

def words(name: String) -> Result[List[String]]
  out: List[String] = []
  for v in try JSON.list(try JSON.get(try expected(), name))
    out = out.push(try JSON.string(v))
  end

  ok(out)
end

## `body` over a new folder, removed after it whatever `body` gave.
def with_dir(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-table-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

## The table of a CSV file that holds `text`.
def written(dir: String, text: String) -> Result[Table.Table]
  try File.write_text("#{dir}/t.csv", text)
  Table.read("#{dir}/t.csv")
end

def error_text[T](r: Result[T]) -> String
  case r
    ok(_) -> ""
    error(e) -> e.message
  end
end

def test_a_csv_file_is_a_table_of_its_columns_in_file_order() -> Result[Void]
  t = try Table.read(FILE)
  try Test.eq(t.names, ["x1", "x2", "count", "label", "name", "gap"])
  try Test.eq(t.rows, 6)
  try Test.eq(try Table.floats(t, "x1"), try numbers("x1"))
  try Test.eq(try Table.floats(t, "x2"), try numbers("x2"))
  try Test.eq(try Table.floats(t, "count"), try numbers("count"))
  try Test.eq(try Table.floats(t, "label"), try numbers("label"))
  try Test.eq(try Table.texts(t, "name"), try words("name"))
  ok()
end

def test_a_column_that_cannot_be_read_as_asked_is_an_error_that_names_it() -> Result[Void]
  t = try Table.read(FILE)
  gap = error_text(Table.floats(t, "gap"))
  try Test.assert(
    gap.include?("gap") and gap.include?("non-number") and gap.include?("row 3 is \"\""),
    "the error said: #{gap}",
  )

  name = error_text(Table.floats(t, "name"))
  try Test.assert(name.include?("name") and name.include?("row 1 is \"ann\""), "the error said: #{name}")
  try Test.assert(error_text(Table.texts(t, "x1")).include?("x1"), "a numeric column read as text")
  try Test.assert(error_text(Table.floats(t, "nope")).include?("nope"), "a missing column")
  try Test.assert(error_text(Table.read("test/fixtures/table/scores.txt")).include?(".parquet"), "an unknown format")
  Test.error?(Table.read("test/fixtures/table/missing.csv"))
end

## The numeric columns are the features and one column the target: inputs `[rows, features]` row by row, targets `[rows]`.
def test_a_dataset_is_the_feature_columns_and_the_target_column() -> Result[Void]
  t = try Table.read(FILE)
  x1 = try numbers("x1")
  x2 = try numbers("x2")
  for backend in [S.REF] + try S.ggml_backends()
    d = try Data.from_table(t, ["x1", "x2"], "label", backend, Tensor.Dtype.I32)
    try Test.eq(Tensor.shape(d.inputs), [6, 2])
    try Test.eq(Tensor.shape(d.targets), [6])
    try Test.eq(Tensor.dtype(d.targets), Tensor.Dtype.I32)
    try Test.eq(Tensor.backend(d.inputs), backend)
    want: List[Tensor.Values] = []
    for i in 0...6
      want = want + [Tensor.Values.Float(x1[i]), Tensor.Values.Float(x2[i])]
    end

    try Test.eq(try Tensor.to_list(d.inputs), want)
    try Test.eq(try Tensor.to_list(d.targets), [0, 1, 1, 0, 2, 2].map { |c| Tensor.Values.Int(c) })
    regression = try Data.from_table(t, ["x1"], "x2", backend)
    try Test.eq(Tensor.dtype(regression.targets), Tensor.Dtype.F32)
  end

  ok()
end

## The loader batches a dataset of a file as it does one of tensors.
def test_a_dataset_of_a_file_goes_through_the_loader() -> Result[Void]
  t = try Table.read(FILE)
  d = try Data.from_table(t, ["x1", "x2"], "label", S.REF, Tensor.Dtype.I32)
  loader = Data.Loader(dataset: d, batch_size: 4, shuffle: false, prefetch: 0)
  try Test.eq(Data.count(loader), 2)
  b = try Data.batch(loader, Data.order(loader, 1), 1)
  try Test.eq(Tensor.shape(b.inputs), [2, 2])
  try Test.eq(try Tensor.to_list(b.targets), [Tensor.Values.Int(2), Tensor.Values.Int(2)])
  ok()
end

## Each refusal names what is wrong.
def test_a_dataset_needs_columns_it_can_read() -> Result[Void]
  t = try Table.read(FILE)
  for bad in [["x1", "gap"], ["x1", "name"], ["x1", "nope"]]
    said = error_text(Data.from_table(t, bad, "label", S.REF))
    try Test.assert(said.include?(bad[1]), "an input #{bad[1]}: #{said}")
  end

  bare = error_text(Data.from_table(t, [], "label", S.REF))
  try Test.assert(bare.include?("input"), "no input: #{bare}")
  text = error_text(Data.from_table(t, ["x1"], "name", S.REF))
  Test.assert(text.include?("name"), "a text target: #{text}")
end

## A target that is not whole, or too large, cannot be class ids: the error names the column and the row.
def test_a_target_that_is_not_a_class_id_is_an_error_that_names_its_row() -> Result[Void]
  t = try Table.read(FILE)
  fraction = error_text(Data.from_table(t, ["x2"], "x1", S.REF, Tensor.Dtype.I32))
  try Test.assert(
    fraction.include?("x1") and fraction.include?("row 1") and fraction.include?("whole"),
    "said: #{fraction}",
  )

  with_dir do |dir|
    huge = try written(dir, "a,b\n1,2\n3,1e30\n")
    said = error_text(Data.from_table(huge, ["a"], "b", S.REF, Tensor.Dtype.I32))
    try Test.assert(said.include?("b") and said.include?("row 2") and said.include?("I32"), "said: #{said}")
    try Test.eq(Tensor.shape((try Data.from_table(huge, ["a"], "b", S.REF)).targets), [2])
    over = try written(dir, "a,b\n1,1e400\n")
    both = error_text(Table.floats(over, "b"))
    try Test.assert(both.include?("b") and both.include?("row 1"), "an infinite cell is text: #{both}")
    ok()
  end
end

## What a CSV file may look like: CRLF line ends, a byte order mark, a quoted line break, a blank line between rows, and no
## row at all; and what it may not: a short row (counted from the first row after the header), a name used twice.
def test_the_shapes_of_a_csv_file_are_read_or_refused() -> Result[Void]
  with_dir do |dir|
    crlf = try written(dir, "a,b\r\n1,x\r\n2,y\r\n")
    try Test.eq(crlf.rows, 2)
    try Test.eq(try Table.floats(crlf, "a"), [1.0, 2.0])
    try Test.eq(try Table.texts(crlf, "b"), ["x", "y"])
    mark = try String.from_bytes(try Bytes.from_list([239, 187, 191]))
    bom = try written(dir, mark + "a,b\n1,x\n")
    try Test.eq(bom.names, ["a", "b"])
    quoted = try written(dir, "a,b\n1,\"two\nlines\"\n2,y\n")
    try Test.eq(try Table.texts(quoted, "b"), ["two\nlines", "y"])
    blank = try written(dir, "a,b\n1,x\n\n2,y\n")
    try Test.eq(blank.rows, 2)
    try Test.eq(try Table.floats(blank, "a"), [1.0, 2.0])
    header_only = try written(dir, "a,b\n")
    try Test.eq(header_only.rows, 0)
    empty = error_text(Data.from_table(header_only, ["a"], "b", S.REF))
    try Test.assert(empty.include?("no rows"), "said: #{empty}")
    try Test.assert(error_text(written(dir, "")).include?("no header"), "an empty file")
    short = error_text(written(dir, "a,b\n1,2\n3\n"))
    try Test.assert(short.include?("row 2") and short.include?("1 cells"), "said: #{short}")
    twice = error_text(written(dir, "a,a\n1,2\n"))
    Test.assert(twice.include?("used twice"), "said: #{twice}")
  end
end

## The same columns from a Parquet file: 32 and 64-bit numbers as floats, booleans as 1.0 and 0.0, text as it is, and a null
## is an error that names the column and the row.
def test_a_parquet_file_is_a_table_of_its_columns_in_file_order() -> Result[Void]
  t = try Table.read(PARQUET)
  try Test.eq(t.names, ["x1", "x2", "count", "label", "flag", "name", "gap", "wild"])
  try Test.eq(t.rows, 6)
  for name in ["x1", "x2", "count", "label"]
    try Test.eq(try Table.floats(t, name), try numbers(name))
  end

  try Test.eq(try Table.floats(t, "flag"), [1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
  try Test.eq(try Table.texts(t, "name"), try words("name"))
  gap = error_text(Table.floats(t, "gap"))
  try Test.assert(gap.include?("gap") and gap.include?("null") and gap.include?("row 3"), "the error said: #{gap}")
  wild = error_text(Table.floats(t, "wild"))
  try Test.assert(
    wild.include?("wild") and wild.include?("finite") and wild.include?("row 2"),
    "the error said: #{wild}",
  )

  gap_text = error_text(Table.texts(t, "gap"))
  try Test.assert(gap_text.include?("gap"), "the error said: #{gap_text}")
  try Test.assert(error_text(Table.texts(t, "x1")).include?("x1"), "a numeric column read as text")
  try Test.assert(error_text(Table.floats(t, "name")).include?("name"), "a text column read as numbers")
  Test.assert(error_text(Table.floats(t, "nope")).include?("nope"), "a missing column")
end

## A dataset of the Parquet file is the dataset of the CSV file with the same columns.
def test_a_dataset_of_a_parquet_file_is_the_one_of_the_csv_file() -> Result[Void]
  csv = try Data.from_table(try Table.read(FILE), ["x1", "x2", "count"], "label", S.REF, Tensor.Dtype.I32)
  parquet = try Data.from_table(try Table.read(PARQUET), ["x1", "x2", "count"], "label", S.REF, Tensor.Dtype.I32)
  try Test.eq(try Tensor.to_list(parquet.inputs), try Tensor.to_list(csv.inputs))
  try Test.eq(try Tensor.to_list(parquet.targets), try Tensor.to_list(csv.targets))
  flagged = try Data.from_table(try Table.read(PARQUET), ["flag", "x1"], "label", S.REF)
  try Test.eq(Tensor.shape(flagged.inputs), [6, 2])
  nulls = error_text(Data.from_table(try Table.read(PARQUET), ["x1", "gap"], "label", S.REF))
  Test.assert(nulls.include?("gap"), "said: #{nulls}")
end

## A file over the 1 MiB that `File.read_text` and `File.read_bytes` take at once is read in chunks: a CSV file of 1.2 MB
## written here, and `big.parquet` (2.2 MiB, `make_big.py`).
def test_a_csv_file_over_one_mib_is_read_whole() -> Result[Void]
  with_dir do |dir|
    path = "#{dir}/t.csv"
    try File.write_text(path, "payload,id\n")
    for i in 0...1200
      try File.append_text(path, "#{"x" * 1000},#{i}\n")
    end

    try Test.assert((try File.stat(path)).size_bytes > 1048576, "the file is not over 1 MiB")
    t = try Table.read(path)
    try Test.eq(t.rows, 1200)
    ids = try Table.floats(t, "id")
    try Test.eq(ids[1199], 1199.0)
    payload = try Table.texts(t, "payload")
    Test.eq(payload[1199].size, 1000)
  end
end

def test_a_parquet_file_over_one_mib_is_read_whole() -> Result[Void]
  try Test.assert((try File.stat(BIG_PARQUET)).size_bytes > 2 * 1048576, "the fixture is not over 2 MiB")
  t = try Table.read(BIG_PARQUET)
  try Test.eq(t.names, ["id", "payload"])
  try Test.eq(t.rows, BIG_ROWS)
  ids = try Table.floats(t, "id")
  try Test.eq(ids[BIG_ROWS - 1], (BIG_ROWS - 1).to_f)
  payload = try Table.texts(t, "payload")
  try Test.eq(payload.size, BIG_ROWS)
  Test.eq(payload[BIG_ROWS - 1].size, BIG_WIDTH)
end
