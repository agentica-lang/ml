## A table read from a CSV file (`table.agc`) and the dataset `Data.from_table` makes of its columns: the fixture
## `fixtures/table/scores.csv` (written by `make_fixtures.py`, next to the values it must read as in `scores.json`) has
## float, integer, text (with a comma and quotes in it) and incomplete columns of 6 rows.

import Data from "../data"
import S from "support"
import Table from "../table"
import Tensor from "pkg/tensor/tensor"

FILE = "test/fixtures/table/scores.csv"

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
  try Test.assert(gap.include?("gap") and gap.include?("row 3"), "the error said: #{gap}")
  try Test.assert(error_text(Table.floats(t, "name")).include?("name"), "a text column read as numbers")
  try Test.assert(error_text(Table.texts(t, "x1")).include?("x1"), "a numeric column read as text")
  try Test.assert(error_text(Table.floats(t, "nope")).include?("nope"), "a missing column")
  try Test.assert(error_text(Table.read("test/fixtures/table/scores.txt")).include?(".csv"), "an unknown format")
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

def test_a_dataset_needs_columns_it_can_read() -> Result[Void]
  t = try Table.read(FILE)
  try Test.error?(Data.from_table(t, ["x1", "gap"], "label", S.REF))
  try Test.error?(Data.from_table(t, ["x1", "name"], "label", S.REF))
  try Test.error?(Data.from_table(t, ["x1", "nope"], "label", S.REF))
  try Test.error?(Data.from_table(t, [], "label", S.REF))
  Test.error?(Data.from_table(t, ["x1"], "name", S.REF))
end

## A target that is not whole cannot be class ids, and a file that is not a rectangle of one header is refused at read.
def test_ragged_rows_and_repeated_names_and_fractional_classes_are_errors() -> Result[Void]
  t = try Table.read(FILE)
  broken = error_text(Data.from_table(t, ["x2"], "x1", S.REF, Tensor.Dtype.I32))
  try Test.assert(broken.include?("x1") and broken.include?("whole"), "the error said: #{broken}")
  dir = "/tmp/agentica-table-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  try File.write_text("#{dir}/short.csv", "a,b\n1,2\n3\n")
  try File.write_text("#{dir}/twice.csv", "a,a\n1,2\n")
  short = error_text(Table.read("#{dir}/short.csv"))
  twice = error_text(Table.read("#{dir}/twice.csv"))
  _ = Path.remove(dir, 5000)
  try Test.assert(short.include?("row 3"), "the error said: #{short}")
  Test.assert(twice.include?("a"), "the error said: #{twice}")
end
