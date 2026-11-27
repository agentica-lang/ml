## A table of named columns read from a file, for `Data.from_table`: a CSV file (the `csv` package of
## agentica-lang/packages) with a header row, read whole. A column is decoded when it is asked for, through the `load`
## function of the table and not into a field of each column: that is the shape a format that holds its columns apart
## (Parquet) fits, which decodes only the columns a dataset uses and, later, a part of the rows. A column is numbers when
## every cell is a finite number, else text (so an empty cell, or `1e400`, makes its column text); a column read as the
## other kind is an error that names the first cell that does not fit. Rows are counted from 1: row 1 is the first one
## after the header.

import Csv from "pkg/csv/csv"

enum Column
  Numbers(List[Float])
  Texts(List[String])
end

## `names` are the columns in file order, `rows` the number of rows, and `load` decodes one column by name.
struct Table
  names: List[String]
  rows: Int
  load: Fn[String, Result[Column]]
end

## The table of the CSV file at `path`: the first row names the columns, every row has as many cells, and a name is
## used once. A file that ends with a line break has no extra row, a byte order mark at its start is dropped, and a line
## with nothing on it is skipped when the table has more than one column.
def read(path: String) -> Result[Table]
  return error("#{path}: a table is a .csv file") if not path.end_with?(".csv")

  lines = try Csv.parse(try File.read_text(path))
  return error("#{path}: no header row") if lines.empty?

  names = lines[0]
  return error("#{path}: a column name is used twice in #{names}") if names.uniq.size != names.size

  rows = lines.drop(1).select { |row| names.size == 1 or row != [""] }
  for i in 0...rows.size
    if rows[i].size != names.size
      return error("#{path}: row #{i + 1} has #{rows[i].size} cells, the header #{names.size}")
    end
  end

  ok(Table(names: names, rows: rows.size, load: ->(name: String) { column_of(names, rows, name) }))
end

## The column `name` as numbers.
def floats(t: Table, name: String) -> Result[List[Float]]
  case try loaded(t, name)
    Column.Numbers(xs) -> ok(xs)
    Column.Texts(xs) -> error("column #{name} has a non-number cell (#{first_cell(xs)})")
  end
end

## The column `name` as text.
def texts(t: Table, name: String) -> Result[List[String]]
  case try loaded(t, name)
    Column.Texts(xs) -> ok(xs)
    Column.Numbers(_) -> error("column #{name} is numbers, not text")
  end
end

private

def loaded(t: Table, name: String) -> Result[Column]
  return error("no column #{name}, the table has #{t.names}") if not t.names.include?(name)
  t.load.call(name)
end

## The cells of the column `name` as numbers when every one is, else as they are.
def column_of(names: List[String], rows: List[List[String]], name: String) -> Result[Column]
  at = names.index(name).unwrap!
  cells = rows.map { |row| row[at] }
  numbers: List[Float] = []
  for cell in cells
    x = number(cell)
    return ok(Column.Texts(cells)) if x.none?
    numbers = numbers.push(x.unwrap!)
  end

  ok(Column.Numbers(numbers))
end

## The first cell of a text column that is not a number, with its row: the one that made the column text.
def first_cell(cells: List[String]) -> String
  for i in 0...cells.size
    return "row #{i + 1} is #{cells[i].inspect}" if number(cells[i]).none?
  end

  "no cell"
end

## The finite number in `cell`, if there is one.
def number(cell: String) -> Optional[Float]
  case cell.strip.to_f
    ok(x) ->
      return none if x - x != 0.0
      some(x)
    error(_) -> none
  end
end
