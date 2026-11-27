## Reads `test/fixtures/table/scores.parquet` (run from `train/`) into a dataset of two features and a class id, and takes a
## batch of it through a `Loader`.

import Data from "../data"
import Table from "../table"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  table = Table.read("test/fixtures/table/scores.parquet").unwrap!
  puts(table.names)
  puts(Table.texts(table, "name").unwrap!)
  puts(Table.floats(table, "gap"))
  puts(Table.floats(table, "flag").unwrap!)
  dataset = Data.from_table(table, ["x1", "x2"], "label", Tensor.Backend.Reference, Tensor.Dtype.I32).unwrap!
  loader = Data.Loader(dataset: dataset, batch_size: 4, shuffle: false, prefetch: 0)
  puts(Data.count(loader))
  batch = Data.batch(loader, Data.order(loader, 1), 0).unwrap!
  puts(Tensor.shape(batch.inputs))
  puts(Tensor.to_list(batch.targets).unwrap!)
end
