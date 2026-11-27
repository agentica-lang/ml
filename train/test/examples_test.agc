## The programs of `examples/` print what their `.expected` file holds. Run from `train/`, with `agentica` on the PATH.

def output_of(name: String) -> Result[String]
  p = try Process.spawn_captured("agentica", ["examples/#{name}.agc"])
  out = try Process.output(p, 120000)
  code = try Process.wait(p, 120000)
  return error("#{name} exited with #{code}: #{out.stderr}") if code != 0
  ok(out.stdout)
end

def test_the_table_examples_print_their_goldens() -> Result[Void]
  for name in ["table_csv", "table_parquet"]
    try Test.eq(try output_of(name), try File.read_text("examples/#{name}.expected"))
  end

  ok()
end
