## Writes the StableHLO text of every case of `stablehlo_cases.agc` into FIXTURES (the snapshots the tests check), and
## each case's plan for IREE as `NAME.plan.json` into PLANS. `make iree-goldens` runs it, then IREE. From `train/`:
##
## ```sh
## agentica tools/stablehlo_emit.agc FIXTURES PLANS
## ```

import Cases from "stablehlo_cases"

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("stablehlo_emit: #{e.message}")
      System.exit(2)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: FIXTURES PLANS") if args.size != 2
  try Dir.mkdir(args[0])
  try Dir.mkdir(args[1])
  for c in try Cases.cases()
    try File.write_text(try Path.join([args[0], "#{c.name}.mlir"]), c.text)
    try File.write_text(try Path.join([args[1], "#{c.name}.plan.json"]), JSON.encode(c.plan))
  end

  ok()
end
