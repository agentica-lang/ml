## The libm functions of `mathx` against Python's `math`. Run from `asr/`.

import Mathx from "../mathx"

INPUTS = [0.001, 0.5, 1.0, 2.0, 3.7, 6.4, 100.0, 701.5]
PEER = (
  "import json, math, sys; xs = json.loads(sys.argv[1]); print(json.dumps([[f(x) for x in xs] for f in (math.exp, math.log, math.log10, math.sin, math.cos)]))"
)

private

def test_mathx_matches_python_math() -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", "-c", PEER, JSON.encode(INPUTS)])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("python3 exited with #{code}: #{out.stderr}") if code != 0
  want = try JSON.list(try JSON.parse(out.stdout))
  got = [
    INPUTS.map { |x| Mathx.exp(x) },
    INPUTS.map { |x| Mathx.log(x) },
    INPUTS.map { |x| Mathx.log10(x) },
    INPUTS.map { |x| Mathx.sin(x) },
    INPUTS.map { |x| Mathx.cos(x) },
  ]

  try Test.eq(want.size, got.size)
  for f in 0...got.size
    row = try JSON.list(want[f])
    for i in 0...INPUTS.size
      w = try JSON.float(row[i])
      if (got[f][i] - w).abs > 1.0e-15 * (1.0 + w.abs)
        return error("function #{f} of #{INPUTS[i]} is #{got[f][i]}, Python says #{w}")
      end
    end
  end

  ok()
end
