"""Runs StableHLO programs on IREE (CPU) and writes the outputs as goldens.

    python tools/iree_goldens.py PLANS FIXTURES

PLANS holds NAME.plan.json (from train/tools/stablehlo_emit.agc) and FIXTURES holds NAME.mlir. For each plan, the
program runs once per entry of "sets", on the arguments of the plan: a run sets the arguments its entry names, and
the "carry" pairs [output, argument] feed the outputs of a run into the next. FIXTURES/NAME.golden.json gets the
IREE version, the SHA-256 of the NAME.mlir it ran (so a test sees a golden that its text outgrew), the arguments of
the first run and the outputs of every run.

It needs iree-compile and iree-run-module on PATH: `make iree-goldens` runs it from the venv `make iree-venv` makes.
Standard library only.
"""

import ast
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
from importlib import metadata

DTYPES = {"f32": ("<f4", "f"), "i32": ("<i4", "i")}
NPY_MAGIC = b"\x93NUMPY"


def write_npy(path, arg):
    descr, code = DTYPES[arg["dtype"]]
    convert = int if code == "i" else float
    payload = struct.pack("<%d%s" % (len(arg["data"]), code), *[convert(x) for x in arg["data"]])
    shape = tuple(arg["shape"])
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %r, }" % (descr, shape)
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    path.write_bytes(NPY_MAGIC + b"\x01\x00" + struct.pack("<H", len(header)) + header.encode() + payload)


def read_npy(path):
    raw = path.read_bytes()
    if raw[:6] != NPY_MAGIC or raw[6:8] != b"\x01\x00":
        sys.exit("%s is not a version 1 .npy file" % path)
    (size,) = struct.unpack("<H", raw[8:10])
    header = ast.literal_eval(raw[10 : 10 + size].decode())
    dtype = next(name for name, (descr, _) in DTYPES.items() if descr == header["descr"])
    code = DTYPES[dtype][1]
    body = raw[10 + size :]
    data = list(struct.unpack("<%d%s" % (len(body) // struct.calcsize(code), code), body))
    return {"dtype": dtype, "shape": list(header["shape"]), "data": [float(x) for x in data]}


def round_f32(arg):
    if arg["dtype"] == "f32":
        arg["data"] = [struct.unpack("<f", struct.pack("<f", x))[0] for x in arg["data"]]
    return arg


def run(cmd):
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit("%s failed:\n%s%s" % (cmd[0], done.stdout, done.stderr))


def golden(name, plans, fixtures, work):
    plan = json.loads((plans / (name + ".plan.json")).read_text())
    module = work / (name + ".vmfb")
    run(
        [
            "iree-compile",
            "--iree-input-type=stablehlo",
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            str(fixtures / (name + ".mlir")),
            "-o",
            str(module),
        ]
    )
    args = [round_f32(dict(a)) for a in plan["args"]]
    first = [dict(a) for a in args]
    runs = []
    for sets in plan["sets"]:
        for index, data in sets.items():
            args[int(index)]["data"] = round_f32({"dtype": args[int(index)]["dtype"], "data": data})["data"]

        cmd = ["iree-run-module", "--module=" + str(module), "--device=local-task", "--function=main"]
        for i, arg in enumerate(args):
            write_npy(work / ("arg%d.npy" % i), arg)
            cmd.append("--input=@" + str(work / ("arg%d.npy" % i)))

        for path in sorted(work.glob("out*.npy")):
            path.unlink()
        cmd += ["--output=@" + str(work / ("out%03d.npy" % i)) for i in range(plan["outputs"])]
        run(cmd)
        outputs = [read_npy(work / ("out%03d.npy" % i)) for i in range(plan["outputs"])]
        runs.append({"outputs": outputs})
        for out, arg in plan["carry"]:
            args[arg]["data"] = outputs[out]["data"]

    version = json.dumps(metadata.version("iree-base-compiler"))
    digest = hashlib.sha256((fixtures / (name + ".mlir")).read_bytes()).hexdigest()
    lines = ["{", ' "iree": %s,' % version, ' "mlir_sha256": "%s",' % digest, ' "args": [', ",\n".join("  " + json.dumps(a) for a in first), " ],"]
    lines += [' "runs": [', ",\n".join("  " + json.dumps(r) for r in runs), " ]", "}"]
    (fixtures / (name + ".golden.json")).write_text("\n".join(lines) + "\n")


def main():
    plans, fixtures = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as tmp:
        for plan in sorted(plans.glob("*.plan.json")):
            golden(plan.name[: -len(".plan.json")], plans, fixtures, pathlib.Path(tmp))


if __name__ == "__main__":
    main()
