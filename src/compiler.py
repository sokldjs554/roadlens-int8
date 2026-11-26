"""Compile TFLite for a specific Arm target and inspect emitted CPU/NPU boundaries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tensorflow.lite.python import schema_py_generated as schema


TARGET = "ethos-u55-128"
SYSTEM = "Ethos_U55_High_End_Embedded"
MEMORY = "Shared_Sram"


def operators(path):
    """Decode the primary FlatBuffer subgraph, including Vela custom command streams."""
    data = Path(path).read_bytes()
    if len(data) < 8 or data[4:8] != b"TFL3":
        raise ValueError(f"Expected a TFLite FlatBuffer: {Path(path).name}")
    model = schema.Model.GetRootAsModel(data)
    if model.SubgraphsLength() != 1:
        raise ValueError("This adapter expects one primary inference graph")
    graph = model.Subgraphs(0)
    names = {val: key for key, val in vars(schema.BuiltinOperator).items() if isinstance(val, int)}
    rows = []
    for i in range(graph.OperatorsLength()):
        node = graph.Operators(i)
        code = model.OperatorCodes(node.OpcodeIndex())
        name = names[code.BuiltinCode()]
        if name == "CUSTOM":
            name = code.CustomCode().decode("utf-8")
        rows.append(dict(index=i, op=name, inputs=[int(node.Inputs(j)) for j in range(node.InputsLength())],
                         outputs=[int(node.Outputs(j)) for j in range(node.OutputsLength())]))
    return rows


def placement(rows):
    producers = {}
    crossings = []
    for row in rows:
        device = "npu" if row["op"] == "ethos-u" else "cpu"
        for tensor in row["inputs"]:
            if tensor in producers:
                index, previous = producers[tensor]
                if previous != device:
                    crossings.append(dict(tensor=tensor, producer=index, consumer=row["index"],
                                          direction=f"{previous}->{device}"))
        for tensor in row["outputs"]:
            producers[tensor] = row["index"], device
    return dict(npu_partitions=sum(row["op"] == "ethos-u" for row in rows),
                host_ops=[row["op"] for row in rows if row["op"] != "ethos-u"],
                crossing_edges=crossings)


def compile_model(path, dest):
    path, dest = Path(path).resolve(), Path(dest)
    source = operators(path)
    vela = shutil.which("vela") or str(Path(sys.executable).with_name("vela"))
    flags = ["--accelerator-config", TARGET, "--config", "Arm/vela.ini",
             "--system-config", SYSTEM, "--memory-mode", MEMORY, "--tensor-allocator", "Greedy",
             "--show-cpu-operations", "--verbose-performance", "--enable-debug-db"]
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as temp:
        result = subprocess.run([vela, *flags, "--output-dir", temp, path], capture_output=True,
                                text=True, timeout=180, check=False)
        if result.returncode:
            raise RuntimeError(f"Vela failed ({result.returncode}): {result.stderr or result.stdout}")
        compiled = Path(temp) / (path.stem + "_vela.tflite")
        rows = operators(compiled)
        report = dict(target=TARGET, system=SYSTEM, memory_mode=MEMORY,
                      compiler_flags=flags, input_ops=source, compiled_ops=rows, **placement(rows))
        # Publish this run only after its compiled graph is readable.
        Path(temp, "vela.log").write_text(result.stdout + result.stderr)
        Path(temp, "placement.json").write_text(json.dumps(report, indent=2) + "\n")
        for file in Path(temp).iterdir():
            os.replace(file, dest / file.name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = compile_model(args.model, args.destination)
    print(json.dumps({key: result[key] for key in ("target", "npu_partitions", "host_ops", "crossing_edges")}, indent=2))
