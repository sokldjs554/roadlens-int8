"""Time a prepared PNG in memory through the binary road mask on CPU."""
import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np
from PIL import __version__ as pillow_version

from .bench import CUTOFF, MODELS, SEED, cpu_name, write_csv, write_json
from .data import decode_rgb
from .runtime import RoadRuntime


def profile(archive_path, out):
    art = Path(__file__).resolve().parents[1] / "artifacts"
    with archive_path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != json.loads((art / "training.json").read_text())["data_sha256"]:
            raise ValueError("Unexpected KITTI ZIP checksum")
    if out.exists():
        raise FileExistsError(out)
    ids = json.loads((art / "split.json").read_text())["test"]
    with ZipFile(archive_path) as archive:
        frames = {key: archive.read(f"data_road/training/image_2/{key}.png") for key in ids}
    runtimes = {m: RoadRuntime(art / f"{m}.tflite") for m in MODELS}
    for runtime in runtimes.values():
        for key in ids[:20]:
            runtime.predict(decode_rgb(frames[key]))
    rows, summary = [], {}
    rng = random.Random(SEED)
    mask_hashes = {}
    for repeat in range(10):
        for key in rng.sample(ids, len(ids)):
            for name in rng.sample(MODELS, len(MODELS)):
                png = frames[key]
                runtime = runtimes[name]
                start = perf_counter_ns()
                rgb = decode_rgb(png)
                prepared = perf_counter_ns()
                prob, invoke = runtime.predict(rgb)
                predicted = perf_counter_ns()
                mask = prob >= CUTOFF
                end = perf_counter_ns()
                digest = hashlib.sha256(mask.tobytes()).hexdigest()
                if mask_hashes.setdefault((key, name), digest) != digest:
                    raise ValueError(f"Unstable mask for {key} / {name}")
                rows.append(dict(model=name, scene=key, repeat=repeat + 1, prepare_ns=prepared-start,
                                 invoke_ns=invoke, adapter_ns=predicted-prepared-invoke,
                                 threshold_ns=end-predicted, total_ns=end-start))
    for name in MODELS:
        subset = [row for row in rows if row["model"] == name]
        summary[name] = {}
        for metric in ("prepare", "invoke", "adapter", "threshold", "total"):
            latency = [row[metric + "_ns"] / 1e6 for row in subset]
            summary[name][metric] = dict(zip(("p50_ms", "p95_ms"), np.percentile(latency, [50, 95])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "profile"
        stage.mkdir()
        write_csv(stage / "timings.csv", rows)
        write_json(stage / "summary.json", dict(cpu=cpu_name(), pillow=pillow_version, evaluation=len(ids),
                   warmup=20, repeats=10, threads=1, seed=SEED, models=summary))
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profile(args.archive, args.out)
