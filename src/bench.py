"""Paired KITTI road segmentation benchmark. Run from the repository root."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns
import zipfile

import numpy as np
from PIL import Image, ImageFilter

from src.data import load_sample
from src.runtime import RoadRuntime

MODELS = ("fp32", "int8")
VARIANTS = ("clean", "dim", "occluded")
LABELS = {"um": "표시 차선 도로 (UM)", "umm": "다차선 도로 (UMM)", "uu": "무표시 도로 (UU)",
          "dim": "합성 저조도", "occluded": "합성 중앙 가림"}
SEED = 731
CUTOFF = 0.5


def altered(rgb, variant):
    """Run the same deterministic sensor perturbation at either image resolution."""
    if variant == "clean":
        return rgb
    if variant == "dim":
        return rgb * np.float32(.35)
    if variant != "occluded":
        raise ValueError(f"Unknown variant {variant}")
    # Fixed sensor obstruction covers 1/4 of width and 1/4 of height.
    h, w = rgb.shape[:2]
    result = rgb.copy()
    result[h // 2:3 * h // 4, 3 * w // 8:5 * w // 8] = .25
    return result


def counts(prob, target, valid):
    pred = prob >= CUTOFF
    return dict(tp=int(np.sum(pred & target & valid)), fp=int(np.sum(pred & ~target & valid)),
                fn=int(np.sum(~pred & target & valid)), tn=int(np.sum(~pred & ~target & valid)))


def scores(confusion):
    tp, fp, fn, tn = (confusion[c] for c in ("tp", "fp", "fn", "tn"))
    return dict(iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                accuracy=(tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else None)


def pooled(rows, model):
    return scores({c: sum(r[f"{c}_{model}"] for r in rows) for c in ("tp", "fp", "fn", "tn")})


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def linux_rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Expected Linux /proc VmRSS")


def cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.machine()


def worker(args, ids):
    import tensorflow as tf

    # Small, identical pre-load footprint. Model memory excludes the later scene cache.
    with zipfile.ZipFile(args.archive) as archive:
        first = load_sample(archive, ids[0])[0]
    before = linux_rss_mib()
    path = args.artifacts / f"{args.worker}.tflite"
    runtime = RoadRuntime(path)
    expected = "float32" if args.worker == "fp32" else "int8"
    metadata = runtime.describe()
    if metadata["input"]["dtype"] != expected:
        raise ValueError(f"{path} does not have expected {expected} input")
    for _ in range(args.warmup):
        runtime.predict(first)
    resident = linux_rss_mib()
    # The allocation/invoke log and the interpreter execution plan record delegation.
    delegates = sum(op["op_name"] == "DELEGATE" for op in runtime.interpreter._get_ops_details())
    info = dict(**metadata, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rss_mib=resident, rss_before_mib=before, rss_delta_mib=resident - before,
                delegated_partitions=delegates, tensorflow=tf.__version__, numpy=np.__version__)
    with zipfile.ZipFile(args.archive) as archive:
        inputs = {id: load_sample(archive, id)[0] for id in ids}
    rng = np.random.default_rng(SEED)
    timings = []
    for repeat in range(1, args.repeats + 1):
        for pos in rng.permutation(len(ids)):
            id = ids[pos]
            start = perf_counter_ns()
            _, invoke_ns = runtime.predict(inputs[id])
            whole_ns = perf_counter_ns() - start
            timings.append(dict(model=args.worker, scene=id, repeat=repeat, invoke_ns=invoke_ns,
                                frame_ns=whole_ns))
    np.savez(args.scratch / f"{args.worker}.npz",
             **{f"{id}_{v}": runtime.predict(altered(inputs[id], v))[0] for id in ids for v in VARIANTS})
    write_csv(args.scratch / f"{args.worker}.csv", timings)
    write_json(args.scratch / f"{args.worker}.json", info)


def overlay(base, mask):
    plane = Image.fromarray(mask).resize(base.size, Image.Resampling.NEAREST).convert("L")
    tint = Image.new("RGB", base.size, "#44e6cf")
    filled = Image.composite(Image.blend(base, tint, .48), base, plane)
    edge = np.asarray(plane) > np.asarray(plane.filter(ImageFilter.MinFilter(3)))
    return Image.composite(Image.new("RGB", base.size, "#b1fff0"), filled, Image.fromarray(edge))


def scene_key(row):
    return f"{row['source_id']}_{row['variant']}"


def make_gallery(archive, rows, predictions, web):
    clean = [r for r in rows if r["variant"] == "clean" and r["iou_int8"] is not None]
    selected = []
    for kind in ("um", "umm", "uu"):
        order = sorted((r for r in clean if r["road_type"] == kind),
                       key=lambda r: (r["iou_int8"], r["source_id"]))
        selected.append((order[(len(order)-1)//2], "도로 유형별 INT8 IoU 중앙 순위"))
    worst = min(clean, key=lambda r: (r["iou_int8"], r["source_id"]))
    if worst not in [r for r, _ in selected]:
        selected.append((worst, "원본 장면 최저 INT8 IoU"))
    for variant in ("dim", "occluded"):
        stress = [r for r in rows if r["variant"] == variant and r["iou_int8"] is not None]
        row = min(stress, key=lambda r: (r["iou_int8"], r["source_id"]))
        selected.append((row, "해당 합성 조건 최저 INT8 IoU"))
    assets = web / "scenes"
    assets.mkdir(exist_ok=True, parents=True)
    gallery = []
    for row, reason in selected:
        id, variant = row["source_id"], row["variant"]
        key = scene_key(row)
        with Image.open(io.BytesIO(archive.read(f"data_road/training/image_2/{id}.png"))) as raw:
            source = raw.convert("RGB")
        if variant != "clean":
            values = altered(np.asarray(source, np.float32) / 255, variant)
            source = Image.fromarray(np.rint(values * 255).astype(np.uint8))
        _, target, valid = load_sample(archive, id)
        pictures = {"input": source, "truth": overlay(source, target & valid)}
        for model in MODELS:
            pictures[model] = overlay(source, (predictions[model][key] >= CUTOFF) & valid)
        scene = dict(id=id if variant == "clean" else key, source_id=id,
                     condition=LABELS[row["road_type"] if variant == "clean" else variant],
                     selection=reason, iou_fp32=row["iou_fp32"], iou_int8=row["iou_int8"],
                     pixel_disagreement=row["pixel_disagreement"])
        for name, pic in pictures.items():
            path = assets / f"{key}_{name}.jpg"
            pic.save(path, quality=87, optimize=True, subsampling=0)
            scene[name] = path.relative_to(web).as_posix()
        gallery.append(scene)
    return gallery


def paired_rows(archive, ids, predictions):
    rows = []
    for id in ids:
        _, target, valid = load_sample(archive, id)
        size = int(np.sum(valid))
        if not size:
            raise ValueError(f"No labelled pixels in {id}")
        for variant in VARIANTS:
            key = f"{id}_{variant}"
            p, q = (predictions[m][key] for m in MODELS)
            changed = int(np.sum(((p >= CUTOFF) != (q >= CUTOFF)) & valid))
            absolute = float(np.abs(p - q)[valid].sum(dtype=np.float64))
            row = dict(source_id=id, road_type=id.split("_")[0], variant=variant,
                       valid_pixels=size, road_area_fraction=float((target & valid).mean()),
                       changed_pixels=changed, pixel_disagreement=changed / size,
                       abs_probability_error_sum=absolute, mean_abs_probability_error=absolute / size)
            for m, prob in zip(MODELS, (p, q)):
                confusion = counts(prob, target, valid)
                for k, value in (confusion | scores(confusion)).items():
                    row[f"{k}_{m}"] = value
            rows.append(row)
    return rows


def group(name, kind, rows):
    return dict(name=name, kind=kind, n=len(rows), **{m: pooled(rows, m)["iou"] for m in MODELS})


def aggregate(args, ids):
    args.web.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(args.archive) as archive:
        with np.load(args.scratch / "fp32.npz") as p, np.load(args.scratch / "int8.npz") as q:
            predictions = dict(fp32=p, int8=q)
            rows = paired_rows(archive, ids, predictions)
            gallery = make_gallery(archive, rows, predictions, args.web)
    clean = [r for r in rows if r["variant"] == "clean"]
    groups = [group(LABELS[k], "natural", [r for r in clean if r["road_type"] == k])
              for k in ("um", "umm", "uu")]
    by_area = sorted(clean, key=lambda r: (r["road_area_fraction"], r["source_id"]))
    low_area = by_area[:int(np.ceil(len(clean) / 4))]
    groups.append(group("도로 면적 하위 25%", "natural", low_area))
    for variant in ("dim", "occluded"):
        groups.append(group(LABELS[variant], "synthetic", [r for r in rows if r["variant"] == variant]))
    models = {}
    timings = []
    for m in MODELS:
        with (args.scratch / f"{m}.csv").open() as f:
            timing = list(csv.DictReader(f))
        timings.extend(timing)
        info = json.loads((args.scratch / f"{m}.json").read_text())
        models[m] = info | pooled(clean, m)
        for field, prefix in (("invoke_ns", ""), ("frame_ns", "frame_")):
            ms = np.array([int(r[field]) for r in timing]) / 1e6
            for percentile in (50, 95):
                models[m][f"{prefix}p{percentile}_ms"] = float(np.percentile(ms, percentile))
    pixels = sum(r["valid_pixels"] for r in clean)
    result = dict(device=f"{cpu_name()} · {platform.system()} {platform.machine()}",
                  n=len(clean), repeats=args.repeats, warmup=args.warmup, threads=1, batch_size=1,
                  timing_order_seed=SEED, threshold=CUTOFF, total_valid_pixels=pixels,
                  pixel_disagreement=sum(r["changed_pixels"] for r in clean) / pixels,
                  mean_abs_probability_error=sum(r["abs_probability_error_sum"] for r in clean) / pixels,
                  low_area_max_fraction=low_area[-1]["road_area_fraction"],
                  models=models, groups=groups, scenes=gallery)
    write_csv(args.artifacts / "latency.csv", timings)
    write_csv(args.artifacts / "scene_metrics.csv", rows)
    write_json(args.artifacts / "benchmark.json", result)
    write_json(args.web / "results.json", result)
    print(json.dumps({m: {k: models[m][k] for k in ("iou", "accuracy", "p50_ms", "p95_ms", "rss_mib")}
                      for m in MODELS}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--artifacts", default=Path("artifacts"), type=Path)
    parser.add_argument("--web", default=Path("web"), type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.repeats, args.warmup) < 1:
        parser.error("warmup and repeats must be positive")
    split = json.loads((args.artifacts / "split.json").read_text())
    ids = split["test"]
    if not ids or len(set(ids)) != len(ids) or set(ids) & (set(split["train"]) | set(split["cal"])):
        raise ValueError("Expected unique held-out IDs disjoint from training and calibration")
    if args.worker:
        worker(args, ids)
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    with tempfile.TemporaryDirectory() as temp:
        args.scratch = Path(temp)
        for model in MODELS:
            subprocess.run([sys.executable, "-m", "src.bench", *sys.argv[1:], "--worker", model,
                            "--scratch", temp], env=env, check=True)
        aggregate(args, ids)


if __name__ == "__main__":
    main()
