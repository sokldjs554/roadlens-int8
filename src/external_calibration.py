"""Evaluate the seven original KITTI-calibrated models on a pinned CamVid test split."""
import argparse
import csv
import hashlib
import io
import json
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .bench import counts, scores, write_csv, write_json
from .calibration_experiment import ARTIFACTS, MARGIN_PP, sha256
from .data import HEIGHT, WIDTH, decode_rgb
from .runtime import RoadRuntime
from .train import convert, keras

REV = "bb68b64e72cb004b3ed927fee0f5624dd8097498"
SEQUENCES = {"0001TP": 62, "Seq05VD": 171}


def labels(png):
    with Image.open(io.BytesIO(png)) as mask:
        ids = np.asarray(mask)
        if mask.mode not in ("L", "P") or ids.max() > 11:
            raise ValueError("Expected CamVid class IDs 0 through 11")
        small = np.asarray(mask.resize((WIDTH, HEIGHT), Image.Resampling.NEAREST))
    return small == 3, small != 11


def input_hash(image):
    return hashlib.sha256(np.rint(image * 255).astype(np.uint8).tobytes()).hexdigest()


def load_external(repo, kitti_hashes):
    def blob(path):
        return subprocess.check_output(["git", "-C", str(repo), "show", f"{REV}:{path}"])

    samples, manifest = {}, []
    for line in blob("CamVid/test.txt").decode("ascii").splitlines():
        image_path, label_path = [s.removeprefix("/SegNet/") for s in line.split()]
        frame = Path(image_path).stem
        if (image_path != f"CamVid/test/{frame}.png" or label_path != f"CamVid/testannot/{frame}.png"
                or frame in samples):
            raise ValueError("Invalid or repeated test pair")
        photo, annotation = blob(image_path), blob(label_path)
        x = decode_rgb(photo)
        truth, valid = labels(annotation)
        digest = input_hash(x)
        samples[frame] = x, truth, valid
        manifest.append(dict(frame=frame, sequence=frame.split("_")[0], input_sha256=digest,
                             image_sha256=hashlib.sha256(photo).hexdigest(),
                             label_sha256=hashlib.sha256(annotation).hexdigest()))
    hashes = {row["input_sha256"] for row in manifest}
    if (Counter(r["sequence"] for r in manifest) != SEQUENCES or len(hashes) != len(samples)
            or hashes & kitti_hashes):
        raise ValueError("External coverage or image independence check failed")
    return samples, manifest


def aggregate(rows):
    return scores({k: sum(row[k] for row in rows) for k in ("tp", "fp", "fn", "tn")})["iou"]


def run(archive, repo, out):
    if out.exists():
        raise FileExistsError(out)
    metadata = json.loads((ARTIFACTS / "calibration/result.json").read_text())
    for path, field in [(archive, "archive_sha256"), (ARTIFACTS / "road.keras", "model_sha256"),
                        (ARTIFACTS / "split.json", "split_sha256")]:
        if sha256(path) != metadata[field]:
            raise ValueError(f"Original input changed: {path.name}")
    split = json.loads((ARTIFACTS / "split.json").read_text())
    kitti_ids = [key for subset in ("train", "cal", "test") for key in split[subset]]
    if len(set(kitti_ids)) != 289:
        raise ValueError("Expected all 289 distinct KITTI images")
    with zipfile.ZipFile(archive) as source:
        kitti = {key: decode_rgb(source.read(f"data_road/training/image_2/{key}.png")) for key in kitti_ids}
    samples, manifest = load_external(repo, {input_hash(x) for x in kitti.values()})
    selections = json.loads((ARTIFACTS / "calibration/selected_ids.json").read_text())
    with (ARTIFACTS / "calibration/runs.csv").open() as stream:
        published = list(csv.DictReader(stream))
    model = keras.models.load_model(ARTIFACTS / "road.keras", compile=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staging = Path(tmp) / "results"
        staging.mkdir()
        models = []
        for choice, recorded in zip(selections, published, strict=True):
            size, seed = choice["count"], choice["seed"]
            if int(recorded["count"]) != size or recorded["seed"] != ("" if seed is None else str(seed)):
                raise ValueError("Condition differs from the original experiment")
            convert(model, [kitti[key] for key in choice["ids"]], Path(tmp))
            binary = Path(tmp) / "int8.tflite"
            digest = sha256(binary)
            if digest != recorded["int8_sha256"]:
                raise ValueError(f"Different model bytes for {size} images, seed {seed}")
            path = Path(tmp) / f"{size}-{seed}.tflite"
            binary.rename(path)
            models.append((size, seed, path, digest))

        scene_rows, group_rows, reference, base_frames = [], [], {}, {}
        for size, seed, path, digest in models:
            runtime = RoadRuntime(path)
            rows = []
            for frame, (image, truth, valid) in samples.items():
                prob, _ = runtime.predict(image)
                confusion = counts(prob, truth, valid)
                iou = scores(confusion)["iou"]
                if size == 26:
                    base_frames[frame] = iou
                delta = None if iou is None or base_frames[frame] is None else (iou - base_frames[frame]) * 100
                rows.append(dict(count=size, seed=seed, sequence=frame.split("_")[0], frame=frame,
                                 **confusion, iou=iou, delta_vs_26_pp=delta))
            scene_rows.extend(rows)
            for group in ("all", *SEQUENCES):
                subset = rows if group == "all" else [r for r in rows if r["sequence"] == group]
                iou = aggregate(subset)
                if size == 26:
                    reference[group] = iou
                delta = 100 * (iou - reference[group])
                group_rows.append(dict(count=size, seed=seed, group=group, frames=len(subset),
                                       iou=iou, delta_vs_26_pp=delta, within_margin=delta >= -MARGIN_PP,
                                       model_sha256=digest))
        overall = [r for r in group_rows if r["group"] == "all"]
        worst = min((r for r in scene_rows if r["count"] < 26 and r["delta_vs_26_pp"] is not None),
                    key=lambda r: (r["delta_vs_26_pp"], r["count"], r["seed"], r["frame"]))
        write_csv(staging / "frames.csv", manifest)
        write_csv(staging / "scenes.csv", scene_rows)
        write_csv(staging / "runs.csv", group_rows)
        write_json(staging / "summary.json", dict(camvid_commit=REV, kitti_images=len(kitti),
                   duplicate_inputs=0, sequences=SEQUENCES, frames=len(samples), models=len(models),
                   margin_pp=MARGIN_PP, sufficient={str(n): all(r["within_margin"] for r in overall
                   if r["count"] == n) for n in (3, 9)}, worst=worst, worst_baseline=base_frames[worst["frame"]]))
        staging.rename(out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "repo", "out"):
        cli.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    run(args.archive, args.repo, args.out)
