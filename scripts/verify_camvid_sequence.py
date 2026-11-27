"""Recompute 0001TP counts and reproduce its fixed first, middle and last frames."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "0001TP"
CAMVID_REV = "bb68b64e72cb004b3ed927fee0f5624dd8097498"
CAMVID_URL = f"https://raw.githubusercontent.com/alexgkendall/SegNet-Tutorial/{CAMVID_REV}/CamVid"
CONFIGS = {"26": ("26", ""), "3_seed42": ("3", "42")}
FIELDS = ("tp", "fp", "fn", "tn")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def rates(values):
    tp, fp, fn = (values[key] for key in ("tp", "fp", "fn"))
    union, road = tp + fp + fn, tp + fn
    return dict(iou=tp / union if union else None,
                recall=tp / road if road else None,
                missed_road_fraction=fn / road if road else None)


def audit_counts(scenes_path, frames_path):
    manifest = [row for row in read_csv(frames_path) if row["sequence"] == SEQUENCE]
    ids = [row["frame"] for row in manifest]
    if len(ids) != 62 or len(set(ids)) != 62:
        raise ValueError("Expected exactly 62 distinct 0001TP frames in the manifest")
    if any(re.fullmatch(r"0001TP_\d+", frame) is None for frame in ids):
        raise ValueError("Unexpected sequence frame ID")
    ordered = sorted(ids, key=lambda frame: int(frame.rsplit("_", 1)[1]))
    # The upper middle is used for an even-sized sequence: positions 1, 32, 62.
    positions = (0, len(ordered) // 2, len(ordered) - 1)
    selected = [dict(position=index + 1, frame=ordered[index]) for index in positions]
    source = read_csv(scenes_path)
    groups, totals = {}, {}
    for name, (count, seed) in CONFIGS.items():
        rows = [row for row in source if row["sequence"] == SEQUENCE
                and row["count"] == count and row["seed"] == seed]
        if len(rows) != 62 or {row["frame"] for row in rows} != set(ids):
            raise ValueError(f"Missing, duplicate or unexpected frames for {name}")
        by_frame = {}
        for row in rows:
            if any(re.fullmatch(r"\d+", row[key]) is None for key in FIELDS):
                raise ValueError(f"Counts must be nonnegative integers: {row['frame']}")
            values = {key: int(row[key]) for key in FIELDS}
            if not 0 < sum(values.values()) <= 192 * 64:
                raise ValueError(f"Invalid evaluated pixel total: {row['frame']}")
            by_frame[row["frame"]] = values
        sums = {key: sum(row[key] for row in by_frame.values()) for key in FIELDS}
        totals[name] = dict(**sums, **rates(sums), frames=len(rows),
                            ground_truth_road=sums["tp"] + sums["fn"],
                            valid_pixels=sum(sums.values()),
                            predicted_road=sums["tp"] + sums["fp"],
                            empty_prediction_frames=sum(
                                row["tp"] + row["fp"] == 0 for row in by_frame.values()))
        groups[name] = by_frame
    for frame in ordered:
        baseline, candidate = (groups[key][frame] for key in CONFIGS)
        if (baseline["tp"] + baseline["fn"] != candidate["tp"] + candidate["fn"]
                or sum(baseline.values()) != sum(candidate.values())):
            raise ValueError(f"Ground truth or evaluated region differs: {frame}")
    result = dict(sequence=SEQUENCE, frames=62, totals=totals,
                  formulas={"iou": "sum(TP) / (sum(TP) + sum(FP) + sum(FN))",
                            "recall": "sum(TP) / (sum(TP) + sum(FN))"},
                  selected=selected,
                  selection_rule="Numeric frame order; positions 1, N//2+1, N; N=62",
                  input_sha256={"scenes.csv": digest(scenes_path),
                                "frames.csv": digest(frames_path)})
    return result, groups, {row["frame"]: row for row in manifest}


def source_bytes(folder, frame, manifest, cache):
    path = cache / folder / f"{frame}.png"
    data = path.read_bytes() if path.exists() else None
    if data is None:
        with urlopen(f"{CAMVID_URL}/{folder}/{frame}.png", timeout=30) as response:
            data = response.read()
    field = "image_sha256" if folder == "test" else "label_sha256"
    if hashlib.sha256(data).hexdigest() != manifest[field]:
        raise ValueError(f"Source checksum mismatch: {folder}/{frame}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def draw_panel(path, frame, position, rgb, truth, valid, predictions, details):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)

    def mask_image(mask):
        colors = np.full((64, 192, 3), (25, 39, 52), dtype=np.uint8)
        colors[mask & valid] = (61, 194, 167)
        colors[~valid] = (154, 163, 174)
        y, x = np.indices(valid.shape)
        colors[(~valid) & ((x + y) % 6 < 2)] = (195, 201, 208)
        return Image.fromarray(colors)

    input_image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
    tiles = [input_image, mask_image(truth), mask_image(predictions["26"]),
             mask_image(predictions["3_seed42"])]
    titles = ["Input", "Road ground truth", "26-image INT8", "3-image INT8 (seed 42)"]
    notes = ["192 x 64 RGB, bilinear", f"Road: {int((truth & valid).sum()):,} pixels"]
    for key in CONFIGS:
        row = details[key]
        notes.append(f"Road: {row['tp'] + row['fp']:,} pixels | IoU: {100 * rates(row)['iou']:.2f}%")
    margin, gap, tile_width, tile_height = 28, 16, 384, 128
    width = 2 * margin + 4 * tile_width + 3 * gap
    canvas = Image.new("RGB", (width, 366), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), f"CamVid / {frame}", fill="#182735", font=font(27))
    draw.text((margin, 58), f"Fixed position {position}/62 | Threshold >= 0.5 | Exact model-input geometry",
              fill="#52616d", font=font(17))
    for index, (tile, title, note) in enumerate(zip(tiles, titles, notes, strict=True)):
        x = margin + index * (tile_width + gap)
        draw.text((x, 100), title, fill="#182735", font=font(20))
        canvas.paste(tile.resize((tile_width, tile_height), Image.Resampling.NEAREST), (x, 133))
        draw.text((x, 273), note, fill="#354754", font=font(15))
    common = int((truth & valid & ~predictions["26"] & ~predictions["3_seed42"]).sum())
    changed = int(((predictions["26"] != predictions["3_seed42"]) & valid).sum())
    draw.text((margin, 311), f"Missed by both: {common:,} road pixels | Changed predictions: {changed:,} valid pixels",
              fill="#182735", font=font(18))
    draw.text((margin, 340), "Teal = road | Dark = non-road | Gray hatch = ignored label 11 | Counts use valid pixels only",
              fill="#52616d", font=font(14))
    canvas.save(path)


def reproduce_panels(result, groups, manifest, candidate_model, cache, stage):
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(ROOT))
    from src.runtime import RoadRuntime

    published = read_csv(ROOT / "artifacts/calibration/runs.csv")
    paths = {"26": ROOT / "artifacts/int8.tflite", "3_seed42": candidate_model}
    model_hashes, runtimes = {}, {}
    for key, (count, seed) in CONFIGS.items():
        records = [row for row in published if row["count"] == count and row["seed"] == seed]
        if len(records) != 1 or digest(paths[key]) != records[0]["int8_sha256"]:
            raise ValueError(f"Model does not match the published {key} condition")
        model_hashes[key] = digest(paths[key])
        runtimes[key] = RoadRuntime(paths[key])

    arrays, checks = {}, []
    for selection in result["selected"]:
        frame, position = selection["frame"], selection["position"]
        photo = source_bytes("test", frame, manifest[frame], cache)
        label = source_bytes("testannot", frame, manifest[frame], cache)
        with Image.open(io.BytesIO(photo)) as image:
            original_size = list(image.size)
            rgb = np.asarray(image.convert("RGB").resize((192, 64), Image.Resampling.BILINEAR),
                             dtype=np.float32) / 255.0
        with Image.open(io.BytesIO(label)) as image:
            if image.mode not in ("L", "P") or np.asarray(image).max() > 11:
                raise ValueError(f"Expected CamVid class IDs 0 through 11: {frame}")
            ids = np.asarray(image.resize((192, 64), Image.Resampling.NEAREST))
        if hashlib.sha256(np.rint(rgb * 255).astype(np.uint8).tobytes()).hexdigest() != manifest[frame]["input_sha256"]:
            raise ValueError(f"Preprocessed input checksum mismatch: {frame}")
        truth, valid = ids == 3, ids != 11
        predictions, details, ignored_predictions = {}, {}, {}
        for key, runtime in runtimes.items():
            probabilities, _ = runtime.predict(rgb)
            pred = probabilities >= 0.5
            # Compute the confusion matrix independently of the benchmark helpers.
            values = dict(tp=int((pred & truth & valid).sum()),
                          fp=int((pred & ~truth & valid).sum()),
                          fn=int((~pred & truth & valid).sum()),
                          tn=int((~pred & ~truth & valid).sum()))
            if values != groups[key][frame]:
                raise ValueError(f"Reproduced counts differ from the CSV: {frame}, {key}: {values}")
            details[key], predictions[key] = values, pred
            ignored_predictions[key] = int((pred & ~valid).sum())
            arrays[f"{frame}_{key}_probabilities"] = probabilities
            arrays[f"{frame}_{key}_prediction"] = pred
        arrays[f"{frame}_rgb"] = rgb
        arrays[f"{frame}_truth"] = truth
        arrays[f"{frame}_valid"] = valid
        draw_panel(stage / f"{frame}.png", frame, position, rgb, truth, valid, predictions, details)
        checks.append(dict(frame=frame, position=position, original_size=original_size,
                           counts=details, ignored_road_predictions=ignored_predictions,
                           common_missed_road=int((truth & valid & ~predictions["26"] & ~predictions["3_seed42"]).sum()),
                           changed_valid_pixels=int(((predictions["26"] != predictions["3_seed42"]) & valid).sum()),
                           source_hashes=manifest[frame]))
    np.savez_compressed(stage / "masks.npz", **arrays)
    result["inference"] = dict(frames=len(checks), model_sha256=model_hashes,
                               camvid_commit=CAMVID_REV, threshold=0.5,
                               input_shape=[64, 192, 3], road_label=3, ignored_label=11,
                               tensor_contract={key: runtime.describe() for key, runtime in runtimes.items()},
                               versions={name: importlib.metadata.version(name) for name in
                                         ("tensorflow-cpu", "keras", "numpy", "pillow")},
                               frames_verified=checks,
                               files_sha256={path.name: digest(path) for path in sorted(stage.iterdir())})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scenes", type=Path, default=ROOT / "artifacts/external/scenes.csv")
    cli.add_argument("--frames", type=Path, default=ROOT / "artifacts/external/frames.csv")
    cli.add_argument("--output", required=True, type=Path, help="New output directory")
    cli.add_argument("--render", action="store_true", help="Run both models on the three fixed frames")
    cli.add_argument("--candidate-model", type=Path,
                     default=ROOT / "artifacts/calibration/int8-3-seed42.tflite")
    cli.add_argument("--camvid-root", type=Path, default=ROOT / "work/camvid-review/source",
                     help="Cache containing test/ and testannot/; missing PNGs are downloaded and hash-checked")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Choose a new output directory; existing results will not be replaced")
    result, groups, manifest = audit_counts(args.scenes, args.frames)
    result["script_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "review"
        stage.mkdir()
        if args.render:
            reproduce_panels(result, groups, manifest, args.candidate_model, args.camvid_root, stage)
        (stage / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stage.rename(args.output)
    print(json.dumps(dict(output=str(args.output), totals=result["totals"], selected=result["selected"],
                          reproduced_frames=result.get("inference", {}).get("frames", 0)), indent=2))


if __name__ == "__main__":
    main()
