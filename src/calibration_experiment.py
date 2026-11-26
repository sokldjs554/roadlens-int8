"""Compare nested calibration subsets using the published weights and clean KITTI holdout."""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .bench import CUTOFF, counts, overlay, pooled, scores, write_csv, write_json
from .data import load_sample, select_calibration_ids
from .runtime import RoadRuntime
from .train import convert, keras

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
SEEDS = (17, 42, 731)
MARGIN_PP = 0.5


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def show_scene(path, scene, sample, baseline, candidate, count, seed):
    rgb, truth, valid = sample
    source = Image.fromarray(np.rint(rgb * 255).astype("uint8"))
    changed = (baseline != candidate) & valid
    panels = [
        ("RGB input", source), ("Road label", overlay(source, truth & valid)),
        ("26-image INT8", overlay(source, baseline & valid)),
        (f"{count}-image INT8, seed {seed}", overlay(source, candidate & valid)),
        ("Changed predictions", Image.composite(Image.new("RGB", source.size, "#ff5252"),
                                               source, Image.fromarray(changed))),
    ]
    w, h = source.size
    canvas = Image.new("RGB", (w * 2, (h * 2 + 26) * 5 + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"KITTI Road | {scene}", fill="black")
    for i, (title, image) in enumerate(panels):
        y = 28 + i * (h * 2 + 26)
        draw.text((8, y), title, fill="black")
        canvas.paste(image.resize((w * 2, h * 2), Image.Resampling.NEAREST), (0, y + 18))
    canvas.save(path, quality=90)


def run(archive_path, out, seeds=SEEDS):
    split = json.loads((ARTIFACTS / "split.json").read_text())
    cal, test = split["cal"], split["test"]
    if (len(set(cal)) != len(cal) or len(cal) != 26 or len(set(test)) != len(test)
            or len(test) != 62 or set(test) & set(cal)):
        raise ValueError("Expected the released disjoint 26/62 calibration and evaluation sets")
    train_meta = json.loads((ARTIFACTS / "training.json").read_text())
    if sha256(archive_path) != train_meta["data_sha256"]:
        raise ValueError("KITTI archive checksum differs from the training record")
    model_path = ARTIFACTS / "road.keras"
    model = keras.models.load_model(model_path, compile=False)
    with zipfile.ZipFile(archive_path) as archive:
        rgb = {key: load_sample(archive, key)[0] for key in cal}
        samples = {key: load_sample(archive, key) for key in test}
    selections = [dict(count=26, seed=None, ids=cal)]
    for size in (3, 9):
        for seed in seeds:
            selections.append(dict(count=size, seed=seed, ids=select_calibration_ids(cal, size, seed)))
    published = json.loads((ARTIFACTS / "benchmark.json").read_text())["models"]["int8"]["iou"]
    baseline_masks, scene_rows, summary = {}, [], []
    largest = None
    # Build in a sibling staging directory so a failed conversion cannot publish partial results.
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        for selection in selections:
            size, seed = selection["count"], selection["seed"]
            convert(model, [rgb[key] for key in selection["ids"]], Path(tmp))
            path = Path(tmp) / "int8.tflite"
            runtime = RoadRuntime(path)
            if runtime.describe()["input"]["dtype"] != "int8":
                raise ValueError("Expected an INT8 model")
            rows = []
            for key in test:
                image, truth, valid = samples[key]
                prob, _ = runtime.predict(image)
                mask = prob >= CUTOFF
                if size == 26:
                    baseline_masks[key] = mask
                changed = int(np.count_nonzero((mask != baseline_masks[key]) & valid))
                n = int(valid.sum())
                confusion = counts(prob, truth, valid)
                row = dict(count=size, seed=seed, scene=key, road_type=key.split("_")[0],
                           valid_pixels=n, changed_vs_26=changed, iou=scores(confusion)["iou"])
                row.update({f"{k}_int8": v for k, v in confusion.items()})
                rows.append(row)
                if size != 26:
                    rank = (-changed / n, size, seed, key)
                    if largest is None or rank < largest[0]:
                        largest = (rank, row, mask)
            scene_rows.extend(rows)
            overall = pooled(rows, "int8")["iou"]
            uu = pooled([r for r in rows if r["road_type"] == "uu"], "int8")["iou"]
            if size == 26:
                reference = overall
            loss = (reference - overall) * 100
            summary.append(dict(count=size, seed=seed, overall_iou=overall, uu_iou=uu,
                                loss_vs_26_pp=loss, within_margin=loss <= MARGIN_PP,
                                int8_sha256=sha256(path)))
        _, row, mask = largest
        key = row["scene"]
        show_scene(stage / "scene.jpg", key, samples[key], baseline_masks[key], mask, row["count"], row["seed"])
        write_csv(stage / "runs.csv", summary)
        write_csv(stage / "scenes.csv", scene_rows)
        write_json(stage / "selected_ids.json", selections)
        write_json(stage / "result.json", dict(
            model_sha256=sha256(model_path), split_sha256=sha256(ARTIFACTS / "split.json"),
            archive_sha256=train_meta["data_sha256"], published_iou=published,
            reproduced=f"{100*published:.2f}" == f"{100*reference:.2f}", margin_pp=MARGIN_PP,
            sufficient={str(n): all(r["within_margin"] for r in summary if r["count"] == n)
                        if f"{100*published:.2f}" == f"{100*reference:.2f}" else None for n in (3, 9)},
            example=row, example_baseline_iou=next(r["iou"] for r in scene_rows if r["count"] == 26
                                                    and r["scene"] == key)))
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--extended", action="store_true",
                        help="Repeat with the original three seeds plus fixed seeds 0 through 16")
    args = parser.parse_args()
    seeds = (*SEEDS, *range(17)) if args.extended else SEEDS
    run(args.archive, args.out, seeds)
