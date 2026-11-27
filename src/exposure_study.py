"""Evaluate a training-derived exposure rule on the frozen KITTI INT8 model."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from .bench import CUTOFF, altered, counts, pooled, scores, write_csv, write_json
from .data import decode_rgb, load_sample
from .runtime import RoadRuntime

ART = Path(__file__).resolve().parents[1] / "artifacts"
VARIANTS = ("clean", "dim", "occluded", "gamma2")


def brightness(rgb):
    return float(np.dot(rgb.mean(axis=(0, 1), dtype=np.float64), [0.2126, 0.7152, 0.0722]))


def correct(rgb, threshold, target):
    level = brightness(rgb)
    gain = min(3.0, target / level) if 0 < level < threshold else 1.0
    if gain == 1.0:
        return rgb, gain
    return np.minimum(rgb * np.float32(gain), 1.0), gain


def variant_image(rgb, variant):
    return rgb * rgb if variant == "gamma2" else altered(rgb, variant)


def example_panel(rgb, adjusted, truth, valid, before, after):
    base = Image.fromarray(np.uint8(rgb * 255))
    corrected = Image.fromarray(np.uint8(adjusted * 255))
    panels = [("Input", base), ("Brightness rule", corrected)]
    for label, mask in (("Baseline INT8", before >= CUTOFF), ("Rule + INT8", after >= CUTOFF),
                        ("Road label", truth)):
        tint = Image.blend(base, Image.new("RGB", base.size, "#44e6cf"), .58)
        panels.append((label, Image.composite(tint, base, Image.fromarray(mask & valid))))
    changed = ((before >= CUTOFF) != (after >= CUTOFF)) & valid
    panels.append(("Changed pixels", Image.composite(Image.new("RGB", base.size, "#ff5252"),
                                                     base, Image.fromarray(changed))))
    row_height = base.height + 17
    sheet = Image.new("RGB", (base.width * 2, row_height * 3), "#10172e")
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        x, y = index % 2 * base.width, index // 2 * row_height
        draw.text((x + 3, y), label, fill="#f2f5ff")
        sheet.paste(panel, (x, y + 17))
    return sheet.resize((sheet.width * 2, sheet.height * 2), Image.Resampling.NEAREST)


def evaluate(archive_path, out):
    if out.exists():
        raise FileExistsError(out)
    with archive_path.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive_sha != json.loads((ART / "training.json").read_text())["data_sha256"]:
        raise ValueError("Unexpected KITTI archive checksum")
    split = json.loads((ART / "split.json").read_text())
    if set(split["train"]) & set(split["test"]):
        raise ValueError("Training/evaluation overlap")
    runtime = RoadRuntime(ART / "int8.tflite")
    with (ART / "scene_metrics.csv").open(newline="") as stream:
        published = {(r["source_id"], r["variant"]): r for r in csv.DictReader(stream)}
    rows, panels = [], {}
    worst = None
    with ZipFile(archive_path) as archive:
        levels = [brightness(decode_rgb(archive.read(f"data_road/training/image_2/{key}.png")))
                  for key in split["train"]]
        threshold, target = np.percentile(levels, [10, 50]).tolist()
        for key in split["test"]:
            source, truth, valid = load_sample(archive, key)
            for variant in VARIANTS:
                rgb = variant_image(source, variant)
                adjusted, gain = correct(rgb, threshold, target)
                raw, _ = runtime.predict(rgb)
                corrected, _ = runtime.predict(adjusted)
                row = dict(scene=key, variant=variant, gain=gain)
                for name, prob in (("base", raw), ("rule", corrected)):
                    c = counts(prob, truth, valid)
                    if name == "base" and variant != "gamma2":
                        if any(n != int(published[key, variant][k + "_int8"]) for k, n in c.items()):
                            raise ValueError(f"Baseline differs at {key}/{variant}")
                    row.update({f"{k}_{name}": n for k, n in c.items()})
                    row[f"iou_{name}"] = scores(c)["iou"]
                row["delta_pp"] = 100 * (row["iou_rule"] - row["iou_base"])
                rows.append(row)
                if variant == "clean" and (worst is None or row["delta_pp"] < worst["delta_pp"]):
                    worst = row
                    panels["worst_clean"] = example_panel(rgb, adjusted, truth, valid, raw, corrected)
                if (key, variant) == ("um_000028", "dim"):
                    panels["known_dim"] = example_panel(rgb, adjusted, truth, valid, raw, corrected)
    groups = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        base, rule = [pooled(subset, m)["iou"] for m in ("base", "rule")]
        groups[variant] = dict(n=len(subset), triggered=sum(r["gain"] > 1 for r in subset),
                               base=base, rule=rule, delta_pp=100 * (rule - base))
    limits = dict(clean=-.2, occluded=-.5, dim=5., gamma2=-.5)
    passes = {v: groups[v]["delta_pp"] >= floor for v, floor in limits.items()}
    passes["worst_clean"] = worst["delta_pp"] >= -5
    summary = dict(archive_sha256=archive_sha, model_sha256=hashlib.sha256((ART / "int8.tflite").read_bytes()).hexdigest(),
                   train=len(levels), threshold=threshold, target=target, groups=groups, worst_clean=worst,
                   checks=passes, candidate=all(passes.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "study"
        stage.mkdir()
        write_json(stage / "summary.json", summary)
        write_csv(stage / "scenes.csv", rows)
        for name, panel in panels.items():
            panel.save(stage / f"{name}.png")
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.archive, args.out)
