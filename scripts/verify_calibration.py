"""Reconvert the seven models and check the published CSVs against raw INT8 outputs.

Run from the repository root: python -m scripts.verify_calibration --archive work/data/data_road.zip
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import tensorflow as tf

from src.data import load_sample, select_calibration_ids
from src.train import convert

ART = Path("artifacts")
RESULTS = ART / "calibration"
FIELDS = ("tn_int8", "fp_int8", "fn_int8", "tp_int8")


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def check(ok, context):
    if not ok:
        raise ValueError(f"Calibration verification failed: {context}")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iou(table):
    tn, fp, fn, tp = table
    return tp / (tp + fp + fn)


def evaluate(path, images):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    check(inp["dtype"] == out["dtype"] == np.int8, path)
    scale, zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    # Smallest signed output code whose dequantized probability reaches 0.5.
    first_road = int(np.ceil(0.5 / out_scale + out_zero))
    for scene, (rgb, truth, valid) in images.items():
        codes = np.clip(np.rint(rgb / scale) + zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], codes[None])
        interpreter.invoke()
        road = interpreter.get_tensor(out["index"])[0, :, :, 0] >= first_road
        mask = road[valid]
        # 0 = TN, 1 = FP, 2 = FN, 3 = TP. Count labels independently of src.bench.
        table = np.bincount(2 * truth[valid].astype(np.uint8) + mask, minlength=4)
        yield scene, mask, table


def verify(archive_path):
    split = json.loads((ART / "split.json").read_text())
    check(digest(archive_path) == json.loads((ART / "training.json").read_text())["data_sha256"], "archive")
    cal, test = split["cal"], split["test"]
    check(len(set(cal)) == len(cal) == 26 and len(set(test)) == len(test) == 62, "split sizes")
    check(set(cal).isdisjoint(test) and set(split["train"]).isdisjoint(cal + test), "split overlap")
    model = tf.keras.models.load_model(ART / "road.keras", compile=False)
    with ZipFile(archive_path) as archive:
        rgb = {key: load_sample(archive, key)[0] for key in cal}
        images = {key: load_sample(archive, key) for key in test}
    selections = {(26, ""): cal}
    for seed in (17, 42, 731):
        ids = select_calibration_ids(cal, 9, seed)
        check(ids[:3] == select_calibration_ids(cal, 3, seed), f"nested seed {seed}")
        check(set(x.split("_")[0] for x in ids[:3]) == {"um", "umm", "uu"}, f"types, seed {seed}")
        check(len(set(ids)) == len(ids) == 9 and set(ids) <= set(cal), f"IDs, seed {seed}")
        for size in (3, 9):
            selections[size, str(seed)] = ids[:size]
    listed = json.loads((RESULTS / "selected_ids.json").read_text())
    check(len(listed) == 7 and selections == {
        (r["count"], str(r["seed"]) if r["seed"] is not None else ""): r["ids"] for r in listed}, "saved IDs")
    runs = read_csv(RESULTS / "runs.csv")
    scenes = read_csv(RESULTS / "scenes.csv")
    check(len(runs) == 7 and len(scenes) == 7 * 62, "CSV sizes")
    by_run = {(int(r["count"]), r["seed"]): r for r in runs}
    by_scene = {(int(r["count"]), r["seed"], r["scene"]): r for r in scenes}
    release = {r["source_id"]: r for r in read_csv(ART / "scene_metrics.csv") if r["variant"] == "clean"}
    check(set(release) == set(test), "release evaluation IDs")
    baseline = {}
    best = None
    passes = {3: [], 9: []}
    with TemporaryDirectory() as tmp:
        for (count, seed), ids in selections.items():
            path = Path(tmp) / "int8.tflite"
            convert(model, [rgb[key] for key in ids], path.parent)
            run = by_run[count, seed]
            check(digest(path) == run["int8_sha256"], (count, seed, "model bytes"))
            if count == 26:
                check(path.read_bytes() == (ART / "int8.tflite").read_bytes(), "v1.0.1 model bytes")
            totals, uu = np.zeros(4, np.int64), np.zeros(4, np.int64)
            for key, mask, table in evaluate(path, images):
                row = by_scene[count, seed, key]
                check(table.tolist() == [int(row[k]) for k in FIELDS], (count, seed, key, "confusion"))
                check(mask.size == int(row["valid_pixels"]) and iou(table) == float(row["iou"]), (key, "scene IoU"))
                if count == 26:
                    baseline[key] = mask
                    check(table.tolist() == [int(release[key][k]) for k in FIELDS], (key, "v1.0.1"))
                changed = int(np.count_nonzero(mask != baseline[key]))
                check(changed == int(row["changed_vs_26"]), (count, seed, key, "changed pixels"))
                if count != 26:
                    rank = (-changed / mask.size, count, int(seed), key)
                    if best is None or rank < best:
                        best = rank
                totals += table
                if key.startswith("uu_"):
                    uu += table
            check(iou(totals) == float(run["overall_iou"]) and iou(uu) == float(run["uu_iou"]), (count, seed, "pooled IoU"))
            if count == 26:
                reference = iou(totals)
            loss_pp = 100 * (reference - iou(totals))
            check(loss_pp == float(run["loss_vs_26_pp"]), (count, seed, "difference from baseline"))
            within = bool(loss_pp <= 0.5)
            check(str(within) == run["within_margin"], (count, seed, "0.5 pp criterion"))
            if count != 26:
                passes[count].append(within)
            print(f"{count:2} {seed or 'all':>3} {100*iou(totals):.6f}% {100*iou(uu):.6f}%")
    result = json.loads((RESULTS / "result.json").read_text())
    check(result["reproduced"] and reference == result["published_iou"], "published IoU")
    check(result["margin_pp"] == 0.5 and result["sufficient"] ==
          {str(n): all(flags) for n, flags in passes.items()}, "decision")
    example = result["example"]
    check(best[1:] == (example["count"], example["seed"], example["scene"]), "maximum disagreement")
    print("Verified 7 model hashes, 434 scene rows, both pooled IoUs and the maximum-disagreement scene.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    verify(parser.parse_args().archive)
