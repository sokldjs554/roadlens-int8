import csv
import io
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from src.external_calibration import ARTIFACTS, MARGIN_PP, SEQUENCES, labels


def mask_bytes(ids):
    stream = io.BytesIO()
    Image.fromarray(ids).save(stream, format="PNG")
    return stream.getvalue()


def test_camvid_road_pavement_void():
    ids = np.tile([3, 4, 11], (4, 1)).astype(np.uint8)
    truth, valid = labels(mask_bytes(ids))
    assert tuple(truth[0, [0, 96, -1]]) == (True, False, False)
    assert tuple(valid[0, [0, 96, -1]]) == (True, True, False)
    ids[0, 0] = 255
    with pytest.raises(ValueError, match="class IDs"):
        labels(mask_bytes(ids))


def test_published_external_results():
    def read(path):
        with (ARTIFACTS / path).open() as stream:
            return list(csv.DictReader(stream))

    frames = read("external/frames.csv")
    scenes = read("external/scenes.csv")
    runs = read("external/runs.csv")
    summary = json.loads((ARTIFACTS / "external/summary.json").read_text())
    assert Counter(r["sequence"] for r in frames) == SEQUENCES
    assert len({r["input_sha256"] for r in frames}) == len(frames) == 233
    original = {(r["count"], r["seed"]): r["int8_sha256"] for r in read("calibration/runs.csv")}
    assert len(original) == 7 and len(scenes) == 1631 and len(runs) == 21
    reference = {r["group"]: float(r["iou"]) for r in runs if r["count"] == "26"}
    for row in runs:
        condition = row["count"], row["seed"]
        assert row["model_sha256"] == original[condition]
        subset = [s for s in scenes if (s["count"], s["seed"]) == condition
                  and row["group"] in ("all", s["sequence"])]
        ids = {s["frame"] for s in subset}
        assert len(subset) == len(ids) == int(row["frames"])
        assert ids == {f["frame"] for f in frames if row["group"] in ("all", f["sequence"])}
        tp, fp, fn = [sum(int(s[k]) for s in subset) for k in ("tp", "fp", "fn")]
        iou = tp / (tp + fp + fn)
        assert iou == float(row["iou"])
        delta = (iou - reference[row["group"]]) * 100
        assert delta == float(row["delta_vs_26_pp"])
        assert (delta >= -MARGIN_PP) == (row["within_margin"] == "True")
    for size in ("3", "9"):
        flags = [r["within_margin"] == "True" for r in runs if r["group"] == "all" and r["count"] == size]
        assert len(flags) == 3 and all(flags) == summary["sufficient"][size]
