"""Validate the exposure rule, frozen baseline and release decision from raw counts."""
import csv
import json
from pathlib import Path

import numpy as np

from src.exposure_study import correct

ART = Path(__file__).resolve().parents[1] / "artifacts"


def test_rule_handles_black_bright_and_extremely_dark():
    for level in (0., 0.8):
        x = np.full((2, 3, 3), level, dtype=np.float32)
        y, gain = correct(x, 0.25, 0.4)
        assert gain == 1 and y is x
    x = np.full((2, 3, 3), 0.01, dtype=np.float32)
    x[0, 0, 0] = 0.9
    y, gain = correct(x, 0.25, 0.4)
    assert gain == 3 and y.max() == 1 and x.max() < 1
    np.testing.assert_allclose(y[1], 0.03)


def test_pooled_results_and_rejection_match_raw_confusions():
    with (ART / "exposure/scenes.csv").open() as f:
        rows = list(csv.DictReader(f))
    with (ART / "scene_metrics.csv").open() as f:
        published = {(r["source_id"], r["variant"]): r for r in csv.DictReader(f)}
    summary = json.loads((ART / "exposure/summary.json").read_text())
    test_ids = set(json.loads((ART / "split.json").read_text())["test"])
    assert len(rows) == 248
    for v, group in summary["groups"].items():
        subset = [r for r in rows if r["variant"] == v]
        assert {r["scene"] for r in subset} == test_ids
        for mode in ("base", "rule"):
            tp, fp, fn = [sum(int(r[f"{k}_{mode}"]) for r in subset) for k in ("tp", "fp", "fn")]
            assert tp / (tp + fp + fn) == group[mode]
        if v != "gamma2":
            for r in subset:
                assert all(r[f"{k}_base"] == published[r["scene"], v][f"{k}_int8"]
                           for k in ("tp", "fp", "fn", "tn"))
    worst = min((r for r in rows if r["variant"] == "clean"), key=lambda r: float(r["delta_pp"]))
    assert worst["scene"] == summary["worst_clean"]["scene"] and float(worst["delta_pp"]) < -5
    assert summary["checks"] == {"clean": True, "dim": True, "occluded": True,
                                  "gamma2": True, "worst_clean": False}
    assert not summary["candidate"]
