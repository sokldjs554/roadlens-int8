"""Check published measurements against their underlying samples."""
import csv
import json
from pathlib import Path

import numpy as np


ART = Path(__file__).resolve().parents[1] / "artifacts"


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def test_extended_sweep_reproduces_applicant_runs():
    original = rows(ART / "calibration/runs.csv")
    extended = rows(ART / "calibration_extended/runs.csv")
    assert len(extended) == 41
    assert all(row in extended for row in original)
    for size in ("3", "9"):
        subset = [row for row in extended if row["count"] == size]
        assert {int(row["seed"]) for row in subset} == {17, 42, 731, *range(17)}
        assert all(float(row["loss_vs_26_pp"]) <= 0.5 for row in subset)


def test_pipeline_raw_samples_and_percentiles():
    measured = rows(ART / "pipeline/timings.csv")
    summary = json.loads((ART / "pipeline/summary.json").read_text())
    ids = json.loads((ART / "split.json").read_text())["test"]
    expected = {(scene, repeat) for scene in ids for repeat in range(1, 11)}
    for model, metrics in summary["models"].items():
        runs = [row for row in measured if row["model"] == model]
        assert len(runs) == len(expected)
        assert {(row["scene"], int(row["repeat"])) for row in runs} == expected
        for row in runs:
            parts = [int(row[key + "_ns"]) for key in ("prepare", "invoke", "adapter", "threshold")]
            assert min(parts) >= 0 and sum(parts) == int(row["total_ns"])
        for key, quantiles in metrics.items():
            values = [int(row[key + "_ns"]) / 1e6 for row in runs]
            np.testing.assert_allclose(np.percentile(values, [50, 95]),
                                       [quantiles["p50_ms"], quantiles["p95_ms"]])
