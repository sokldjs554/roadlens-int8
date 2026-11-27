import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.data import HEIGHT, WIDTH, decode_rgb, load_sample
from src.infer import compare_image
from src.runtime import RoadRuntime


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "web/scenes/um_000032_clean_input.jpg"


def cli(source, output):
    return subprocess.run([sys.executable, "-m", "src.infer", str(source), str(output)],
                          cwd=ROOT, text=True, capture_output=True)


def test_cli_predicts_both_masks(tmp_path):
    destination = tmp_path / "pair.png"
    result = cli(INPUT, destination)
    assert result.returncode == 0, result.stderr
    rgb = decode_rgb(INPUT.read_bytes())
    with Image.open(destination) as image:
        assert image.size == (WIDTH, HEIGHT) and image.mode == "P"
        labels = np.array(image)
    for bit, name in enumerate(("fp32", "int8")):
        reference, _ = RoadRuntime(ROOT / "artifacts" / f"{name}.tflite").predict(rgb)
        np.testing.assert_array_equal((labels >> bit) & 1, reference >= 0.5)
    assert json.loads(result.stdout)["different_pixels"] == int(np.isin(labels, [1, 2]).sum())


def test_cli_rejects_unreadable_and_preserves_existing_output(tmp_path):
    invalid = tmp_path / "damaged.png"
    invalid.write_bytes(b"broken")
    output = tmp_path / "result.png"
    bad = cli(invalid, output)
    assert bad.returncode != 0 and not output.exists()
    assert "roadlens:" in bad.stderr and "Traceback" not in bad.stderr
    output.write_bytes(b"existing result")
    occupied = cli(INPUT, output)
    assert occupied.returncode != 0 and output.read_bytes() == b"existing result"

@pytest.mark.skipif(not os.environ.get("KITTI_ROAD_ZIP"), reason="Set KITTI_ROAD_ZIP for the 62-image audit")
def test_saved_masks_match_all_heldout_reference_counts(tmp_path):
    with (ROOT / "artifacts/scene_metrics.csv").open(newline="") as source:
        clean = {row["source_id"]: row for row in csv.DictReader(source) if row["variant"] == "clean"}
    ids = json.loads((ROOT / "artifacts/split.json").read_text())["test"]
    assert set(clean) == set(ids) and len(ids) == 62
    with ZipFile(os.environ["KITTI_ROAD_ZIP"]) as archive:
        for scene in ids:
            path = tmp_path / f"{scene}.png"
            path.write_bytes(archive.read(f"data_road/training/image_2/{scene}.png"))
            output = tmp_path / f"{scene}-pair.png"
            compare_image(path, output)
            with Image.open(output) as image:
                labels = np.array(image)
            _, truth, valid = load_sample(archive, scene)
            for bit, model in enumerate(("fp32", "int8")):
                road = (labels[valid] >> bit) & 1
                table = np.bincount(2 * truth[valid].astype(np.uint8) + road, minlength=4)
                assert table.tolist() == [int(clean[scene][f"{key}_{model}"])
                                          for key in ("tn", "fp", "fn", "tp")], (scene, model)
