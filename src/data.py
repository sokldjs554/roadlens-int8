"""KITTI road images, valid pixels and a drive-disjoint labelled holdout."""

import io
import random
import zipfile
from collections import Counter
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

HEIGHT = 64
WIDTH = 192
SEED = 731
CLASSES = ("um", "umm", "uu")


def load_sample(archive: zipfile.ZipFile, id: str):
    """Return RGB float32 [0,1], foreground bool, valid-region bool in camera view."""
    kind, number = id.split("_")
    base = "data_road/training"
    with Image.open(io.BytesIO(archive.read(f"{base}/image_2/{id}.png"))) as image:
        x = np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255.0
    with Image.open(io.BytesIO(archive.read(f"{base}/gt_image_2/{kind}_road_{number}.png"))) as mask:
        rgb = np.asarray(mask.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.NEAREST))
    return x, rgb[..., 2] > 0, rgb[..., 0] > 0


def build_split(archive, mapping):
    """Group globally by raw drive. Use category counts alone to balance each split."""
    by_drive = {}
    for kind in CLASSES:
        lines = mapping.read(f"training/{kind}_mapping.txt").decode("ascii").splitlines()
        for i, line in enumerate(lines):
            drive, _ = line.split()
            by_drive.setdefault(drive, []).append(f"{kind}_{i:06d}")
    mapped = [id for ids in by_drive.values() for id in ids]
    official = {PurePosixPath(path).stem for path in archive.namelist()
                if path.startswith("data_road/training/image_2/") and path.endswith(".png")}
    if len(mapped) != len(official) or set(mapped) != official:
        raise ValueError("Road archive IDs and official drive mapping disagree")
    drives = sorted(by_drive)
    counts = {drive: Counter(id.split("_")[0] for id in ids) for drive, ids in by_drive.items()}
    totals = Counter(id.split("_")[0] for id in mapped)
    rng = random.Random(SEED)
    # Choose small drive subsets by their counts, without reading image pixels or labels.
    best = float("inf")
    for _ in range(6000):
        order = rng.sample(drives, len(drives))
        n_cal, n_test = round(len(drives) * .1), round(len(drives) * .2)
        candidate = {"cal": order[:n_cal], "test": order[n_cal:n_cal+n_test],
                     "train": order[n_cal+n_test:]}
        loss = sum((sum(counts[d][k] for d in candidate[p]) - totals[k] * frac) ** 2
                   for p, frac in (("cal", .1), ("test", .2)) for k in CLASSES)
        if loss < best:
            selected, best = candidate, loss
    split = {p: sorted(id for drive in ds for id in by_drive[drive]) for p, ds in selected.items()}
    split["drives"] = {p: sorted(ds) for p, ds in selected.items()}
    return split


def select_calibration_ids(cal_ids, count, seed):
    """Sample a repeatable prefix, starting with one image of each road type."""
    if len(set(cal_ids)) != len(cal_ids):
        raise ValueError("Calibration IDs must be unique")
    if not len(CLASSES) <= count <= len(cal_ids):
        raise ValueError("Choose at least one image per type and at most the full set")
    groups = {kind: [] for kind in CLASSES}
    for image_id in sorted(cal_ids):
        kind = image_id.split("_", 1)[0]
        if kind not in groups:
            raise ValueError(f"Unknown road type in {image_id}")
        groups[kind].append(image_id)
    if not all(groups.values()):
        raise ValueError("Calibration requires um, umm and uu images")

    rng = random.Random(seed)
    first = [rng.choice(groups[kind]) for kind in CLASSES]
    rest = sorted(set(cal_ids).difference(first))
    rng.shuffle(rest)
    return (first + rest)[:count]
