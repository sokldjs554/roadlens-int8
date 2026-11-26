import json
from pathlib import Path

import pytest

from src.data import select_calibration_ids


SPLIT = json.loads((Path(__file__).resolve().parents[1] / "artifacts/split.json").read_text())


@pytest.mark.parametrize("seed", [17, 42, 731])
def test_calibration_prefixes(seed):
    cal = SPLIT["cal"].copy()
    full = select_calibration_ids(cal, 26, seed)
    for count in (3, 9, 26):
        ids = select_calibration_ids(cal, count, seed)
        assert len(ids) == len(set(ids)) == count
        assert ids == full[:count]
        assert {key.split("_")[0] for key in ids[:3]} == {"um", "umm", "uu"}
        assert set(ids) <= set(cal)
        assert set(ids).isdisjoint(SPLIT["test"])
        assert ids == select_calibration_ids(cal, count, seed)
    assert cal == SPLIT["cal"]


def test_seed_changes_selection():
    assert len({tuple(select_calibration_ids(SPLIT["cal"], 3, s)) for s in (17, 42, 731)}) == 3


@pytest.mark.parametrize("ids,count", [
    (SPLIT["cal"], 2), (SPLIT["cal"], 27), (SPLIT["cal"] * 2, 9),
    (["um_1", "um_2", "uu_1"], 3),
])
def test_invalid_calibration_pool(ids, count):
    with pytest.raises(ValueError):
        select_calibration_ids(ids, count, 17)
