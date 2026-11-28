import numpy as np
import pytest

from scripts.compare_camvid_fp32 import aggregate_counts, count_pixels


def test_counts_include_all_four_cells_and_exclude_void():
    # Four valid cells cover TP, FP, FN, TN. Both extra positive predictions
    # are ignored, including one whose truth happens to be positive.
    pred = np.array([[True, True, False], [False, True, True]])
    truth = np.array([[True, False, True], [False, True, False]])
    valid = np.array([[True, True, True], [True, False, False]])
    assert count_pixels(pred, truth, valid) == dict(tp=1, fp=1, fn=1, tn=1)


def test_masks_must_not_broadcast_or_use_integer_inversion():
    mask = np.ones((2, 3), dtype=bool)
    with pytest.raises(ValueError, match="Boolean.*same.*shape"):
        count_pixels(mask, mask[:1], mask)
    with pytest.raises(ValueError, match="Boolean.*same.*shape"):
        count_pixels(mask.astype(np.uint8), mask, mask)


def test_pooled_scores_use_pixel_sums_not_mean_frame_scores():
    # IoUs are 1/2 and 9/90; their mean (0.3) is not pooled IoU (10/92).
    rows = [dict(tp=1, fp=1, fn=0, tn=8),
            dict(tp=9, fp=0, fn=81, tn=10)]
    result = aggregate_counts(rows)
    assert {key: result[key] for key in ("tp", "fp", "fn", "tn")} == {
        "tp": 10, "fp": 1, "fn": 81, "tn": 18,
    }
    assert result["iou"] == pytest.approx(10 / 92)
    assert result["recall"] == pytest.approx(10 / 91)
    assert result["valid_pixels"] == 110
    assert result["ground_truth_road"] == 91


def test_undefined_scores_are_not_invented():
    result = aggregate_counts([dict(tp=0, fp=0, fn=0, tn=12)])
    assert result["iou"] is None
    assert result["recall"] is None
