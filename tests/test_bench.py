import unittest

import numpy as np

from src.bench import altered, counts, pooled, scores


class MetricTests(unittest.TestCase):
    def test_threshold_and_ignore_mask(self):
        prob = np.array([[.5, 0., 1.], [.4, .9, 1.]])
        mask = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        valid = np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)
        c = counts(prob, mask, valid)
        self.assertEqual(c, dict(tp=1, fp=1, fn=1, tn=1))
        self.assertEqual(scores(c), dict(iou=1/3, accuracy=.5))

    def test_pool_counts_before_dividing_and_empty_union(self):
        rows = [dict(tp_fp32=10, fp_fp32=0, fn_fp32=0, tn_fp32=1),
                dict(tp_fp32=0, fp_fp32=2, fn_fp32=0, tn_fp32=0)]
        self.assertEqual(pooled(rows, "fp32")["iou"], 10/12)
        self.assertIsNone(scores(dict(tp=0, fp=0, fn=0, tn=10))["iou"])

    def test_perturbation_geometry_and_immutable_input(self):
        x = np.ones((64, 192, 3), np.float32)
        occ = altered(x, "occluded")
        self.assertEqual(np.sum(occ[..., 0] < 1), 16*48)
        self.assertTrue((occ[32:48, 72:120] == .25).all())
        self.assertTrue((altered(x, "dim") == .35).all())
        self.assertTrue((x == 1.).all())


if __name__ == "__main__":
    unittest.main()
