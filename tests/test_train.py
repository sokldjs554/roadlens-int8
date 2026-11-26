import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data import HEIGHT, WIDTH, load_sample
from src.train import build_model, convert, masked_loss


ROOT = Path(__file__).resolve().parents[1]


def png(a):
    stream = io.BytesIO()
    Image.fromarray(a).save(stream, format="PNG")
    return stream.getvalue()


class TrainingTests(unittest.TestCase):
    def test_rgb_ground_truth_and_road_lane_distinction(self):
        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        rgb[..., 1] = 128
        mask = np.zeros_like(rgb)
        mask[:, :64] = [255, 0, 0]
        mask[:, 64:128] = [255, 0, 255]
        with zipfile.ZipFile(io.BytesIO(), "w") as archive:
            archive.writestr("data_road/training/image_2/um_000000.png", png(rgb))
            archive.writestr("data_road/training/gt_image_2/um_road_000000.png", png(mask))
            archive.writestr("data_road/training/gt_image_2/um_lane_000000.png", png(rgb))
            image, road, valid = load_sample(archive, "um_000000")
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(image.shape, (64, 192, 3))
        np.testing.assert_allclose(image[..., 1], 128/255)
        np.testing.assert_array_equal(road, mask[..., 2] > 0)
        np.testing.assert_array_equal(valid, mask[..., 0] > 0)

    def test_loss_ignores_predictions_at_void_pixels(self):
        labels = tf.constant([[[[1.], [0.]]]])
        valid = tf.constant([[[[1.], [0.]]]])
        a = tf.constant([[[[.9], [.98]]]])
        b = tf.constant([[[[.9], [.01]]]])
        self.assertAlmostEqual(float(masked_loss(labels, valid, a)),
                               float(masked_loss(labels, valid, b)))

    def test_actual_quantizer_rejects_bad_calibration_and_preserves_published_pair(self):
        # Missing RGB dimension reaches the real CONV_2D calibration kernel.
        invalid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        model = build_model()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            previous = {name: (ROOT / "artifacts" / name).read_bytes()
                        for name in ("fp32.tflite", "int8.tflite")}
            for name, contents in previous.items():
                (out / name).write_bytes(contents)
            with self.assertRaisesRegex(RuntimeError, "CONV_2D.*failed to prepare"):
                convert(model, [invalid], out)
            for name, contents in previous.items():
                self.assertEqual((out / name).read_bytes(), contents)
            rgb = np.random.default_rng(731).random((HEIGHT, WIDTH, 3), dtype=np.float32)
            convert(model, [rgb], out)
            for name, contents in previous.items():
                restored = tf.lite.Interpreter(model_path=str(out / name), num_threads=1)
                restored.allocate_tensors()
                self.assertNotEqual((out / name).read_bytes(), contents)

    def test_committed_partitions_have_unique_drives_and_frame_ids(self):
        split = json.loads((ROOT / "artifacts/split.json").read_text())
        partitions = ("train", "cal", "test")
        ids = [id for p in partitions for id in split[p]]
        drives = [drive for p in partitions for drive in split["drives"][p]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(set(ids)), 289)
        self.assertEqual(len(drives), len(set(drives)))
        self.assertEqual(len(set(drives)), 37)
        for p in partitions:
            self.assertEqual({id.split("_")[0] for id in split[p]}, {"um", "umm", "uu"})

    def test_both_exports_execute_and_quantized_graph_has_no_float_tensors(self):
        x = np.random.default_rng(9).random((1, HEIGHT, WIDTH, 3), dtype=np.float32)
        outputs = {}
        for name, dtype in (("fp32", np.float32), ("int8", np.int8)):
            itp = tf.lite.Interpreter(model_path=str(ROOT / "artifacts" / f"{name}.tflite"), num_threads=1)
            itp.allocate_tensors()
            inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
            self.assertEqual(inp["dtype"], dtype)
            self.assertEqual(out["dtype"], dtype)
            np.testing.assert_array_equal(inp["shape"], x.shape)
            np.testing.assert_array_equal(out["shape"], [1, HEIGHT, WIDTH, 1])
            if name == "int8":
                self.assertTrue(all(not np.issubdtype(t["dtype"], np.floating)
                                    for t in itp.get_tensor_details()))
                scale, zero = inp["quantization"]
                self.assertGreater(scale, 0)
                qx = np.clip(np.rint(x / scale) + zero, -128, 127).astype(np.int8)
            itp.set_tensor(inp["index"], x if name == "fp32" else qx)
            itp.invoke()
            y = itp.get_tensor(out["index"]).astype(np.float32)
            if name == "int8":
                scale, zero = out["quantization"]
                y = (y - zero) * scale
            self.assertTrue(np.all(np.isfinite(y) & (y >= 0) & (y <= 1)))
            outputs[name] = y
        self.assertLess(np.mean(np.abs(outputs["int8"] - outputs["fp32"])), .05)


if __name__ == "__main__":
    unittest.main()
