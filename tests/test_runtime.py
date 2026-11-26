from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.runtime import INPUT_SHAPE, OUTPUT_SHAPE, InferenceError, ModelError, RoadRuntime


def tensor(shape, dtype, scale, zero):
    return dict(index=0, shape=shape, dtype=dtype, quantization=(scale, zero),
                quantization_parameters={"scales": [scale], "zero_points": [zero]})


class Stub:
    def __init__(self, dtype=np.float32, fail=False):
        self.fail = fail
        self.input = tensor(INPUT_SHAPE, dtype, 1 / 255, -128)
        self.output = tensor(OUTPUT_SHAPE, dtype, 1 / 256, -128)

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [self.input]

    def get_output_details(self):
        return [self.output]

    def set_tensor(self, index, value):
        self.last = value

    def invoke(self):
        if self.fail:
            raise RuntimeError("simulated interpreter failure")

    def get_tensor(self, index):
        return np.zeros(OUTPUT_SHAPE, dtype=self.output["dtype"])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros(INPUT_SHAPE[1:], dtype=np.float32)

    def test_quantization_dequantization_and_endpoints(self):
        stub = Stub(np.int8)
        runtime = RoadRuntime("test", lambda path: stub)
        self.rgb[0, 0] = [0., .5, 1.]
        prob, elapsed = runtime.predict(self.rgb)
        np.testing.assert_array_equal(stub.last[0, 0, 0], [-128, -1, 127])
        np.testing.assert_array_equal(prob, np.full(OUTPUT_SHAPE[1:3], .5))
        self.assertGreaterEqual(elapsed, 0)

    def test_invalid_inputs_do_not_poison_next_valid_frame(self):
        runtime = RoadRuntime("test", lambda path: Stub())
        for bad in (self.rgb[1:], self.rgb.astype(np.uint8), self.rgb + np.nan,
                    self.rgb - .01, self.rgb + 1.1):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                runtime.predict(bad)
        self.assertEqual(runtime.predict(self.rgb)[0].shape, OUTPUT_SHAPE[1:3])

    def test_failed_invoke_reopens_and_recovers_on_next_frame(self):
        stubs = iter((Stub(fail=True), Stub()))
        runtime = RoadRuntime("test", lambda path: next(stubs))
        with self.assertRaises(InferenceError):
            runtime.predict(self.rgb)
        self.assertIsNone(runtime.interpreter)
        runtime.predict(self.rgb)

    def test_unloadable_model_and_bad_tensor_metadata(self):
        def corrupt(_path):
            raise ValueError("bad FlatBuffer")
        with self.assertRaises(ModelError):
            RoadRuntime("bad", corrupt)
        for field, value in (("dtype", np.int16), ("shape", (1, 48, 192, 3)),
                             ("quantization", (0., 0))):
            stub = Stub(np.int8)
            stub.input[field] = value
            with self.subTest(field=field), self.assertRaises(ModelError):
                RoadRuntime("bad", lambda path: stub)

    def test_actual_corrupt_flatbuffer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.tflite"
            path.write_bytes(b"corrupt FlatBuffer model")
            with self.assertRaises(ModelError):
                RoadRuntime(path)

    def test_shipped_models_accept_next_frame_after_input_rejection(self):
        root = Path(__file__).resolve().parents[1] / "artifacts"
        for name in ("int8", "fp32"):
            with self.subTest(model=name):
                runtime = RoadRuntime(root / f"{name}.tflite")
                before, _ = runtime.predict(self.rgb)
                with self.assertRaises(ValueError):
                    runtime.predict(self.rgb.astype(np.uint8))
                after, _ = runtime.predict(self.rgb)
                np.testing.assert_array_equal(before, after)
                self.assertTrue(np.isfinite(after).all())

    def test_invalid_output_rejected(self):
        stub = Stub()
        stub.get_tensor = lambda index: np.full(OUTPUT_SHAPE, np.nan, np.float32)
        with self.assertRaises(InferenceError):
            RoadRuntime("test", lambda path: stub).predict(self.rgb)


if __name__ == "__main__":
    unittest.main()
