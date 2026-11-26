"""Batch-one road segmentation with an exclusive, single-threaded TFLite interpreter."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import numpy as np

INPUT_SHAPE = (1, 64, 192, 3)
OUTPUT_SHAPE = (1, 64, 192, 1)


class ModelError(RuntimeError):
    """The model could not be loaded or did not satisfy the inference contract."""


class InferenceError(RuntimeError):
    """Inference failed for this frame. No previous prediction is returned."""


def _interpreter(path):
    from tensorflow import lite

    return lite.Interpreter(model_path=str(path), num_threads=1)


def _check_tensor(details, shape):
    if len(details) != 1 or tuple(details[0]["shape"]) != shape:
        raise ValueError(f"Expected exactly one tensor with shape {shape}")
    info = details[0]
    dtype = np.dtype(info["dtype"])
    if dtype not in (np.dtype(np.float32), np.dtype(np.int8)):
        raise ValueError(f"Unsupported I/O type {dtype}")
    if dtype == np.int8:
        scale, zero = info["quantization"]
        if not np.isfinite(scale) or scale <= 0 or int(zero) != zero or not -128 <= zero <= 127:
            raise ValueError(f"Invalid INT8 quantization parameters {scale}, {zero}")
        params = info["quantization_parameters"]
        if len(params["scales"]) != 1 or len(params["zero_points"]) != 1:
            raise ValueError("Expected per-tensor INT8 quantization at the model boundary")
    return info


class RoadRuntime:
    """Own one instance per serial inference worker, with lazy recovery after failure."""

    def __init__(self, model_path, interpreter_factory=None):
        self.path = Path(model_path)
        self.factory = interpreter_factory or _interpreter
        self.interpreter = None
        self._load()

    def _load(self):
        try:
            interpreter = self.factory(self.path)
            interpreter.allocate_tensors()
            self.input = _check_tensor(interpreter.get_input_details(), INPUT_SHAPE)
            self.output = _check_tensor(interpreter.get_output_details(), OUTPUT_SHAPE)
            if self.input["dtype"] != self.output["dtype"]:
                raise ValueError("Expected matching input and output types")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise ModelError(f"Cannot load {self.path.name}: {exc}") from exc
        self.interpreter = interpreter

    def describe(self):
        def tensor(detail):
            scale, zero = detail["quantization"]
            return dict(shape=list(map(int, detail["shape"])), dtype=np.dtype(detail["dtype"]).name,
                        scale=float(scale), zero_point=int(zero))

        return dict(input=tensor(self.input), output=tensor(self.output))

    def predict(self, rgb):
        """Return an H×W probability map and invoke-only elapsed nanoseconds."""
        if not isinstance(rgb, np.ndarray) or rgb.shape != INPUT_SHAPE[1:] or rgb.dtype != np.float32:
            raise ValueError("Expected float32 RGB with shape (64, 192, 3)")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Expected finite RGB values in [0, 1]")
        if self.interpreter is None:
            self._load()
        batch = rgb[np.newaxis]
        if self.input["dtype"] == np.int8:
            scale, zero = self.input["quantization"]
            batch = np.clip(np.rint(batch / scale) + zero, -128, 127).astype(np.int8)
        try:
            self.interpreter.set_tensor(self.input["index"], batch)
            start = perf_counter_ns()
            self.interpreter.invoke()
            elapsed = perf_counter_ns() - start
            raw = self.interpreter.get_tensor(self.output["index"])
            if raw.shape != OUTPUT_SHAPE or raw.dtype != self.output["dtype"]:
                raise ValueError("Output tensor shape/type changed")
            prob = raw[0, :, :, 0].astype(np.float32)
            if raw.dtype == np.int8:
                scale, zero = self.output["quantization"]
                prob = (prob - zero) * scale
            if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                raise ValueError("Expected finite output probabilities in [0, 1]")
        except (ValueError, RuntimeError, OSError) as exc:
            self.interpreter = None
            raise InferenceError(f"Frame failed on {self.path.name}: {exc}") from exc
        return prob, elapsed
