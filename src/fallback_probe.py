"""A tiny diagnostic graph that forces a host island between quantized convolutions."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
import numpy as np
import tensorflow as tf

from src.compiler import compile_model


def build_probe(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kernel = tf.constant([[[[.75]]]], tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([1, 8, 8, 1], tf.float32)])
    def graph(x):
        first = tf.nn.conv2d(x, kernel, 1, 'SAME')
        return tf.nn.conv2d(tf.sin(first), kernel, 1, 'SAME')

    x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(1, 8, 8, 1)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([graph.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: iter([[x], [-x]])
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model = out / "sin_probe.tflite"
    model.write_bytes(converter.convert())
    result = compile_model(model, out)
    if result['npu_partitions'] < 2 or 'SIN' not in result['host_ops'] or not result['crossing_edges']:
        raise RuntimeError("Expected host SIN between accelerated regions, inspect placement.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(build_probe(args.out)["host_ops"])
