"""Train and convert the KITTI road segmentation CNN. Run as python -m src.train."""
import argparse
import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "3")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .data import HEIGHT, SEED, WIDTH, build_split, load_sample


def build_model():
    layers = keras.layers

    def conv(x, channels, stride=1):
        x = layers.Conv2D(channels, 3, strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=.9)(x)
        return layers.ReLU(max_value=6)(x)

    def decode(x, skip, channels):
        x = layers.Resizing(skip.shape[1], skip.shape[2], interpolation="nearest")(x)
        return conv(layers.Concatenate()([x, skip]), channels)

    rgb = keras.Input((HEIGHT, WIDTH, 3), name="rgb")
    half = conv(rgb, 12, 2)
    quarter = conv(half, 20, 2)
    eighth = conv(quarter, 32, 2)
    low = conv(conv(eighth, 48, 2), 48)
    x = decode(low, eighth, 32)
    x = decode(x, quarter, 20)
    x = decode(x, half, 12)
    x = layers.Conv2D(1, 1)(x)
    x = layers.Resizing(HEIGHT, WIDTH, interpolation="nearest")(x)
    road = layers.Activation("sigmoid", name="road")(x)
    return keras.Model(rgb, road, name="road_segmentation")


def masked_loss(label, valid, pred):
    eps = tf.constant(1e-6, tf.float32)
    p = tf.clip_by_value(pred, eps, 1 - eps)
    bce = -(label * tf.math.log(p) + (1 - label) * tf.math.log1p(-p))
    axes = (1, 2, 3)
    loss = tf.reduce_sum(bce * valid, axis=axes) / tf.maximum(tf.reduce_sum(valid, axis=axes), 1.)
    overlap = tf.reduce_sum(p * label * valid, axis=axes)
    mass = tf.reduce_sum((p + label) * valid, axis=axes)
    return tf.reduce_mean(loss + .4 * (1 - (2 * overlap + eps) / (mass + eps)))


def fit(model, samples, epochs=30, batch_size=12):
    x, y, valid = (np.stack(a).astype("float32") for a in zip(*samples))
    rng = np.random.default_rng(SEED)
    optimizer = keras.optimizers.Adam(learning_rate=.002)

    @tf.function(reduce_retracing=True)
    def step(images, labels, known):
        with tf.GradientTape() as tape:
            loss = masked_loss(labels, known, model(images, training=True))
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    history = []
    for epoch in range(epochs):
        optimizer.learning_rate.assign(.0002 + .0018 * .5 * (1 + np.cos(np.pi * epoch / epochs)))
        ids = rng.permutation(len(x))
        total = 0.
        for start in range(0, len(x), batch_size):
            batch = ids[start:start + batch_size]
            images, labels, known = x[batch], y[batch], valid[batch]
            flip = rng.random(len(batch)) < .5
            images[flip] = images[flip, :, ::-1]
            labels[flip] = labels[flip, :, ::-1]
            known[flip] = known[flip, :, ::-1]
            aug_shape = (len(batch), 1, 1, 1)
            images = np.clip(images * rng.uniform(.85, 1.15, aug_shape)
                             + rng.uniform(-.07, .07, aug_shape), 0., 1.).astype("float32")
            total += float(step(images, labels[..., None], known[..., None])) * len(batch)
        history.append(total / len(x))
        print(f"Epoch {epoch + 1:2d}/{epochs} loss {history[-1]:.5f}", flush=True)
    return history


def convert(model, calibration, out):
    @tf.function(input_signature=[tf.TensorSpec((1, HEIGHT, WIDTH, 3), tf.float32, name="rgb")])
    def serve(rgb):
        return model(rgb, training=False)

    # Freeze variables to constants so batch norm is folded and both exports share one graph.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    graph = convert_variables_to_constants_v2(serve.get_concrete_function())
    converted = {}
    for precision in ("fp32", "int8"):
        converter = tf.lite.TFLiteConverter.from_concrete_functions([graph])
        if precision == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.representative_dataset = lambda: ([x[None]] for x in calibration)
        converted[precision] = converter.convert()
    for precision, flatbuffer in converted.items():
        (out / f"{precision}.tflite").write_bytes(flatbuffer)


def keras_parity(model, test, archive, out):
    runner = tf.lite.Interpreter(model_path=str(out / "fp32.tflite"), num_threads=1)
    runner.allocate_tensors()
    inp, output = runner.get_input_details()[0], runner.get_output_details()[0]
    largest, error_sum, elements = 0., 0., 0
    for id in test:
        x = load_sample(archive, id)[0][None]
        expected = model(x, training=False).numpy()
        runner.set_tensor(inp["index"], x)
        runner.invoke()
        diff = np.abs(runner.get_tensor(output["index"]) - expected)
        if not np.all(np.isfinite(diff)):
            raise RuntimeError(f"Nonfinite parity difference for {id}")
        largest = max(largest, float(np.max(diff)))
        error_sum += diff.sum(dtype=np.float64)
        elements += diff.size
    if largest > 2e-4:
        raise RuntimeError(f"Keras and FP32 TFLite parity exceeded tolerance: {largest}")
    return {"frames": len(test), "pixels": elements, "max_abs_error": largest,
            "mean_abs_error": float(error_sum / elements), "max_abs_tolerance": 2e-4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    # Keep the published weights, models and metadata intact if conversion or parity fails.
    with tempfile.TemporaryDirectory(dir=args.out, prefix=".training-") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(args.data) as archive, zipfile.ZipFile(args.mapping) as mapping:
            split = build_split(archive, mapping)
            (stage / "split.json").write_text(json.dumps(split, indent=2) + "\n")
            train = [load_sample(archive, id) for id in split["train"]]
            model = build_model()
            start = time.perf_counter()
            history = fit(model, train, args.epochs)
            elapsed = time.perf_counter() - start
            model.save(stage / "road.keras")
            convert(model, [load_sample(archive, id)[0] for id in split["cal"]], stage)
            restored = keras.models.load_model(stage / "road.keras", compile=False)
            parity = keras_parity(restored, split["test"], archive, stage)
        meta = {"seed": SEED, "epochs": args.epochs, "parameters": model.count_params(),
                "tensorflow": tf.__version__, "keras": keras.__version__, "loss_by_epoch": history,
                "training_seconds": elapsed, "keras_fp32_parity": parity}
        for source, path in (("data", args.data), ("mapping", args.mapping)):
            with path.open("rb") as f:
                meta[source + "_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        (stage / "training.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(parity))
        for path in sorted(stage.iterdir()):
            os.replace(path, args.out / path.name)


if __name__ == "__main__":
    main()
