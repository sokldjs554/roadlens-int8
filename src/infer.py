"""Run the released FP32 and INT8 models on one image and save both masks in one PNG."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .bench import CUTOFF
from .data import decode_rgb
from .runtime import InferenceError, ModelError, RoadRuntime


ART = Path(__file__).resolve().parents[1] / "artifacts"
# Palette index bits represent FP32 (bit 0) and INT8 (bit 1) road predictions.
PALETTE = [16, 23, 46, 255, 151, 82, 178, 138, 255, 89, 210, 222]


def compare_image(source, output):
    if output.suffix.lower() != ".png":
        raise ValueError("Output must have the .png extension")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    rgb = decode_rgb(source.read_bytes())
    masks = []
    for name in ("fp32", "int8"):
        probability, _ = RoadRuntime(ART / f"{name}.tflite").predict(rgb)
        masks.append(probability >= CUTOFF)
    fp32, int8 = masks
    labels = fp32.astype(np.uint8) | (int8.astype(np.uint8) << 1)
    image = Image.fromarray(labels)
    image.putpalette(PALETTE)
    output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output.parent) as stage:
        prepared = Path(stage) / "mask.png"
        image.save(prepared)
        # Publish the complete PNG exclusively, including when two jobs finish together.
        os.link(prepared, output)
    return dict(output=str(output), width=image.width, height=image.height,
                fp32_road_pixels=int(fp32.sum()), int8_road_pixels=int(int8.sum()),
                different_pixels=int(np.count_nonzero(fp32 != int8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path, help="New comparison PNG")
    args = parser.parse_args()
    try:
        result = compare_image(args.image, args.output)
    except (OSError, ValueError, ModelError, InferenceError, Image.DecompressionBombError) as error:
        parser.exit(1, f"roadlens: {error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
