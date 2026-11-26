"""Encode the saved KITTI gallery as six three-second comparison cards."""
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


WEB = Path(__file__).resolve().parents[1] / "web"
FONT_DIR = Path("/usr/share/fonts/opentype/noto")


def font(size, bold=False):
    weight = "Bold" if bold else "Regular"
    return ImageFont.truetype(str(FONT_DIR / f"NotoSansCJK-{weight}.ttc"), size, index=1)


def card(scene, index, total):
    image = Image.new("RGB", (1280, 640), "#10172e")
    draw = ImageDraw.Draw(image)
    draw.text((26, 10), f"RoadLens   /   {scene['id']}   ·   {scene['condition']}",
              font=font(31, True), fill="#f2f5ff")
    draw.text((1250, 24), f"{index + 1} / {total}", font=font(21), fill="#bac8ff", anchor="rt")
    for slot, (key, label) in enumerate((("input", "입력"), ("truth", "도로 정답"),
                                         ("fp32", "FP32"), ("int8", "INT8"))):
        x, y = 26 + (slot % 2) * 624, 78 + (slot // 2) * 251
        if key in ("fp32", "int8"):
            label += f"   도로 IoU {scene['iou_' + key]:.2%}"
        draw.text((x, y), label, font=font(24, True), fill="#afbcff" if slot < 2 else "#ffb899")
        with Image.open(WEB / scene[key]) as source:
            fitted = ImageOps.contain(source.convert("RGB"), (602, 190), Image.Resampling.LANCZOS)
            image.paste(fitted, (x + (602 - fitted.width) // 2, y + 42))
    draw.text((26, 585), "저장된 추론 결과  ·  KITTI Road  ·  CC BY-NC-SA 3.0",
              font=font(22), fill="#bdc9e8")
    return image


def main():
    scenes = json.loads((WEB / "results.json").read_text())["scenes"]
    with tempfile.TemporaryDirectory(dir=WEB) as tmp:
        stage = Path(tmp)
        for i, scene in enumerate(scenes):
            card(scene, i, len(scenes)).save(stage / f"{i:03}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1/3", "-i",
                        str(stage / "%03d.png"), "-vf", "fps=24", "-c:v", "libx264",
                        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(stage / "roadlens-replay.mp4")], check=True)
        (stage / "roadlens-replay.mp4").replace(WEB / "roadlens-replay.mp4")


if __name__ == "__main__":
    main()
