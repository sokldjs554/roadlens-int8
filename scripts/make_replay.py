"""Encode the saved KITTI gallery as six comparison cards followed by a three-second benchmark summary."""
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


def summary(benchmark):
    fp = benchmark["models"]["fp32"]
    quant = benchmark["models"]["int8"]
    latency_drop = 100 * (1 - quant["p50_ms"] / fp["p50_ms"])
    size_drop = 100 * (1 - quant["bytes"] / fp["bytes"])
    lines = [
        f"자체 평가 {benchmark['n']}장 · 도로 IoU {fp['iou']:.2%} → {quant['iou']:.2%}",
        f"CPU 순수 추론 p50 {fp['p50_ms']:.3f} → {quant['p50_ms']:.3f}ms · {latency_drop:.1f}% 감소",
        f"모델 파일 크기 {size_drop:.1f}% 감소",
    ]
    image = Image.new("RGB", (1280, 640), "#10172e")
    draw = ImageDraw.Draw(image)
    draw.text((1232, 20), "결과 요약   /   FP32 → INT8", font=font(33, True),
              fill="#afbcff", anchor="rt")
    body = font(50, True)
    for text, y in zip(lines, (98, 192, 286)):
        if draw.textbbox((48, y), text, font=body)[2] > image.width - 32:
            raise ValueError(f"Summary line exceeds video safe area: {text}")
        draw.text((48, y), text, font=body, fill="#f2f5ff")
    return image


def encode_cards(pattern, destination):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1/3", "-i", str(pattern),
                    "-vf", "fps=24", "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(destination)], check=True)


def main():
    scenes = json.loads((WEB / "results.json").read_text())["scenes"]
    benchmark = json.loads((WEB.parent / "artifacts" / "benchmark.json").read_text())
    with tempfile.TemporaryDirectory(dir=WEB) as tmp:
        stage = Path(tmp)
        for i, scene in enumerate(scenes):
            card(scene, i, len(scenes)).save(stage / f"{i:03}.png")
        encode_cards(stage / "%03d.png", stage / "scenes.mp4")
        summary(benchmark).save(stage / "summary-000.png")
        encode_cards(stage / "summary-%03d.png", stage / "summary.mp4")
        # Keep the original scene frames intact and start the summary on a keyframe.
        (stage / "clips.txt").write_text("file 'scenes.mp4'\nfile 'summary.mp4'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-i", "clips.txt",
                        "-c", "copy", "-movflags", "+faststart", "roadlens-replay.mp4"],
                       cwd=stage, check=True)
        (stage / "roadlens-replay.mp4").replace(WEB / "roadlens-replay.mp4")


if __name__ == "__main__":
    main()
