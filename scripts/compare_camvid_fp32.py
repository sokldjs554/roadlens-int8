"""Compare the published FP32 and 26-image INT8 models on 0001TP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_camvid_sequence import (  # noqa: E402
    CAMVID_REV, FIELDS, SEQUENCE, audit_counts, digest, rates, source_bytes,
)
from src.runtime import RoadRuntime  # noqa: E402

FIRST_FRAME = "0001TP_008550"
EXPECTED_FIRST = dict(tp=0, fp=0, fn=2560, tn=9018)
EXPECTED_INT8 = dict(tp=3952, fp=525, fn=117494, tn=587417)


def load_frame(frame, manifest, cache):
    """Decode one shared RGB array and class-ID masks from verified source PNGs."""
    photo = source_bytes("test", frame, manifest, cache)
    label = source_bytes("testannot", frame, manifest, cache)
    with Image.open(io.BytesIO(photo)) as image:
        original_size = image.size
        rgb = np.asarray(image.convert("RGB").resize((192, 64), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    with Image.open(io.BytesIO(label)) as image:
        if image.size != original_size or image.mode not in ("L", "P") or np.asarray(image).max() > 11:
            raise ValueError(f"Expected aligned CamVid class IDs 0 through 11: {frame}")
        ids = np.asarray(image.resize((192, 64), Image.Resampling.NEAREST))
    actual = hashlib.sha256(np.rint(rgb * 255).astype(np.uint8).tobytes()).hexdigest()
    if actual != manifest["input_sha256"]:
        raise ValueError(f"Preprocessed input checksum mismatch: {frame}")
    # Both runtimes receive this exact array, and neither can modify it in place.
    rgb.setflags(write=False)
    return rgb, ids == 3, ids != 11


def count_pixels(pred, truth, valid):
    """Count all four cells independently, excluding label 11 from every cell."""
    if any(mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape != pred.shape
           for mask in (pred, truth, valid)):
        raise ValueError("Boolean masks must have the same two-dimensional shape")
    return dict(tp=int((pred & truth & valid).sum()),
                fp=int((pred & ~truth & valid).sum()),
                fn=int((~pred & truth & valid).sum()),
                tn=int((~pred & ~truth & valid).sum()))


def aggregate_counts(rows):
    """Compute pooled metrics from integer pixel sums, never mean frame scores."""
    totals = {key: sum(row[key] for row in rows) for key in FIELDS}
    return dict(**totals, **rates(totals), frames=len(rows),
                ground_truth_road=totals["tp"] + totals["fn"],
                valid_pixels=sum(totals.values()),
                empty_valid_prediction_frames=sum(row["tp"] + row["fp"] == 0 for row in rows))


def load_model(name, benchmark):
    artifact = "int8" if name == "int8_26" else "fp32"
    path = ROOT / "artifacts" / f"{artifact}.tflite"
    reference = benchmark["models"][artifact]
    sha256 = digest(path)
    if sha256 != reference["sha256"] or path.stat().st_size != reference["bytes"]:
        raise ValueError(f"Model differs from the published artifact: {name}")
    model = RoadRuntime(path)
    contract = model.describe()
    expected_dtype = "int8" if artifact == "int8" else "float32"
    for boundary in ("input", "output"):
        if contract[boundary] != reference[boundary] or contract[boundary]["dtype"] != expected_dtype:
            raise ValueError(f"Unexpected tensor contract: {name}/{boundary}")
    return model, dict(file=f"artifacts/{artifact}.tflite", sha256=sha256,
                       bytes=path.stat().st_size, tensor_contract=contract)


def report(summary):
    lines = ["# 0001TP: FP32와 26장 보정 INT8 비교", "", summary["conclusion"], "",
             "## 62프레임의 픽셀 합계", "",
             "| 모델 | TP | FP | FN | TN | IoU | 도로 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("fp32", "FP32"), ("int8_26", "INT8 · 26장 보정")):
        row = summary["totals"][name]
        counts = " | ".join(f"{row[key]:,}" for key in FIELDS)
        lines.append(f"| {label} | {counts} | {100 * row['iou']:.4f}% | {100 * row['recall']:.4f}% |")
    lines += ["", "IoU와 재현율은 프레임별 점수의 평균이 아니라 TP·FP·FN 합계로 계산했다. "
              "두 모델의 정답 도로는 각각 121,446픽셀, 유효 영역은 각각 709,388픽셀이다.", "",
              "## 검증 조건", "",
              "- INT8 첫 프레임 `0001TP_008550`: TP 0, FP 0, FN 2,560, TN 9,018. "
              "기존 CSV 및 고정한 예상값과 모두 일치한 뒤 전체 비교를 진행했다.",
              "- INT8 62프레임의 네 값 모두 기존 `scenes.csv`와 일치했다(248개 정수 대조). "
              "누적 합계도 TP 3,952, FP 525, FN 117,494, TN 587,417과 일치했다.",
              "- 프레임마다 RGB 배열을 한 번 만들고 같은 읽기 전용 배열을 두 모델에 전달했다. "
              "첫 프레임의 INT8 검산은 전체 비교에 앞서 한 번 추가로 수행했다.",
              "- RGB: 192×64 bilinear, float32/255. 라벨: nearest, road=3, ignore=11. "
              "예측 임계값은 두 모델 모두 probability ≥ 0.5였다.",
              "- 원본 이미지·라벨 124개의 SHA-256과 전처리 RGB 62개의 SHA-256, "
              "두 모델의 해시·입출력 규격을 확인했다. 재학습·재보정·임계값 변경은 없었다.", "",
              "이 비교가 확인하는 범위는 고정한 0001TP 62프레임과 기존 전처리에서의 두 모델 성능이다. "
              "조명·카메라 구도·화면비 등 실패 원인을 구분하려면 별도 실험이 필요하다.", "",
              "## 재현", "", "```bash",
              "python scripts/compare_camvid_fp32.py --check-first",
              "python scripts/compare_camvid_fp32.py --output work/camvid-fp32-repeat",
              "```", "",
              "출력 경로는 새 디렉터리여야 한다. 원본 PNG는 고정한 CamVid 커밋에서 내려받아 "
              "`work/camvid-review/source`에 검증 후 캐시한다. `--camvid-root`로 캐시 위치를 지정할 수 있다.", "",
              "[검산 코드](../../../scripts/compare_camvid_fp32.py) · "
              "[프레임별 124행](frames.csv) · [합계·해시·실행 환경](summary.json) · "
              "[기존 INT8 CSV](../scenes.csv) · [입력 매니페스트](../frames.csv)", ""]
    return "\n".join(lines)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    action = cli.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-first", action="store_true", help="Check only the first INT8 frame")
    action.add_argument("--output", type=Path, help="New directory for the complete comparison")
    cli.add_argument("--scenes", type=Path, default=ROOT / "artifacts/external/scenes.csv")
    cli.add_argument("--frames", type=Path, default=ROOT / "artifacts/external/frames.csv")
    cli.add_argument("--camvid-root", type=Path, default=ROOT / "work/camvid-review/source")
    args = cli.parse_args()
    if not __debug__:
        cli.error("Run without -O so the INT8 reproduction assertions remain enabled")
    if args.output is not None and args.output.exists():
        cli.error("Choose a new output directory; existing results will not be replaced")

    result, groups, manifest = audit_counts(args.scenes, args.frames)
    benchmark_path = ROOT / "artifacts/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    int8, int8_metadata = load_model("int8_26", benchmark)
    assert sorted(manifest)[0] == FIRST_FRAME, "Unexpected first frame"
    first_arrays = load_frame(FIRST_FRAME, manifest[FIRST_FRAME], args.camvid_root)
    rgb, truth, valid = first_arrays
    probabilities, _ = int8.predict(rgb)
    first_counts = count_pixels(probabilities >= 0.5, truth, valid)
    assert first_counts == groups["26"][FIRST_FRAME] == EXPECTED_FIRST, (
        f"First INT8 frame differs: {first_counts}; CSV={groups['26'][FIRST_FRAME]}"
    )
    assert int((truth & valid).sum()) == 2560, "Unexpected first-frame road area"
    first_check = dict(frame=FIRST_FRAME, **first_counts, matches_csv=True)
    print(json.dumps({"first_int8_check": first_check}), flush=True)
    if args.check_first:
        return

    # FP32 is loaded only after the first INT8 frame has reproduced successfully.
    fp32, fp32_metadata = load_model("fp32", benchmark)
    models = {"int8_26": int8, "fp32": fp32}
    rows = []
    for index, frame in enumerate(sorted(manifest), 1):
        rgb, truth, valid = first_arrays if frame == FIRST_FRAME else load_frame(
            frame, manifest[frame], args.camvid_root)
        for name, model in models.items():
            probabilities, _ = model.predict(rgb)
            pred = probabilities >= 0.5
            counts = count_pixels(pred, truth, valid)
            if name == "int8_26":
                assert counts == groups["26"][frame], (
                    f"INT8 counts differ for {frame}: {counts}; CSV={groups['26'][frame]}"
                )
            rows.append(dict(sequence=SEQUENCE, frame=frame, model=name, **counts,
                             iou=rates(counts)["iou"], recall=rates(counts)["recall"],
                             ground_truth_road=int((truth & valid).sum()), valid_pixels=int(valid.sum())))
        if index % 10 == 0 or index == len(manifest):
            print(f"Verified INT8 and compared FP32: {index}/{len(manifest)} frames", flush=True)

    totals = {name: aggregate_counts([row for row in rows if row["model"] == name]) for name in models}
    assert {key: totals["int8_26"][key] for key in FIELDS} == EXPECTED_INT8, "Unexpected INT8 totals"
    assert all(totals[name][key] == totals["int8_26"][key] for name in models
               for key in ("frames", "ground_truth_road", "valid_pixels")), "Evaluation regions differ"
    delta = {key: 100 * (totals["int8_26"][key] - totals["fp32"][key]) for key in ("iou", "recall")}
    conclusion = (f"0001TP에서 FP32 재현율은 {100 * totals['fp32']['recall']:.2f}%, "
                  f"INT8은 {100 * totals['int8_26']['recall']:.2f}%였고, "
                  f"INT8−FP32 차이는 {delta['recall']:+.4f}%p였다.")
    summary = dict(sequence=SEQUENCE, frames=len(manifest), totals=totals,
                   int8_minus_fp32_pp=delta, conclusion=conclusion,
                   validation=dict(first_int8_check=first_check, matched_int8_frames=len(manifest),
                                   matched_int8_count_values=4 * len(manifest),
                                   source_png_sha256_checks=2 * len(manifest),
                                   preprocessed_rgb_sha256_checks=len(manifest)),
                   formulas=result["formulas"],
                   preprocessing=dict(rgb="RGB -> bilinear 192x64 -> float32 / 255",
                                      labels="Class IDs -> nearest 192x64", road_label=3, ignored_label=11,
                                      same_readonly_rgb_array_for_both_models=True, threshold=0.5),
                   camvid_commit=CAMVID_REV, interpreter_threads=1,
                   models=dict(int8_26=int8_metadata, fp32=fp32_metadata),
                   input_sha256=dict(**result["input_sha256"], benchmark_json=digest(benchmark_path)),
                   code_sha256={str(path.relative_to(ROOT)): digest(path) for path in
                                (Path(__file__).resolve(), ROOT / "scripts/verify_camvid_sequence.py",
                                 ROOT / "src/runtime.py")},
                   versions=dict(python=platform.python_version(), **{
                       name: importlib.metadata.version(name) for name in ("tensorflow-cpu", "numpy", "pillow")}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        stage = Path(temporary) / "comparison"
        stage.mkdir()
        with (stage / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "README.md").write_text(report(summary), encoding="utf-8")
        summary["files_sha256"] = {path.name: digest(path) for path in sorted(stage.iterdir())}
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
        stage.rename(args.output)
    print(json.dumps({"totals": totals, "int8_minus_fp32_pp": delta, "conclusion": conclusion},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
