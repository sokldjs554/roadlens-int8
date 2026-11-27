# RoadLens · 전방 카메라 도로 분할의 INT8 실행 분석

**[지원자 보정 실험 결과](artifacts/calibration/README.md) · [21초 결과 영상](https://roadlens-kitti-int8.ykh13571.chatgpt.site/replay.html) · [공개 데모](https://roadlens-kitti-int8.ykh13571.chatgpt.site) · [소스 다운로드](https://roadlens-kitti-int8.ykh13571.chatgpt.site/downloads/roadlens-source.zip) · [직무 요건별 근거표](docs/evidence.md) · [실험 원본](artifacts)**

**지원자 직접 실험 · INT8 보정 이미지는 몇 장 필요한가?** 같은 가중치와 평가 62장을 고정하고 보정 3·9·26장을 비교했다. 도로 유형별로 한 장씩 고른 3장은 세 시드 모두 26장 기준 대비 전체 도로 IoU 하락 허용폭 0.5%p를 충족했다. 개별 장면 `umm_000083`에서는 3장 모델의 IoU가 1.75%p 낮아졌다. [장면 비교와 실험 결과](artifacts/calibration/README.md) · [7개 조건 원시 CSV](artifacts/calibration/runs.csv)

후속 검증에서 Codex는 보정 선택을 20개 시드로 확장하고 CPU에서 PNG 디코딩부터 마스크 출력까지 측정했다. [시드별 편차·CPU 구간별 측정·재현 방법](docs/calibration_review.md)

실제 주행 이미지에서 도로를 분할하는 79,569개 파라미터의 CNN을 학습하고, 같은 가중치를 FP32·완전 정수 INT8 TFLite로 변환했다. 동일한 평가 62장에서 INT8은 모델 파일을 **71.2%** 줄이고 이 CPU의 순수 추론 지연시간 중앙값을 **56.0%** 줄였다. 원본 장면의 도로 IoU는 77.33%와 77.55%였다. 공개 데모는 **저장된 추론 결과를 재생**한다.

![FP32와 INT8의 도로 경계 및 실측 비교](artifacts/checks/desktop-preview.png)

| 평가 및 실행 조건 | FP32 | INT8 |
|---|---:|---:|
| 도로 IoU · 이미지 평면 유효 픽셀 합산 | 77.331% | 77.547% |
| 픽셀 정확도 | 95.207% | 95.242% |
| CPU invoke p50 / p95 | 0.971 / 1.108 ms | 0.427 / 0.481 ms |
| 모델 파일 크기 | 322,536 B | 92,832 B |
| 워밍업 직후 전체 프로세스 RSS | 545.34 MiB | 545.11 MiB |
| 모델 로드 전 대비 RSS 증가 | 7.03 MiB | 5.41 MiB |

KITTI Road 라벨 289장을 **원본 주행 ID로 분리**했다. 학습 201장(26개 주행), 보정 26장(4개), 평가 62장(7개)이다. Linux x86-64, AMD EPYC 9V74, TensorFlow Lite 2.19.1, CPU/XNNPACK, 배치 1, 스레드 1. 모델마다 별도 프로세스에서 20회 워밍업한 뒤 같은 순서로 62장 × 10회 호출해 측정했다. CPU p50에는 디코딩·리사이즈·입출력 양자화를 제외했다. 모델 로드 시점 RSS는 모델 저장용량이나 보드 SRAM 사용량과 다르다. [원시 시간 CSV](artifacts/latency.csv) · [정확도·메모리 측정 기준](docs/evaluation.md)

[회사 SDK·국내외 13개 사례와의 최종 대조](docs/final_comparison.md)에는 기능, 모델 변환, 실행 장치, 평가 및 데모의 근거를 정리했다. Codex 후속 실험에서 [입력 밝기 규칙](docs/exposure.md)은 합성 저조도에서 개선됐지만 정상 장면 하락 한도를 넘어 기본 경로에 적용하지 않았다.

## 어디에서 실패했나

원본 무표시 도로의 INT8 IoU는 70.89%였다. 중앙을 가린 합성 입력에서는 74.73%, RGB 밝기를 35%로 줄인 합성 저조도에서는 33.06%였다. 데모의 `um_000028_dim`에서는 두 모델 모두 정답 도로 2,252픽셀 전부를 놓쳤다. 이 데이터는 실제 야간이나 소형 객체 검출 결과를 제공하지 않는다. 검출 임계값은 0.5로 고정했다. [장면별 정답·예측 통계](artifacts/scene_metrics.csv) · [개별 조건과 장면 선택](docs/evaluation.md)

## NPU를 위해 확인한 범위

| 구간 | 검증 근거 |
|---|---|
| CPU 실제 추론 | Keras 복원 결과와 FP32 TFLite의 최대 확률 차이 0.00000370. FP32·INT8 TFLite 모두 CPU에서 실행 |
| Arm Ethos-U55-128용 컴파일 | 실제 Vela 4.4.1 출력에서 FP32는 CPU 연산 17개, INT8은 NPU 구간 1개·CPU 연산 0개 |
| 지원하지 않는 연산 | 별도 SIN 진단 그래프에서 NPU 2구간, CPU 3연산과 경계 2개를 검출 |
| 컴파일러 추정 | 500 MHz 예제 설정에서 INT8 628,069 cycles, SRAM 276 KiB. NPU 보드에서의 실행·시간·전력은 미측정 |

[CPU 실행기](src/runtime.py) · [컴파일 어댑터](src/compiler.py) · [원본 그래프와 컴파일러 결과](artifacts/vela) · [CPU/NPU 경계 해석](docs/runtime.md)

## 재현

Linux x86-64와 Python 3.12에서 실행한다. Pytest는 커밋된 두 모델과 컴파일 산출물을 읽으며 KITTI 다운로드 없이 실행할 수 있다.

```bash
git clone https://github.com/sokldjs554/roadlens-int8.git
cd roadlens-int8
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
python -m pytest -q
```

### 새 이미지 한 장에서 두 모델 실행

저장소에 포함된 전방 카메라 이미지로 바로 실행할 수 있다. 입력 파일을 자기 PNG 또는 JPEG로 바꾸면 두 모델이 로컬 CPU에서 새로 추론한다.

```bash
python -m src.infer web/scenes/um_000032_clean_input.jpg work/paired-mask.png
```

결과 PNG는 192×64다. 남색은 두 모델 모두 배경, 청록은 모두 도로, 주황은 FP32만 도로, 보라는 INT8만 도로인 픽셀이다. 표준 출력에는 모델별 도로 픽셀 수와 판정이 다른 픽셀 수가 기록된다. 정답 없는 입력에서는 IoU를 계산할 수 없다. 기존 출력 파일 또는 읽을 수 없는 입력을 지정하면 오류 코드로 종료한다. 샘플 JPEG는 데모용 축소본이므로 평가 원본 PNG와 픽셀값이 다를 수 있다.

### 데이터와 코드의 흐름

| 단계 | 코드 | 출력과 검증 |
|---|---|---|
| 주행 단위 분리 → CNN 학습 → FP32·INT8 변환 | [데이터](src/data.py) · [학습·변환](src/train.py) | 분할 ID, 가중치, TFLite 모델 |
| 새 이미지 → 정규화 → 추론 → 마스크 저장 | [단일 이미지 CLI](src/infer.py) · [CPU 실행기](src/runtime.py) | 두 마스크를 함께 복원할 수 있는 PNG, 입력·모델·출력 오류 검사 |
| 고정 평가 → 보정 실험 → 실행 구간 분석 | [벤치마크](src/bench.py) · [보정](src/calibration_experiment.py) · [프로파일러](src/profile_pipeline.py) | 같은 62장의 픽셀 통계, 시드·호출별 CSV |
| NPU 컴파일 → 공개 결과 조회 | [Vela 어댑터](src/compiler.py) · [사이트 빌드](scripts/build_site.py) | 연산 배치, 저장된 장면, 소스 ZIP과 배포 해시 |

동일한 평가 집합에서 추론을 다시 실행한다. 아래 명령은 기록된 원본을 보존하면서 새 결과를 `work/recheck`에 저장한다.

```bash
mkdir -p work/recheck/artifacts
curl -fL --retry 3 https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip -o work/data_road.zip
cp artifacts/{fp32.tflite,int8.tflite,split.json} work/recheck/artifacts/
python -m src.bench --archive work/data_road.zip --artifacts work/recheck/artifacts --web work/recheck/gallery
python -m src.compiler artifacts/int8.tflite work/recheck/vela-int8
python -m src.compiler artifacts/fp32.tflite work/recheck/vela-fp32
python -m src.fallback_probe work/recheck/probe
```

모델 학습부터 재현할 때는 공식 매핑 ZIP도 받는다. [학습·보정·변환 명령과 원본 해시](docs/training.md)

```bash
curl -fL --retry 3 https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road_mapping.zip -o work/devkit_road_mapping.zip
python -m src.train --data work/data_road.zip --mapping work/devkit_road_mapping.zip --out work/retrain
```

## 데모 검증과 버전 대조

Git checkout에서 `python scripts/build_site.py`를 실행하면 `dist/`에 정적 데모, 근거 문서, Git 커밋의 소스 ZIP과 SHA-256 목록이 생긴다. `python -m http.server 8000 --directory dist`로 로컬에서 열 수 있다. 배포본의 `release.json`에서 테스트할 커밋을 확인한다. 아래 브라우저 검사는 시스템 Chromium과 별도로 설치한 Playwright 1.62.0을 사용한다.

```bash
python -m pip install playwright==1.62.0
python tests/replay_smoke.py https://roadlens-kitti-int8.ykh13571.chatgpt.site/
python tests/browser_smoke.py https://roadlens-kitti-int8.ykh13571.chatgpt.site/
python scripts/verify_release.py https://roadlens-kitti-int8.ykh13571.chatgpt.site/
```

제출본에서 공식 ZIP을 지정해 실행한 검사 34개가 통과했다. ZIP 없이 실행하면 62장 마스크 대조 한 건을 건너뛴다. 전체 검사 명령은 `KITTI_ROAD_ZIP=work/data_road.zip python -m pytest -q`다. 보정 입력 shape 오류로 실제 변환 실패를 유발한 후 기존 모델 보존과 재시도를 검사했다. CPU 런타임의 손상 FlatBuffer·잘못된 RGB·호출 오류 회복도 검사했다. [제출본 테스트 출력](artifacts/checks/submission-pytest.txt) · [브라우저 검사 코드](tests/browser_smoke.py) · [모바일 레이아웃](artifacts/checks/mobile-preview.png) · [배포본 해시](https://roadlens-kitti-int8.ykh13571.chatgpt.site/release.json)

## 출처와 기여

지원자는 과제와 기준을 정하고 저장소를 개설했으며, [보정 장수 비교 실험](artifacts/calibration/README.md)을 설계·구현하고 결과를 검산했다. Codex는 기준 모델·벤치마크·데모를 제작하고, 지원자 실험을 재현해 공개 버전에 통합했다. 기존 저장소 6곳은 커밋 고정 코드·테스트·결과·데모를 조사해 실험 방법을 참고했다. [공고와 역할 기록](docs/brief.md) · [선행 저장소 비교](docs/prior.md) · [국내외 디자인·모델군 조사](docs/references.md) · [KITTI 데이터 이용 조건](NOTICE.md)
