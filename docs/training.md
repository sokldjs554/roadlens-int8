# KITTI 도로 분할 모델 학습과 변환

전방 카메라 RGB 한 장에서 카메라 좌표계의 도로 확률 맵을 예측한다. Keras 모델, FP32 TFLite와 INT8 TFLite는 같은 학습 가중치에서 생성했다. 학습 원본 ZIP은 470,992,343바이트이며 배포 파일에서 제외했다. [KITTI Road](https://www.cvlibs.net/datasets/kitti/eval_road.php) · [데이터 이용 조건](https://www.cvlibs.net/datasets/kitti/)

## 재현

Linux x86-64와 Python 3.12.14에서 실행했다. 저장소 루트에서 아래 명령을 실행하면 별도 디렉터리에 결과가 생긴다. TensorFlow CPU 2.19.1, Keras 3.15.1, NumPy 2.1.3, Pillow 13.0.0을 사용했다.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock
mkdir -p work/data
curl -fL --retry 3 -o work/data/data_road.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip
curl -fL --retry 3 -o work/data/devkit_road_mapping.zip https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road_mapping.zip
sha256sum work/data/*.zip
python -m src.train --data work/data/data_road.zip --mapping work/data/devkit_road_mapping.zip --out work/reproduced
python -m pytest -q tests/test_train.py
```

| 원본 | SHA-256 |
|---|---|
| data_road.zip | `45508c51ba09020235eb714e71c7d3ef15b4551be06d996772ecc8a20fc34a34` |
| devkit_road_mapping.zip | `cd75648c0825832c045723ec009386d8495804b5ec1f2b428d2cd61da070fdb6` |

학습 결과로 `road.keras`, `fp32.tflite`, `int8.tflite`, `split.json`, `training.json`이 저장된다. 시드와 결정론적 연산을 고정했다. CPU 명령어 집합이나 라이브러리 빌드에 따라 부동소수점 연산 결과에 차이가 생길 수 있다.

## 어떤 프레임을 어디에 썼는가

공식 라벨이 있는 289장을 원본 주행 ID로 묶었다. 각 유형의 공식 매핑 텍스트에서 0부터 센 줄 번호가 학습 프레임 번호다. 도로 유형이 달라도 같은 주행 ID는 한 그룹에만 둔다. 37개 주행의 순서를 고정 시드 731로 6,000회 섞고, 유형별 이미지 수가 보정 10%, 평가 20%에 가까운 조합을 선택했다. 선택 기준에는 이미지, 정답 픽셀 또는 모델 점수가 들어가지 않는다. [공식 주행 매핑](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road_mapping.zip) · [사용한 ID 목록](../artifacts/split.json)

| 용도 | 주행 수 | 도시 표시 도로 UM | 도시 다차로 UMM | 도시 무표시 도로 UU | 이미지 수 |
|---|---:|---:|---:|---:|---:|
| 가중치 학습 | 26 | 67 | 65 | 69 | 201 |
| INT8 보정 | 4 | 9 | 10 | 7 | 26 |
| 최종 평가 | 7 | 19 | 21 | 22 | 62 |

좌측 컬러 카메라의 RGB를 192×64(너비×높이)로 bilinear 리사이즈하고 255로 나눈다. 정답은 `{유형}_road_{번호}.png`를 nearest-neighbor 리사이즈한다. RGB의 빨강 채널이 0보다 큰 픽셀만 학습과 평가에 사용하며 파랑 채널이 0보다 큰 픽셀이 도로다. UM에 함께 제공되는 차로 마스크는 사용하지 않는다. 공개 벤치마크의 공식 테스트 이미지는 라벨이 없어서 이 프로젝트의 지표는 위 내부 분할에서 산출한다. [데이터 로더](../src/data.py) · [KITTI 공식 평가 도구](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_road.zip)

## 모델과 학습 조건

RGB 입력에서 3×3 stride-2 합성곱 네 단계로 공간 해상도를 줄인다. 채널 수는 12, 20, 32, 48이다. 디코더는 최근접 확대와 인코더 특징 연결을 세 번 수행하고 1×1 합성곱, 마지막 2배 확대, sigmoid로 픽셀별 확률을 생성한다. 합성곱 블록은 배치 정규화와 ReLU6를 사용한다. 배치 정규화의 이동 통계를 포함한 파라미터 수는 79,569개다. [모델 구현](../src/train.py)

| 조건 | 값 |
|---|---|
| 목표 함수 | 유효 픽셀의 프레임별 평균 이진 교차 엔트로피 + 0.4 × soft Dice loss |
| 학습 | Adam, 배치 12, 고정 30에포크, 코사인 학습률 스케줄 0.002에서 0.0002 부근까지 |
| 증강 | 좌우 반전 확률 0.5, 프레임별 밝기 배율 0.85~1.15와 오프셋 −0.07~0.07, 픽셀값을 [0, 1]로 제한 |
| 모델 선택 | 최종 에포크 가중치 사용. 보정·평가 프레임은 가중치 학습에 쓰지 않음 |
| 실행 | TensorFlow 결정론적 연산, intra-op 스레드 3, inter-op 스레드 1로 실행 |

## 변환 검증과 실패 재현

고정 입력 `[1,64,192,3]`에서 추론 그래프를 동결하고 두 TFLite를 생성했다. FP32는 기본 변환, INT8은 보정 프레임 26장과 `TFLITE_BUILTINS_INT8`을 지정한 사후 양자화다. INT8은 입력과 출력까지 signed int8이며, 바이어스와 형상 텐서에는 int32 등이 쓰인다. 출력 형상은 `[1,64,192,1]`이다. INT8 텐서는 실행기가 기록한 scale과 zero point를 읽어 변환한다. 공개한 파일 크기는 FP32 322,536바이트, INT8 92,832바이트다. [검증 테스트](../tests/test_train.py)

디스크에서 복원한 Keras 모델과 FP32 TFLite에 동일한 평가 62장을 입력했다. 유효 영역 밖까지 포함한 출력 확률 761,856개의 최대 절대 차이는 0.00000369549, 평균 절대 차이는 0.0000000351478이었다. 최대 허용 차이는 0.0002다. 도로 IoU와 픽셀 정확도에는 별도로 유효 영역과 확률 임계값 0.5를 적용한다. [변환 검증 원본](../artifacts/training.json) · [평가 기록](../artifacts/benchmark.json)

아래 회귀 테스트는 임시 디렉터리에 두 기존 모델을 복사하고, RGB 채널 차원을 누락한 보정 입력을 실제 TFLite 변환기에 전달한다. INT8 보정의 CONV_2D 준비 단계가 실패하는지와 기존 두 모델의 바이트가 보존되는지 확인한다. 같은 모델에 정상 RGB 입력을 공급해 변환을 재시도하고 FP32와 INT8 인터프리터에서 텐서를 할당한다. 학습 명령도 변환과 Keras 대조가 모두 성공한 뒤 결과 파일을 출력 위치에 반영한다.

```bash
python -m pytest -q tests/test_train.py -k actual_quantizer
```
