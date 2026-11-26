# CPU 실행과 NPU 연산 배치

동일한 학습 가중치에서 FP32와 완전 정수 INT8 TFLite를 생성했다. AMD EPYC 9V74 CPU에서 실행한 파일은 각각 `artifacts/fp32.tflite`, `artifacts/int8.tflite`다. 이 두 파일을 별도로 Arm Vela 4.4.1에 넣어 Ethos-U55-128용 실행 그래프를 생성했다. 프로젝트의 실제 NPU 장비·시뮬레이터 실행 횟수는 0이다. 보스반도체 NPU의 지원 연산, 속도, 정확도는 미검증이다.

## 지원 여부를 확인한 방법

| 컴파일 입력 | 입력 TFLite 연산 | 출력 그래프의 NPU 구간 | CPU 연산 | CPU↔NPU 텐서 연결 |
|---|---:|---:|---:|---:|
| 도로 모델 FP32 | 17 | 0 | 17 | 0 |
| 도로 모델 INT8 | 17 | 1 | 0 | 0 |
| 혼합 정밀도 진단 모델 | 5 | 2 | 3 | 2 |

`src/compiler.py`는 Vela가 실제 생성한 FlatBuffer에서 각 연산의 이름 및 입력·출력 텐서를 읽는다. Ethos-U 명령 스트림을 담은 custom op 하나를 NPU 구간 하나로 센다. 서로 다른 장치에 배치된 생산자와 소비자의 연결을 집계한다. 입력 이미지 전달과 최종 출력 회수는 이 개수에서 제외한다. [FP32 배치](../artifacts/vela/fp32/placement.json) · [INT8 배치](../artifacts/vela/int8/placement.json)

FP32 도로 모델의 Conv2D·최근접 확대·concat·logistic은 CPU 그래프에 남는다. INT8 도로 모델은 하나의 NPU 구간으로 묶인다. Vela 내부 lowering 후 NPU 연산 수 19개와 입력 FlatBuffer 연산 수 17개는 집계 단계가 다르다. 레이어별 시간 모델과 버퍼 분석은 컴파일러 원본에 보존했다. [Vela 로그](../artifacts/vela/int8/vela.log) · [레이어별 CSV](../artifacts/vela/int8/int8_per-layer.csv)

`src/fallback_probe.py`는 두 양자화 합성곱 사이에 부동소수점 SIN이 있는 작은 진단 모델을 별도로 만들고 컴파일한다. 출력에서 첫 NPU 구간, CPU의 DEQUANTIZE → SIN → QUANTIZE, 두 번째 NPU 구간과 두 경계가 관측됐다. 도로 모델의 정확도나 지연시간 집계에는 진단 모델이 들어가지 않는다. [진단 그래프와 텐서 번호](../artifacts/vela/probe/placement.json)

## 컴파일러 추정값

Arm 예제 시스템 설정 Ethos_U55_High_End_Embedded, 500 MHz, Shared_Sram, Greedy 할당, 배치 1이다. INT8에 대해 총 **628,069 추정 사이클**, **1.256ms 추정 시간**, **276.00 KiB 추정 SRAM**, **95.03 KiB 추정 off-chip flash**를 기록했다. 실제 보드에서는 드라이버, DMA 및 호스트 동기화 비용, 메모리 대역폭과 클럭이 달라진다. CPU 프로세스 RSS와 Vela의 SRAM 추정은 다른 층의 메모리를 측정한다. [원본 요약 CSV](../artifacts/vela/int8/int8_summary_Ethos_U55_High_End_Embedded.csv) · [Arm Vela](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela)

## 런타임 계약

CPU 어댑터는 인터프리터를 한 작업자에 귀속하고 배치 1, 단일 스레드로 연다. 모델 입출력 자료형과 shape, INT8 scale·zero point를 검사한다. 입력을 양자화하고 호출 시간을 따로 기록한 뒤 출력을 같은 float 확률 범위로 복원한다. 손상 모델·범위 밖 입력·잘못된 출력·호출 오류를 구분하며, 실패한 세션은 다음 프레임에서 다시 로드한다. 실제 배포용 NPU 드라이버, 명령 제출과 보드에서의 CPU fallback 실행은 향후 장비 검증 영역이다. [코드](../src/runtime.py) · [회귀 테스트](../tests/test_runtime.py)
