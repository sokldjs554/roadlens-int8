# 데모와 모델 선정에 참고한 자료

2026년 11월 26일 확인. 아래 내용은 Codex가 공식 제품 페이지, 문서, 저자 논문을 읽고 정리한 설계 근거다. 인용된 제품의 성능 수치는 이번 프로젝트의 실험값에 포함하지 않는다.

## 데모에서 어떤 정보를 먼저 보여 줄 것인가

| 참고한 화면 | 관찰한 정보 구성 | 적용할 요소와 이유 |
| --- | --- | --- |
| [MORAI 시나리오 기반 검증](https://www.morai.ai/ko/use-case-scenario) | 도로에서 마주치는 조건을 시나리오 단위로 제시한다. | 장면을 바꾸면 해당 입력과 두 예측을 함께 갱신한다. 어려운 조건마다 표본 수를 병기해 한 장의 인상을 일반화하지 않도록 한다. |
| [현대차그룹·42dot 자율주행 미디어데이](https://org.hyundai.com/worldwide/en/newsroom/detail/0000001273) | 자료 설명과 함께 원본 이미지와 문서를 받을 수 있다. | 지표에서 원본 파일로 이동할 수 있게 해 면접관이 기록의 출처를 검산할 수 있도록 한다. |
| [Waymo Open Dataset](https://waymo.com/open/about/) | 도로 장면과 라벨을 시각적으로 연결하고 데이터 설명에 센서·주석 종류를 명시한다. | 실제 평가 입력에 정답과 예측을 연결한다. 분류만 평가했다면 박스를 검출 결과처럼 합성할 이유가 없다. |
| [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/hub/walkthrough.html) | 대상 단말, 런타임, 모델 변환, 단말 추론 및 프로파일링을 실행 단계별로 보여 준다. | 수치 옆에 측정 장치와 실행 공급자를 표시한다. 그래프의 실행 후보와 프로파일이 기록한 공급자를 분리할 수 있다. |
| [Netron](https://github.com/lutzroeder/netron) | 모델 그래프를 열어 연산과 텐서 관계를 살펴보는 도구다. | 지원 여부 상세 정보에는 모델 파일에 기록된 실제 연산명을 유지한다. 모델 그래프와 실행 로그를 대조하기 쉽다. |
| [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) | 실행 시나리오와 시스템별로 결과를 비교하며 제출 세부 정보로 이동할 수 있다. | 측정 단위와 배치, 반복 수를 명시하고 동일 조건의 두 모델을 같은 순서로 놓는다. 본 실험은 독립 측정이며 MLPerf 인증 결과가 아니다. |
| [OpenVINO Benchmark Tool](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/benchmark-tool.html) | 지연시간과 처리량을 서로 다른 운용 목표로 제시하고 실행 옵션을 출력한다. | 단일 이미지 지연시간에 ms 단위를 붙이고, 어떤 구간을 재고 워밍업을 어떻게 제외했는지 결과 파일에서 밝힌다. |

화면의 중심은 입력 이미지다. 두 모델에는 같은 측정 단위와 정답을 적용한다. 모바일에서는 장면과 두 예측이 함께 읽히도록 두 결과의 표시 순서를 고정한다. 작은 글자색이나 막대 길이만으로 우열을 나타내면 비슷한 수치와 실패 사례를 구분하기 어렵다.

## 모델군별 검토

아래 모델명은 구조를 검토하기 위한 대표 사례다. 실제 실행 여부와 결과는 저장소의 실험 원본을 따른다.

| 범위 | 검토한 계산과 시나리오 | 정밀도 변경 시 확인할 점과 이 프로젝트의 범위 |
| --- | --- | --- |
| 비전 CNN | 합성곱의 지역 수용 영역으로 시각 특징을 추출한다. MobileNetV3는 모바일 CPU를 목표로 depthwise convolution과 경량 블록을 사용한 예다. | 고정 입력 CNN은 합성곱·활성화와 양자화 경계를 추적하고 한 프레임당 시간을 재기 좋다. 개별 장치에 적합한 레이어 구성과 실제 속도는 실험으로 판정한다. [MobileNetV3 원 논문](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) |
| 비전 Transformer | ViT는 이미지를 패치로 나누고 토큰 사이의 self-attention으로 장면 정보를 결합한다. | 해상도를 높이면 토큰 수와 표준 attention 행렬이 커진다. 배포 전 MatMul·Softmax·LayerNorm의 지원 정밀도와 그래프 경계를 각각 확인해야 한다. CNN FP32와 CNN INT8 사이의 효과를 확인하는 첫 실험에는 구조 변경이 추가 변수가 된다. [ViT 원 논문](https://arxiv.org/abs/2010.11929) |
| LLM | 장면 메타데이터를 문장으로 요약할 때 쓸 수 있다. AWQ는 대표 입력 활성값을 활용해 가중치 전용 저비트 양자화에서 중요한 채널의 오차를 줄이는 접근이다. | 디코딩 단계의 메모리와 처리량, 입력 길이별 첫 토큰 시간과 생성 품질을 함께 측정할 수 있다. 현재 이미지 라벨만으로 텍스트 요약의 정확성을 평가하기 어려워 조사 범위로 둔다. [AWQ 원 논문](https://arxiv.org/abs/2306.00978) |
| VLM | SmolVLM-256M은 이미지와 텍스트 질문을 받아 답변을 생성한다. 예를 들어 도로 장면 묘사를 질의할 수 있다. | 이미지 인코더와 언어 디코더의 정밀도를 각각 바꾸고 같은 질문에서 정답률과 환각 사례를 검토해야 한다. 장면 질의·응답 정답셋이 필요한 별도 실험이다. [SmolVLM 모델 카드](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| VLA | OpenVLA는 이미지와 지시를 받아 로봇 조작 행동을 예측한다. 저자들은 양자화 구성에서 과제 성공률도 실험했다. | 행동 공간과 로봇 상태, 폐루프 과제 성공률을 확인해야 한다. 도로 이미지 분류 정답은 행동의 성공 여부를 제공하지 않아 이 프로젝트의 데모에서 제어 성능을 주장할 수 없다. [OpenVLA 저자 페이지](https://openvla.github.io/) |
| 오디오 | Whisper는 오디오를 log-Mel 표현으로 바꿔 음성을 전사한다. 차량 음성 명령 전사에는 적합하고 사이렌 탐지에는 별도의 소리 이벤트 라벨과 검출기가 필요하다. | 전사기라면 양자화 전후 WER와 실시간 계수, 경보음 검출기라면 이벤트 재현율·오경보를 평가한다. 동기화된 오디오와 정답이 없는 이미지 평가는 오디오 모델 검증 근거를 제공하지 않는다. [Whisper 코드와 구조](https://github.com/openai/whisper) |

ONNX Runtime의 정적 양자화는 대표 입력에 따라 활성값의 양자화 파라미터를 산출한다. 그래서 보정용 이미지와 정확도 평가 이미지를 분리해야 한다. Q/DQ 노드가 삽입된 모델을 얻은 뒤에는 실제 실행 프로파일에서 연산 공급자와 그래프 최적화 결과를 확인할 수 있다. CPU에서 관측한 공급자는 그 세션에 대한 증거다. 보스반도체 NPU의 지원 연산·정확도·속도는 해당 SDK와 장비에서 별도로 검증해야 한다. [ONNX Runtime 양자화 문서](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) · [ONNX Runtime 프로파일링 문서](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
