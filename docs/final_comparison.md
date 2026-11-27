# 보스반도체 AI SW 직무와 RoadLens의 최종 대조

2026년 11월 27일 기준. 채용 판단 기준은 서울대학교가 안내한 2026 하반기 신입 공고와 회사 포스터다. 회사 SDK 문서는 업무를 구체화하는 제품 사례로 사용했다. 아래 외부 수치는 작성자 보고값이다. 외부 보드나 서비스를 RoadLens로 다시 벤치마크하지 않았다. [서울대 공지](https://cse.snu.ac.kr/en/community/notice/25870) · [회사 채용 페이지](https://bossemi.ninehire.site/job_posting/PeSqBKiq) · [원본 포스터](https://image.ninehire.com/job-posting/2e441e61-de42-11f1-8efd-a72e55879d43/1268a5e0-de64-11f1-bb65-41c900d9757c.jpg)

## 공고에서 요구하는 일

| 공고 항목 | RoadLens에서 열어볼 근거 | 판단과 남은 차이 |
|---|---|---|
| 자율주행·Physical AI 모델 개발·최적화 | 고정 크기 192×64 전방 RGB를 도로/배경으로 분할하는 CNN. 주행 ID 분리, 학습, 변환, 새 입력 실행까지 제공한다. [학습](training.md) · [단일 이미지 실행기](../src/infer.py) | 카메라 인지 부분 구현. 다중 객체 구분, 시간축 추적, 경로 계획·제어, 폐루프 주행은 미구현. |
| CNN·Transformer·LLM/VLM/VLA 등 모델군 | CNN을 CPU와 정수 연산 배치까지 추적했다. 다른 모델군의 입력·연산·평가 변경 사항은 [선정 근거](references.md)에 있다. | 실행한 계열은 CNN. 언어·행동 정답 없는 KITTI Road로 문답 품질이나 과제 성공률을 평가할 수 없다. |
| 비전 모델 양자화·성능 최적화 | 같은 가중치의 FP32·INT8, 지원자 보정 3·9·26장 비교, Codex 20시드 후속 검증. [보정 장면과 원시 결과](calibration_review.md) | 직접 증거가 가장 강하다. 보정 수에 대한 결론은 한 모델, 한 원본 분할, 명시한 허용폭 안에서 유효하다. |
| LLM/VLM·오디오 양자화 | 회사 Model Zoo의 Qwen3-VL·Whisper-Tiny와 공개 구현의 입력·출력 및 평가 방식을 아래에서 비교했다. | 멀티모달 양자화 및 ASR는 조사 범위. WER, 언어 생성 품질, 토큰 지연시간 실측 없음. |
| NPU 기반 모델 실행 SW·런타임 | TFLite CPU 인터프리터, 입출력·scale·zero point 검사, 호출 오류 복구. Arm Vela 출력에서 INT8 1 NPU 구간과 혼합 모델의 두 CPU↔NPU 경계를 검산했다. [런타임과 산출물](runtime.md) | CPU 런타임과 Arm용 오프라인 컴파일 구현. Eagle-N 장치 컴파일·로드·메모리 전송·동기화·실행·전력·프로파일은 미검증. |
| 학습 의지·주도적 문제 해결 | 지원자가 판정 기준을 정하고 사례별 픽셀 불일치를 검산했다. 원래 3시드와 Codex 후속 실험을 구분한다. [판정 계획](calibration_protocol.md) · [작성자 커밋](https://github.com/sokldjs554/roadlens-int8/commit/d3396f8588c907a7993db7e0958c82d1a2d2d248) | 설계 판단과 코드 이력을 확인할 수 있다. 후속 20시드·밝기 실험은 Codex 기여다. |
| 열린 소통·동료 협업 | 공개 소스, 재현 명령, 역할 기록과 원시 결과. [기여 기록](brief.md) | 문서로 판단을 전달한 흔적. 실제 사람과의 팀 협업은 별도 경험으로 확인할 사항이다. |
| 학력·전공·근무 가능일 | 4년제 이상 졸업·졸업예정, 관련 전공, 졸업예정자는 2027년 1월 근무 가능. | 지원자 본인과 증빙으로 확인. |
| 전문연구요원·우대 | 전문연구요원 대상자 지원 가능. 국가유공자·보훈대상자 법령상 우대. | 공고에는 AI SW만의 별도 기술 우대 목록이 없다. 본인 해당 여부는 프로젝트로 판단할 수 없다. |

## 회사의 실제 모델과 가장 가까운 비교

회사의 **Panoptic DeepLab** 문서는 Cityscapes 가중치, 512×1024 입력, 배치 1, 19개 의미 범주와 객체 중심·오프셋 출력을 설명한다. 임의 이미지에서 TTNN 결과와 PyTorch 참조 결과를 별도 디렉터리에 저장한다. backbone·ASPP·두 head·전체 모델의 수치 비교 테스트와 장치 성능 테스트가 분리돼 있다. 문서에는 주로 Blackhole 20코어 그리드에서 테스트했다고 명시돼 있다. 모델 문서의 각 수치를 모든 Eagle-N 보드에 적용할 수 없다. [회사 분할 모델 가이드](https://developers.bos-semi.com/docs/Validated_Models/Vision/Panoptic_DeepLab/)

RoadLens도 동일 입력의 두 출력, 정답, 픽셀별 차이, 원본 CSV를 연결한다. 여기서 두 출력은 **CPU의 FP32와 INT8**이고, 회사 예제는 **참조 구현과 장치용 구현**이다. RoadLens의 도로 IoU 77.55%는 이진 분할의 이미지 평면 통계다. 회사의 19종 분할, KITTI 공식 BEV MaxF, 타 장치 FPS와 직접 비교할 수 없다. 회사 모델 목록에는 ViT-base, Qwen3-VL, Whisper-Tiny 등도 있다. ViT 가이드는 배치 5, ImageNet 검증과 E2E 성능 검사를 구분한다. [BOS SDK 모델 목록](https://developers.bos-semi.com/docs/Validated_Models/) · [BOS ViT](https://developers.bos-semi.com/docs/Validated_Models/Vision/ViT-base/) · [BOS 공식 데모 영상 목록](https://www.bos-semi.com/aimodels) · [KITTI 공식 평가 지표](https://www.cvlibs.net/datasets/kitti/eval_road.php)

## 국내외 구현·서비스의 기능, 평가와 데모

개인·교육 프로젝트, 연구 구현과 상용 SDK를 나눠 해석했다. 표는 링크된 문서에서 확인한 기능을 적는다. 외부 구현의 전체 테스트 통과 여부나 예외 복구율은 독립 검증하지 않았다.

| 공개 사례 | 입력 → 구현·실행 대상 | 평가·실패 처리에서 확인한 근거 | 실제로 볼 수 있는 데모·재현 경로 | RoadLens 판단 |
|---|---|---|---|---|
| 국내 TSTL 교육 프로젝트 | 교통표지·신호등 → ONNX/TensorRT → Jetson Nano·ROS 트랙 주행 | 인식과 차량 동작을 통합하고 단계별 패키지를 공개 | 학습·Jetson 최적화·ROS 폴더와 주행 프로젝트 설명. [팀 저장소](https://github.com/winterbloooom/TSTL-object-detection) | 물리 출력까지 이어지는 증거는 TSTL이 직접적이다. RoadLens는 제어 대신 마스크 변화와 평가 분할을 추적한다. |
| 국내 SIF Warboy 참가팀 | 카메라 자세 추정 → Warboy NPU → 자세 교정 응용 | 참가자가 장치 환경, 최적화·실행 흐름과 CPU 비교를 설명 | 참가자 회고의 구현 과정·발표 이미지. [참가자 원문](https://htaekjung.tistory.com/43) | 실물 NPU 경험이 별도 강점이다. RoadLens의 Vela 추정 사이클로 대신할 수 없다. |
| F1Tenth Vision Pipeline | 카메라 → 차선·검출 → 차량 좌표 거리. Jetson TensorRT | FP32 2.785ms, FP16 1.356ms로 모델 시간을 보고. 카메라 보정과 좌표 변환도 명시 | 카메라 코드, 장면 이미지, 통합 결과 및 학습 손실 자료. [수업 프로젝트](https://github.com/neelmulayPenn/F1Tenth-Vision-Pipeline) | RoadLens는 속도와 마스크 품질을 함께 제시한다. 거리 추정·실시간 카메라는 검증하지 않았다. |
| Self-Driving-ish | 검출·차선·도로·깊이를 TensorRT로 조합 | Xavier NX에서 모든 기능 6.8FPS, 분할·깊이 제외 24.4FPS. 구성 변경이 FPS를 바꾼다. | 영상과 카메라 실행·빌드 방법. [저장소](https://github.com/iwatake2222/self-driving-ish_computer_vision_system) | RoadLens의 세 마스크를 같은 장면에서 비교하도록 한 근거. 브라우저는 저장 결과, 새 입력은 로컬 CLI다. |
| JetDriveRT | YOLOP → ONNX/TensorRT·CUDA·C++ 상주 프로세스. Orin Nano | 전처리+전달 11.143→2.027ms, 프레임 스케줄링과 capture-to-result p95를 구분 | 영상 파이프라인, 실행 절차와 구간별 측정표. [구현·벤치마크](https://github.com/1569084106/JetDriveRT) | RoadLens는 PNG 바이트→마스크 구간과 순수 호출을 분리했다. 디코딩 전 카메라 지연·프레임 드롭·비동기 큐는 미측정. |
| PTQ4ViT | ViT·DeiT·Swin의 가중치·활성값 PTQ | 32·128장, 6·8비트, 일부 모델에서 32장 선택을 20회 반복해 평균·편차 제시 | 실험 스크립트와 정확도·양자화 시간표. [연구자 구현](https://github.com/hahnyuan/PTQ4ViT) | RoadLens는 3·9·26장, 전체·무표시도로·개별 장면을 비교한다. 다중 구조 검증과 양자화 알고리즘 자체의 범위는 이 연구가 넓다. |
| Rockchip PP-LiteSeg | ONNX 도로 장면 분할 → RKNN i8/fp → 지원 Rockchip 보드 | 칩 선택, 정밀도 옵션, Linux·Android 빌드와 보드 실행 경로 분리 | 변환 명령, C/C++ 예제·입력과 출력 예시. [공식 예제](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/ppseg) | RoadLens의 이식 근거는 변환 산출물과 CPU fallback 분석까지이며 보드용 실행 증거가 부족하다. |
| FuriosaAI Warboy SDK | 장치 추론 작업에 프로파일러 연결 | 실행 trace를 수집해 병목을 조사하는 방법을 제공 | 환경 변수, 예제 실행, trace JSON. [공식 프로파일러](https://developer.furiosa.ai/docs/latest/en/software/profiler.html) | RoadLens의 호출별 CPU 원시 시간은 재검산 가능하다. 장치 내부 이벤트 시간은 기록하지 못했다. |
| Qualcomm AI Hub Workbench | 사용자 모델 → 대상 기기 컴파일 → 클라우드 실기기 추론 | 모델 정확도 검사와 실행 유닛, 메모리·지연시간·레이어 프로파일을 구분 | 단계별 예제와 작업별 결과 화면. [공식 워크스루](https://workbench.aihub.qualcomm.com/docs/hub/walkthrough.html) | RoadLens는 설치 없이 실패 사례와 원시 파일을 열 수 있다. 공개 화면에서 임의 입력을 업로드해 장치에서 실행하는 서비스는 아니다. |
| TinyChatEngine | 양자화 LLM 및 VILA-7B 이미지 질의 → ARM/x86 등 대상 런타임 | 저비트 가중치 레이아웃·변환과 ISA 의존성이 명시됨 | MacBook의 VILA 데모와 자체 이미지·모델 다운로드 명령. [저장소](https://github.com/mit-han-lab/TinyChatEngine) | 도로 IoU로 이미지 질의 정확도나 토큰 생성 지연을 대신 평가할 수 없다. LLM/VLM 미구현. |
| OpenVLA | 카메라·언어 지시 → 행동 토큰 → 로봇 동작 | 여러 로봇·과제에서 rollout과 성공 여부, 일반화 조건을 검증 | 실제 로봇 영상, 체크포인트, 학습·추론 코드. [저자 데모](https://openvla.github.io/) | RoadLens가 검증한 Physical AI 범위는 시각 인지다. 실제 주행 안전성이나 행동 성공률로 확대하지 않는다. |
| whisper.cpp | 오디오 → 정수 양자화 Whisper → 전사 | 파일 크기·메모리와 하드웨어 의존성을 기술. 음성 평가에는 참조 전사 필요 | 로컬 오디오 CLI와 모델 변환·양자화 예제. [구현](https://github.com/ggml-org/whisper.cpp) | 오디오와 동기화된 정답이 없어 범위를 확장하지 않았다. |
| Zero-DCE | 저조도 입력 → 학습한 픽셀별 곡선 → 향상된 이미지 | 무참조 이미지 향상이 목적. 도로 마스크 품질은 별도 평가가 필요 | 테스트 코드와 저조도 전후 이미지. [저자 구현](https://github.com/Li-Chongyi/Zero-DCE) | 실패 입력에 전처리를 적용해볼 근거. RoadLens에서는 가벼운 자체 규칙을 사용해 기존 INT8 모델의 도로 IoU와 정상 장면 회귀를 검사했다. 외부 모델을 이식하지 않았다. |

## 제출본에서 확인한 기능

| 기능 | 증거 및 범위 |
|---|---|
| 입력·모델·비교 결과 | 로컬 CLI는 새 PNG/JPEG에서 두 모델을 실행해 마스크와 차이 픽셀 수를 출력한다. 손상 입력·기존 파일 덮어쓰기·호출 실패 복구를 검사한다. [CLI](../src/infer.py) · [입출력 테스트](../tests/test_infer.py) · [런타임 테스트](../tests/test_runtime.py) |
| 평가·최적화 | 학습 201장, 보정 26장, 평가 62장을 주행별로 분리했다. 기준 실행은 모델당 워밍업 20회, 평가 62장×10회다. 후속 CPU 경로 실험에서 순수 추론 p50은 55.6%, 메모리의 PNG부터 마스크까지 p50은 6.0% 감소했다. [구간별 측정](calibration_review.md) |
| 실패·한계 | 합성 밝기 저하에서 기준 IoU가 0인 장면을 공개했다. 보정 이미지 수 실험의 하락 사례와 입력 밝기 규칙의 정상 장면 회귀는 각각 다른 문제다. [밝기 규칙 판정](exposure.md) |
| 데모·버전·출처 | PC·모바일 레이아웃, 저장 결과 모드, 슬라이더·조건 선택·21초 재생, 소스 ZIP과 SHA 목록. 지원자 원실험과 Codex 후속 작업을 구분한다. [공개 데모](https://roadlens-kitti-int8.ykh13571.chatgpt.site/) · [배포 매니페스트](https://roadlens-kitti-int8.ykh13571.chatgpt.site/release.json) |

**결정.** 제출 주제는 비전 양자화와 측정 가능한 실행 경로로 유지한다. 이번 추가 구현은 기존 실패 장면을 검증 가능한 전처리 가설로 연결하는 데 사용했다. 가장 큰 잔여 격차는 Eagle-N에서 참조 출력과 장치 출력을 동일 입력으로 대조하고 실제 메모리·장치 지연을 측정한 기록이다. 이 결과에는 해당 실측이 없다.
