KITTI 주행 분리 평가 62장의 FP32·INT8 실험과 공개 데모를 고정한 버전입니다.

- 최종 소스 커밋은 `a3d3272f9cea8d8bc887498c33ce09ec5c2fcf81`입니다. GitHub main, 배포 서비스 저장 버전 2, 공개 매니페스트, 제공한 소스 ZIP의 커밋이 일치합니다.
- 공개 데모에서 로그인 없이 받아온 38개 파일의 SHA-256이 로컬 빌드와 일치했습니다. 소스 ZIP SHA-256은 `034fa6aaa3f2950b47e257cb6c79643b5a9487eef489f1f727f05fbf1d981b05`입니다.
- 단위·회귀 테스트 19개가 통과했습니다. 공개 URL은 클라우드 Chrome에서 열어 입력, 양쪽 예측, 장면 선택 및 전체 지표 표시를 확인했습니다.
- 자동화용 Chromium이 실행 환경의 HTTPS 인증서를 신뢰하지 않아, 시스템 인증서로 내려받아 해시 검증한 배포 파일을 로컬에서 제공하고 두 화면 크기로 회귀 테스트했습니다. Chromium 156, Playwright 1.62, 1365×850 데스크톱과 360×760 터치 에뮬레이션에서 장면 6개·4개 보기·슬라이더·수치와 탐색을 검사했습니다. 실물 스마트폰은 사용하지 않았습니다.
- 실측 환경은 AMD EPYC 9V74 CPU입니다. Arm Ethos-U55-128 결과는 Vela 컴파일과 비용 추정입니다. 실제 보스반도체 NPU와 Arm 보드에서의 실행 기록은 없습니다.

[공개 데모](https://roadlens-kitti-int8.ykh13571.chatgpt.site/) · [직무 근거표](https://roadlens-kitti-int8.ykh13571.chatgpt.site/evidence/index.html) · [전체 소스·모델·실험 원본 ZIP](https://roadlens-kitti-int8.ykh13571.chatgpt.site/downloads/roadlens-source.zip)
