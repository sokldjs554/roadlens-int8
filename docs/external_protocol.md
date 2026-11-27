# 외부 주행에서 보정 장수 비교 · 결과 확인 전 고정한 계획

원래 지원자의 질문을 다른 촬영 주행에서 재검증하는 Codex 후속 실험이다. 모델과 판정 기준은 지원자의 [기존 실험](calibration_protocol.md)을 따른다.

- 데이터는 SegNet 저자가 배포한 CamVid test 목록 전부다. 고정 리비전은 `bb68b64e72cb004b3ed927fee0f5624dd8097498`이며 Cambridge 주행 영상에서 추출한 두 시퀀스(0001TP, Seq05VD) 총 233프레임이다. 학습, 보정, 모델 선택에 CamVid를 쓰지 않는다. 목록과 파일의 Git 객체를 확인하고, KITTI 학습·보정·평가 289장 전체와 리사이즈된 RGB 입력의 SHA-256 교집합을 검사한다. 중복 또는 결측이면 실행을 중단한다.
- 원래 가중치와 KITTI 보정 ID를 고정한다. 3장과 9장은 원래 시드 17·42·731, 26장은 기준 1회다. 7개 INT8 변환 결과의 SHA-256이 각각 원래 실험과 일치해야 평가한다.
- 같은 RGB 전처리(192×64 비율 변형, bilinear, 0~1), CPU 1스레드, 모델 출력 임계값 0.5를 사용한다. 마스크는 최근접 이웃으로 줄인다. SegNet CamVid 라벨 ID 3(road)은 양성, 11(unlabelled)은 제외하고 나머지 0~10은 음성이다. pavement는 보도이며 음성이다. 다른 라벨 ID는 오류다.
- 각 조건에서 유효 픽셀의 TP·FP·FN을 전부 합산한 도로 IoU를 쓴다. 같은 외부 데이터의 26장 보정 대비 하락폭이 반올림 전 0.5%p 이하인지 판정하며 세 시드 모두 만족해야 해당 장수가 통과한다. 절대 IoU, 시드별 변화와 시퀀스별 결과도 공개한다. 별도 시퀀스에는 같은 허용폭을 참고용으로 표시하며 전체 판정 규칙을 바꾸지 않는다.
- 프레임별 최악의 예는 26장 대비 IoU 하락이 가장 큰 조건으로 고른다(동률이면 장수·시드·ID 오름차순). 연속 영상의 프레임은 독립 표본이 아니므로 233개 주행으로 해석하지 않는다. KITTI와 CamVid는 촬영 환경·시야·라벨의 평가 영역이 다르다. 각 데이터 안에서 짝을 맞춘 3·9·26장 비교가 주 평가다.

[분할 목록](https://github.com/alexgkendall/SegNet-Tutorial/blob/bb68b64e72cb004b3ed927fee0f5624dd8097498/CamVid/test.txt) · [라벨 순서](https://github.com/alexgkendall/SegNet-Tutorial/blob/bb68b64e72cb004b3ed927fee0f5624dd8097498/Scripts/test_segmentation_camvid.py) · [CamVid 원저자 데이터 설명](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) · [KITTI 수집 차량](https://www.cvlibs.net/datasets/kitti/setup.php)
