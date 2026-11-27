# CamVid 외부 주행에서도 보정 이미지 3장이 충분했을까?

**전체 도로 IoU 하락 허용폭 0.5%p를 3장과 9장 모두 세 시드에서 충족했다.** 반면 26장 기준 모델의 도로 IoU 자체는 한 시퀀스에서 **3.24%**에 그쳤다. 보정 장수를 줄였을 때의 상대 손실과 새 환경에서의 인지 성능은 서로 다른 질문이다.

Codex가 지원자의 [KITTI 보정 장수 실험](calibration_review.md)을 다른 촬영 데이터에서 반복했다. CamVid test의 카메라 영상 2개 시퀀스에서 추출한 233프레임을 모두 썼다. KITTI 289장(학습·보정·평가)과 동일하게 전처리한 RGB의 SHA-256 중복은 0건이었다. 두 데이터의 별도 수집 이력과 입력 해시를 함께 확인했다. 재학습·재보정 데이터 추가·임계값 조정은 없었다. 7개 INT8 파일 해시는 지원자의 원래 기록과 각각 일치했다. [평가 전 고정한 계획](external_protocol.md) · [계획 커밋](https://github.com/sokldjs554/roadlens-int8/commit/f88e303) · [원본 모델 해시](../artifacts/calibration/runs.csv)

## 동일한 233프레임에서의 도로 IoU

| KITTI 보정 장수 | 시드 17 | 시드 42 | 시드 731 | 26장 대비 변화 범위 | 허용폭 충족 |
|---|---:|---:|---:|---:|---:|
| 26장 기준 | 56.310% | 동일 모델 | 동일 모델 | 기준 | 1개 모델 |
| 3장 | 56.299% | 56.278% | 56.302% | −0.0321 ~ −0.0080%p | 3/3 |
| 9장 | 56.317% | 56.326% | 56.302% | −0.0082 ~ +0.0157%p | 3/3 |

판정은 시드별로 유효 픽셀의 TP·FP·FN을 전체 프레임에 걸쳐 더한 후 계산했다. 26장은 시드 선택이 없는 단일 기준이다. 1·2장 보정이나 무작위 세 장의 충분성은 검사하지 않았다. 원래와 같이 3장은 KITTI의 세 도로 유형에서 한 장씩 선택했다.

## 전체 점수에 가려진 실패

| 시퀀스 | 프레임 | 26장 기준 IoU | 3장 IoU 범위 | 9장 IoU 범위 | 26장 기준 도로 재현율 |
|---|---:|---:|---:|---:|---:|
| 0001TP | 62 | **3.24%** | 3.20~3.24% | 3.20~3.22% | 3.25% |
| Seq05VD | 171 | **65.67%** | 65.63~65.67% | 65.67~65.69% | 73.58% |

0001TP에서 기준 모델은 실제 도로 121,446픽셀 중 **3,952픽셀**만 검출했고 **117,494픽셀**을 놓쳤다. 세 장과 스물여섯 장의 점수가 비슷해도 둘 다 같은 도로를 놓칠 수 있다. 개별 프레임에서 가장 큰 하락은 Seq05VD_f00630이었다. 시드 42의 3장 모델은 **74.28%**, 26장 기준은 **76.37%**로 **2.10%p** 차이다.

CamVid는 영상에서 추출한 인접 프레임을 포함한다. 독립적인 촬영 시퀀스는 두 개다. KITTI와 카메라·화면비·라벨 범위가 다르며 기존 입력 계약에 맞춰 480×360을 192×64로 변환했다. CamVid에서는 road(3)만 양성, unlabelled(11)는 평가 제외다. 기존 KITTI 평가의 77.55%와 이 평가의 56.31% 차이를 양자화 손실로 해석할 수 없다. 이번 실험에서 실패 원인을 특정하거나 장치 지연시간을 측정하지 않았다.

## 원시 결과와 재현

[조건·시퀀스별 21행](../artifacts/external/runs.csv) · [프레임별 혼동행렬 1,631행](../artifacts/external/scenes.csv) · [입력·라벨 해시 233행](../artifacts/external/frames.csv) · [최종 판정 JSON](../artifacts/external/summary.json) · [평가 코드](../src/external_calibration.py) · [CamVid 저자 배포본](https://github.com/alexgkendall/SegNet-Tutorial/tree/bb68b64e72cb004b3ed927fee0f5624dd8097498/CamVid)

RoadLens 저장소의 학습 환경에서 기존 KITTI ZIP과 SegNet 저장소를 지정한다. 스크립트는 고정한 커밋의 Git 객체에서 이미지와 라벨을 읽는다. 출력 경로는 새 디렉터리여야 한다.

```bash
git clone https://github.com/alexgkendall/SegNet-Tutorial.git work/segnet
python -m src.external_calibration --archive work/data_road.zip --repo work/segnet --out work/camvid-repeat
diff -qr artifacts/external work/camvid-repeat
```
