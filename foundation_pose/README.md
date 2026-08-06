# foundation_pose — 과일 6DoF 자세추정 (FoundationPose 기반)

기존 파이프라인의 **orientation 이 안 잡히는 문제**를 FoundationPose 로 대체해 보는 모듈.
기존 코드(`fruit-manipulation/`, `record/`)는 **하나도 건드리지 않는다.**

- 논문: [FoundationPose (CVPR 2024)](https://arxiv.org/abs/2312.08344) · [프로젝트](https://nvlabs.github.io/FoundationPose/) · [코드](https://github.com/NVlabs/FoundationPose)

---

## 1. 왜 지금 orientation 이 안 잡히나

현재 경로는 이렇다:

```
live_bbox_gui.py (SAM2 마스크) → /inhand/bbox_corners (8코너 OBB)
        → fruit_pose_bridge.py → PCA 로 주축 뽑아 quaternion
```

`record/fruit_pose_bridge.py` 는 8코너에서 **PCA 주축**으로 방향을 만든다. 오렌지처럼
둥근 물체는 점군의 공분산이 등방(isotropic)에 가까워 **주축 방향이 프레임마다 임의로
튄다.** 알고리즘 버그가 아니라 입력이 방향 정보를 안 담고 있는 것이다.

## 2. FoundationPose 는 뭐가 다른가

깊이 점군의 모양이 아니라 **CAD 모델을 렌더링해서 RGB-D 와 정합**한다. 회전 가설을
여러 개 뿌리고(refine) 점수를 매겨(score) 고른다. 즉 **표면 텍스처가 회전을 결정한다.**

### ⚠ 그래도 남는 한계 (반드시 읽을 것)

**구에 가까운 물체의 회전은 기하학적으로 관측 불가능하다.** 깊이만 보면 오렌지는
어떻게 돌려도 같은 반구다. 그래서 회전을 잡아주는 건 오직 표면 텍스처(꼭지, 반점,
색 얼룩)뿐이다. 이건 FoundationPose 의 한계가 아니라 문제 자체의 성질이다.

| 물체 | 회전 관측 가능성 |
|---|---|
| 오렌지·사과처럼 매끈한 구 | **거의 불가** — 텍스처가 유일한 단서 |
| 꼭지·굴곡 있는 배·바나나 | 가능 (형상이 비대칭) |
| 텍스처 뚜렷한 물체 | 잘 됨 |

그래서 이 모듈은 **회전이 실제로 얼마나 안정적인지 측정하는 것**까지가 목적이다.
`--compare` 로 기존 방식과 나란히 띄워 quaternion 흔들림을 직접 비교할 수 있다.

정말로 그 오렌지의 회전이 필요하면 model-free(참조영상 ~16장 + Neural Object Field)
쪽이 맞다. 근사 구 메시로는 한계가 있다.

## 3. 세그멘테이션 — 논문은 어떻게 하나

논문 원문:

> "Given the RGBD image, the object is detected using an off-the-shelf method
> such as **Mask R-CNN or CNOS**"

그리고 BOP 리더보드에서는 CNOS 를 썼다. **핵심은 마스크가 첫 프레임에만 필요하다는
것이다.** 이후에는 직전 자세로 렌더링한 것과 현재 프레임을 함께 refiner 에 넣어
추적하므로 매 프레임 세그멘테이션이 없다.

> "at each timestamp, we send the cropped current frame and the rendering using
> the previous pose to the pose refinement module"

논문에 XMem 같은 video object segmentation 은 안 나온다.

**→ 우리는 SAM2 를 쓰므로 그대로 맞는다.** 기존 SAM2 체크포인트
(`fruit-manipulation/sam2.1_hiera_tiny.pt`)로 첫 프레임 마스크만 만들고,
그 뒤는 FoundationPose 트래킹에 넘긴다. 매 프레임 SAM2 를 돌리는 지금 방식보다
오히려 가볍다.

## 4. 구조 — 왜 도커로 쪼갰나

FoundationPose 는 nvdiffrast 와 PyTorch3D 를 **nvcc 로 소스 빌드**해야 한다.
그런데 이 PC 에는 CUDA 툴킷이 없고(`nvcc: command not found`), 호스트에 새로 깔면
이미 잘 돌고 있는 **torch 2.6+cu124 / SAM2 환경이 깨질 위험**이 있다.

그래서 추론만 공식 도커 이미지 안에 가두고, ROS2·SAM2 는 호스트에 그대로 뒀다:

```
 호스트                                   컨테이너 (wenbowen123/foundationpose)
 ─────────────────────────────           ────────────────────────────────────
 fp_ros_node.py                          fp_server.py
  · ROS2 구독 (color/depth/info)   TCP    · FoundationPose
  · 첫 프레임 SAM2 마스크        ──5577─► · nvdiffrast + PyTorch3D
  · /fruit/pose, /fruit/size 발행  ◄────  · register() / track_one()
```

컨테이너는 `--network=host` 라 `127.0.0.1:5577` 로 그냥 붙는다. `/home` 을 마운트하므로
메시·가중치 경로가 양쪽에서 동일하다. 프로토콜은 길이 접두 + pickle (양쪽 다 파이썬).

## 5. 파일

| 파일 | 역할 | 어디서 도나 |
|---|---|---|
| `setup.sh` | 이미지·저장소·가중치·확장빌드·메시 준비 (한 번만) | 호스트 |
| `run_foundation_pose.sh` | **원커맨드 런처** | 호스트 |
| `fp_server.py` | FoundationPose TCP 추론 서버 | 컨테이너 |
| `fp_ros_node.py` | ROS2 브리지 + SAM2 초기 마스크 | 호스트 |
| `make_fruit_mesh.py` | 과일 근사 메시(OBJ+텍스처) 생성 | 호스트 |
| `FoundationPose/` | 업스트림 저장소 (클론, git 제외) | — |
| `weights/`, `assets/` | 가중치·메시 (git 제외) | — |

## 6. 사용법

```bash
# 최초 1회 (도커 이미지 ~20GB 받음)
bash foundation_pose/setup.sh

# 실행 — 한 줄
bash foundation_pose/run_foundation_pose.sh

# 전제조건만 점검
bash foundation_pose/run_foundation_pose.sh --check

# 기존 방식과 동시 비교 (/fruit_fp/* 로 발행)
bash foundation_pose/run_foundation_pose.sh --compare
```

기본 모드는 `/fruit/pose` · `/fruit/size` 로 발행하는 **드롭인**이라
`record/fruit_overlay.py` 를 원본 그대로 재사용한다.

메시 크기를 실측에 맞추려면:

```bash
python3 foundation_pose/make_fruit_mesh.py --diameter 0.075 -o foundation_pose/assets/orange.obj
```

## 7. 실측 결과 (2026-08-06, 오렌지 1개, RTX 4080 SUPER)

카메라 앞 오렌지 1개로 90초 연속 측정. 근사 구 메시(지름 70mm, 비대칭 텍스처), model-based.

| 항목 | 결과 |
|---|---|
| 발행 주기 | **6~12 Hz** |
| 위치 | **[+0.023, −0.059, +0.334] m — 90초간 변동 ≤1mm** |
| 회전 | 초기 ~26초 요동 → **이후 수렴·고정** |

회전 quaternion 변화:

```
t+0  ~ t+26s   [-0.402,+0.341,-0.184,+0.830]   ← 프레임마다 완전히 다름
               [+0.515,+0.695,-0.248,-0.436]
               [+0.122,+0.961,-0.231,+0.092]      (구 대칭성 때문에 자세가 방황)
t+26 ~ t+43s   [+0.957,-0.123,+0.262,+0.026]   ← 여기서 락
               [+0.961,-0.141,+0.237,+0.023]
               [+0.965,-0.127,+0.229,+0.015]      (성분 변동 ±0.02 이내로 안정)
```

**해석**

- **위치는 그냥 쓰면 된다.** ±1mm 는 기존 PCA 중심보다 확실히 낫다.
- **회전은 "절대값"이 아니라 "일관성"으로 봐야 한다.** 수렴한 자세는 오렌지의 실제
  방향이라는 보장이 없다(구라서 기준이 없음). 하지만 한 번 락되면 계속 그 값을
  유지하므로 **상대 회전 추적**은 된다. 기존 PCA 방식은 끝까지 계속 튄다.
- 초기 수렴에 시간이 걸리는 건 구 대칭성 탓이다. 꼭지처럼 눈에 띄는 특징이 카메라를
  향하면 훨씬 빨리 잡힌다.

**다음에 해볼 것**
- 실측 지름을 맞춘 메시 (`--diameter`) — 지금은 70mm 가정
- 그 오렌지 사진으로 텍스처를 갈아끼우면 초기 수렴이 빨라진다
- 회전이 정말 중요하면 model-free(참조영상 16장 + NeRF) 경로

## 8. 전제조건

- 제어 PC 에서 realsense 가 떠 있고 **정렬 깊이**가 켜져 있을 것
  (`ros2 param set /camera/camera align_depth.enable true`)
- `nvidia-container-toolkit` (setup.sh 가 확인해 준다)
- GPU: FoundationPose + SAM2 동시 상주. RTX 4080 16GB 면 충분하다.
