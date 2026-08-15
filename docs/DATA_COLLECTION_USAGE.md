# 데이터 수집 사용법 (글러브 텔레옵 → HDF5 + 영상)

글러브로 KISTAR 손을 텔레옵하며 과일 조작 시연을 수집한다. 팔은 고정.
발판으로 **텔레옵 제어권**과 **로깅 시작/종료**를 제어하고, RealSense 화면을 보며 원할 때 영상 녹화.

## 구성 (데이터 흐름)
```
[제어 PC] 로봇/핸드/paxini 스택(shm+nd) ─ /hand/* /paxini/* /franka/right/* ─┐
          RealSense ─ /front_cam/front/color/* ───────────────┐              │
[이 PC]                                                      ▼              ▼
  glove_teleop.py ── /hand/right/q_target(engage시) ──▶            ros2_hdf5_recorder.py ──▶ record/logs/exp_*.h5
  live_bbox_gui(ros env) ─ /inhand/bbox_corners ─▶ fruit_pose_bridge ─ /fruit/pose,/fruit/size ─┘
  realsense_view.py ─ 화면표시 + record/videos/*.mp4
  foot_pedal_glove.py ─ /teleop/hand_engage/right(engage) + /record/enable(로깅)
```

## 발판 사용법 (독립 매핑)
| 페달 | 동작 |
|---|---|
| **오른쪽** | 텔레옵 **ENGAGE** — 글러브가 손 타겟 스트리밍 시작 (제어권 ON) |
| **왼쪽** | 텔레옵 **DISENGAGE** — 홀드(마지막 자세 유지, 손 안 풀림) |
| **중간** | **로깅 S/E 토글** — 누르면 에피소드 수집 시작, 다시 누르면 Demo_N 저장 (`--sync`면 영상도 같이) |

## 공통 환경 (ROS 터미널마다 맨 위)
```bash
source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
```

## 실행 순서

### 0) 제어 PC (먼저)
- 로봇/핸드 스택: `shm` → `nd`  (→ /hand/*, /paxini/*, /franka/right/* 발행)
- RealSense: `rs`  (→ /front_cam/front/* 발행). fruit 쓰려면 정렬깊이 필요:
  `ros2 param set /front_cam/front align_depth.enable true`

### 이 PC — 터미널별
```bash
# 터미널 1 — 글러브 텔레옵 (USB 글러브 연결 상태)
python3 ~/Desktop/vive_franka_teleop/tools/glove_teleop.py

# 터미널 2 — (과일 쓸 때) fruit-manip. ros conda env + 카메라 네임스페이스
conda run -n ros --no-capture-output python /media/js/T9/fruit-manipulation/live_bbox_gui.py --namespace /front_cam/front

# 터미널 3 — (과일) pose 브리지
python3 ~/Desktop/vive_franka_teleop/record/fruit_pose_bridge.py

# 터미널 4 — RealSense 화면 보기 + 영상 녹화 (발판 로깅과 동기)
python3 ~/Desktop/vive_franka_teleop/record/realsense_view.py --sync

# 터미널 5 — HDF5 레코더 (먼저 --check 로 수신 점검!)
python3 ~/Desktop/vive_franka_teleop/record/ros2_hdf5_recorder.py --check
python3 ~/Desktop/vive_franka_teleop/record/ros2_hdf5_recorder.py

# 터미널 6 — 발판 (input 그룹 필요: sudo usermod -aG input $USER 후 재로그인)
python3 ~/Desktop/vive_franka_teleop/record/foot_pedal_glove.py
```

### 수집 절차
1. **터미널 5 `--check`** 로 필요한 토픽이 ✓ 뜨는지 확인 (안 뜨면 해당 노드/스택 기동).
2. 발판 **오른쪽** → 텔레옵 engage. 글러브로 손 움직여 물체 잡을 준비.
3. 발판 **중간** → 로깅 시작(에피소드). (`--sync`면 영상도 이때 시작)
4. 과일 조작 시연.
5. 발판 **중간** → 로깅 종료 → `Demo_N` 저장(영상도 정지).
6. 3~5 반복해서 데모 여러 개 수집.
7. 끝나면 발판 **왼쪽**(disengage) → 레코더 **Ctrl+C**(파일 마무리).

## 저장 결과
- 로봇/글러브/과일 데이터: `record/logs/exp_YYYYMMDD_HHMMSS.h5` (`Demo_0`, `Demo_1`, … 100Hz)
- RGB(`--rgb-on`): `record/logs/exp_YYYYMMDD_HHMMSS_image/Demo_N/000000.jpg …` (h5 옆, 같은 이름 + `_image`)
- 화면 영상: `record/videos/rs_*.mp4`

### HDF5 필드 (Demo_N/)

**h5 는 이름순으로 나열되므로 번호가 곧 표시 순서입니다.** 보기 편하게
**로봇핸드(01~11) → 로봇암(12~20) → 글러브(21~23) → 과일(30~34) → 이미지(40~42) → 클락(50/51)** 로 묶었습니다.

| 키 | dim | 소스 |
|---|---|---|
| **로봇 핸드** | | |
| 01_hand_mode / 02_hand_servo_on | 1/1 | /hand/right/mode |
| 03_hand_j_pos / 04_hand_j_tar / 05_hand_j_cur | 16/16/16 | /hand/right/joint_states(pos·effort) · /hand/right/q_target(**액션**) |
| 06_hand_j_kin / 07_hand_j_tac | 12/4 | /hand/right/kin · /hand/right/tac_legacy |
| 08_hand_tip_pos / 09_hand_tip_quat | 12/16 | /hand/right/fingertip_poses |
| 10_hand_paxini_ft / 11_hand_paxini_raw | 12/1524 | /paxini/right/ft · /paxini/right/raw |
| **로봇 암** | | |
| 12_franka_Arm_j_pos / 13_franka_Arm_j_tar / 14_franka_Arm_j_vel / 15_franka_Arm_j_tq | 7 each | /franka/right/joint_states · /franka/right/target_joint_states |
| 16_franka_Arm_C_pos / 17_franka_Arm_C_quat | 3/4 | /franka/right/ee_pose |
| 18_franka_Arm_tar_pos / 19_franka_Arm_tar_quat | 3/4 | /franka/right/ee_target_world |
| 20_franka_Arm_speed_factor | 1 | /franka/right/speed_factor |
| **글러브** | | |
| 21_glove_g_pos | 16 | /glove/right/q_raw |
| 22_glove_paxini_ft / 23_glove_paxini_raw | 12/1524 | /glove/paxini/right/ft · /glove/paxini/right/raw |
| **과일** | | |
| 30_fruit_pos / 31_fruit_quat / 32_fruit_size | 3/4/3 | /fruit/pose · /fruit/size |
| 33_fruit_type / 34_fruit_corners | 1/24 | /fruit/type · /inhand/bbox_corners (8코너 OBB) |
| **이미지 (`--rgb-on`)** | | |
| 40_image / 41_rgb_time / 42_rgb_stamp | 1 / n_frames / n_frames | 레코더가 만듦 (FIELDS 아님) |
| **클락** | | |
| 50_real_time_demo / 51_real_time_global | 1/1 | 시간 |

`/fruit/type_name`(String)은 float 배열에 못 담아 **`Demo_N.attrs["fruit_type_name"]`** 에 넣습니다(예: `"lemon"`).
`/fruit/reset`·`/fruit/set_type` 은 사람이 보내는 **명령** 토픽이라 상태 데이터가 아니어서 제외했습니다.

> ⚠️ **2026-08-06 이전에 수집한 h5 는 옛 번호**입니다(`20_paxini_ft`, `16_glove_g_pos`, `18/19_real_time_*`,
> `42/43_rgb_*`, 팔은 `10~15`+`22~24`). 옛 파일은 그대로 두고 새로 수집하는 파일만 위 번호를 씁니다 —
> 두 세대를 같이 학습에 쓰려면 로더에서 이름을 매핑하세요.

> 필드 추가/제거는 `record/ros2_hdf5_recorder.py` 상단 `FIELDS` 표만 수정.
> 프랑카 토픽 이름이 스택마다 다를 수 있으니(로봇 스택 `/franka/right/*` vs vive 스택 `/franka/ee_pose_r`) `--check`로 확인 후 맞추기.

## RGB 영상 저장 (`--rgb-on`)

기본은 RGB를 저장하지 않습니다(저차원 데이터만). 이미지가 필요하면 옵션으로 켭니다.

```bash
python3 record/ros2_hdf5_recorder.py --rgb-on                 # 전부 저장 (≈30Hz)
python3 record/ros2_hdf5_recorder.py --rgb-on --rgb-every 3   # 3장마다 1장 (≈10Hz, 용량 1/3)
python3 record/ros2_hdf5_recorder.py --check --rgb-on         # RGB 수신되는지 먼저 점검
python3 record/ros2_hdf5_recorder.py --rgb-on --rgb-topic /other/cam/compressed
```

**설계 (실측 + 타 그룹 관행 조사 기반)**
- 카메라가 **이미 JPEG**(`CompressedImage`, `rgb8; jpeg compressed bgr8`)로 발행 → **재인코딩 없이 바이트를 그대로** `.jpg` 파일로 씁니다. CPU 비용 0, 세대손실 0. (ALOHA는 카메라가 raw를 주기 때문에 `cv2.imencode(q=50)`으로 인코딩하지만, 우리는 그럴 필요가 없습니다)
- **이미지는 h5 밖에 둡니다.** h5 는 저차원만 담아 가벼워지고(스크립트로 훑기 빠름), 이미지는 뷰어로 바로 열리고 rsync 로 부분 백업이 됩니다. h5 에는 스텝별 **파일 번호(`40_image`)만** 남습니다.
- ⚠️ 100Hz 스텝마다 프레임을 복사하면 **41GB/시간**. **고유 프레임만 한 번 저장 + 스텝별 번호**로 3.47배 절감. 이건 **DROID가 쓰는 방식**이고(LeRobot·RLDS·ALOHA는 한 클럭만 쓰고 중복/드롭), 추론 때 정책이 "마지막 도착 프레임"을 보는 것과 분포가 일치해서 더 충실합니다.

**용량 (실측)**

| `jpeg_quality` | 프레임 | 시간당 |
|---|---|---|
| **95 (현재 기본)** | 106~112KB | **~12 GB** |
| 85 | 64KB | 6.8 GB |
| **80 (권장)** | 56KB | **5.9 GB** |
| 50 (ALOHA 선택) | 33KB | 3.5 GB |

용량을 줄이는 가장 좋은 방법은 **발행측 품질을 낮추는 것**입니다(레코더 CPU 0, 이중압축 없음):
```bash
ros2 param list | grep -i jpeg     # 파라미터 이름 확인 후
ros2 param set /front_cam/front .front.color.image_raw.jpeg_quality 80
```
`--rgb-every 3`으로도 1/3이 되지만, 시간 해상도가 10Hz로 떨어집니다.

**저장 위치**

```
record/logs/exp_20260806_143000.h5                        # 저차원 + 이미지 번호
record/logs/exp_20260806_143000_image/Demo_0/000000.jpg   # 폴더명 = h5 이름 + "_image"
                                      Demo_0/000001.jpg
                                      Demo_1/000000.jpg   # 데모마다 번호 0부터 다시
```
h5 를 옮길 때 `_image` 폴더도 **같이** 옮겨야 합니다(짝이 깨지면 이미지 매칭 불가).

**데이터셋 (Demo_N/)**

| 키 | shape | 내용 |
|---|---|---|
| `40_image` | (N,) int32 | **각 100Hz 스텝이 보는 JPEG 파일 번호** → `Demo_N/{번호:06d}.jpg` (-1 = 아직 없음) |
| `41_rgb_time` | (n_frames,) f64 | 수신 시각 — `51_real_time_global`과 **같은 시계** |
| `42_rgb_stamp` | (n_frames,) f64 | 카메라 캡처 시각(ROS header) — 전송 지연 확인용 |

`41/42`의 축은 스텝이 아니라 **파일 번호**입니다 → `41_rgb_time[40_image[step]]`.
파일 attrs에 `image_dir`(폴더명) + `rgb_width/height/fx/fy/cx/cy/format/every` + **시계 앵커 `t0_perf`/`t0_unix`**가 기록됩니다.
`51_real_time_global`은 `perf_counter`(프로세스 단조시계)라 그 자체로는 다른 프로세스·rosbag과 대조가 안 됩니다 — 앵커로 변환하세요:
```python
wall = f.attrs["t0_unix"] + (g["51_real_time_global"][t] - f.attrs["t0_perf"])
```

**저차원 데이터셋은 `chunks=(100, D)` + gzip4 + shuffle로 저장합니다.** h5py 자동 chunking에 맡기면 시간축을 잘게 쪼개 한 스텝 읽기에 수십 chunk를 풀어야 하고(실측 **2237µs → 61µs, 37배 차이**), 학습 DataLoader 속도를 그대로 깎습니다.

**읽는 법**
```python
import h5py, cv2
from pathlib import Path
p = Path("record/logs/exp_20260806_143000.h5")
f = h5py.File(p, "r"); g = f["Demo_0"]
t = g["51_real_time_global"][:]          # 100Hz 타임라인
idx = g["40_image"][:].ravel()           # 스텝 -> JPEG 파일 번호
img_dir = p.parent / f.attrs["image_dir"] / "Demo_0"
img = cv2.imread(str(img_dir / f"{idx[step]:06d}.jpg"))   # BGR, idx==-1 이면 프레임 없음
```

**검증 결과**: 디코딩 (480,640,3) 정상 / 스텝↔프레임 지연 평균 **16.6ms**(30Hz 반주기 = 정상) / 프레임 간격 33.4ms.

## 학습 state / action 스펙 (저장 ≠ 학습 입력)

**HDF5에는 최대한 다 저장하고, 학습에는 골라 쓴다.** 저장은 되돌릴 수 없지만 선택은 나중에 바꿀 수 있으므로.

| HDF5 키 | dim | 저장 | 학습 state | 비고 |
|---|---|---|---|---|
| `03_hand_j_pos` | 16 | ✅ | ✅ | 손 현재 각도 |
| `06_hand_j_kin` (**paxini ft**) | 12 | ✅ | ✅ | **촉각은 ft 만 학습에 사용** |
| `07_hand_j_tac` (**paxini raw**) | 1524 | ✅ | ❌ | 보관·분석용. 차원이 너무 커 그대로 넣으면 과적합·지배 |
| `30_fruit_pos` | 3 | ✅ | ✅ | 과일 위치 |
| `31_fruit_quat` | 4 | ✅ | ✅→가공 | 원본 quat 대신 **누적 unwrap 각도의 sin/cos** 권장 |
| `32_fruit_size` | 3 | ✅ | ✅ (a,b) | |
| `05_hand_j_cur` | 16 | ✅ | 선택 | 전류. 넣을지는 실험으로 결정 |
| `08/09_hand_tip_*` | 12/16 | ✅ | 선택 | 손끝 pos/quat |
| `10~12_franka_r_*` | 3/4/7 | ✅ | ❌ | 이번 데모는 팔 고정이라 상수 |
| `21_glove_g_pos` | 16 | ✅ | 🚫 **금지** | 액션이 글러브의 결정적 함수 → **정답 유출**. 디버깅용으로만 |
| `04_hand_j_tar` | 16 | ✅ | — | **action (정책 출력)** |

- **action** = `04_hand_j_tar` 16차원 (절대 타겟, count)
- 학습 변환 시 **10~20Hz로 다운샘플** (100Hz 원본이면 PRED_HORIZON=16이 0.16초로 무의미)
- 각 차원 z-score 정규화 (obs_norm / act_norm 저장)

⚠️ **수집 전 필수 확인**: `06_hand_j_kin`(paxini ft)이 실제로 접촉에 반응해야 미끄러짐·리커버리가 학습됩니다. 값이 계속 0이면 촉각 없이 학습하는 셈이 됩니다.
