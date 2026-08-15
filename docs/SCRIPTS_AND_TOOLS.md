# 스크립트 · 파이썬 파일 실행 가이드

이 저장소의 **모든 실행 진입점**을 한 곳에 정리한다.
원칙: `scripts/*.sh` 는 **env 를 내부에서 처리**하므로 그냥 `bash scripts/xxx.sh` 로 쓰면 되고,
`*.py` 를 직접 돌릴 때만 아래 공통 env 를 먼저 export 한다.

## 공통 env (py 를 직접 실행할 때만 필요)

```bash
source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
```

> `Permission denied` 가 나면 실행비트 문제다 → `chmod +x scripts/*.sh Visualization/*.py`
> **GUI 프로그램은 sudo 로 돌리지 말 것** (root 는 X 접근이 막혀 창이 안 뜬다).

---

# 1. 자주 쓰는 것 — 이것만 알면 된다

| 하고 싶은 것 | 명령 |
|---|---|
| 상태 확인 (뭐가 떠 있나) | `bash run.sh status` |
| 글러브 텔레옵 + 촉각 | `bash scripts/run_glove_paxini.sh` |
| 글러브 텔레옵 + **촉각 보조(DexFIT)** | `bash scripts/run_glove_paxini.sh --DexFIT` |
| 발판 (제어권 + 로깅 토글) | `python3 record/foot_pedal_glove.py` |
| 데이터 수집 (HDF5) | `python3 record/ros2_hdf5_recorder.py` |
| 데이터 수집 + RGB | `python3 record/ros2_hdf5_recorder.py --rgb-on` |
| **과일 인식 → 통합 시각화** | `bash scripts/run_fruit_viz_all.sh` |
| 시각화만 | `bash scripts/run_live_viz.sh` |
| 학습된 정책 배포 | `cd diffusion_policy && python3 run_kist_vtdp.py --model_type 000` |

---

# 2. 통합 런처

### `run.sh` — 전체 스택 통합 런처
```bash
bash run.sh status          # 지금 뭐가 떠 있는지 (글러브 중복 확인 필수)
bash run.sh --help
```

### `scripts/start_teleop.sh` — 입력 장치 선택 텔레옵
```bash
bash scripts/start_teleop.sh          # vive / glove / both 중 선택
```

---

# 3. 글러브 텔레옵 (`tools/`)

### `scripts/run_glove_paxini.sh` ⭐ — 글러브 + Paxini 촉각 동시 기동
```bash
bash scripts/run_glove_paxini.sh                   # 오른손
bash scripts/run_glove_paxini.sh left              # 왼손
bash scripts/run_glove_paxini.sh --DexFIT          # 촉각 보조 ON
bash scripts/run_glove_paxini.sh --DexFIT --dry-run # 손 안 움직이고 값만 확인 (첫 실행 권장)
```
내부에서 `tools/paxini_uart_node.py`(→`/glove/paxini/<side>/{ft,raw}`) + `tools/glove_teleop.py` 를 함께 띄우고,
기존 프로세스를 먼저 정리한다(UART 를 exclusive 로 잡으므로 중복 실행 금지).

**DexFIT (촉각 보조)**: 손가락 Fz 로 그 손가락 굽힘 관절에 추가 각도를 준다.
`target[j] = clamp(글러브각도[j] + GAIN*SCALE*Fz)`,
`Fz[0]엄지→관절 2,3` `Fz[1]검지→5,6,7` `Fz[2]중지→9,10,11` `Fz[3]약지→13,14,15`.
게인·상한·데드존·매핑은 **`tools/glove_teleop.py` 상단 "── 6. 촉각 보조 (DexFIT)"** 에서 조정.
⚠️ 검지(관절 5·6·7)는 서보가 명령을 3.45배 증폭하므로 `PAXINI_MAX_EXTRA` 를 300~500 으로 낮춰 시작할 것.

### `tools/glove_teleop.py` — 글러브 → 핸드 타겟 (본체)
```bash
python3 tools/glove_teleop.py                 # 자동 포트 탐지
python3 tools/glove_teleop.py --dry-run       # 타겟 미발행 (값만)
python3 tools/glove_teleop.py --side left --port /dev/ttyUSB1
GLOVE_DIAG=1 python3 tools/glove_teleop.py    # 내부 발행 통계 5초마다
```
발행: `/glove/<side>/q_raw`(글러브 raw), `/hand/<side>/q_target`(핸드 타겟)
구독: `/hand/<side>/joint_states`, `/teleop/hand_engage/<side>`(발판 제어권)

### 글러브 진단용 모니터
| 파일 | 용도 |
|---|---|
| `tools/glove_joint_monitor.py` | 16관절 각도 라이브 (관절 이름 + rad/count) |
| `tools/glove_paxini_monitor.py` | 글러브 16채널 + Paxini ft 를 토픽으로 구독해 표시 |
| `tools/glove_raw_uart.py` | UART 순수 raw (필터·클램프·램프 전부 없음) |
| `tools/glove_tactile_monitor.py` | 글러브 16채널 + Paxini 4손가락 동시 raw |
| `tools/paxini_uart_node.py` | Paxini UART → ROS2 발행 (`--topic-prefix /glove/paxini`) |
| `tools/paxini_writer.py` | Paxini → SHM(key 0x7951) writer |

---

# 4. 데이터 수집 (`record/`)

### `record/foot_pedal_glove.py` — 발판
```bash
python3 record/foot_pedal_glove.py            # 오른손
python3 record/foot_pedal_glove.py --side left
```
| 페달 | 동작 |
|---|---|
| **오른쪽** | 텔레옵 ENGAGE (`/teleop/hand_engage/<side>` = true) |
| **왼쪽** | DISENGAGE (홀드 — 손 안 풀림) |
| **중간** | 로깅 시작/종료 토글 (`/record/enable`) |

최초 1회 권한: `sudo usermod -aG input $USER` 후 재로그인 (또는 udev 0666).

### `record/ros2_hdf5_recorder.py` ⭐ — HDF5 레코더
```bash
python3 record/ros2_hdf5_recorder.py --check          # 어떤 토픽이 오는지 3초 점검
python3 record/ros2_hdf5_recorder.py                  # 수집 (발판 중간으로 S/E)
python3 record/ros2_hdf5_recorder.py --rgb-on         # RGB(JPEG) 도 저장 (~12GB/시간)
python3 record/ros2_hdf5_recorder.py --rgb-on --rgb-every 3   # 3장마다 1장 (1/3 용량)
python3 record/ros2_hdf5_recorder.py --check --rgb-on
```
- 저장: `record/logs/exp_YYYYMMDD_HHMMSS.h5` (에피소드 = `Demo_N` 그룹, 100Hz)
- 실행 중 **`k` 키** = 현재 파일 저장하고 다음 에피소드부터 새 파일 (배치 분리)
- 자세한 스키마/사용법: `docs/DATA_COLLECTION_USAGE.md`

### `record/check_perception.py` — 수집 전 프리플라이트
```bash
python3 record/check_perception.py            # 5초 점검
python3 record/check_perception.py --sec 10
```
카메라(color/depth/camera_info)와 과일 토픽이 실제로 오는지 Hz 로 확인.

### `record/realsense_view.py` — 카메라 뷰어 + mp4 녹화
```bash
python3 record/realsense_view.py              # 창에서 r=녹화토글, q=종료
python3 record/realsense_view.py --sync       # 발판 로깅과 동기 녹화
python3 record/realsense_view.py --no-window  # 창 없이 녹화만
```

---

# 5. 과일 인식 + 시각화

### `scripts/run_fruit_viz_all.sh` ⭐ — 과일 인식 → 시각화 순차 기동
```bash
bash scripts/run_fruit_viz_all.sh                 # 기본 lemon
bash scripts/run_fruit_viz_all.sh --fruit peach
bash scripts/run_fruit_viz_all.sh --check         # 전제조건 점검만
bash scripts/run_fruit_viz_all.sh --keep-overlay  # 기존 fruit_overlay 창도 함께
bash scripts/run_fruit_viz_all.sh --no-tactile    # 촉각 끄기
```
1) `foundation_pose/run_foundation_pose.sh --no-overlay` (SAM2 로 과일 인식 → FoundationPose 6DoF)
2) `/fruit/pose` 수신 대기 (최대 180초)
3) `Visualization/live_viz.py` 통합 시각화
Ctrl+C 시 docker `fp_server` 포함 전부 정리.

### `scripts/run_live_viz.sh` — 통합 시각화만
```bash
bash scripts/run_live_viz.sh --selftest     # ROS 없이 렌더 검증 (PNG 저장)
bash scripts/run_live_viz.sh                # 실시간
bash scripts/run_live_viz.sh --no-tactile   # 과일만
bash scripts/run_live_viz.sh --vmax 2.0     # 촉각 색상 상한
bash scripts/run_live_viz.sh --tactile-topic /glove/paxini/right/raw   # 글러브 촉각으로
```

### `Visualization/live_viz.py` — 통합 시각화 본체
```
┌────────────────────────┬──────────────────┐
│ 카메라 + 과일 바운딩박스│ Paxini 촉각 3D   │
│  + XYZ축 + 수치        │  (4개 파트)      │
├──────┬──────┬──────┬───┴──────────────────┤
│Thumb │Index │Middle│ Ring                 │  ← 손가락 4칸
│  각 칸에 Tx, Ty, Fz 3줄 (링버퍼 6초)      │
└──────┴──────┴──────┴──────────────────────┘
```
- 과일 포즈: `record/fruit_overlay.py` 의 `Overlay` 클래스를 임포트해 재사용
- 촉각: `Visualization/tactile_render.py` 의 `Scene3D`(Open3D) — 지문 CAD 4개에 127 탁셀 색상 + 상위 2개 힘 화살표
- 키: `q`/`ESC` 종료, `s` 스냅샷(`Visualization/snapshots/`)
- 에셋: `Visualization/assets/{fingertip-*.stl, taxel_m2826_127.csv}`

**구독 토픽 (기본값)**
```
/front_cam/front/color/image_raw/compressed   카메라
/front_cam/front/color/camera_info            K
/fruit/pose      /fruit/size                  과일 6DoF + 크기
/paxini/right/raw   (1524 = 4x127x3)          촉각
/hand/right/kin     (12 = 4x(Fz,Tx,Ty))       FT
```

### `Visualization/tactile_render.py` — 촉각 렌더러 (라이브러리)
직접 실행하지 않고 `live_viz.py` 가 임포트한다. 상수(`VMAX_DEFAULT`, 색상, `PART_COLORS`) 조정은 이 파일 상단.

### 과일 포즈 파이프라인 (별도 저장소성 디렉터리, git 미추적)
```bash
bash foundation_pose/run_foundation_pose.sh --check          # 사전 점검
bash foundation_pose/run_foundation_pose.sh --fruit lemon    # 실행
bash foundation_pose/run_foundation_pose.sh --fruit lemon --no-overlay   # 자체 창 끄기
```
⚠️ `--fruit` 를 **값 없이** 주면 조용히 lemon 으로 떨어진다. 항상 이름을 붙일 것.

### 구형(SAM2 bbox) 과일 경로 — 참고용
| 파일 | 용도 |
|---|---|
| `record/run_fruit_viz.sh` | republish + live_bbox_gui + bridge + overlay 원커맨드 |
| `record/fruit_pose_bridge.py` | `/inhand/bbox_corners`(8코너) → `/fruit/pose` + `/fruit/size` (PCA) |
| `record/fruit_overlay.py` | 영상 위 6DoF 오버레이 (`--selftest` 로 ROS 없이 검증 가능) |
| `record/fruit_viz.py` | 과일 위치/크기 matplotlib 시각화 |

---

# 6. 학습 정책 배포 (`diffusion_policy/`, git 미추적)

```bash
export KIST_VTDP_REPO=~/Desktop/kist-vtdp-wrapper
cd diffusion_policy
python3 run_kist_vtdp.py --list_models                  # 모델 표
python3 run_kist_vtdp.py --model_type 000 --self_test    # ROS/로봇 없이 계약 검증
python3 run_kist_vtdp.py --model_type 000 --dry_run      # 전체 상태머신, 발행만 안 함
python3 run_kist_vtdp.py --model_type 000                # 실전
```
`000` = 레몬 · 시각+촉각 · J 405.45 · **유일하게 실기 검증됨**.

⚠️ **정지는 홀드이지 이완이 아니다.** 힘을 빼려면 다른 터미널에 미리 준비:
```bash
ros2 topic pub -1 /hand/right/cmd_servo std_msgs/Bool "{data: false}"
```
⚠️ 실행 전 `bash run.sh status` 로 **`glove_teleop` 이 0개**인지 확인 (q_target 발행자 둘이면 손이 튐).

---

# 7. Vive 트래커 (팔 텔레옵)

| 명령 | 용도 |
|---|---|
| `bash scripts/start_teleop_pipeline.sh` | 헤드리스 파이프라인(viz_node + teleop_delta) |
| `bash scripts/run_viz.sh` | viz_node + RViz (GUI) |
| `bash scripts/run_tracker_read.sh` | 트래커 절대 포즈 + 델타 터미널 표시 |
| `bash scripts/run_viz_and_read.sh` | RViz + 터미널 6DoF 동시 |
| `bash scripts/run_clutch.sh` | 키보드 클러치 (1=engage, space=hold, q=종료) |
| `bash scripts/run_delta_monitor.sh` | ABS/INC/WORLD 델타 비교 |
| `bash scripts/run_delta_viz.sh` | 델타 3D 경로 + 시계열 그래프 |
| `bash scripts/run_relative_pose.sh` | engage 기준 상대 포즈 발행 |
| `sudo bash scripts/fix_ros_net.sh` | LAN 프로파일 고정 (1회) |

---

# 8. 자주 겪는 문제

| 증상 | 원인 / 조치 |
|---|---|
| `Permission denied` | 실행비트 없음 → `chmod +x`. sudo 는 해결책이 아니다 |
| GUI 창이 안 뜸 | root 로 실행했거나 `DISPLAY` 불일치 → 일반 사용자, `DISPLAY=:1` 확인 |
| 토픽이 하나도 안 보임 | `config/fastdds_lan_only.xml` 의 IP·`ROS_DOMAIN_ID=9`·`ROS_LOCALHOST_ONLY=0` 확인 |
| 글러브가 2초마다 멈춤 | **중복 실행**이 원인인 경우가 많다 → `bash run.sh status` 로 `glove_teleop` 개수 확인 |
| 손이 튐 | `q_target` 발행자가 둘 (글러브 + 정책) → 하나만 실행 |
| 촉각이 전부 0 | Paxini 노드 미실행 또는 `/dev/ttyACM0` 권한 → `sudo chmod 666 /dev/ttyACM0` |

---

# 9. git 미추적 디렉터리 (용량·별도 저장소)

`.gitignore` 에 의해 아래는 **코드만 있고 파일은 추적되지 않는다.** 새 PC로 옮길 땐 별도 복사 필요:
`hdf5/`, `record/logs/`, `record/snapshots/`, `fruit-manipulation/`, `foundation_pose/`,
`diffusion_policy/`, `Obj/`, `vendor/`, `build/`, `install/`, `log/`

→ 이관용 zip 패키지 만드는 법과 새 PC 세팅은 **`Visualization/PACKAGE_README.md`** 참고.
