# vive_franka_teleop

**Vive Tracker(팔) + TUMI 글러브(손) → Franka(FR3) 2팔 + 4지 핸드 실시간 텔레옵.**
입력 장치를 **옵션으로 골라서**(`vive` / `glove` / `both`) 한 명령으로 기동한다.

- **팔(vive)**: SteamVR로 트래커 포즈를 읽어 engage 기준 6DoF 상대 델타를 계산 → 제어 PC의
  **현재 EE 포즈에 얹어 "절대 EE 타겟 포즈"** 로 변환해 전송.
- **손(glove)**: 글러브 시리얼 16관절을 **1:1 매핑**해 핸드 q_target 으로 전송(램프·EMA·rate limit).
- **USB 3구 풋스위치**: 왼=STOP · 오른=GO · 중간=로깅 S/E 토글.

```
Vive PC (192.168.0.1)                                   제어 PC (192.168.0.100)
SteamVR ─ viz_node ─ /vive/{l,r}/pose ─┐
                                        └ teleop_delta ─ /franka_{r,l}/ee_target_world ─LAN▶ IK+임피던스 ─ FR3 2팔
        /franka/ee_pose_{r,l} ◀───────────┘ (engage 순간 EE 앵커 T_ee0 캡처)
글러브(USB 시리얼) ─ glove_teleop ─ /hand/<side>/q_target + cmd_servo|mode ─LAN▶ hand_target_receiver ─ 4지 핸드
풋스위치 ─ foot_pedal ─ /teleop/engage/{l,r}(팔) + /teleop/hand_engage/<side>(손) + /record/enable(로깅)
```

변환식(engage 시 EE 앵커 `T_ee0`, 트래커 델타 `Δp/ΔR`):
`T_target = T_ee0 · [ R_align·ΔR·R_alignᵀ | ee_scale·R_align·Δp ]` → 트래커 자기좌표계 이동/회전이 EE 자기좌표계로 매핑(기본 `R_align`=단위, `ee_scale`=1.0).

트래커 매핑: **tracker_1(LHR-7B9A3BA9)→right**, **tracker_2(LHR-F4A94AD1)→left** (시리얼 고정, SteamVR 인덱스 무관).

---

## 실행 방법 (ROS 기동부터)

### 0) 준비 (최초 1회)
- **SteamVR 실행 + 트래커 2개 초록불(tracking)** 확인 (vive/both 모드).
- **글러브 USB 연결** (glove/both 모드). 포트는 CH340 자동 탐지 — 꽂은 자리가 바뀌어도 찾음.
  단 **CH340은 시리얼번호가 없어 2개를 동시에 꽂으면 `by-id`가 겹친다** → 왼손 추가 시
  `/dev/serial/by-path/...` 로 좌/우를 고정해 `--port` 로 넘겨야 한다. (현재는 오른손 1개)
- **풋스위치 권한**: `sudo usermod -aG input $USER` → 재로그인 (또는 실행 셸에서 `newgrp input`).
- **네트워크**(1회, sudo): `sudo bash ~/Desktop/vive_franka_teleop/scripts/fix_ros_net.sh` (enp6s0→192.168.0.1 고정).
- (코드 수정 후) 빌드: `cd ~/franka_ros2_ws && colcon build --packages-select vive_3d_viz --symlink-install`

### 공통 환경 — 새 터미널마다 맨 위에
```bash
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
```

### 터미널 1 (T1) — 통합 기동 (입력 장치 선택)
```bash
bash ~/Desktop/vive_franka_teleop/scripts/start_teleop.sh both     # vive 팔 + 글러브 손 (기본)
bash ~/Desktop/vive_franka_teleop/scripts/start_teleop.sh vive     # vive 팔만
bash ~/Desktop/vive_franka_teleop/scripts/start_teleop.sh glove    # 글러브 손만 (현재 오른손 1개)
```
모드에 따라 아래를 순서대로 띄운다(**안전 순서**: 정리 → 페달 → vive → 글러브).

| 모드 | 띄우는 것 |
|---|---|
| `vive` | 페달 + `viz_node`(트래커→`/vive/{l,r}/pose`) + `teleop_delta`(→`/franka_{r,l}/ee_target_world`) |
| `glove` | 페달 + `glove_teleop --side right` (vive 파이프라인은 내림) |
| `both` | 위 전부 |

**페달을 먼저 띄우는 이유**: 페달이 STOP 을 latched(TRANSIENT_LOCAL)로 깔아두면, 뒤에 뜨는
`glove_teleop`(기본 engage=True, 뜨자마자 서보 ON+램프)이 그 STOP 을 받아 **손이 안 움직인다**.
팔(`teleop_delta`)은 기본 상태가 disengaged 라 애초에 안전.

| 페달 | 동작 |
|---|---|
| **왼쪽** | **STOP** — 팔 disengage + 손 disengage (홀드, 손 힘 안 풀림) |
| **오른쪽** | **GO** — 팔 engage + 손 engage (스트리밍 시작) |
| **중간** | **로깅 S/E 토글** (`/record/enable`: true=에피소드 시작, false=저장) |

풋스위치는 `/dev/input/...FootSwitch-event-kbd` 직접 read + EVIOCGRAB(독점) → 포커스/한영 무관.
**페달은 한 프로세스만 잡을 수 있다** — `record/foot_pedal_glove.py`, `scripts/foot_pedal_teleop.py`와 동시 실행 불가(런처가 자동 정리).
로그: `/tmp/foot_pedal.log`, `/tmp/glove_right.log`, `/tmp/viz_node.log`, `/tmp/teleop_delta.log`.

> 개별 실행도 가능: `bash scripts/start_teleop_pipeline.sh`(vive만) · `python3 scripts/foot_pedal.py --mode vive` ·
> `python3 tools/glove_teleop.py --side right [--dry-run]` · 트래커를 눈으로 보려면 `bash scripts/run_viz.sh`(RViz).

### 터미널 2 (T2) — EE 모니터
```bash
python3 ~/Desktop/vive_franka_teleop/scripts/ee_monitor.py
```
왼/오 **EE 타겟(우리가 쏨)** 과 **실제 EE pose(제어PC)** 를 실시간 표로. `⚠ZERO`(원점근처=재engage 점프원인) / `⚠STALE`(수신끊김) 표시.

---

## 토픽 / 메시지
| 방향 | 토픽 | 타입 | 비고 |
|---|---|---|---|
| 발행(팔) | `/franka_{r,l}/ee_target_world` | geometry_msgs/PoseStamped | **절대 EE 타겟** (franka base `fr3_link0_{r,l}`), engage 중에만 |
| 발행(팔,디버그) | `/teleop/delta/{left,right}` | std_msgs/String(JSON) | Δp·ΔR·engaged·valid (로컬 모니터용) |
| 클러치(팔) | `/teleop/engage/{left,right}` | std_msgs/Bool | true=engage / false=stop |
| 구독(팔) | `/franka/ee_pose_{r,l}` | geometry_msgs/PoseStamped | 제어PC 실제 EE(200Hz), engage 시 앵커 |
| 발행(손) | `/hand/<side>/q_target` | std_msgs/Float32MultiArray[16] | 관절 타겟(엔코더 카운트), side=right\|left |
| 발행(손) | `/hand/<side>/cmd_servo` · `/hand/<side>/cmd_mode` | std_msgs/Bool · Int32 | 서보 on/off · mode(1=position) |
| 클러치(손) | `/teleop/hand_engage/<side>` | std_msgs/Bool | latched. glove_teleop 게이팅 |
| 발행(글러브) | `/glove/<side>/q_raw` | std_msgs/Float32MultiArray[16] | 글러브 엔코더 raw |
| 로깅 | `/record/enable` | std_msgs/Bool | true=에피소드 시작 / false=저장 |
| 트래커 | `/vive/{left,right}/pose` · `/vive/{...}/valid` | PoseStamped · Bool | vive_world, +Y up |

> ⚠️ 손 토픽은 제어 PC 개편으로 **per-side 네이밍**(`/hand/right/q_target`)이다. 구 네이밍
> (`/hand/q_target_r` 등)을 쓰는 `scripts/foot_pedal_teleop.py`·`scripts/hand_target_test.py`는
> 현재 제어 PC와 맞지 않는다(미갱신). 페달은 `scripts/foot_pedal.py` 를 쓸 것.

파라미터(`config/teleop_params.yaml`): `ee_scale`(트래커→로봇 위치 스케일, 1.0=1:1), `publish_ee_target`, `ee_timeout`, `r_align_{right,left}`(트래커↔로봇 정렬 3×3, 기본 단위).
제어 PC 수신측 계약: `docs/VIVE_PC_HANDOFF.md`(팔) · `docs/GLOVE_PC_HANDOFF.md`(손) · 수집 절차 `docs/DATA_COLLECTION_USAGE.md`.

## 네트워크 (전용 LAN)
- 이 PC enp6s0 = `192.168.0.1/24`, 제어 PC = `192.168.0.100/24`, 직결 GbE.
- 공통 환경의 `FASTRTPS_DEFAULT_PROFILES_FILE`(`config/fastdds_lan_only.xml`)가 DDS를 enp6s0만 쓰게 강제(WiFi 우회 차단).
- 제어 PC: `config/fastdds_lan_only_NEIGHBOR.xml` 사용.
- 확인: `ping 192.168.0.100`, `ros2 topic list`에 양쪽 토픽.

## 폴더 구조 (핵심)
```
scripts/
  start_teleop.sh            ★T1: 통합 런처 (both|vive|glove) — 페달·vive·글러브 기동
  foot_pedal.py              ★통합 페달 (--mode, 왼=STOP 오른=GO 중간=로깅)
  ee_monitor.py              T2: 왼/오 EE 타겟·실제 EE 모니터
  start_teleop_pipeline.sh   vive 파이프라인만 (viz_node + teleop_delta)
  run_viz.sh                 viz_node + RViz(GUI)
  run_clutch.sh              키보드 클러치 (1=engage, space=stop)  ※페달과 동시 사용 금지
  run_tracker_read.sh / run_delta_monitor.sh / run_delta_viz.sh   (선택) 델타/트래커 확인 도구
  fix_ros_net.sh             네트워크 고정(sudo)
  foot_pedal_teleop.py / hand_target_test.py   (구 네이밍, 미갱신 — 위 ⚠️ 참고)
tools/glove_teleop.py        글러브 시리얼 → 핸드 q_target (1:1, 램프·EMA·자동재연결)
record/                      데이터 수집 (ros2_hdf5_recorder.py, foot_pedal_glove.py, 인지/시각화 도구)
src/vive_3d_viz/             핵심 ROS2 패키지 (viz_node, teleop_delta) — franka_ros2_ws/src 에 심볼릭
config/                      viz_params.yaml, teleop_params.yaml, fastdds_lan_only*.xml
docs/                        VIVE_PC_HANDOFF(팔 계약) · GLOVE_PC_HANDOFF(손 계약) · DATA_COLLECTION_USAGE(수집)
```
> colcon 빌드는 `~/franka_ros2_ws` 에서. `vendor/`(libsurvive, legacy)는 git 미포함(.gitignore).

## 단위
- 위치(EE 타겟/트래커): 미터 [m] · 회전: rotvec[rad] / quaternion · 핸드 `q_target`: 엔코더 카운트

## License
Copyright (c) 2026 **KIST Prime Lab — Jaesung Lee** (jay.lee@kist.re.kr). See [LICENSE](LICENSE).
