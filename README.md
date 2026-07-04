# vive_franka_teleop

Vive Tracker 2개 → Franka(FR3) 2팔 + 4지 핸드 **실시간 텔레옵**.
SteamVR/OpenVR로 트래커 포즈를 읽어 engage(클러치) 기준 6DoF 상대 델타를 계산하고,
제어 PC가 잡고 있는 **현재 EE 포즈에 얹어 "절대 EE 타겟 포즈"** 로 변환해 ROS2(DDS, 전용 LAN)로 전송한다.
USB 3구 풋스위치로 클러치(정지/전송)와 핸드(Init/Grasp) 조작.

```
Vive PC (192.168.0.1)                                   제어 PC (192.168.0.100)
SteamVR ─ viz_node ─ /vive/{l,r}/pose ─┐
                                        └ teleop_delta ─ /franka_{r,l}/ee_target_world ─LAN▶ IK+임피던스 ─ FR3 2팔
        /franka/ee_pose_{r,l} ◀───────────┘ (engage 순간 EE 앵커 T_ee0 캡처)
foot_pedal_teleop ─ /teleop/engage/{l,r} (클러치)
                  └ /hand/q_target_{r,l} + /hand/cmd_servo|mode ─LAN▶ hand_target_receiver ─ 4지 핸드
```

변환식(engage 시 EE 앵커 `T_ee0`, 트래커 델타 `Δp/ΔR`):
`T_target = T_ee0 · [ R_align·ΔR·R_alignᵀ | ee_scale·R_align·Δp ]` → 트래커 자기좌표계 이동/회전이 EE 자기좌표계로 매핑(기본 `R_align`=단위, `ee_scale`=1.0).

트래커 매핑: **tracker_1(LHR-7B9A3BA9)→right**, **tracker_2(LHR-F4A94AD1)→left** (시리얼 고정, SteamVR 인덱스 무관).

---

## 실행 방법 (ROS 기동부터)

### 0) 준비 (최초 1회)
- **SteamVR 실행 + 트래커 2개 초록불(tracking)** 확인.
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

### 터미널 1 (T1) — Vive→ROS 연결 + EE 타겟 전송
```bash
bash ~/Desktop/vive_franka_teleop/scripts/start_teleop_pipeline.sh
```
env·프로파일 내부 처리하고 **viz_node**(트래커→`/vive/{l,r}/pose`) + **teleop_delta**(→`/franka_{r,l}/ee_target_world`)를 헤드리스로 기동. 로그: `/tmp/viz_node.log`, `/tmp/teleop_delta.log`.
> 트래커를 눈으로 보려면 대신 `bash scripts/run_viz.sh`(RViz) — 단 이건 teleop_delta를 안 띄우므로 전송이 필요하면 T1을 쓴다.

### 터미널 2 (T2) — 풋스위치 (클러치 + 핸드)
```bash
python3 ~/Desktop/vive_franka_teleop/scripts/foot_pedal_teleop.py
```
| 페달 | 동작 |
|---|---|
| **왼쪽** | 클러치 **STOP** (양팔 disengage → 로봇 홀드) |
| **오른쪽** | **델타 EE 전송** (양팔 engage → 트래커 따라 EE 타겟 전송) |
| **중간** | 핸드 **Init↔Grasp** 토글 (시작 Init, 2초 보간) |

풋스위치를 직접 read(EVIOCGRAB 독점)하므로 포커스/한영 무관. 실행 시 **핸드 서보 자동 ON + mode1 + Init 이동** → 실행 전 손을 Init 근처에. 종료(Ctrl-C) 시 클러치 STOP + 서보 OFF.
> ⚠️ `run_clutch.sh`(키보드 클러치)·`hand_target_test.py`와 **동시 실행 금지** (engage/q_target 발행 충돌).

### 터미널 3 (T3) — EE 모니터
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
| 클러치 | `/teleop/engage/{left,right}` | std_msgs/Bool | true=engage / false=stop |
| 구독(팔) | `/franka/ee_pose_{r,l}` | geometry_msgs/PoseStamped | 제어PC 실제 EE(200Hz), engage 시 앵커 |
| 발행(핸드) | `/hand/q_target_{r,l}` | std_msgs/Float32MultiArray[16] | 관절 타겟(엔코더 카운트) |
| 발행(핸드) | `/hand/cmd_servo_{r,l}` · `/hand/cmd_mode_{r,l}` | std_msgs/Bool · Int32 | 서보 on/off · mode(1=position) |
| 트래커 | `/vive/{left,right}/pose` · `/vive/{...}/valid` | PoseStamped · Bool | vive_world, +Y up |

파라미터(`config/teleop_params.yaml`): `ee_scale`(트래커→로봇 위치 스케일, 1.0=1:1), `publish_ee_target`, `ee_timeout`, `r_align_{right,left}`(트래커↔로봇 정렬 3×3, 기본 단위).
제어 PC 수신측 계약: `docs/VIVE_PC_HANDOFF.md`.

## 네트워크 (전용 LAN)
- 이 PC enp6s0 = `192.168.0.1/24`, 제어 PC = `192.168.0.100/24`, 직결 GbE.
- 공통 환경의 `FASTRTPS_DEFAULT_PROFILES_FILE`(`config/fastdds_lan_only.xml`)가 DDS를 enp6s0만 쓰게 강제(WiFi 우회 차단).
- 제어 PC: `config/fastdds_lan_only_NEIGHBOR.xml` 사용.
- 확인: `ping 192.168.0.100`, `ros2 topic list`에 양쪽 토픽.

## 폴더 구조 (핵심)
```
scripts/
  start_teleop_pipeline.sh   T1: viz_node + teleop_delta 데몬 기동
  foot_pedal_teleop.py       T2: 3구 풋스위치 (클러치+핸드)
  ee_monitor.py              T3: 왼/오 EE 타겟·실제 EE 모니터
  hand_target_test.py        핸드 단독 테스트 (s/f=서보, m=mode1, g/i/0=타겟, 2초 보간)
  run_viz.sh                 viz_node + RViz(GUI)
  run_clutch.sh              키보드 클러치 (1=engage, space=stop)  ※페달과 동시 사용 금지
  run_tracker_read.sh / run_delta_monitor.sh / run_delta_viz.sh   (선택) 델타/트래커 확인 도구
  fix_ros_net.sh             네트워크 고정(sudo)
src/vive_3d_viz/             핵심 ROS2 패키지 (viz_node, teleop_delta) — franka_ros2_ws/src 에 심볼릭
config/                      viz_params.yaml, teleop_params.yaml, fastdds_lan_only*.xml
docs/VIVE_PC_HANDOFF.md      제어 PC 수신측 인터페이스 계약
```
> colcon 빌드는 `~/franka_ros2_ws` 에서. `vendor/`(libsurvive, legacy)는 git 미포함(.gitignore).

## 단위
- 위치(EE 타겟/트래커): 미터 [m] · 회전: rotvec[rad] / quaternion · 핸드 `q_target`: 엔코더 카운트

## License
Copyright (c) 2026 **KIST Prime Lab — Jaesung Lee** (jay.lee@kist.re.kr). See [LICENSE](LICENSE).
