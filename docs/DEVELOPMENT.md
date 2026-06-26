# 개발 정리 — Vive → Franka 텔레옵

> 운영(실행) 가이드는 [`../README.md`](../README.md), 제어 PC 수신측 인터페이스는
> [`TELEOP_INTERFACE.md`](TELEOP_INTERFACE.md). 이 문서는 **무엇을 어떻게 만들었고 왜 그렇게
> 정했는지**(아키텍처·컴포넌트·설계 결정·현황) 를 정리한다.
> 최종 갱신: 2026-06-10

---

## 1. 한 줄 요약 / 데이터 흐름

Vive Tracker 2개의 6DoF 포즈를 SteamVR로 읽어, **클러치(engage) 기준 상대 델타**로
바꿔 전용 LAN(ROS2/DDS)으로 옆 제어 PC에 100Hz로 보내 Franka 양팔을 텔레옵한다.

```
Vive PC (192.168.0.1, enp6s0)                              제어 PC (192.168.0.100)
┌───────────────────────────────────────────────────┐
│ SteamVR/OpenVR                                       │
│   └ viz_node ── /vive/{left,right}/pose  (절대,60Hz) │
│                 /vive/{left,right}/valid             │
│                 /vive/markers, TF vive_world         │
│        │                                             │
│        └ teleop_delta ── /teleop/delta/{left,right} ─┼──LAN──▶  Franka 2팔 (T_ee0·dT 적용)
│              ▲ 클러치   /teleop/engage/{left,right}  │
│   (delta_monitor / tracker_read = 디버그 관찰용)     │
└───────────────────────────────────────────────────┘
```

매핑(authoritative = `config/viz_params.yaml`):
**`LHR-7B9A3BA9` → right arm**, **`LHR-F4A94AD1` → left arm**.

---

## 2. 컴포넌트

### ROS2 패키지 (`src/`, `~/franka_ros2_ws/src`에 심볼릭 연결)
| 패키지 | 상태 | 내용 |
|---|---|---|
| **`vive_3d_viz`** | **현재 사용** | SteamVR/OpenVR(triad_openvr) 기반. `viz_node`(포즈/마커/TF) + `teleop_delta`(델타) + `list_devices`. |
| `vive_tracker_ros2` | 대안 | OpenVR 트래커 노드 단독 + mock 노드. |
| `vive_teleop_3d` | legacy | libsurvive 기반(SteamVR 불필요). `vendor/libsurvive` 필요. |

### 노드 (vive_3d_viz)
| 노드 | 발행/구독 | 핵심 |
|---|---|---|
| `viz_node` | → `/vive/<arm>/pose`(PoseStamped, `vive_world`, BEST_EFFORT, 60Hz), `/vive/<arm>/valid`(Bool), `/vive/markers`, TF `vive_world→vive_<arm>` | **시작 시점에만** 장치 탐색 → 트래커 추가/페어링 후 재시작 필요. 시리얼→friendly 매핑은 `tracker_name_map`. |
| `teleop_delta` | 구독 `/vive/<arm>/{pose,valid}`, `/teleop/engage/<arm>`(Bool) · 발행 `/teleop/delta/<arm>`(String JSON, RELIABLE, 100Hz) | engage 시 anchor 캡처 → 매 틱 **anchor 기준 델타** + 절대포즈 동봉 발행. |

### 스크립트 (`scripts/`, 대부분 `bash <스크립트>` 한 줄)
| 스크립트 | 용도 |
|---|---|
| `start_teleop_pipeline.sh` | 헤드리스 파이프라인 기동(viz_node + teleop_delta를 setsid 데몬, 로그 `/tmp/*.log`) |
| `run_viz.sh` | viz_node + RViz(GUI) |
| `run_clutch.sh` / `teleop_clutch_key.py` | 키보드 클러치 (`1`=engage, `space`=정지, `q`=종료). 양팔 동시. |
| `run_tracker_read.sh` / `tracker_read.py` | 트래커 절대 포즈 터미널 실시간 (pos[m]+RPY/quat/axis-angle) |
| **`delta_monitor.py`** | **델타 비교 모니터(신규).** ABS(anchor 누적=A) vs INC(직전 대비 증분=B) vs WORLD(절대) 양팔 라이브. A/B 판단·드리프트 관찰용. |
| `run_viz_and_read.sh` | RViz + 터미널 6DoF 동시 |
| `fix_ros_net.sh` | enp6s0 네트워크 고정(sudo, 1회) |

### 설정 (`config/`)
- `teleop_params.yaml`: `arms=left,right`, `rate_hz=100`, `pos_scale=1.0`, `pose_timeout=0.3`, `reliable=true`, `auto_engage=false`
- `viz_params.yaml`: `tracker_name_map`(시리얼→arm), `publish_tf=true`, `frame_id=vive_world`, `rate_hz=60`
- `fastdds_lan_only.xml`(이 PC) / `fastdds_lan_only_NEIGHBOR.xml`(제어 PC) — DDS를 LAN(enp6s0)만 쓰게 강제

---

## 3. 핵심 설계 결정 (무엇을, 왜)

### 3-1. 델타 규약 = **anchored(engage 기준 절대 델타)**, step 증분 아님
engage 순간 anchor `A=(p0,q0)` 캡처 후, 매 틱:
```
dp  = R0ᵀ (p - p0)      # anchor 로컬 프레임 기준 위치 델타 [m]
dq  = q0⁻¹ ⊗ q          # anchor 기준 회전 델타
rot = axis-angle(dq)    # rotvec [rad]
```
즉 `dT = A⁻¹·T(t)`. (`teleop_delta_node.py:200-203`)

**왜 anchored(A)인가 — step 증분(B) 대비:**
| | A: anchored (채택) | B: step 증분 |
|---|---|---|
| 드리프트(누적오차) | 없음(고정 기준 절대계산) | 매 스텝 노이즈 영구 누적 |
| 프레임 드랍/지연 | 다음 틱이 기준점에서 재계산→자동복구 | 놓친 증분 영구 손실→오프셋 고정 |
| 글리치 1프레임 | 그 틱만 튀고 원복 | 누적 포즈에 영구 주입 |
| 1:1 대응 | 손 원위치→로봇 원위치 보장 | 드리프트로 안 맞음 |
| 클러치 재engage | 재engage=재anchor (의도와 일치) | 누적분 잔존 |

A의 유일한 주의점: ① engage 순간 anchor 프레임이 깨끗해야 함, ② 트래킹 로스트 후 절대 점프
가능(제어 PC velocity clamp 권장), ③ 빠른 동작 시 큰 Δ → 출력단 보간 필요. 부드러움은
A 위에 출력 보간을 얹어 해결(B의 드리프트는 구조적이라 해결 불가).

### 3-2. 안전 = 클러치 우선, 무효 시 hold
- 시작 상태 = **정지(disengage)**. `auto_engage=false`(켜면 정지가 즉시 취소돼 위험).
- `engaged==false` 또는 `valid==false`(standby/시야상실/stale>`pose_timeout`)면 **`pos`/`rot`=0** 발행 → 제어 PC는 현재 자세 hold.
- 종료 시 자동으로 정지 publish.

### 3-3. 메시지에 절대포즈 동봉 (2026-06-10)
델타(`pos`/`rot`) 외에 `abs_pos`/`abs_quat`(vive_world 절대, normalize)도 매 틱 발행 →
제어는 델타로, 절대포즈는 캘리브/디버그로. (`/vive/<arm>/pose` 토픽은 그대로 유지)

### 3-4. 네트워크/QoS
- 랩 표준 env: `ROS_DOMAIN_ID=9`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_LOCALHOST_ONLY=0`.
- 전용 직결 GbE(192.168.0.0/24) + FastDDS 화이트리스트 XML로 WiFi 우회 차단.
- 델타 토픽 RELIABLE/KEEP_LAST(1), pose 토픽 BEST_EFFORT(viz와 매칭).

### 3-5. 단위/좌표계
- `pos`/`abs_pos`: m · `rot`: rad rotvec(|rot|=각, 방향=축) · `abs_quat`: 단위 quat (x,y,z,w).
- `vive_world` = SteamVR 월드, 오른손계 +Y up. **트래커 잡는 방향 ≠ 로봇 EE 방향**이면 제어
  PC에서 고정 정렬회전 `R_align` 1회 구해 적용(→ TELEOP_INTERFACE.md).

---

## 4. 검증된 현황 (2026-06-09 ~ 06-10)

- 실가동 텔레옵 확인, 델타 수학 단위테스트 통과. 델타 100Hz·양팔 `engaged/valid` 채워짐 확인.
- `delta_monitor.py` 라이브 동작 확인(오늘): 100Hz 수신, STANDBY 때 ABS/INC=0, WORLD 갱신 정상.
- **하드웨어(라이브 `list_devices`로 항상 재확인):**
  - 트래커: `tracker_1=LHR-7B9A3BA9`→right, `tracker_2=LHR-F4A94AD1`→left
  - 동글: `D29531955E`↔`LHR-F4A94AD1`, `BECB785339`↔`LHR-7B9A3BA9` (1동글=1트래커)
  - 베이스: `LHB-EC40A549`, `LHB-9E35D5A7` (캘리브 정상)
  - README의 `LHR-9AB9C3A4`/`LHR-46A73216`는 **옛 트래커, 무효**.

---

## 5. 알려진 함정 / 운용 노하우

- **`tracker_name_map`은 반드시 `left`/`right`** 로. `teleop_delta`는 `/vive/left|right/pose`만
  구독 → friendly명을 `*_hand` 등으로 주거나 map 없이 띄우면 `/vive/tracker_1|2`로 발행돼
  **델타가 계속 0**(engage해도 "no valid tracker pose"). (수차례 빠진 함정)
- **viz_node는 시작 시점에만 장치 탐색** → 트래커 추가/페어링 후 노드 재시작.
- **동글 핫플러그하면 SteamVR가 못 잡음** → vrserver/vrmonitor 재시작 필요.
- **트래커 standby**(절전): OpenVR consumer 떠 있으면 깨움 유지. 완전 off는 전원버튼/흔들기로만.
- **2번째 트래커 페어링**: SteamVR 상태창 ≡ → Devices → Pair Controller (트래커 버튼만으론 안 붙음).
- ros2 CLI가 `!rclpy.ok()`로 깨지면 `ros2 daemon stop && ros2 daemon start`.
- 스크립트에서 ROS `setup.bash` 소싱 전 `set -u` 금지(`AMENT_TRACE_SETUP_FILES` unbound로 죽음).

---

## 6. 남은 작업 / 미해결

- [ ] **제어 PC 델타 적용 검증** — 송신측은 이미 anchored(A). 제어 PC가 받은 델타를 매 틱
      **누적(integrate)** 하면 "step 증분"처럼 드리프트함. 반드시 `T_target = T_ee0·dT`
      (engage 상승엣지에서 `T_ee0` 1회 캡처, 누적 금지)로 적용해야 함.
      → `delta_monitor.py`로 ABS vs INC 비교하며 확인. (상세: TELEOP_INTERFACE.md)
- [ ] **프레임 정렬 `R_align`** 캘리브 — 트래커 파지 방향 ↔ 로봇 EE 방향 정합.
- [ ] 빠른 동작/트래킹 로스트 대비 제어 PC측 velocity/accel clamp.

---

## 7. 관련 문서
- [`../README.md`](../README.md) — 실행/네트워크/토픽 운영 가이드
- [`TELEOP_INTERFACE.md`](TELEOP_INTERFACE.md) — 제어 PC 수신측 인터페이스(메시지 포맷·적용법)
