# Vive 텔레옵 PC 연동 안내 (제어 PC 시스템 변경 사항)

> 대상: Vive tracker PC 담당자
> 작성: 2026-07-02 · 제어 PC 저장소: `Dual_Arm_Hand_Ctrl` (구 `Dual_Arm_Hand_Imp_Ctrl_V1.0` 후속)

## 요약

제어 PC가 새 시스템으로 바뀌면서 **텔레옵 입력 인터페이스가 "델타 JSON"에서 "절대 EE 타겟 포즈"로 표준화**되었습니다.
델타→절대 포즈 변환은 **Vive PC 쪽에서** 수행해 주세요. 변환 로직의 참고 구현(파이썬 노드)을 함께 제공합니다 —
그대로 실행하거나 기존 코드에 통합하시면 됩니다.

역할 분담: **제어 PC = 범용 입구만** (절대 EE 수신 → IK → 임피던스, 어떤 장치든 동일) / **장치 특화 로직(트래커 델타·engage·스케일) = Vive PC**.

## 1. 제어 PC 쪽 무엇이 바뀌었나 (간단 요약)

| | 구 시스템 (V1.0) | 신 시스템 (Dual_Arm_Hand_Ctrl) |
|---|---|---|
| 텔레옵 입력 | `/teleop/delta/*` JSON → 제어 PC의 `vive_teleop`이 누적+IK | **절대 EE 타겟** `/franka_r\|l/ee_target_world` 수신 → 공용 IK(스텝 클램프·특이점 감쇠·조인트 리밋 내장) |
| 임피던스 게인 | 고강성 (600 계열) | DROID 저강성 (손목 떨림 방지, 컴플라이언트) |
| 상태 토픽 | `/shm/arm/*` | `/franka/ee_pose_r\|l`, `/franka/joint_states` 등 (3절) |

⚠️ 구 V1.0의 제어측 `vive_teleop` 노드는 새 시스템과 호환되지 않습니다. **Vive PC에서는 델타 발행 노드 + 아래 변환만** 실행하세요.

## 2. 보내주실 것 (신규 계약)

| 항목 | 값 |
|---|---|
| 토픽 | `/franka_r/ee_target_world` (오른팔), `/franka_l/ee_target_world` (왼팔) |
| 타입 | `geometry_msgs/PoseStamped` — position [m] + orientation quaternion |
| 내용 | **절대 EE 목표 포즈** (델타 아님) |
| 기준 프레임 | 현재는 **franka base 기준** (world→base 캘리브레이션 전) |
| 주기 | 15~100Hz 연속 스트리밍 (engage 중일 때만) |
| engage off / 트래킹 invalid | **발행 중단** — 제어 PC가 마지막 타겟에 수렴 후 자동 유지(hold) |

### 변환 방법 (권장 로직 — 참고 구현 제공)

1. 제어 PC가 발행하는 `/franka/ee_pose_r|l`(현재 실제 EE 포즈, 200Hz)를 구독
2. **engage(트리거) 상승엣지** 순간의 EE 포즈를 기준 `T_ee0`로 캡처
3. 매 프레임: `T_target = T_ee0 · [R_align·ΔR·R_alignᵀ | scale·R_align·Δp]`
   - `scale` 기본 0.3, `R_align` = 트래커 쥔 방향 ↔ 로봇 프레임 정렬 회전 (1회 결정)
4. `T_target`을 PoseStamped로 발행

### 참고 구현 (그대로 실행 가능)

제어 PC 저장소의 `ros2/trajectory_receiver/tools/vive_delta_adapter.py` 를 Vive PC로 복사해서 실행하면
기존 `/teleop/delta/right|left` JSON을 받아 위 변환을 수행합니다 (의존성: rclpy + numpy):

```bash
python3 vive_delta_adapter.py --ros-args -p arm:=r -p scale:=0.3
# R_align 조정: -p r_align:="[r11,r12,r13, r21,r22,r23, r31,r32,r33]"
```

기존 델타 발행 노드(`vive_teleop_delta`)는 그대로 두고 이 노드만 추가로 띄우면 됩니다.
(자체 코드에 통합하셔도 됩니다 — 핵심은 2절의 계약만 지키면 됩니다)

## 3. 제어 PC가 제공하는 상태 토픽 (구독용)

| 토픽 | 타입 | 주기 | 용도 |
|---|---|---|---|
| `/franka/ee_pose_r`, `_l` | PoseStamped | 200Hz | **T_ee0 캡처용** (FK 직접 풀 필요 없음) |
| `/franka/joint_states` | JointState | 200Hz | 관절 상태 |
| `/paxini/ft_r`, `_l` | Float32MultiArray [4×3] | 90Hz | 손가락 촉각 합력 [N] |

(구 `/shm/arm/*` 토픽은 더 이상 없습니다)

## 4. 네트워크 설정 (기존과 동일)

| 항목 | 값 |
|---|---|
| `ROS_DOMAIN_ID` | `9` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| 네트워크 | 같은 서브넷 (직결 LAN 권장), UDP 7400번대 개방 |

## 5. 연동 확인 순서

1. 양쪽 PC 4절 환경변수 확인
2. Vive PC: 델타 노드 + 변환 노드 실행
3. 제어 PC에서 `ros2 topic info /franka_r/ee_target_world` → **Publisher count ≥ 1** 확인
4. `ros2 topic echo /franka_r/ee_target_world` → engage 상태에서 트래커 움직이면 포즈 변화 확인
5. 제어 PC가 IK 수신 노드를 켜면 팔이 따라옴 — **첫 가동은 아주 천천히** 움직여 방향 확인 (뒤틀리면 `R_align` 조정)

문제 시 제어 PC 담당자에게: 3·4번 결과와 engage/valid 상태를 알려주세요.
