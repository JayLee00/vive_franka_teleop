# Vive Tracker delta 송신 에이전트용 브리핑

**대상:** `vive_teleop_delta` 등 `/teleop/delta/*` JSON을 publish하는 Vive PC 쪽 개발/에이전트  
**제어 PC 프로젝트:** `Dual_Arm_Hand_Imp_Ctrl_Tele_V1.0` (`vive_teleop` 패키지)  
**기본 송신 스펙:** [`TELEOP_INTERFACE.md`](TELEOP_INTERFACE.md)  
**제어 PC 상세 개발 기록:** [`VIVE_TELEOP_DEV.md`](VIVE_TELEOP_DEV.md)

---

## 1. 한 줄 요약 (송신 쪽이 알아야 할 것)

**JSON 토픽·포맷·누적 delta 의미는 바뀌지 않았습니다.**  
제어 PC는 내부 IK만 **DLS → GeoFIK weighted IK**로 바꿨고, **delta latch**로 동일 delta 반복 시 target 재계산을 막습니다.

송신 에이전트는 **지금과 동일하게** engage 기준 **누적** `pos`/`rot`을 ~100Hz로내면 됩니다.

---

## 2. 송신 프로토콜 (변경 없음 — 반드시 유지)

| 항목 | 값 |
|------|-----|
| 토픽 R | `/teleop/delta/right` |
| 토픽 L | `/teleop/delta/left` |
| 타입 | `std_msgs/String` (JSON) |
| QoS | RELIABLE, ~100 Hz |
| `ROS_DOMAIN_ID` | **9** (Vive PC · 제어 PC 동일) |

### JSON (필수 필드)

```json
{
  "arm": "right",
  "stamp": 1781088016.0,
  "engaged": true,
  "valid": true,
  "pos": [dx, dy, dz],
  "rot": [rx, ry, rz],
  "abs_pos": [x, y, z],
  "abs_quat": [qx, qy, qz, qw]
}
```

| 필드 | 송신 의미 | 제어 PC 사용 |
|------|-----------|--------------|
| `pos` | engage 순간 트래커 기준 **누적** 위치 [m] | **사용** (× scale) |
| `rot` | engage 순간 트래커 기준 **누적** rotvec [rad] | **사용** (× scale) |
| `engaged` | 클러치 on/off | **사용** |
| `valid` | 트래커 추적 유효 | **사용** |
| `abs_pos`, `abs_quat` | 절대 pose (참고) | **미사용** |

### 송신 수식 (제어 PC와 정합)

```text
engage 상승엣지:  pos=0, rot=0 리셋
이후 매 프레임:    pos/rot = (현재 트래커 pose) − (engage 시 트래커 pose)   ← 누적값
disengage:        pos=0, rot=0
```

**프레임당 증분이 아님.** 트래커를 멈추면 **같은 pos/rot이 반복**되는 것이 정상입니다.

---

## 3. 제어 PC에서 바뀐 것 (송신 변경 불필요)

### ver1 → ver2 (최근)

| 구분 | 이전 | 현재 |
|------|------|------|
| IK | Jacobian DLS (libfranka Model) | **GeoFIK analytic + q7 Brent weighted IK** |
| 특이점 | manip 미고려 → target 진동 가능 | manip/neutral/current 점수로 해 선택 |
| delta 동일 시 | (버그) 500Hz IK 재적분 → drift | **delta latch** — IK 재실행 안 함 |
| `abs_pos` | 미사용 | 미사용 (동일) |
| 로봇 제어 | `franka_arm.cpp` 관절 **임피던스** | **동일** |
| 입력 모델 | engage 기준 **누적 delta** EE 목표 | **동일** |

제어 PC 수식:

```text
engage:     T_ee0 ← 그 순간 Franka O_T_EE (SHM)
매 메시지:  T_desired = T_ee0 × T( pos×TELEOP_SCALE, rot×TELEOP_SCALE )
delta 변경 시만: GeoFIK IK → Arm_q_tar
delta 동일:     latched Arm_q_tar 유지
```

현재 `TELEOP_SCALE = 0.3` (제어 PC `teleop_settings.py`).

### 송신 쪽이 **절대 바꾸면 안 되는 것**

- `pos`/`rot`을 **프레임당 증분**으로 바꾸기 (제어 PC `DELTA_MODE=cumulative`와 불일치 → 발산)
- engage 없이 nonzero delta 보내기
- `engaged:false`인데 delta가 0이 아닌 값

### 송신 쪽 **권장** (변경 없어도 됨)

- LPF는 **송신측**(`vive_teleop_delta`)에서 적용 권장
- `valid:false` / `engaged:false` 시 `pos`/`rot` **정확히 0**

---

## 4. 제어 PC 실행 순서 (송신 전제 조건)

제어 PC에서 아래가 **모두** 떠 있어야 delta가 로봇으로 연결됩니다.

```text
1) sudo ./build/test/Dual_Arm_Hand_Imp_Ctrl_V1_0     # SHM + Franka + Hand
2) Vive PC: vive_teleop_delta                         # ← 송신
3) ros2 launch vive_teleop dual_vive_teleop.launch.py # 구독 + IK
```

`trajectory_receiver`와 `vive_teleop` **동시 실행 금지** (둘 다 `Arm_q_tar` 사용).

---

## 5. 연동 검증 (송신 에이전트 체크리스트)

### Vive PC에서

```bash
export ROS_DOMAIN_ID=9
ros2 topic hz /teleop/delta/right
ros2 topic echo /teleop/delta/right --once
```

- ~100 Hz
- `engaged:true` + 움직임 시 `pos`/`rot` 변화
- 정지 시 **동일 값 반복**

### 제어 PC에서

```bash
export ROS_DOMAIN_ID=9
ros2 topic info /teleop/delta/right -v    # Subscription ≥ 1 (vive_teleop_node_r)
ros2 node list | grep vive_teleop
ros2 param get /vive_teleop_node_r delta_mode   # cumulative
```

### 양방향

- 제어 PC `/joint_states_r`, `/joint_states_l` 이 Vive PC에서 보이면 네트워크 OK

---

## 6. ★ 알려진 이슈: 제어 PC 코드 변경 후 delta가 “끊김”

### 증상 (보고됨)

- 제어 PC에서 `vive_teleop` **빌드/코드 변경 후**
- Vive 쪽에서 delta가 **로봇에 반영되지 않거나**, 토픽이 **안 보이는 것처럼** 느껴짐

### 원인 후보 (우선순위)

| # | 원인 | 설명 |
|---|------|------|
| 1 | **제어 PC 구독 노드 미기동** | `colcon build`만 하고 `dual_vive_teleop.launch.py` 재시작 안 함 → publish는 되지만 **구독자 0** |
| 2 | **`source install/setup.bash` 누락** | 옛 바이너리 실행 또는 launch 실패 |
| 3 | **C++ 메인(SHM) 미실행** | `vive_teleop_node`는 떠도 SHM 없으면 IK 결과 미반영 |
| 4 | **`ROS_DOMAIN_ID` 불일치** | 한쪽만 9가 아니면 토픽 discovery 실패 |
| 5 | **FastDDS LAN 설정** | WiFi 우회 시 제어 PC에서 토픽 안 보임 (`fastdds_lan_only.xml`) |
| 6 | **Vive PC 프로세스 종료/재시작 필요** | 제어 PC만 재시작했을 때 discovery 꼬임 → **양쪽 ROS 노드 재시작** |
| 7 | **송신 노드가 subscriber 수에 반응** | (구현에 따라) 구독자 없으면 publish 중단하는 로직이 있으면 #1과 동일 증상 |

**중요:** 제어 PC IK 변경은 **토픽 이름·JSON 스키마를 바꾸지 않음**.  
“끊김”은 대부분 **ROS 프로세스/환경 재시작** 문제이지, 프로토콜 불일치가 아닐 가능성이 큼.

### 복구 절차 (권장 순서)

**제어 PC:**

```bash
cd ~/Dual_Arm_Hand_Imp_Ctrl_V1.0/ros2
colcon build --packages-select vive_teleop
source install/setup.bash
export ROS_DOMAIN_ID=9

# 1) C++ 메인 실행 중인지 확인
# 2) 텔레옵 launch 재시작
ros2 launch vive_teleop dual_vive_teleop.launch.py
```

**Vive PC:**

```bash
export ROS_DOMAIN_ID=9
# vive_teleop_delta 재시작
ros2 topic hz /teleop/delta/right
```

**양쪽 확인:**

```bash
# 제어 PC
ros2 topic info /teleop/delta/right -v
# Publisher: vive_teleop_delta (Vive PC)
# Subscription: vive_teleop_node_r (제어 PC)  ← 둘 다 있어야 함
```

여전히 안 되면 **Vive PC · 제어 PC ROS 노드 전부 재시작** (FastDDS discovery 초기화).

### 이슈 트래킹용 태그

```text
[ISSUE] post-rebuild delta disconnect
- repro: colcon build vive_teleop 후 launch 미재시작 / DOMAIN_ID / SHM / FastDDS
- expected: Publisher+Subscriber 모두 /teleop/delta/* 에 존재
- sender action: vive_teleop_delta 재시작, hz 확인
- receiver action: dual_vive_teleop.launch.py + Dual_Arm_Hand main 재시작
```

---

## 7. 송신 에이전트에게 묻지 않아도 되는 것

- GeoFIK / q7 Brent / weighted score — **전부 제어 PC 내부**
- `Arm_q_tar` / SHM / 임피던스 — **제어 PC `franka_arm.cpp`**
- `R_align` (트래커↔EE 축 정렬) — **아직 미구현** (송신 포맷과 별개)

---

## 8. 송신 변경이 필요해지는 경우 (미래)

| 변경 | 송신 쪽 작업 |
|------|--------------|
| `R_align` 추가 | 트래커 로컬 delta를 EE 프레임으로 회전 후 `pos`/`rot` 전송 (스펙 합의 필요) |
| 프레임당 증분 모드 | 제어 PC `DELTA_MODE=incremental` + 송신도 **증분**으로 변경 (현재 **비권장**) |
| 토픽명 변경 | `teleop_settings.py` `R_DELTA_TOPIC` / `L_DELTA_TOPIC` 와 동시 변경 |

현재 버전에서는 **송신 코드 변경 없음**.

---

## 9. 문의 시 제어 PC에 전달할 정보

1. `ros2 topic hz /teleop/delta/right` (Vive PC)
2. `ros2 topic info /teleop/delta/right -v` (양쪽)
3. engage 시 JSON 샘플 1개
4. 제어 PC `ros2 launch vive_teleop dual_vive_teleop.launch.py` 로그 앞 10줄
5. 끊김 시각 전후 **어떤 프로세스를 재시작했는지**

---

## 10. 관련 문서

| 문서 | 용도 |
|------|------|
| [`TELEOP_INTERFACE.md`](TELEOP_INTERFACE.md) | 송신 JSON·네트워크 공식 스펙 |
| [`VIVE_TELEOP_DEV.md`](VIVE_TELEOP_DEV.md) | 제어 PC ver1/ver2 개발·IK·이슈 기록 |
| `ros2/vive_teleop/teleop_settings.py` | scale, delta_mode, IK weight (제어 PC만) |
