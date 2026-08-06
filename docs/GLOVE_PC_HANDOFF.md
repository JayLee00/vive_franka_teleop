# 글러브 PC 연동 안내 — 글러브로 KISTAR 핸드 텔레옵

> 대상: 글러브를 물린 PC 담당자
> 작성: 2026-08-03 · 제어 PC 저장소: `Dual_Arm_Hand_Ctrl`
> 참고 구현: 같은 폴더의 [`glove_pc_publisher.py`](glove_pc_publisher.py) (단독 파일, 복사해서 쓰세요)

## 요약

**제어 PC는 수정할 게 없습니다.** 이미 떠 있는 `hand_target_receiver` 노드가 범용 입구
`/hand/<side>/q_target`을 받아 SHM에 쓰고 EtherCAT이 손으로 내보냅니다.
글러브 PC는 **시리얼을 읽어 그 토픽으로 발행**만 하면 됩니다.

역할 분담 (Vive PC와 동일한 원칙):

| | 담당 |
|---|---|
| 시리얼 파싱 · 관절 매핑 · 스케일 · 필터 · 램프 | **글러브 PC** |
| q_target 수신 → SHM → EtherCAT → 손, 상태 발행 | 제어 PC (무수정) |

```
[글러브 PC]                                    [제어 PC]
 /dev/ttyUSB0                              shm (EtherCAT) + nd
   │ 16채널 CSV 100Hz                            ▲
   ▼                                             │ SHM Hand_q_tar
 glove_pc_publisher.py ──/hand/right/q_target──► hand_target_receiver
                       ──/glove/right/q_raw───► (SHM Glove_q_pos, 로깅용)
                       ◄─/hand/right/joint_states── shm_state_publisher
                            (램프 기준 + 연결 확인)
```

## 1. 보내주실 것 (계약)

| 항목 | 값 |
|---|---|
| 토픽 | `/hand/right/q_target` (오른손), `/hand/left/q_target` (왼손) |
| 타입 | `std_msgs/Float32MultiArray`, `data[16]` |
| 단위 | **count** ⚠️ rad 아님. 1 count = π/8192 rad. rad→count = ×(8192/π)≈2607.6 |
| 주기 | 15~100Hz 연속 스트리밍 |
| QoS | best_effort, keep_last 1 |
| 발행 중단 시 | 제어 PC가 **마지막 타겟 유지(hold)** — 손이 힘 풀리지 않습니다 |

같이 보내주시면 좋은 것 (필수 아님):

| 토픽 | 타입 | 용도 |
|---|---|---|
| `/glove/<side>/q_raw` | Float32MultiArray[16] | 글러브 엔코더 **raw**. 제어 PC가 SHM `Glove_q_pos`에 기록 → HDF5 로거가 로봇 데이터와 같은 타임라인에 저장. 명령이 아니라 telemetry라 제어권 게이트 없음 |

서보/모드 (텔레옵 시작할 때 1회):

| 토픽 | 타입 | 값 |
|---|---|---|
| `/hand/<side>/cmd_mode` | `std_msgs/Int32` | `1` = position |
| `/hand/<side>/cmd_servo` | `std_msgs/Bool` | `true` = 서보 on |

### ⚠️ 핸드 안전 순서 (반드시 지킬 것)

```
1) /hand/<side>/joint_states 로 현재 손 자세를 읽는다
2) 그 값을 q_target 으로 1회 발행한다
3) cmd_servo true
4) 현재 자세 → 글러브 타겟으로 1~2초 램프
5) 이후 연속 스트리밍
```

**서보를 먼저 켜면** SHM `Hand_q_tar` 초기값 0으로 손가락이 튑니다.
참고 구현은 이 순서를 그대로 구현해 뒀습니다 (`RAMP_SEC`).

## 2. 제어 PC가 제공하는 상태 토픽 (구독용)

| 토픽 | 타입 | 주기 | 용도 |
|---|---|---|---|
| `/hand/<side>/joint_states` | JointState | 200Hz | 현재 손 16관절 [count] + effort=모터전류. **램프 기준·연결 확인용** |
| `/hand/<side>/target_joint_states` | JointState | 200Hz | 제어 PC가 실제로 쓰고 있는 q_tar — 내가 보낸 게 반영됐는지 확인 |
| `/hand/<side>/mode` | Int32MultiArray | 200Hz | `data[2] = {mode, servo_on}` |
| `/paxini/<side>/ft` | Float32MultiArray[12] | 90Hz | 손가락 촉각 합력 [N] (paxini_writer 켜져 있을 때) |

⚠️ 상태 토픽은 **best_effort**로 발행됩니다 → 구독도 best_effort여야 매칭됩니다.

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy
SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
```

## 3. 네트워크 설정 (양쪽 PC 동일해야 함)

| 항목 | 값 |
|---|---|
| `ROS_DOMAIN_ID` | `9` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` ← **1이면 다른 PC와 통신 안 됩니다** |
| 네트워크 | 같은 서브넷, UDP 7400번대 개방 |

```bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```

참고 구현은 `FORCE_ENV = True`로 이 3개를 스크립트가 직접 설정합니다.

## 4. 실행

의존성: `rclpy` + `pyserial` (제어 PC 저장소 코드에 의존하지 않는 단독 파일)

```bash
pip3 install pyserial

# 0) 배선/네트워크 점검 — 타겟 발행 안 함
python3 glove_pc_publisher.py --check

# 1) 손 안 움직이고 매핑값만 확인
python3 glove_pc_publisher.py --dry-run

# 2) 실전
python3 glove_pc_publisher.py
python3 glove_pc_publisher.py --side left --port /dev/ttyUSB1
```

`--check` 출력 예:

```
=== 1. 글러브 시리얼 ===
  ✓ 198프레임 / 2초 (~99Hz)
=== 2. 제어 PC 연결 (/hand/right/joint_states) ===
  ✓ 412개 수신 — 제어 PC 연결 정상
```

## 5. 글러브 데이터 스펙 (2026-08-03 실측)

| 항목 | 값 |
|---|---|
| 포트 | `/dev/ttyUSB0`, **115200** baud |
| 형식 | `j0,j1,...,j15\n` — **정수 16개** 콤마 구분 |
| 주기 | ~100Hz |
| 촉각 | **없음** (현 펌웨어는 엔코더만) |
| 노이즈 | 채널당 std ≈ 20 count |
| 캘리브 | **글러브 펌웨어에서 이미 완료** → 호스트에서 1:1 그대로 씀 |

> 구 `DexFIT_Final/dexfit-glove-teleoperation/TUMI_Glove_Publisher.py`는
> `16 joint + 21 tactile = 37필드`를 기대하므로 **현 펌웨어와 호환되지 않습니다**
> (모든 줄을 파싱 실패로 버립니다). 참고하지 마세요.

### 관절 순서 (글러브 채널 = 핸드 관절, 동일 순서)

| idx | 관절 | idx | 관절 |
|---|---|---|---|
| 0 | thumb_cmc_opposition | 8 | middle_mcp_abduction |
| 1 | thumb_cmc_abduction | 9 | middle_mcp_flexion |
| 2 | thumb_mcp | 10 | middle_pip |
| 3 | thumb_ip | 11 | middle_dip |
| 4 | index_mcp_abduction | 12 | ring_mcp_abduction |
| 5 | index_mcp_flexion | 13 | ring_mcp_flexion |
| 6 | index_pip | 14 | ring_pip |
| 7 | index_dip | 15 | ring_dip |

### 핸드 하드리밋 [count] — 보내기 전 클램프 필수

| 관절 | 리밋 |
|---|---|
| 1 (엄지 외전) | `(-4096, 4096)` |
| 4, 8, 12 (손가락 외전) | `(-1000, 1000)` |
| 3, 7, 11, 15 (엄지 IP·손가락 DIP) | `(-2048, 4096)` |
| 그 외 | `(0, 4096)` |

## 6. 튜닝 — `JOINTS` 표

참고 구현 상단의 표에서 관절별로 조절합니다:

```python
#          hand_idx, 이름,                  glove_ch, scale, offset, on
    (  5, "index_mcp_flexion",      5,      1.0,    0.0,  True ),
#                                           ↑scale  ↑offset
```

`target = clamp(glove[glove_ch] * scale + offset)`

- `scale` — `1.0`=그대로, `1.1`=10% 더 꽉 쥠, `-1.0`=방향 반전, `0.0`=안 움직임
- `offset` — 상수 보정 [count]
- `on` — `False`면 `offset` 값으로 고정 (그 관절만 끔)
- `glove_ch` — 손가락 묶기 (예: 약지 13~15의 채널을 9~11로 → 중지와 동기화)

> ⚠️ **`scale`은 0을 기준으로 곱합니다.** 쉴 때 값이 큰 관절(예: 엄지 j0 ≈ 3665)에
> `1.1`을 걸면 손을 가만히 둬도 타겟이 +366 count 밀립니다.
> 쉴 때 0 근처인 굽힘 관절(j5, j9, j13, j10, j14 …)부터 올리세요.
> 전체에 걸고 싶으면 중심 기준 스케일(`center + (v-center)*scale`)로 식을 바꿔야 합니다.

기타 안전 파라미터:

| 변수 | 기본 | 의미 |
|---|---|---|
| `RAMP_SEC` | 2.0 | 시작/재연결 시 현재 자세→타겟 램프 [s] |
| `MAX_STEP` | 100 | 1주기당 관절 최대 변화 [count] (100Hz면 10000 count/s) |
| `EMA_ALPHA` | 0.3 | 저역통과. `1.0`=필터 없음. 손가락 떨리면 낮추기 |
| `STALE_SEC` | 0.5 | 글러브 수신 끊기면 타겟 홀드 |
| `RECONNECT_SEC` | 1.0 | USB 재열거 시 재연결 주기 (재연결 후 램프 다시 태움) |

## 7. 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| `제어 PC 미연결 — joint_states 안 옴` | 제어 PC에서 `shm` + `nd` 떠 있는지 / 양쪽 `ROS_DOMAIN_ID`·`RMW` 동일한지 / `ROS_LOCALHOST_ONLY=0` / UDP 7400번대 방화벽 |
| `글러브 수신 없음` | `ls /dev/ttyUSB*`, 보레이트 115200, 사용자가 `dialout` 그룹인지 (`sudo usermod -aG dialout $USER` 후 재로그인) |
| `글러브 시리얼 끊김` 반복 | **USB 케이블/커넥터 접촉 불량.** 2~3초마다 끊기면 램프가 계속 재시작돼 추종이 안 됩니다 — 소프트웨어로 못 덮으니 하드웨어부터 잡으세요. `sudo dmesg \| grep -i ttyUSB` 확인 |
| 손이 안 움직임 | `--dry-run` 아닌지 / `ros2 topic echo /hand/right/mode` 가 `[1, 1]`인지 / 제어 PC가 `require_control:=true`면 8절 참고 |
| 손가락이 떨림 | `EMA_ALPHA` 낮추기 (0.3 → 0.15) |
| 손이 굼뜸 | `MAX_STEP` 올리기 (100 → 200) |
| 보낸 타겟이 반영 안 됨 | `ros2 topic echo /hand/right/target_joint_states --field position` 로 제어 PC가 실제 쓰고 있는 값 확인 |

## 8. 제어권 (`require_control:=true`로 운영할 때만)

제어 PC를 `require_control:=true`로 띄우면 Sequence Arbiter 제어권이 있어야 q_target이 반영됩니다.
(기본값은 `false`라 보통은 신경 쓸 필요 없습니다.)

```bash
ros2 service call /sequence/request_control dual_arm_msgs/srv/RequestControl "{client_id: 1}"
# 텔레옵 중 3초 주기로 하트비트 발행 (끊기면 자동 회수)
ros2 topic pub /sequence/heartbeat std_msgs/msg/Int32 "{data: 1}" -r 0.5
```

`/glove/<side>/q_raw`(telemetry)는 제어권과 무관하게 항상 기록됩니다.

## 9. 같은 손에 두 곳에서 보내지 마세요

`/hand/<side>/q_target`은 입구가 하나입니다. 글러브 PC와 GPU PC가 동시에 스트리밍하면
서로 덮어써서 손이 두 목표 사이에서 진동합니다. 한 시점에 한 곳만 보내세요
(운영 시엔 8절 제어권으로 강제).
