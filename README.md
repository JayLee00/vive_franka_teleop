# vive_franka_teleop

Vive Tracker 2개 → Franka(FR3) 2팔 **실시간 델타 텔레옵**.
SteamVR/OpenVR로 트래커 포즈를 읽어, engage(클러치) 기준 6DoF 상대 델타를
ROS2(DDS, 전용 LAN)로 옆 제어 PC에 100Hz로 보냄.

```
Vive PC (192.168.0.1)                              제어 PC (192.168.0.100)
SteamVR ─ viz_node ─ /vive/{left,right}/pose ─ teleop_delta ─ /teleop/delta/{left,right} ──LAN──▶ Franka 2팔
                                                    ▲ 클러치 /teleop/engage/{left,right}
```

## 폴더 구조
```
~/Desktop/vive_franka_teleop/
├── src/                      ROS2 패키지 (franka_ros2_ws/src 에 심볼릭 연결됨)
│   ├── vive_3d_viz/            ← 현재 쓰는 핵심 (viz_node + teleop_delta, SteamVR 기반)
│   ├── vive_tracker_ros2/      ← OpenVR 트래커 노드 (대안)
│   └── vive_teleop_3d/         ← legacy (libsurvive 기반, vendor/libsurvive 필요)
├── vendor/libsurvive/        legacy 용 (231M)
├── scripts/                  실행 스크립트 (대부분 bash <스크립트> 한 줄로 실행)
│   ├── start_teleop_pipeline.sh   헤드리스 파이프라인 기동(viz_node+teleop_delta 데몬)
│   ├── run_viz.sh                 viz_node + RViz(GUI)
│   ├── run_clutch.sh              키보드 클러치 실행 (space=정지, 1=재개)
│   ├── run_tracker_read.sh        터미널 6DoF 실시간
│   ├── run_delta_monitor.sh       델타 ABS(anchor누적) vs INC(증분) 실시간 비교
│   ├── run_delta_viz.sh           델타 3D 경로 + 시계열 그래프 (matplotlib)
│   ├── run_viz_and_read.sh        RViz + 터미널 6DoF 동시
│   ├── teleop_clutch_key.py       (클러치 본체)
│   ├── tracker_read.py            (6DoF 리더 본체)
│   ├── delta_monitor.py           (델타 모니터 본체)
│   ├── delta_viz.py               (델타 시각화 본체: 3D 경로+그래프)
│   ├── fix_ros_net.sh             네트워크(enp6s0->ros2) 고정 (sudo)
│   └── legacy/                    옛 설치/점검 스크립트
├── config/
│   ├── viz_params.yaml            트래커 시리얼→arm 매핑
│   ├── teleop_params.yaml         arms, rate, auto_engage 등
│   ├── fastdds_lan_only.xml       이 PC: DDS를 LAN(enp6s0)만 쓰게
│   └── fastdds_lan_only_NEIGHBOR.xml  제어 PC용
└── docs/TELEOP_INTERFACE.md   제어 PC 수신측 인터페이스 안내

# colcon 빌드는 기존 워크스페이스에서: ~/franka_ros2_ws  (install/, build/ 거기 있음)
```

매핑: **tracker_1(LHR-7B9A3BA9)→right arm**, **tracker_2(LHR-F4A94AD1)→left arm**
시리얼 기반이라 SteamVR 인덱스가 바뀌어도 좌/우 고정. `viz_params.yaml` + `viz.launch.py` 기본값 + `run_viz*.sh` 모든 실행 경로에 동일하게 박혀 있음.

> `vendor/libsurvive`(231M, legacy 전용 서드파티)는 git 미포함(.gitignore). legacy 빌드가 필요하면 따로 클론하세요.

---

## 실행 방법 — 어디서 뭘 치는지

### 0) (이동/수정 후 1회) 빌드
```bash
cd ~/franka_ros2_ws
colcon build --packages-select vive_3d_viz vive_tracker_ros2 vive_teleop_3d --symlink-install
```

### 공통 환경 (새 터미널마다 맨 위에)
```bash
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
```

### 터미널 1 — SteamVR
SteamVR 실행, 트래커 2개 **초록불(tracking)** 확인. (이 PC는 별도 ROS 명령 없음)

### 터미널 2 — 파이프라인 기동 (둘 중 하나)
- **헤드리스(권장, 백그라운드 데몬)** — 어디서든:
  ```bash
  bash ~/Desktop/vive_franka_teleop/scripts/start_teleop_pipeline.sh
  ```
- **RViz로 보면서**:
  ```bash
  bash ~/Desktop/vive_franka_teleop/scripts/run_viz.sh
  ```
  (헤드리스 데몬은 자동 정리됨. 이 터미널 닫으면 viz_node도 죽음)

### 터미널 3 — 키보드 클러치 (전송 제어)
```bash
bash ~/Desktop/vive_franka_teleop/scripts/run_clutch.sh
```
- **`1`** = 재개(engage, 현재 자세로 anchor 재캡처) · **`space`** = 정지(hold) · **`q`** = 종료

### (선택) 트래커 좌표 + 델타 실시간 확인 — 터미널 6DoF
```bash
bash ~/Desktop/vive_franka_teleop/scripts/run_tracker_read.sh
```
팔별로 **2줄** 출력:
- **절대(WORLD)**: `/vive/<arm>/pose` → pos[m] + RPY[deg] + quaternion + axis-angle[deg]
- **델타(Δ)**: `/teleop/delta/<arm>` → engage 상태(`eng ✓/·`) + anchor 기준 상대 pos[m]·rot[deg] + |Δp|·|Δθ|
  (teleop_delta 노드가 떠 있어야 나옴. 클러치 미engage 시 델타는 0 = hold)

### (선택) 델타 모니터 — ABS vs INC 실시간 비교
```bash
bash ~/Desktop/vive_franka_teleop/scripts/run_delta_monitor.sh        # 양팔 (오른팔만: 뒤에 right)
```
→ `/teleop/delta/<arm>` 구독해 양팔 동시 표시:
   **ABS**=engage anchor 기준 누적(=송신값) · **INC**=직전 메시지 대비 증분(+속도) · **WORLD**=vive_world 절대 포즈.
   engage(클러치 `1`) 후 트래커를 움직이면 ABS는 누적·INC는 매 스텝 증분으로 뜸. 손을 멈추면 ABS 유지·INC→0.

### (선택) 델타 시각화 — 3D 경로 + 시계열 그래프
```bash
bash ~/Desktop/vive_franka_teleop/scripts/run_delta_viz.sh           # 양팔 중 right (left: 뒤에 left)
```
→ matplotlib 창: **왼쪽 3D** = engage 원점 기준 트래커가 그린 위치 경로(궤적), **오른쪽** = dx/dy/dz·rotvec 시계열.
   engage 후 움직이면 경로가 그려지고 재engage 시 리셋. (GUI 없이 수신만 확인: `... right --selftest`)

### (선택) RViz(3D 트래커) + 터미널 6DoF **동시**
```bash
bash ~/Desktop/vive_franka_teleop/scripts/run_viz_and_read.sh
```
→ RViz 창에 트래커 3D + 같은 터미널에 6DoF 실시간. Ctrl+C 시 둘 다 종료.
   (기존 viz_node 데몬은 자동 정리 — OpenVR 1개만 가능)

> ⚠️ 모든 긴 명령은 래퍼 스크립트(`run_*.sh`)로 감쌌습니다. 터미널에서 긴 명령 붙여넣을 때
> 줄바꿈으로 쪼개지는 문제 방지용 — `bash <스크립트>` 한 줄만 치면 됩니다.

---

## 네트워크 (전용 LAN으로 보내기)
- 이 PC enp6s0 = `192.168.0.1/24`, 제어 PC = `192.168.0.100/24`, 직결 GbE.
- 이 PC 프로파일 고정(1회, sudo): `sudo bash ~/Desktop/vive_franka_teleop/scripts/fix_ros_net.sh`
- WiFi 우회 차단: 위 공통 환경의 `FASTRTPS_DEFAULT_PROFILES_FILE` 가 DDS를 enp6s0만 쓰게 강제.
- **제어 PC**: `config/fastdds_lan_only_NEIGHBOR.xml` 복사 → `export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_lan_only.xml` → ROS 노드 재시작.
- 확인: `ping 192.168.0.100` (이 PC) / `ros2 topic list`에 양쪽 토픽.

## 토픽 / 메시지
- 발행: `/teleop/delta/{left,right}` (std_msgs/String JSON, 100Hz)
  `{arm,stamp,engaged,valid, pos[m], rot[rad rotvec], abs_pos[m], abs_quat}`
- 절대 포즈: `/vive/{left,right}/pose` (PoseStamped, vive_world, +Y up)
- 클러치: `/teleop/engage/{left,right}` (std_msgs/Bool)
- 제어 PC 수신측 상세는 `docs/TELEOP_INTERFACE.md` 참고.

## 단위
- 위치 `pos`/`abs_pos`: 미터 [m]
- 회전 `rot`: 라디안 [rad], rotvec(axis-angle, |rot|=각·방향=축)
- `abs_quat`: 단위 쿼터니언 (x,y,z,w)
