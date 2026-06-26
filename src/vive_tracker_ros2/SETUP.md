# Vive Tracker → ROS2 셋업 (Ubuntu 22.04 + ROS2 Humble)

목표: 트래커 켜고 USB 동글 꽂으면 `/vive/<name>/pose` 토픽에 EE 포즈가 떠야 함.

준비물 체크:
- HTC Vive Tracker (2.0 또는 3.0)
- USB 동글 (Tracker 패키지 동봉) 또는 HTC Wireless Receiver
- Lighthouse base station 1.0 ×2 또는 2.0 ×2 (1개로도 추적은 가능하지만 occlusion 심함)
- Lighthouse 전원 / 마운트 (벽 또는 삼각대)
- (선택) HMD 없이 돌리면 SteamVR 의 null-driver 트릭 필요 — 아래 step 5 참고

---

## 1. Steam 설치

```bash
sudo apt update
sudo apt install -y steam-installer libsdl2-dev libudev-dev
# 첫 실행 시 자동 업데이트가 일어남
steam
```

Steam GUI 가 뜨면 본인 계정으로 **로그인 (수동)**. 계정 없으면 새로 만들어야 함.

> Steam GUI 가 안 뜨면 `~/.steam/error.log` 확인. `libGL` 오류면 `sudo apt install libgl1-mesa-glx libgl1-mesa-dri` 추가.

## 2. SteamVR 설치

Steam GUI → Library → 좌상단 검색 "SteamVR" → Install. 약 1.5GB. 설치 완료 후 일단 닫음.

## 3. udev 규칙 (HTC HID 디바이스를 비루트로 접근)

```bash
sudo curl -L -o /etc/udev/rules.d/60-HTC-Vive-perms.rules \
    https://raw.githubusercontent.com/ValveSoftware/SteamVR-for-Linux/master/60-HTC-Vive-perms.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

(파일이 404 면 SteamVR 설치 시 `~/.steam/steam/steamapps/common/SteamVR/drivers/lighthouse/bin/linux64/` 에 동봉됨 — 거기서 복사)

## 4. pyopenvr (Python 바인딩)

```bash
pip install --user openvr
python3 -c "import openvr; print(openvr.__version__)"
```

triad_openvr 은 이 패키지(`vive_tracker_ros2`) 안에 vendoring 되어 있으므로 별도 설치 불필요.

## 5. HMD 없이 트래커만 쓰기 (null-HMD)

SteamVR 은 기본으로 HMD 가 꽂혀 있다고 가정. 헤드셋 없이 트래커만 쓰려면 null-driver 활성화:

`~/.steam/steam/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings` 편집:
```json
{
  "driver_null": {
    "enable": true,
    ...
  }
}
```

그리고 `~/.steam/steam/config/steamvr.vrsettings` (없으면 SteamVR 한번 실행 후 생김) 에 추가/병합:
```json
{
  "steamvr": {
    "requireHmd": false,
    "forcedDriver": "null",
    "activateMultipleDrivers": true
  }
}
```

## 6. Lighthouse + Tracker 페어링

1. Lighthouse 베이스 전원 ON, 서로 마주보게 또는 90° 이내로 설치. LED 가 안정된 흰색이 되면 OK.
2. USB 동글을 PC 에 꽂음.
3. Tracker 측면 버튼 길게 눌러서 페어링 모드 (LED 파란색 점멸).
4. SteamVR 실행 → 우측 상단 햄버거 메뉴 → Devices → Pair Controller → Vive Tracker 선택 → 진행.
5. 페어링 끝나면 SteamVR 상태창에서 트래커가 **녹색 hexagon** 으로 표시.

> Lighthouse 2 개 다 보일 때만 추적 안정. 한 개만 보이면 yaw drift 생김.

## 7. ROS2 패키지 빌드

```bash
source /opt/ros/humble/setup.bash
cd /home/js/franka_ros2_ws
colcon build --packages-select vive_tracker_ros2 --symlink-install
source install/setup.bash
```

## 8. 실행

랩 표준 env 와 함께:
```bash
source /opt/ros/humble/setup.bash
source /home/js/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0

# SteamVR 이 켜진 상태에서:
ros2 launch vive_tracker_ros2 vive_tracker.launch.py
```

토픽 확인:
```bash
ros2 topic list | grep vive
ros2 topic hz /vive/tracker_1/pose
ros2 topic echo /vive/tracker_1/pose --once
```

시리얼 → 친근 이름 매핑 (오른손/왼손 등):
```bash
ros2 run vive_tracker_ros2 list_devices   # 시리얼 출력
ros2 launch vive_tracker_ros2 vive_tracker.launch.py \
    tracker_name_map:='["LHR-AAAAAAAA:right_hand","LHR-BBBBBBBB:left_hand"]'
```

## 9. SteamVR 없이 ROS2 측만 검증

```bash
ros2 launch vive_tracker_ros2 vive_mock.launch.py
ros2 topic echo /vive/mock_tracker/pose --once
```

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `triad_openvr init failed` | SteamVR 안 켜짐, 또는 null-HMD 설정 안 됨 |
| 트래커는 보이는데 토픽이 안 옴 | DOMAIN/RMW 표준 (DOMAIN=9, fastrtps, LOCALHOST_ONLY=0) 적용 확인 |
| `/vive/*/valid` 가 false 만 떠 | Lighthouse 1개만 보임 또는 occlusion. base station 위치/각도 재조정 |
| yaw 가 천천히 돈다 | Lighthouse 1개만 잡힐 때 정상. 두 개 보이게 |
| SteamVR 이 GPU error | NVIDIA 비번 드라이버 권장. `nvidia-smi` 로 확인 |
| 토픽 hz 낮음 | `rate_hz` 파라미터, 그리고 SteamVR 자체 추적 hz 도 확인 (보통 250Hz까지 가능) |

## Franka 측 연결 (다음 단계 — 별도 노드)

이 패키지는 **EE 포즈 publisher 까지만**. Franka 제어로 넘기려면 다음 노드들이 추가로 필요:

1. **Vive→Franka 좌표 변환 노드** (handeye calibration 결과를 static TF 로 publish)
2. **시작 자세 offset 노드** — 사용자가 트래커를 처음 잡은 순간을 origin 으로 잡아서 현재 EE 위치에 매핑 (안 하면 손이 튀어서 reflex 걸림)
3. **libfranka cartesian pose 노드** (제어 PC 에서 동작): `franka::CartesianPose` 컨트롤러에 1kHz 로 EE 포즈를 전달. 보간/필터 + 워크스페이스 한계 클램프 필수
