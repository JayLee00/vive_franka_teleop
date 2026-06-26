# vive_teleop_3d

HTC Vive Tracker pose publisher via **libsurvive** — no SteamVR required.

병행 운영되는 자매 패키지:
- `vive_tracker_ros2` (OpenVR + SteamVR 기반, `/vive/<name>/pose` 발행)
- `vive_teleop_3d`   (libsurvive 기반, `/survive/<name>/pose` 발행)  ← **본 패키지**

두 패키지를 **동시에 켤 수 없음** (둘 다 USB 동글을 잡으려 함). 둘 중 하나만 켜고 작업.

## 구성

```
vive_teleop_3d/
├── package.xml / setup.py
├── vive_teleop_3d/
│   ├── survive_tracker_node.py     # 메인 노드 (libsurvive SimpleContext)
│   └── survive_list.py             # 디바이스 목록 한 번 출력
├── launch/survive_tracker.launch.py
└── scripts/run_survive.sh          # SteamVR 종료 + env 세팅 + ros2 launch 한방
```

libsurvive 소스: `../vendor/libsurvive/` (clone + make 완료).
udev: `/etc/udev/rules.d/81-vive.rules` 적용됨.

## 토픽

- `/survive/<name>/pose`  — `geometry_msgs/PoseStamped`, frame_id=`survive_world`
- `/survive/<name>/valid` — `std_msgs/Bool`
- TF: `survive_world` → `survive_<name>`

`<name>` 은 libsurvive 가 출력하는 객체 이름 (예: `T20`, 시리얼 등). 친근 이름으로 바꾸려면 `tracker_name_map` 파라미터.

## 실행

```bash
# 가장 간단 — SteamVR 종료 + env 세팅 + 실행 한 스크립트로:
bash /mnt/grasp_data/vive_franka_teleop/vive_teleop_3d/scripts/run_survive.sh

# 친근 이름 매핑하고 싶으면:
bash run_survive.sh --ros-args -p tracker_name_map:='LHR-46A73216:right_hand,LHR-9AB9C3A4:left_hand'

# 디바이스 빠른 확인:
SURV=/home/js/Desktop/vive_franka_teleop/vendor/libsurvive
PYTHONPATH=$SURV/bindings/python:$PYTHONPATH \
LD_LIBRARY_PATH=$SURV/bin:$LD_LIBRARY_PATH \
ros2 run vive_teleop_3d survive_list
```

## SteamVR 와의 전환

SteamVR (vive_tracker_ros2) → libsurvive (vive_teleop_3d):
```bash
# vive_tracker_ros2 노드 / SteamVR 종료 후
bash run_survive.sh
```

libsurvive → SteamVR:
```bash
# libsurvive 노드 종료 (Ctrl+C)
# Steam → Library → SteamVR → Play
# 그 다음 ros2 launch vive_tracker_ros2 vive_tracker.launch.py
```

## libsurvive 의 좌표계

libsurvive 의 world frame 은 **첫 번째 Lighthouse 가 정의**. SteamVR/OpenVR 와 origin/축이 다르므로, 두 패키지의 pose 를 동일시하면 안 됨. Franka 정합은 어차피 별도 handeye 라 영향 없음.

## 쿼터니언 순서

libsurvive: `Rot = (w, x, y, z)` ← w 가 먼저  
ROS2: `(x, y, z, w)` ← w 가 마지막  
노드 안에서 자동 reorder.

## 주의

- 처음 켰을 때 트래커가 Lighthouse 시야 안에 있어도 pose 가 안정화되는데 5~10초 걸릴 수 있음 (libsurvive 의 calibration 학습 단계).
- pose 가 `(0,0,0) + identity` 면 아직 localization 미완. valid=False 로 publish.
- 노드는 매 update event 마다 publish 하므로 rate 는 Vive 자체의 raw rate (보통 ~120-250Hz).
