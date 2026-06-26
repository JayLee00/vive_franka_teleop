# vive_tracker_ros2

HTC Vive Tracker → ROS2 (Humble) pose publisher.

- `tracker_node` — SteamVR 에 연결된 모든 트래커의 포즈를 `geometry_msgs/PoseStamped` 와 TF 로 publish
- `mock_tracker_node` — SteamVR 없이 가짜 포즈 publish (다운스트림 노드 개발/검증용)
- `list_devices` — SteamVR 에 잡힌 디바이스/시리얼 출력

토픽:
- `/vive/<name>/pose` — `geometry_msgs/PoseStamped`, frame_id="vive_world"
- `/vive/<name>/valid` — `std_msgs/Bool`
- TF: `vive_world` → `vive_<name>`

빌드 + 셋업: 자세한 단계는 [SETUP.md](SETUP.md) 참고. 핵심 흐름:
1. Steam + SteamVR 설치 (수동 로그인 필요)
2. `pip install --user openvr`
3. `colcon build --packages-select vive_tracker_ros2`
4. SteamVR 띄우고 트래커 페어링
5. `ros2 launch vive_tracker_ros2 vive_tracker.launch.py`

`vendor/triad_openvr/` 에 TriadSemi/triad_openvr 원본이 들어 있고, 실제 사용되는 사본은 `vive_tracker_ros2/_triad_openvr.py` 입니다. (ament_python 설치를 위해 패키지 안으로 흡수)
