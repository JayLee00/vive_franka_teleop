#!/usr/bin/env bash
# engage 기준 상대 포즈 노드: T_E_O 를 PoseStamped(/vive/relative_pose) + TF 로 발행.
# 읽기 전용 소비자(engage/pose 구독, 로봇 명령 X) — 라이브 텔레옵과 동시 실행 가능.
# 사용: bash run_relative_pose.sh [arm]   예) bash run_relative_pose.sh left
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
ARM="${1:-right}"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
       FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"
exec ros2 run vive_3d_viz relative_pose --ros-args \
  -p pose_topic:="/vive/${ARM}/pose" \
  -p engage_topic:="/teleop/engage/${ARM}" \
  -p output_topic:="/vive/relative_pose/${ARM}" \
  -p timeout_topic:="/vive/relative_pose/${ARM}/timeout" \
  -p engage_frame:="engage_frame_${ARM}" \
  -p relative_child_frame:="tracker_relative_${ARM}"
