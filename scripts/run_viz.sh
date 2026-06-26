#!/usr/bin/env bash
# 터미널 2: Vive viz_node + RViz (GUI). 매핑: tracker_1(LHR-7B9A3BA9)->right, tracker_2(LHR-F4A94AD1)->left
# 헤드리스 viz_node 데몬이 떠 있으면 정리하고 띄움(OpenVR 클라이언트 1개만 가능).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
cd "$WS"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
       FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"

# 기존 viz_node(데몬 포함)만 정리 (teleop_delta 는 건드리지 않음)
self=$$
for d in /proc/[0-9]*; do
  p=${d#/proc/}; [ "$p" = "$self" ] && continue
  cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cl" in /usr/bin/python3\ *vive_3d_viz/lib/vive_3d_viz/viz_node*) kill "$p" 2>/dev/null ;; esac
done

ros2 launch vive_3d_viz viz.launch.py \
  tracker_name_map:="LHR-7B9A3BA9:right,LHR-F4A94AD1:left"
