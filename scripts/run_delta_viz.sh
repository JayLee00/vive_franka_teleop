#!/usr/bin/env bash
# 델타 시각화: 3D 경로 + dx/dy/dz·rotvec 시계열 그래프 (matplotlib, env 내부 처리)
# 사용: bash run_delta_viz.sh [arm]   예) bash run_delta_viz.sh left
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
       FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"
export DISPLAY="${DISPLAY:-:1}"
exec python3 "$HERE/delta_viz.py" "$@"
