#!/usr/bin/env bash
# 델타 모니터: ABS(anchor 누적) vs INC(직전 대비 증분) vs WORLD(절대) 실시간 비교 (env 내부 처리)
# 사용: bash run_delta_monitor.sh [arms]   예) bash run_delta_monitor.sh right
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
       FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"
exec python3 "$HERE/delta_monitor.py" "$@"
