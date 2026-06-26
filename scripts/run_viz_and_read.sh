#!/usr/bin/env bash
# RViz(3D 트래커 시각화) + 터미널 6DoF 실시간 표시 동시.
#   viz_node + RViz : 백그라운드 GUI 창
#   tracker_read    : 이 터미널 포그라운드 (pos[m] + 각도[deg])
# 종료(Ctrl+C) 하면 RViz/viz_node 까지 같이 정리됨.
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
       FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"

# 기존 viz_node(데몬 포함) 정리 — OpenVR 클라이언트는 1개만 가능
self=$$
for d in /proc/[0-9]*; do
  p=${d#/proc/}; [ "$p" = "$self" ] && continue
  cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cl" in /usr/bin/python3\ *vive_3d_viz/lib/vive_3d_viz/viz_node*) kill "$p" 2>/dev/null ;; esac
done
sleep 2

# viz_node + RViz (GUI) 를 백그라운드로
ros2 launch vive_3d_viz viz.launch.py \
  tracker_name_map:="LHR-7B9A3BA9:right,LHR-F4A94AD1:left" >/tmp/viz_rviz.log 2>&1 &
VIZ=$!
cleanup(){ kill "$VIZ" 2>/dev/null; pkill -f "vive_3d_viz/lib/vive_3d_viz/viz_node" 2>/dev/null; }
trap cleanup EXIT INT TERM
echo "RViz 기동 중... (로그 /tmp/viz_rviz.log)   종료: Ctrl+C"
sleep 4

# 터미널 6DoF 실시간 (포그라운드)
python3 "$HERE/tracker_read.py"
