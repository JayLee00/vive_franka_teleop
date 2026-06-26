#!/usr/bin/env bash
# Vive 2트래커 -> 프랑카 2팔 델타 텔레옵 파이프라인 (헤드리스 백그라운드 데몬)
# 매핑: tracker_1(LHR-7B9A3BA9) -> right arm,  tracker_2(LHR-F4A94AD1) -> left arm
#   viz_node     : /vive/right/pose (7B9A3BA9), /vive/left/pose (F4A94AD1)
#   teleop_delta : /teleop/delta/right (-> right arm), /teleop/delta/left (-> left arm)
set +u   # ROS setup.bash 가 미설정 변수를 참조하므로 -u 금지
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"                 # ~/Desktop/vive_franka_teleop
WS=/home/js/franka_ros2_ws                # colcon 워크스페이스 (그대로 사용)
PROFILE="$PROJ/config/fastdds_lan_only.xml"
ENVSRC="source /opt/ros/humble/setup.bash; source $WS/install/setup.bash; \
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
FASTRTPS_DEFAULT_PROFILES_FILE=$PROFILE"
eval "$ENVSRC"

# 1) 기존 vive 노드/RViz 정리
self=$$
for d in /proc/[0-9]*; do
  p=${d#/proc/}; [ "$p" = "$self" ] && continue
  cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cl" in /usr/bin/python3\ *vive_3d_viz*) echo "kill $p"; kill "$p" 2>/dev/null ;; esac
done
pkill -x rviz2 2>/dev/null || true
sleep 2   # OpenVR 연결 해제 대기

# 2) viz_node (헤드리스) — 포즈 발행
setsid bash -c "$ENVSRC; exec ros2 run vive_3d_viz viz_node \
  --ros-args -r __node:=vive_3d_viz_node --params-file $PROJ/config/viz_params.yaml" \
  >/tmp/viz_node.log 2>&1 < /dev/null &
echo "viz_node 기동 (log: /tmp/viz_node.log)"
sleep 5   # 디바이스 디스커버리 + 포즈 발행 대기

# 3) teleop_delta — 델타(위치/각도) 발행
setsid bash -c "$ENVSRC; exec ros2 run vive_3d_viz teleop_delta \
  --ros-args -r __node:=vive_teleop_delta --params-file $PROJ/config/teleop_params.yaml" \
  >/tmp/teleop_delta.log 2>&1 < /dev/null &
echo "teleop_delta 기동 (log: /tmp/teleop_delta.log)"
sleep 2
echo "=== 기동 완료 (profile: $PROFILE) ==="
