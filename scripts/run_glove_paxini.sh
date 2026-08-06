#!/usr/bin/env bash
# 글러브 텔레옵 + 글러브에 달린 Paxini 촉각 + 둘을 같이 보는 모니터를 띄운다.
# 로봇 핸드에도 같은 Paxini 가 달려 있어 토픽이 겹치므로 촉각은 /glove/paxini 로 낸다.
# 사용: bash run_glove_paxini.sh [side]   예) bash run_glove_paxini.sh right
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
       FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"

SIDE="${1:-right}"
PAXINI_PORT=/dev/ttyACM0

# 이 포트는 root:dialout 이라 udev 규칙이 없으면 열리지 않는다. 촉각만 조용히
# 죽고 글러브는 도는 상황을 피하려고 미리 알려준다(중단하지는 않는다).
if [[ ! -r "$PAXINI_PORT" || ! -w "$PAXINI_PORT" ]]; then
    echo "경고: $PAXINI_PORT 읽기/쓰기 불가 — 촉각은 안 뜨고 글러브만 돕니다."
    echo "      sudo chmod 666 $PAXINI_PORT  (임시) 또는 udev 규칙 등록"
fi

echo "글러브 : /glove/$SIDE/q_raw, /hand/$SIDE/q_target"
echo "촉각   : /glove/paxini/$SIDE/ft, /glove/paxini/$SIDE/raw"
echo "모니터 : 글러브16 + 촉각 ft (Ctrl-C 로 전체 종료)"

python3 "$PROJ/tools/paxini_uart_node.py" \
        --side "$SIDE" --port "$PAXINI_PORT" --topic-prefix /glove/paxini &
PAXINI_PID=$!

python3 "$PROJ/tools/glove_teleop.py" --side "$SIDE" &
GLOVE_PID=$!

# Ctrl-C(=모니터 종료) 하면 두 노드도 같이 정리한다.
cleanup() {
    kill "$GLOVE_PID" "$PAXINI_PID" 2>/dev/null
    wait "$GLOVE_PID" "$PAXINI_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# 모니터는 토픽만 구독한다(시리얼 안 건드림). 두 노드가 토픽을 올릴 시간을 준다.
sleep 2
python3 "$PROJ/tools/glove_paxini_monitor.py" --side "$SIDE"
