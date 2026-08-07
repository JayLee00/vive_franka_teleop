#!/usr/bin/env bash
# 글러브 텔레옵 + 글러브에 달린 Paxini 촉각을 함께 띄운다 (env 내부 처리).
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

# 남아있는 글러브/촉각 프로세스를 먼저 정리한다. 둘 다 UART 를 exclusive 로 잡아서
# 중복 실행되면 새로 뜬 쪽이 포트를 못 열고 "글러브 끊김/미연결" 로만 돈다.
# TERM 으로 보내는 이유: glove_teleop.py 는 종료 훅에서 핸드 servo OFF 를 발행한다.
STALE_PATS=(
    "tools/glove_teleop.py"
    "tools/paxini_uart_node.py"
    "tools/glove_tactile_monitor.py"
    "tools/glove_joint_monitor.py"
    "tools/glove_raw_uart.py"
)
for pat in "${STALE_PATS[@]}"; do
    pids="$(pgrep -f "$pat")" || continue
    echo "정리: $pat (pid $(echo $pids | tr '\n' ' '))"
    kill $pids 2>/dev/null
done
# 서보 OFF 발행 + 포트 해제까지 기다린다. 그래도 남으면 강제 종료.
for _ in 1 2 3 4 5 6; do
    sleep 0.25
    pgrep -f "tools/glove_teleop.py|tools/paxini_uart_node.py" >/dev/null || break
done
for pat in "${STALE_PATS[@]}"; do
    pkill -9 -f "$pat" 2>/dev/null
done

# 이 포트는 root:dialout 이라 udev 규칙이 없으면 열리지 않는다. 촉각만 조용히
# 죽고 글러브는 도는 상황을 피하려고 미리 알려준다(중단하지는 않는다).
if [[ ! -r "$PAXINI_PORT" || ! -w "$PAXINI_PORT" ]]; then
    echo "경고: $PAXINI_PORT 읽기/쓰기 불가 — 촉각은 안 뜨고 글러브만 돕니다."
    echo "      sudo chmod 666 $PAXINI_PORT  (임시) 또는 udev 규칙 등록"
fi

echo "글러브 : /glove/$SIDE/q_raw, /hand/$SIDE/q_target"
echo "촉각   : /glove/paxini/$SIDE/ft, /glove/paxini/$SIDE/raw"

python3 "$PROJ/tools/paxini_uart_node.py" \
        --side "$SIDE" --port "$PAXINI_PORT" --topic-prefix /glove/paxini &
PAXINI_PID=$!

# 글러브를 끄면 촉각 노드도 같이 정리한다. 반대로 촉각이 죽어도 글러브는 계속 돈다.
cleanup() {
    kill "$PAXINI_PID" 2>/dev/null
    wait "$PAXINI_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

python3 "$PROJ/tools/glove_teleop.py" --side "$SIDE"
