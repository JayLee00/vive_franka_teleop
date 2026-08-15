#!/usr/bin/env bash
# 전체 스택 통합 런처 — 하위 명령 하나로 각 구성요소를 띄운다.
#
# 기존 스크립트를 대체하지 않고 **호출**한다. 흩어진 진입점과 매번 다시 치던
# ROS 환경변수를 한 곳에 모으는 게 목적이다.
#
#   bash run.sh <명령> [옵션]
#
#   status      전부 점검 (토픽·프로세스·포트) — 먼저 이걸 보세요
#   stop        전부 정리 (누수된 republish 포함)
#
#   camera      compressed → raw republish  (카메라 본체는 제어 PC 에서 `rs`)
#   pedal       발판 (팔/손 제어권 + 로깅 S/E 토글)
#   glove       글러브 → 핸드 텔레옵
#   paxini      글러브쪽 Paxini 촉각 → /glove/paxini/*
#   gp          글러브 + Paxini 함께 (scripts/run_glove_paxini.sh)
#   fruit       FoundationPose 과일 6DoF (창 두 개, 클릭 선택)
#   record      HDF5 로깅 (RGB 포함)
#   dp          Diffusion Policy 실기 배포
#   dp-table    DP 스윕 결과표
#
# 왜 `stop` 이 따로 있나: `ros2 run` 은 래퍼라 그것만 죽으면 자식 republish 가
# 남는다. 이게 쌓이면 같은 컬러 프레임이 N중으로 발행돼 시간동기화가 무너지고
# 과일 자세가 30Hz → 2.5Hz 로 주저앉는다(실측 15개 누적). 작업 전후로 돌리세요.
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS=/home/js/franka_ros2_ws
NS=/front_cam/front
COLOR_C="$NS/color/image_raw/compressed"
COLOR_FAST="$NS/color/image_fast"
SIDE="${SIDE:-right}"

source /opt/ros/humble/setup.bash
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE="$HERE/config/fastdds_lan_only.xml"
export DISPLAY="${DISPLAY:-:1}"

CMD="${1:-status}"; shift 2>/dev/null

# 파이프라인이라 exit 는 항상 0 이다 → 호출측에서 `|| echo ✗` 가 안 먹는다.
# 없으면 여기서 직접 ✗ 를 낸다.
hz()  { local r; r=$(timeout "${HZ_WAIT:-3}" ros2 topic hz "$1" 2>/dev/null \
          | grep -oP 'average rate: \K[\d.]+' | head -1); echo "${r:-✗}"; }
cnt() { pgrep -fc "$1" 2>/dev/null | head -1; }

case "$CMD" in

status)
  echo "── 프로세스 ──"
  printf "  %-26s %s\n" "republish"        "$(cnt 'image_transport/republish')  (1 이 정상, 2 이상이면 중복)"
  printf "  %-26s %s\n" "glove_teleop"     "$(cnt 'glove_teleop.py')"
  printf "  %-26s %s\n" "paxini_uart_node" "$(cnt 'paxini_uart_node.py')"
  printf "  %-26s %s\n" "foot_pedal"       "$(cnt 'foot_pedal')"
  printf "  %-26s %s\n" "fp_ros_node(과일)" "$(cnt 'fp_ros_node.py')"
  printf "  %-26s %s\n" "hdf5_recorder"    "$(cnt 'ros2_hdf5_recorder.py')"
  printf "  %-26s %s\n" "dp run.py"        "$(cnt 'diffusion_policy/run.py')"
  echo "  docker fp_server        : $(docker ps --format '{{.Names}}' 2>/dev/null | grep -cx fp_server)"
  echo ""
  echo "── 장치 ──"
  echo "  글러브 /dev/ttyUSB0 : $([ -e /dev/ttyUSB0 ] && echo 있음 || echo '✗ 없음')"
  echo "  촉각   /dev/ttyACM0 : $([ -e /dev/ttyACM0 ] && ([ -w /dev/ttyACM0 ] && echo '있음(쓰기 가능)' || echo '있음(권한 없음 → sudo chmod 666)') || echo '✗ 없음')"
  echo "  발판   footswitch   : $(ls /dev/input/by-id/*FootSwitch*event-kbd 2>/dev/null | head -1 || echo '✗ 없음')"
  echo ""
  echo "── 토픽 ──"
  for t in "$COLOR_C" "$COLOR_FAST" "$NS/aligned_depth_to_color/image_raw" \
           /glove/$SIDE/q_raw /hand/$SIDE/joint_states /paxini/$SIDE/ft \
           /glove/paxini/$SIDE/ft /fruit/pose /fruit/type /record/enable ; do
    printf "  %-46s %s\n" "$t" "$(hz "$t")"
  done
  ;;

stop)
  echo "정리 중..."
  pkill -f "image_transport/republish"        2>/dev/null
  pkill -f "foundation_pose/fp_ros_node.py"   2>/dev/null
  pkill -f "foundation_pose/fruit_label_node.py" 2>/dev/null
  pkill -f "record/fruit_overlay.py"          2>/dev/null
  pkill -f "run_foundation_pose.sh"           2>/dev/null
  docker rm -f fp_server >/dev/null 2>&1
  sleep 2
  echo "남은 republish: $(cnt 'image_transport/republish')  (0 이면 깨끗)"
  echo "※ glove/paxini/pedal/record/dp 는 각자 터미널에서 Ctrl+C 로 끄세요 (의도치 않은 중단 방지)"
  ;;

camera)
  # 카메라 본체는 제어 PC 에서 뜬다. 여기서는 compressed 를 raw 로 풀어주기만 한다
  # (raw 를 그대로 받으면 LAN 대역 때문에 3Hz 밖에 안 나온다).
  pkill -f "image_transport/republish.*$COLOR_FAST" 2>/dev/null; sleep 1
  echo "republish: $COLOR_C → $COLOR_FAST  (Ctrl+C 로 종료)"
  exec ros2 run image_transport republish compressed raw \
       --ros-args -r "in/compressed:=$COLOR_C" -r "out:=$COLOR_FAST"
  ;;

pedal)
  # 왼=STOP  오른=GO  중간=로깅 S/E (/record/enable 토글)
  echo "발판: 왼=STOP 오른=GO 중간=로깅토글   (권한 필요시 sudo usermod -aG input \$USER)"
  exec python3 "$HERE/scripts/foot_pedal.py" --hand-side "$SIDE" "$@"
  ;;

glove)
  exec python3 "$HERE/tools/glove_teleop.py" --side "$SIDE" "$@"
  ;;

paxini)
  # 로봇 핸드에도 같은 Paxini 가 달려 토픽이 겹치므로 글러브쪽은 prefix 로 가른다
  exec python3 "$HERE/tools/paxini_uart_node.py" --side "$SIDE" \
       --port /dev/ttyACM0 --topic-prefix /glove/paxini "$@"
  ;;

gp)
  exec bash "$HERE/scripts/run_glove_paxini.sh" "$SIDE" "$@"
  ;;

fruit)
  exec bash "$HERE/foundation_pose/run_foundation_pose.sh" "$@"
  ;;

record)
  # --rgb-on 이 없으면 이미지가 저장되지 않는다(기본 꺼짐). 시작/종료는 발판 중간 버튼.
  echo "로깅: 발판 중간 버튼(=/record/enable)으로 에피소드 시작/종료"
  exec python3 "$HERE/record/ros2_hdf5_recorder.py" --rgb-on "$@"
  ;;

dp)
  exec python3 "$HERE/diffusion_policy/run.py" --side "$SIDE" "$@"
  ;;

dp-table)
  cd "$HERE/diffusion_policy" && exec python3 make_table.py "$@"
  ;;

*)
  # 맨 위 주석 블록을 그대로 도움말로 쓴다 (주석 아닌 줄이 나오면 멈춘다)
  awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
  exit 1
  ;;
esac
