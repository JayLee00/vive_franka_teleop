#!/usr/bin/env bash
# 통합 텔레옵 런처 — 입력 장치를 골라서 한 방에 기동.
#
#   bash scripts/start_teleop.sh both     # vive 팔 + 글러브 손 (기본)
#   bash scripts/start_teleop.sh vive     # vive 팔만 (트래커 2개 → 절대 EE 타겟)
#   bash scripts/start_teleop.sh glove    # 글러브 손만 (현재 오른손 1개)
#
# 모드 뒤의 인자는 글러브로 그대로 넘어간다 (run_glove_paxini.sh → glove_teleop.py):
#   bash scripts/start_teleop.sh both --DexFIT             # 촉각보조(글러브 Paxini Fz) 켜기
#   bash scripts/start_teleop.sh glove --DexFIT --dry-run  # 손 안 움직이고 값만 확인
#
# 기동 후 기본으로 대시보드를 띄운다 (vive/EE/글러브/핸드/paxini 한 화면, 제자리 갱신).
#   Ctrl-C = 대시보드만 종료, 텔레옵은 계속 돎.
#   --log = 대신 로그 따라가기 · --no-follow = 띄우고 바로 프롬프트로
#   bash scripts/start_teleop.sh stop      # 텔레옵 전부 종료
#
# 기동 순서(안전): 기존 정리 → 페달(STOP을 latched로 깔아둠) → vive 파이프라인 → 글러브
#   glove_teleop 은 뜨자마자 서보 ON + 램프를 시작하므로, 페달이 먼저 STOP 을
#   TRANSIENT_LOCAL 로 발행해 두면 늦게 뜬 글러브가 그 상태를 받아 손이 안 움직인다.
#
# 로그: /tmp/viz_node.log  /tmp/teleop_delta.log  /tmp/glove_right.log  /tmp/foot_pedal.log
set +u   # ROS setup.bash 가 미설정 변수를 참조하므로 -u 금지

MODE="${1:-both}"
case "$MODE" in
  vive|glove|both) shift ;;
  stop) shift ;;                     # 전부 종료
  -*) MODE=both ;;                   # 모드 생략하고 옵션만 준 경우
  *) echo "사용법: bash $0 [both|vive|glove|stop] [--DexFIT] [--dry-run] [--no-follow]"; exit 1 ;;
esac

# 런처 옵션(글러브로 넘기지 않는다). 기본은 기동 후 대시보드를 띄운다.
VIEW=dash                            # dash | log | none
GLOVE_ARGS=()
for a in "$@"; do
  case "$a" in
    --no-follow) VIEW=none ;;
    --log)       VIEW=log ;;
    --dash)      VIEW=dash ;;
    *) GLOVE_ARGS+=("$a") ;;
  esac
done

ALL_PATS="scripts/foot_pedal.py|tools/glove_teleop.py|tools/paxini_uart_node.py|vive_3d_viz/viz_node|vive_3d_viz/teleop_delta|run_glove_paxini.sh"
if [ "$MODE" = "stop" ]; then
  echo "=== 텔레옵 전부 종료 ==="
  pkill -f "$ALL_PATS" 2>/dev/null
  sleep 2
  pgrep -af "$ALL_PATS" || echo "  모두 종료됨 ✅"
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
WS=/home/js/franka_ros2_ws
PROFILE="$PROJ/config/fastdds_lan_only.xml"
ENVSRC="source /opt/ros/humble/setup.bash; source $WS/install/setup.bash; \
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
FASTRTPS_DEFAULT_PROFILES_FILE=$PROFILE"
eval "$ENVSRC"

echo "=== 통합 텔레옵 기동 (mode=$MODE) ==="

# 1) 기존 프로세스 정리 — 페달은 한 프로세스만 잡을 수 있고(EVIOCGRAB),
#    글러브/파이프라인 중복은 토픽 발행 충돌을 만든다.
#    glove 모드에선 vive 파이프라인도 내린다 — 남아 있으면 이전 engage 상태 그대로
#    팔에 EE 타겟이 계속 나갈 수 있다(페달은 glove 모드에서 팔 engage 를 안 건드림).
self=$$
for d in /proc/[0-9]*; do
  p=${d#/proc/}; [ "$p" = "$self" ] && continue
  cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$cl" in
    *foot_pedal*.py*|*glove_teleop.py*) echo "  정리: $p"; kill "$p" 2>/dev/null ;;
  esac
  if [ "$MODE" = "glove" ]; then
    case "$cl" in
      *vive_3d_viz/viz_node*|*vive_3d_viz/teleop_delta*)
        echo "  정리(vive): $p"; kill "$p" 2>/dev/null ;;
    esac
  fi
done
sleep 1

# 2) 페달 먼저 (STOP 을 latched 로 깔아둠) — 글러브/팔이 늦게 떠도 안 움직임
setsid bash -c "$ENVSRC; exec python3 $HERE/foot_pedal.py --mode $MODE" \
  >/tmp/foot_pedal.log 2>&1 < /dev/null &
echo "  페달 기동 (mode=$MODE, log: /tmp/foot_pedal.log)"
sleep 2

# 3) vive 팔 파이프라인 (viz_node + teleop_delta) — 기존 검증 스크립트 재사용
if [ "$MODE" = "vive" ] || [ "$MODE" = "both" ]; then
  bash "$HERE/start_teleop_pipeline.sh" | sed 's/^/  [vive] /'
fi

# 4) 글러브 (현재 오른손 1개. 왼손 추가 시 --port 로 좌/우 포트 고정 필요)
#    run_glove_paxini.sh 를 쓴다 — 글러브 텔레옵 + 글러브에 달린 Paxini 촉각 노드를 함께 띄운다.
#    --DexFIT 은 /glove/paxini/<side>/ft 를 구독하므로 그 노드가 떠 있어야 동작한다.
#    (로봇 핸드의 /paxini/<side>/ft 와는 다른 것)
if [ "$MODE" = "glove" ] || [ "$MODE" = "both" ]; then
  setsid bash -c "exec bash $HERE/run_glove_paxini.sh right ${GLOVE_ARGS[*]}" \
    >/tmp/glove_right.log 2>&1 < /dev/null &
  echo "  글러브(right) + 촉각 기동 ${GLOVE_ARGS[*]:+[${GLOVE_ARGS[*]}]} (log: /tmp/glove_right.log)"
  sleep 3
fi

echo "=== 기동 완료 ==="
echo "  페달: 왼=STOP  오른=GO  중간=로깅 S/E토글"
echo "  모니터(다른 터미널):  python3 $HERE/ee_monitor.py"
echo "  전부 종료:            bash $0 stop"

# 기동한 것들은 데몬이라 이 터미널과 무관하게 계속 돈다. 기본은 대시보드를 띄워
# vive/EE/글러브/핸드/paxini 수치를 한 화면에서 제자리 갱신으로 보여준다.
#   Ctrl-C = 대시보드만 종료 (텔레옵은 계속 돎).  완전히 끄려면 위 'stop'.
if [ "$VIEW" = "dash" ]; then
  sleep 1
  exec python3 "$HERE/teleop_dashboard.py"
elif [ "$VIEW" = "log" ]; then
  LOGS=(/tmp/foot_pedal.log)
  [ "$MODE" = "glove" ] || [ "$MODE" = "both" ] && LOGS+=(/tmp/glove_right.log)
  [ "$MODE" = "vive" ] || [ "$MODE" = "both" ] && LOGS+=(/tmp/teleop_delta.log)
  echo ""
  echo "--- 실시간 로그 (Ctrl-C = 보기만 종료, 텔레옵은 계속 돎) ---"
  exec tail -n 5 -f "${LOGS[@]}"
fi
echo "  대시보드:     python3 $HERE/teleop_dashboard.py"
echo "  실시간 로그:  tail -f /tmp/foot_pedal.log /tmp/glove_right.log /tmp/teleop_delta.log"
