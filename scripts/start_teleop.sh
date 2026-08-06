#!/usr/bin/env bash
# 통합 텔레옵 런처 — 입력 장치를 골라서 한 방에 기동.
#
#   bash scripts/start_teleop.sh both     # vive 팔 + 글러브 손 (기본)
#   bash scripts/start_teleop.sh vive     # vive 팔만 (트래커 2개 → 절대 EE 타겟)
#   bash scripts/start_teleop.sh glove    # 글러브 손만 (현재 오른손 1개)
#
# 기동 순서(안전): 기존 정리 → 페달(STOP을 latched로 깔아둠) → vive 파이프라인 → 글러브
#   glove_teleop 은 뜨자마자 서보 ON + 램프를 시작하므로, 페달이 먼저 STOP 을
#   TRANSIENT_LOCAL 로 발행해 두면 늦게 뜬 글러브가 그 상태를 받아 손이 안 움직인다.
#
# 로그: /tmp/viz_node.log  /tmp/teleop_delta.log  /tmp/glove_right.log  /tmp/foot_pedal.log
set +u   # ROS setup.bash 가 미설정 변수를 참조하므로 -u 금지

MODE="${1:-both}"
case "$MODE" in
  vive|glove|both) ;;
  *) echo "사용법: bash $0 [both|vive|glove]"; exit 1 ;;
esac

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
if [ "$MODE" = "glove" ] || [ "$MODE" = "both" ]; then
  setsid bash -c "$ENVSRC; exec python3 $PROJ/tools/glove_teleop.py --side right" \
    >/tmp/glove_right.log 2>&1 < /dev/null &
  echo "  글러브(right) 기동 (log: /tmp/glove_right.log)"
  sleep 2
fi

echo "=== 기동 완료 ==="
echo "  페달: 왼=STOP  오른=GO  중간=로깅 S/E토글"
echo "  모니터:  python3 $HERE/ee_monitor.py"
echo "  로그:    tail -f /tmp/foot_pedal.log /tmp/glove_right.log /tmp/teleop_delta.log"
