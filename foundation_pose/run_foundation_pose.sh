#!/usr/bin/env bash
# FoundationPose 과일 6DoF 원커맨드 런처  (run_fruit_viz.sh 의 자세추정 대체판)
#
#   [제어 PC realsense]  /camera/camera/color/image_raw/compressed
#         ▼ republish (compressed→raw)
#   /camera/camera/color/image_fast
#         ▼
#   fp_ros_node.py (호스트)  ── 첫 프레임만 SAM2 마스크 ──┐
#         │                                              │ TCP :5577
#         │                                              ▼
#         │                          [docker] fp_server.py = FoundationPose
#         ▼                                              │
#   /fruit/pose + /fruit/size  ◄─────────────────────────┘
#         ▼
#   fruit_overlay.py (원본 그대로)  →  화면 오버레이
#
# 기존 run_fruit_viz.sh 와의 차이: 자세를 SAM2 OBB 의 PCA 가 아니라
# FoundationPose 의 CAD 정합으로 뽑는다. 오버레이·토픽은 동일하다.
#
# 사용:  bash foundation_pose/run_foundation_pose.sh
#        bash foundation_pose/run_foundation_pose.sh --check     # 전제조건만
#        bash foundation_pose/run_foundation_pose.sh --compare   # /fruit_fp/* 로 발행(기존과 동시 비교)
#
# 종료: Ctrl+C
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
NS=/camera/camera
COLOR_C="$NS/color/image_raw/compressed"
COLOR_FAST="$NS/color/image_fast"
INFO="$NS/color/camera_info"
DEPTH="$NS/aligned_depth_to_color/image_raw"

IMAGE=foundationpose:local     # setup.sh 가 공식 이미지에 sm_86 패치를 구워 만든 것
CPY=/opt/conda/envs/my/bin/python   # 이미지의 torch 는 conda env "my" 에 있다
CONTAINER=fp_server
PORT=5577
MESH="$HERE/assets/orange.obj"
FPROOT="$HERE/FoundationPose"
PUB_NS=/fruit          # 기본은 드롭인(기존 오버레이가 그대로 받음)

source /opt/ros/humble/setup.bash
source /home/js/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"
export DISPLAY="${DISPLAY:-:1}"

while [ $# -gt 0 ]; do
  case "$1" in
    --compare) PUB_NS=/fruit_fp ;;
    --mesh)    MESH="$2"; MESH_EXPLICIT=1; shift ;;
    --fruit)   FRUIT="$2"; shift ;;
    --seg-hz)  SEGHZ="$2"; shift ;;
    --check)   CHECK=1 ;;
  esac
  shift
done
# --fruit 를 주면 카탈로그에서 그 과일의 CAD 를 쓴다 (--mesh 보다 우선순위 낮음)
FRUIT="${FRUIT:-lemon}"
if [ -z "${MESH_EXPLICIT:-}" ] && [ -f "$HERE/fruits.yaml" ]; then
  m=$(/usr/bin/python3 - "$HERE" "$FRUIT" <<'PY' 2>/dev/null
import os, sys, yaml
here, key = sys.argv[1], sys.argv[2]
for c in yaml.safe_load(open(os.path.join(here, "fruits.yaml")))["fruits"]:
    if str(c["id"]) == key or c["name"].lower() == key.lower():
        p = c["mesh"] if os.path.isabs(c["mesh"]) else os.path.join(here, c["mesh"])
        print(p if os.path.isfile(p) else "")
        break
PY
)
  [ -n "$m" ] && MESH="$m"
fi
# 그래도 없으면 assets/ 의 아무 obj
[ -f "$MESH" ] || { alt=$(ls "$HERE"/assets/*.obj 2>/dev/null | head -1); [ -n "$alt" ] && MESH="$alt"; }

hz() { timeout 6 ros2 topic hz "$1" 2>/dev/null | grep -oP 'average rate: [\d.]+' | head -1; }

echo "── 0) 준비물 점검 ──"
MISSING=0
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  ✗ 도커 이미지 없음: $IMAGE (setup.sh 가 만든다)"; MISSING=1
else echo "  ✓ 도커 이미지"; fi
if [ ! -d "$FPROOT" ]; then echo "  ✗ FoundationPose 저장소 없음: $FPROOT"; MISSING=1
else echo "  ✓ FoundationPose 저장소"; fi
if [ ! -d "$FPROOT/weights" ] || [ -z "$(ls -A "$FPROOT/weights" 2>/dev/null)" ]; then
  echo "  ✗ 가중치 없음: $FPROOT/weights"; MISSING=1
else echo "  ✓ 가중치"; fi
if [ ! -f "$MESH" ]; then echo "  ✗ 메시 없음: $MESH"; MISSING=1
else echo "  ✓ 메시 ($(basename "$MESH"))"; fi
if [ "$MISSING" = 1 ]; then
  echo ""; echo "→ 먼저 준비 스크립트를 돌리세요:  bash $HERE/setup.sh"; exit 1
fi

echo "── 1) 카메라 점검 ──"
C=$(hz "$COLOR_C"); I=$(hz "$INFO"); D=$(hz "$DEPTH")
echo "  color(compressed): ${C:-✗ 없음}"
echo "  camera_info      : ${I:-✗ 없음}"
echo "  depth(aligned)   : ${D:-✗ 없음}"
if [ -z "${C:-}" ] || [ -z "${I:-}" ] || [ -z "${D:-}" ]; then
  echo ""
  echo "✗ 카메라(특히 정렬깊이)가 필요합니다. 제어 PC 에서 realsense 를 띄우세요 (rs)."
  echo "  ros2 param set /camera/camera align_depth.enable true"
  exit 1
fi
[ "${CHECK:-0}" = 1 ] && { echo "점검만 수행 — 종료"; exit 0; }

PIDS=()
cleanup() {
  echo ""; echo "정리 중..."
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  wait 2>/dev/null; echo "종료"
}
trap cleanup EXIT INT TERM

echo ""
echo "── 2) republish (compressed → raw) ──"
ros2 run image_transport republish compressed raw \
  --ros-args -r "in/compressed:=$COLOR_C" -r "out:=$COLOR_FAST" >/tmp/fp_republish.log 2>&1 &
PIDS+=($!); sleep 3
R=$(hz "$COLOR_FAST"); echo "  $COLOR_FAST : ${R:-✗ (로그 /tmp/fp_republish.log)}"

echo "── 3) FoundationPose 서버 (docker, GPU) ──"
docker rm -f "$CONTAINER" >/dev/null 2>&1
docker run -d --name "$CONTAINER" --gpus all --network=host --ipc=host \
  --env NVIDIA_DISABLE_REQUIRE=1 -e PYTHONUNBUFFERED=1 \
  -v /home:/home -v /tmp:/tmp \
  -w "$FPROOT" "$IMAGE" \
  "$CPY" "$HERE/fp_server.py" --mesh "$MESH" --port "$PORT" --fp-root "$FPROOT" \
  >/tmp/fp_server_start.log 2>&1
echo "  컨테이너 기동 — 모델 로드 대기(최대 120s)"
for i in $(seq 1 120); do
  docker logs "$CONTAINER" 2>&1 | grep -q "listening on" && break
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "  ✗ 컨테이너가 죽었습니다. 로그:"; docker logs "$CONTAINER" 2>&1 | tail -25; exit 1
  fi
  sleep 1
done
if docker logs "$CONTAINER" 2>&1 | grep -q "listening on"; then
  echo "  ✓ fp_server 준비 완료 (:$PORT)"
else
  echo "  ✗ 시간 초과. 로그:"; docker logs "$CONTAINER" 2>&1 | tail -25; exit 1
fi

echo "── 4) ROS2 브리지 (호스트, SAM2 초기 마스크) ──"
/usr/bin/python3 "$HERE/fp_ros_node.py" \
  --server "127.0.0.1:$PORT" --ns "$PUB_NS" --seg-hz "${SEGHZ:-5}" \
  --color-topic "$COLOR_FAST" --depth-topic "$DEPTH" --info-topic "$INFO" \
  >/tmp/fp_node.log 2>&1 &
PIDS+=($!)
echo "  기동 중... (SAM2 로드 ~10초, 로그 /tmp/fp_node.log)"
sleep 15
grep -E "SAM2 준비|SAM2 마스크|초기 등록|K 수신|접속|실패" /tmp/fp_node.log | tail -5 | sed 's/^/    /'

echo "── 5) 과일 라벨 노드 (/fruit/type) ──"
/usr/bin/python3 "$HERE/fruit_label_node.py" --fruit "$FRUIT" >/tmp/fp_label.log 2>&1 &
PIDS+=($!); sleep 3
grep -E "과일 =|카탈로그|CAD 교체|모르는 과일" /tmp/fp_label.log | tail -3 | sed 's/^/    /'

echo "── 6) fruit_overlay (원본 그대로) ──"
/usr/bin/python3 "$PROJ/record/fruit_overlay.py" \
  --color-topic "$COLOR_C" --info-topic "$INFO" >/tmp/fp_overlay.log 2>&1 &
PIDS+=($!); sleep 3

echo ""
echo "── 상태 ──"
echo "  $PUB_NS/pose : $(hz "$PUB_NS/pose" || echo '✗ 아직 (창에서 과일을 클릭하세요)')"
echo "  /fruit/type  : $(timeout 5 ros2 topic echo /fruit/type --once 2>/dev/null | grep -oP 'data: \K\d+' || echo '✗')"
echo ""
echo "창: 'Fruit 6DoF overlay' (q=종료, s=스냅샷)"
echo "로그: /tmp/fp_node.log (브리지), docker logs $CONTAINER (추정), /tmp/fp_overlay.log"
[ "$PUB_NS" = "/fruit_fp" ] && echo "※ --compare 모드: 기존 오버레이는 /fruit/* 를 보므로 이 자세는 안 보입니다."
echo "Ctrl+C 로 전체 종료"
wait
