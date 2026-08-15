#!/usr/bin/env bash
# 과일 6DoF 시각화 원커맨드 런처
#
#   [제어 PC realsense]  /front_cam/front/color/image_raw/compressed (jpeg, ~20-37Hz)
#         │                                    (raw 는 3Hz 밖에 안 나옴 = LAN 대역 한계)
#         ▼ republish (compressed→raw)
#   /front_cam/front/color/image_fast
#         ▼
#   live_bbox_gui.py  (SAM2 세그먼트 + AprilTag 방향)  →  /inhand/bbox_corners (8코너)
#         ▼
#   fruit_pose_bridge.py  →  /fruit/pose (위치+방향) + /fruit/size ([a,b,c] m)
#         ▼
#   fruit_overlay.py  →  영상 위에 3D 박스 + XYZ축 + 수치 오버레이 (화면)
#
# 사용:  bash record/run_fruit_viz.sh          # 전부 띄움
#        bash record/run_fruit_viz.sh --check  # 전제조건만 점검
#
# 종료: Ctrl+C (이 스크립트가 띄운 것 모두 정리)
set +u          # ROS setup.bash 가 미설정 변수를 참조하므로 -u 금지
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FM="$PROJ/fruit-manipulation"
NS=/front_cam/front
COLOR_C="$NS/color/image_raw/compressed"
COLOR_FAST="$NS/color/image_fast"
INFO="$NS/color/camera_info"
DEPTH="$NS/aligned_depth_to_color/image_raw"

source /opt/ros/humble/setup.bash
source /home/js/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJ/config/fastdds_lan_only.xml"
export DISPLAY="${DISPLAY:-:1}"

hz() { timeout 6 ros2 topic hz "$1" 2>/dev/null | grep -oP 'average rate: [\d.]+' | head -1; }

echo "── 전제조건 점검 ──"
C=$(hz "$COLOR_C"); I=$(hz "$INFO"); D=$(hz "$DEPTH")
echo "  color(compressed): ${C:-✗ 없음}"
echo "  camera_info      : ${I:-✗ 없음}"
echo "  depth(aligned)   : ${D:-✗ 없음}"
if [ -z "${C:-}" ] || [ -z "${I:-}" ]; then
  echo ""
  echo "✗ 카메라가 발행되지 않습니다. 제어 PC 에서 realsense 를 먼저 띄우세요 (rs)."
  echo "  정렬 깊이도 필요:  ros2 param set /front_cam/front align_depth.enable true"
  exit 1
fi
[ -z "${D:-}" ] && echo "  ⚠ depth 없음 → live_bbox_gui 가 bbox 를 발행하지 못합니다(정렬깊이 필수)"
[ "${1:-}" = "--check" ] && { echo "점검만 수행 — 종료"; exit 0; }

PIDS=()
cleanup() { echo ""; echo "정리 중..."; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; echo "종료"; }
trap cleanup EXIT INT TERM

echo ""
echo "── 1) republish (compressed → raw) ──"
ros2 run image_transport republish compressed raw \
  --ros-args -r "in/compressed:=$COLOR_C" -r "out:=$COLOR_FAST" >/tmp/fruit_republish.log 2>&1 &
PIDS+=($!); sleep 3
R=$(hz "$COLOR_FAST"); echo "  $COLOR_FAST : ${R:-✗ (로그 /tmp/fruit_republish.log)}"

echo "── 2) live_bbox_gui (SAM2, GPU) ──"
( cd "$FM" && exec /usr/bin/python3 live_bbox_gui.py \
    --color-topic "$COLOR_FAST" --info-topic "$INFO" --depth-topic "$DEPTH" \
    --frame-id camera_color_optical_frame \
    --no-sam3 --roi-frac 0.5,0.5,0.9,0.9 --depth-band 0.15,1.2 --wait 30 \
) >/tmp/fruit_bbox.log 2>&1 &
PIDS+=($!)
echo "  기동 중... (SAM2 로드 ~10초, 로그 /tmp/fruit_bbox.log)"
sleep 15
grep -E "streaming SAM2|stream tracker unavailable|camera topics up|WARN: no depth" /tmp/fruit_bbox.log | tail -4 | sed 's/^/    /'

echo "── 3) fruit_pose_bridge (8코너 → pose/size) ──"
/usr/bin/python3 "$PROJ/record/fruit_pose_bridge.py" >/tmp/fruit_bridge.log 2>&1 &
PIDS+=($!); sleep 2

echo "── 4) fruit_overlay (화면 오버레이) ──"
/usr/bin/python3 "$PROJ/record/fruit_overlay.py" \
  --color-topic "$COLOR_C" --info-topic "$INFO" >/tmp/fruit_overlay.log 2>&1 &
PIDS+=($!); sleep 3

echo ""
echo "── 상태 ──"
echo "  /inhand/bbox_corners : $(hz /inhand/bbox_corners || echo '✗ 아직 (과일이 ROI+깊이대역 안에 있어야 발행됨)')"
echo "  /fruit/pose          : $(hz /fruit/pose || echo '✗ 아직')"
echo ""
echo "창: 'Fruit 6DoF overlay' (q=종료, s=스냅샷) / live_bbox_gui 창(⚠ s 키 금지 — 크래시)"
echo "로그: /tmp/fruit_bbox.log, /tmp/fruit_bridge.log, /tmp/fruit_overlay.log"
echo "Ctrl+C 로 전체 종료"
wait
