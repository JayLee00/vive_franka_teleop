#!/usr/bin/env python3
"""데이터 수집 전 지각(perception) 프리플라이트 점검.

RealSense(제어 PC가 ROS2로 발행)와 과일 pose가 실제로 잘 들어오는지 수집 전에 확인한다.
토픽 네임스페이스가 셋업마다 다르므로(`/camera/camera` vs `/front_cam/front`) 자동 탐지한다.

점검 항목 (기본 5초):
  RealSense : color 이미지 · depth 이미지 · camera_info  → Hz + 해상도
  과일       : /inhand/bbox_corners(fruit-manip 원본) · /fruit/pose · /fruit/size
              → Hz + 마지막 값(위치[m], 크기 a/b/c[m])
각 항목 ✓/✗ 와 원인 힌트를 출력하고 종료.

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/check_perception.py            # 5초 점검
  python3 record/check_perception.py --sec 10    # 더 길게
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)


def discover(node: Node):
    """ROS 그래프에서 realsense 컬러/뎁스/info 토픽을 자동 탐지."""
    names = dict(node.get_topic_names_and_types())  # {name: [types]}
    color = depth = info = None
    for name, types in names.items():
        low = name.lower()
        if info is None and "sensor_msgs/msg/CameraInfo" in types and "color" in low:
            info = name
        if "image" not in low:
            continue
        is_img = ("sensor_msgs/msg/Image" in types) or ("sensor_msgs/msg/CompressedImage" in types)
        if not is_img:
            continue
        if depth is None and "depth" in low:
            depth = name
        elif color is None and "color" in low and "depth" not in low:
            color = name
    # camera_info fallback: color 없이도 아무 color camera_info
    if info is None:
        for name, types in names.items():
            if "sensor_msgs/msg/CameraInfo" in types:
                info = name; break
    return color, depth, info, names


class Check(Node):
    def __init__(self, sec: float):
        super().__init__("check_perception")
        self.sec = sec
        self.count: dict[str, int] = {}
        self.res: dict[str, str] = {}
        self.fruit_pose = None
        self.fruit_size = None

    def _bump(self, key):
        self.count[key] = self.count.get(key, 0) + 1

    def subscribe_all(self, color, depth, info, all_names):
        # RealSense
        if color:
            typ = CompressedImage if "compressed" in color.lower() else Image
            self.create_subscription(typ, color,
                lambda m: (self._bump("color"), self._img_res("color", m)), QOS)
        if depth:
            self.create_subscription(Image, depth,
                lambda m: (self._bump("depth"), self._img_res("depth", m)), QOS)
        if info:
            self.create_subscription(CameraInfo, info,
                lambda m: (self._bump("info"), self.res.__setitem__("info", f"{m.width}x{m.height}")), QOS)
        # 과일
        if "/inhand/bbox_corners" in all_names:
            self.create_subscription(Float32MultiArray, "/inhand/bbox_corners",
                lambda m: self._bump("bbox"), QOS)
        self.create_subscription(PoseStamped, "/fruit/pose", self._on_fruit_pose, QOS)
        self.create_subscription(Float32MultiArray, "/fruit/size", self._on_fruit_size, QOS)

    def _img_res(self, key, m):
        if key not in self.res:
            if hasattr(m, "width"):
                self.res[key] = f"{m.width}x{m.height}"
            else:
                self.res[key] = f"{len(m.data)}B jpeg"

    def _on_fruit_pose(self, m: PoseStamped):
        self._bump("fruit_pose")
        p = m.pose.position
        self.fruit_pose = (p.x, p.y, p.z, m.header.frame_id)

    def _on_fruit_size(self, m: Float32MultiArray):
        self._bump("fruit_size")
        self.fruit_size = list(m.data)


def line(ok, label, detail):
    return f"  {'✓' if ok else '✗':2s} {label:26s} {detail}"


def main():
    ap = argparse.ArgumentParser(description="수집 전 RealSense + 과일 데이터 점검")
    ap.add_argument("--sec", type=float, default=5.0, help="점검 시간[s] (기본 5)")
    args = ap.parse_args()

    rclpy.init()
    node = Check(args.sec)
    # 토픽 디스커버리 잠깐 대기
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    color, depth, info, all_names = discover(node)
    node.subscribe_all(color, depth, info, all_names)

    print(f"\n{'='*60}\n  지각 프리플라이트 — {args.sec:.0f}초 측정\n{'='*60}")
    print(f"  탐지된 카메라 토픽: color={color or '없음'}")
    print(f"                      depth={depth or '없음'}  info={info or '없음'}")

    t0 = time.perf_counter()
    while rclpy.ok() and time.perf_counter() - t0 < args.sec:
        rclpy.spin_once(node, timeout_sec=0.05)
    dt = time.perf_counter() - t0

    def hz(k): return node.count.get(k, 0) / dt

    print(f"\n── RealSense ──────────────────────────────────────────────")
    print(line(node.count.get("color", 0) > 0, "color 이미지",
               f"{hz('color'):5.1f} Hz  {node.res.get('color','-')}" if color else "토픽 없음"))
    print(line(node.count.get("depth", 0) > 0, "depth(aligned) 이미지",
               f"{hz('depth'):5.1f} Hz  {node.res.get('depth','-')}" if depth else "토픽 없음"))
    print(line(node.count.get("info", 0) > 0, "camera_info(K)",
               f"{hz('info'):5.1f} Hz  {node.res.get('info','-')}" if info else "토픽 없음"))

    print(f"\n── 과일(fruit) ────────────────────────────────────────────")
    print(line(node.count.get("bbox", 0) > 0, "/inhand/bbox_corners",
               f"{hz('bbox'):5.1f} Hz  (fruit-manip live_bbox_gui 원본)"))
    if node.fruit_pose:
        x, y, z, frame = node.fruit_pose
        pose_detail = f"{hz('fruit_pose'):5.1f} Hz  pos=({x:+.3f},{y:+.3f},{z:+.3f})m  [{frame}]"
    else:
        pose_detail = "수신 없음 (fruit_pose_bridge 미실행?)"
    print(line(node.count.get("fruit_pose", 0) > 0, "/fruit/pose", pose_detail))
    if node.fruit_size:
        s = node.fruit_size
        size_detail = f"{hz('fruit_size'):5.1f} Hz  a/b/c=" + "/".join(f"{v:.3f}" for v in s[:3]) + " m"
    else:
        size_detail = "수신 없음"
    print(line(node.count.get("fruit_size", 0) > 0, "/fruit/size", size_detail))

    # 종합 판정
    rs_ok = node.count.get("color", 0) > 0 and node.count.get("depth", 0) > 0
    fruit_ok = node.count.get("fruit_pose", 0) > 0 and node.count.get("fruit_size", 0) > 0
    print(f"\n{'='*60}")
    print(f"  RealSense : {'✓ 정상' if rs_ok else '✗ 문제'}   |   과일 : {'✓ 정상' if fruit_ok else '✗ 문제'}")
    if not rs_ok:
        print("  → RealSense: 제어 PC에서 realsense 드라이버(ros)와 aligned_depth 켜졌는지 확인")
    if not fruit_ok:
        print("  → 과일: (1) live_bbox_gui.py(ros conda env) 실행  (2) fruit_pose_bridge.py 실행 확인")
    print(f"  {'▶ 수집 시작해도 됩니다.' if (rs_ok and fruit_ok) else '▶ 위 문제 해결 후 수집하세요.'}")
    print(f"{'='*60}\n")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
