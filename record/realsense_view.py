#!/usr/bin/env python3
"""RealSense 실시간 뷰어 + 원할 때 mp4 녹화 (텔레옵 모니터용).

제어 PC RealSense가 ROS2로 발행하는 컬러 스트림을 이 PC 화면에 띄우고,
원할 때 그 화면을 mp4로 녹화한다. (HDF5엔 이미지 저장 안 함 — 이건 별도 영상 기록.)

기본은 compressed(jpeg) 토픽 구독(LAN 대역폭 절약). 창에서 키로 제어:
    r      = 녹화 시작/정지 토글
    q/ESC  = 종료

옵션:
    --sync        발판 로깅(/record/enable)에 맞춰 녹화도 자동 시작/정지 (에피소드와 동기)
    --no-window   창 없이 녹화만 (헤드리스; --sync 로만 제어). 서버/테스트용.
    --raw         raw Image 토픽 사용(기본은 compressed)
    --topic-ns    카메라 네임스페이스 (기본 /camera/camera)
    --fps         저장 fps (기본 30)

실행:
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/realsense_view.py            # 보면서 r 로 녹화
  python3 record/realsense_view.py --sync     # 발판 로깅과 동기 녹화도 함께
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool

OUT_DIR = Path(__file__).resolve().parent / "videos"
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)


class RSView(Node):
    def __init__(self, args):
        super().__init__("realsense_view")
        self.args = args
        self.frame = None                  # 최신 BGR 프레임
        self.recording = False
        self.writer = None
        self.rec_path = None

        ns = args.topic_ns.rstrip("/")
        if args.raw:
            self.create_subscription(Image, f"{ns}/color/image_raw", self._on_raw, QOS)
            src = f"{ns}/color/image_raw (raw)"
        else:
            self.create_subscription(CompressedImage, f"{ns}/color/image_raw/compressed",
                                     self._on_compressed, QOS)
            src = f"{ns}/color/image_raw/compressed (jpeg)"
        if args.sync or args.no_window:
            self.create_subscription(Bool, "/record/enable", self._on_enable, 10)
        self.get_logger().info(f"구독: {src}  |  녹화 저장: {OUT_DIR}")

    # ── 이미지 콜백 ──
    def _on_compressed(self, msg: CompressedImage):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR
        if img is not None:
            self.frame = img

    def _on_raw(self, msg: Image):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.frame = img

    def _on_enable(self, msg: Bool):
        if msg.data and not self.recording:
            self.start_rec()
        elif (not msg.data) and self.recording:
            self.stop_rec()

    # ── 녹화 ──
    def start_rec(self):
        if self.frame is None:
            self.get_logger().warn("아직 프레임 없음 — 녹화 시작 보류")
            return
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        h, w = self.frame.shape[:2]
        self.rec_path = OUT_DIR / f"rs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.rec_path), fourcc, float(self.args.fps), (w, h))
        self.recording = True
        self.get_logger().info(f"● 녹화 시작 → {self.rec_path.name} ({w}x{h}@{self.args.fps})")

    def stop_rec(self):
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"■ 녹화 정지 → {self.rec_path}")
        self.writer = None

    def write_if_recording(self):
        if self.recording and self.writer is not None and self.frame is not None:
            self.writer.write(self.frame)


def main():
    ap = argparse.ArgumentParser(description="RealSense 뷰어 + 원할때 mp4 녹화")
    ap.add_argument("--sync", action="store_true", help="/record/enable(발판)에 맞춰 자동 녹화")
    ap.add_argument("--no-window", action="store_true", help="창 없이 녹화만(--sync 제어)")
    ap.add_argument("--raw", action="store_true", help="raw Image 토픽 사용(기본 compressed)")
    ap.add_argument("--topic-ns", default="/camera/camera", help="카메라 네임스페이스")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    rclpy.init()
    node = RSView(args)
    win = "RealSense (r=녹화토글  q=종료)"
    period = 1.0 / max(1.0, args.fps)
    try:
        while rclpy.ok():
            t0 = time.perf_counter()
            rclpy.spin_once(node, timeout_sec=0.0)
            node.write_if_recording()
            if not args.no_window and node.frame is not None:
                disp = node.frame.copy()
                if node.recording:
                    cv2.circle(disp, (18, 18), 8, (0, 0, 255), -1)
                    cv2.putText(disp, "REC", (32, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)
                cv2.imshow(win, disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    node.stop_rec() if node.recording else node.start_rec()
            else:
                # 창 없을 때 페이싱
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_rec()
        if not args.no_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
