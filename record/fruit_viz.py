#!/usr/bin/env python3
"""과일 위치/크기 실시간 시각화.

fruit_pose_bridge.py 가 내는:
    /fruit/pose  (geometry_msgs/PoseStamped)   위치[m] + 방향(quat)
    /fruit/size  (std_msgs/Float32MultiArray)  [a, b, c] 장축..단축 [m]
를 구독해 matplotlib 창에 실시간 표시:
    왼쪽  3D : 과일 중심 위치(카메라 광학 프레임) + 최근 경로 trail
    오른쪽   : 크기 a/b/c 막대 + 현재 위치/크기/수신율 텍스트

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/fruit_viz.py
  python3 record/fruit_viz.py --selftest   # ROS 없이 가짜 데이터로 렌더 확인
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TRAIL = 60


class State:
    def __init__(self):
        self.pos = None            # (x,y,z) m
        self.quat = None           # (x,y,z,w)
        self.size = None           # [a,b,c] m
        self.frame = "-"
        self.trail = deque(maxlen=TRAIL)
        self.last_rx = 0.0
        self.rate = 0.0
        self._prev_t = None

    def update_pose(self, x, y, z, q, frame, t):
        self.pos = (x, y, z); self.quat = q; self.frame = frame
        self.trail.append((x, y, z)); self.last_rx = t
        if self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-6:
                inst = 1.0 / dt
                self.rate = inst if self.rate == 0 else 0.9 * self.rate + 0.1 * inst
        self._prev_t = t


def build_fig(st: State):
    fig = plt.figure(figsize=(11, 5))
    fig.canvas.manager.set_window_title("Fruit pose/size (camera frame)")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axb = fig.add_subplot(1, 2, 2)

    def update(_):
        ax3d.clear(); axb.clear()
        fresh = (time.time() - st.last_rx) < 1.0 and st.pos is not None
        # 3D 위치 + trail
        ax3d.set_title("fruit position [m]" + ("" if fresh else "  (no data)"))
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        if st.trail:
            t = np.array(st.trail)
            ax3d.plot(t[:, 0], t[:, 1], t[:, 2], "-", color="tab:orange", alpha=0.5, lw=1)
        if st.pos is not None:
            c = "tab:red" if fresh else "gray"
            ax3d.scatter(*st.pos, s=120, color=c)
            r = 0.05
            ax3d.set_xlim(st.pos[0]-0.3, st.pos[0]+0.3)
            ax3d.set_ylim(st.pos[1]-0.3, st.pos[1]+0.3)
            ax3d.set_zlim(max(0, st.pos[2]-0.3), st.pos[2]+0.3)
        # 크기 막대
        axb.set_title("size a/b/c [m]")
        if st.size:
            names = ["a(major)", "b", "c(minor)"][:len(st.size)]
            axb.bar(names, st.size[:3], color=["tab:blue", "tab:cyan", "tab:green"][:len(st.size)])
            axb.set_ylim(0, max(0.12, max(st.size[:3]) * 1.2))
        axb.set_ylabel("m")
        # 텍스트
        if fresh:
            x, y, z = st.pos
            sz = "/".join(f"{v:.3f}" for v in (st.size or [])[:3]) or "-"
            txt = (f"pos = ({x:+.3f}, {y:+.3f}, {z:+.3f}) m\n"
                   f"size a/b/c = {sz} m\n"
                   f"frame = {st.frame}\n"
                   f"rate = {st.rate:4.1f} Hz")
        else:
            txt = "no data\n(check fruit_pose_bridge / live_bbox_gui)"
        axb.text(0.02, -0.28, txt, transform=axb.transAxes, family="monospace",
                 va="top", fontsize=10)
        fig.tight_layout()

    return fig, update


def run_ros(st: State):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32MultiArray
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    rclpy.init()
    node = Node("fruit_viz")

    def on_pose(m: PoseStamped):
        p = m.pose.position; o = m.pose.orientation
        st.update_pose(p.x, p.y, p.z, (o.x, o.y, o.z, o.w),
                       m.header.frame_id or "-", time.time())

    def on_size(m: Float32MultiArray):
        st.size = list(m.data)

    node.create_subscription(PoseStamped, "/fruit/pose", on_pose, qos)
    node.create_subscription(Float32MultiArray, "/fruit/size", on_size, qos)

    fig, update = build_fig(st)

    def tick(frame):
        rclpy.spin_once(node, timeout_sec=0.0)
        update(frame)

    ani = FuncAnimation(fig, tick, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def selftest():
    matplotlib.use("Agg")
    st = State()
    fig, update = build_fig(st)
    for i in range(20):
        a = i * 0.2
        st.update_pose(0.1*math.cos(a), 0.1*math.sin(a), 0.3+0.02*math.sin(a),
                       (0, 0, 0, 1), "camera_color_optical_frame", time.time())
        st.size = [0.075, 0.070, 0.065]
        update(i)
    out = "/tmp/fruit_viz_selftest.png"
    fig.savefig(out)
    print(f"selftest OK — 렌더 정상, 저장: {out}")


def main():
    ap = argparse.ArgumentParser(description="과일 위치/크기 실시간 시각화")
    ap.add_argument("--selftest", action="store_true", help="ROS 없이 가짜 데이터로 렌더 확인")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        run_ros(State())


if __name__ == "__main__":
    main()
