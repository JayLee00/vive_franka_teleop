#!/usr/bin/env python3
"""왼팔/오른팔 EE 모니터 — 우리가 쏘는 ee_target 과 제어PC 실제 ee_pose 를 실시간 표로.

  TARGET R/L : /franka_{r,l}/ee_target_world   (우리가 발행, engage 중에만 옴)
  EEPOSE R/L : /franka/ee_pose_{r,l}           (제어PC 실제 EE, 200Hz)

flag:
  ⚠ZERO  = 위치가 원점[0,0,0] 근처 → 재engage 점프 원인 (제어PC EE가 0을 뱉는 중)
  ⚠STALE = 0.5s 넘게 수신 없음 (TARGET 은 disengage 면 정상적으로 STALE)

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
  python3 ~/Desktop/vive_franka_teleop/scripts/ee_monitor.py
"""
import sys
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped

TOPICS = [
    ('TARGET R', '/franka_r/ee_target_world'),
    ('TARGET L', '/franka_l/ee_target_world'),
    ('EEPOSE R', '/franka/ee_pose_r'),
    ('EEPOSE L', '/franka/ee_pose_l'),
]


class EEMonitor(Node):
    def __init__(self) -> None:
        super().__init__('ee_monitor')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,   # 어떤 pub 과도 매칭
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.data = {label: (None, -1.0, 0) for label, _ in TOPICS}   # label -> (pos, t_recv, count)
        for label, topic in TOPICS:
            self.create_subscription(PoseStamped, topic,
                                     lambda m, l=label: self._cb(l, m), qos)
        self.first = True
        self.create_timer(0.1, self._render)

    def _cb(self, label: str, msg: PoseStamped) -> None:
        p = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        _, _, c = self.data[label]
        self.data[label] = (p, time.monotonic(), c + 1)

    def _render(self) -> None:
        now = time.monotonic()
        lines = ['=== EE Monitor (DOMAIN 9)   [Ctrl-C 종료] ===',
                 f"{'':10s}{'x':>9s}{'y':>9s}{'z':>9s}{'age':>8s}   flag"]
        for label, _ in TOPICS:
            p, t, c = self.data[label]
            if p is None:
                lines.append(f"{label:10s}{'--':>9s}{'--':>9s}{'--':>9s}{'--':>8s}   수신없음")
                continue
            age = now - t
            if age > 0.5:
                flag = '⚠STALE'
            elif math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) < 0.05:
                flag = '⚠ZERO'
            else:
                flag = 'ok'
            lines.append(f"{label:10s}{p[0]:9.3f}{p[1]:9.3f}{p[2]:9.3f}{age:7.2f}s   {flag}  (#{c})")
        lines.append('TARGET=우리가 쏨(engage중만) / EEPOSE=제어PC 실제. ⚠ZERO=원점근처=재engage점프원인')
        out = ''.join(ln + '\033[K\n' for ln in lines)
        if not self.first:
            out = f'\033[{len(lines)}F' + out
        sys.stdout.write(out)
        sys.stdout.flush()
        self.first = False


def main() -> None:
    rclpy.init()
    node = EEMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print()


if __name__ == '__main__':
    main()
