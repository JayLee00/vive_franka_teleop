#!/usr/bin/env python3
"""트래커 절대 포즈 + 델타 포즈 실시간 표시.

팔별로 두 줄:
  1줄 절대(WORLD): /vive/<arm>/pose 기준 pos[m] + RPY/quat/axis-angle.
  2줄 델타(Δ):     /teleop/delta/<arm> 기준 engage anchor 상대 pos[m] + rot[deg] (+engaged 표시).
                   teleop_delta 노드가 안 떠 있으면 "(no delta)".
실행:
  source /opt/ros/humble/setup.bash && source /home/js/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 \
         FASTRTPS_DEFAULT_PROFILES_FILE=/home/js/fastdds_lan_only.xml
  python3 /home/js/tracker_read.py
"""
import json
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

ARMS = ['left', 'right']
RAD2DEG = 180.0 / math.pi


def quat_to_euler_deg(x, y, z, w):
    """ZYX(roll-pitch-yaw) Euler [deg]. roll=x축, pitch=y축, yaw=z축."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG


def rotvec_to_quat(rv):
    """rotvec(axis-angle) [rx,ry,rz] (rad) -> 단위 쿼터니언 [x,y,z,w]."""
    ang = math.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
    if ang < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(ang / 2.0) / ang
    return (rv[0] * s, rv[1] * s, rv[2] * s, math.cos(ang / 2.0))


def quat_to_axisangle_deg(x, y, z, w):
    """단위쿼터니언 -> (회전각[deg], 회전축 단위벡터)."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return 0.0, (0.0, 0.0, 1.0)
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    vn = math.sqrt(x * x + y * y + z * z)
    angle = 2 * math.atan2(vn, w) * RAD2DEG
    axis = (x / vn, y / vn, z / vn) if vn > 1e-9 else (0.0, 0.0, 1.0)
    return angle, axis


class TrackerRead(Node):
    def __init__(self):
        super().__init__('tracker_read')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.last = {a: None for a in ARMS}
        self.last_delta = {a: None for a in ARMS}
        for a in ARMS:
            self.create_subscription(PoseStamped, f'/vive/{a}/pose',
                                     lambda m, arm=a: self._cb(arm, m), qos)
            # delta 송신은 RELIABLE 이지만 BEST_EFFORT 구독으로도 매칭됨
            self.create_subscription(String, f'/teleop/delta/{a}',
                                     lambda m, arm=a: self._delta_cb(arm, m), qos)
        self.tty = sys.stdout.isatty()
        self.create_timer(0.1, self._print)  # 10 Hz

    def _cb(self, arm, msg):
        self.last[arm] = msg.pose

    def _delta_cb(self, arm, msg):
        try:
            self.last_delta[arm] = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _fmt(self, arm):
        p = self.last[arm]
        if p is None:
            return f'[{arm:5s}] (no data)'
        px, py, pz = p.position.x, p.position.y, p.position.z
        qx, qy, qz, qw = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        r, pi, yw = quat_to_euler_deg(qx, qy, qz, qw)
        ang, _ = quat_to_axisangle_deg(qx, qy, qz, qw)
        return (f'[{arm:5s}] pos[m] x={px:+.3f} y={py:+.3f} z={pz:+.3f}  | '
                f'RPY[deg] r={r:+6.1f} p={pi:+6.1f} y={yw:+6.1f}  | '
                f'quat {qx:+.3f},{qy:+.3f},{qz:+.3f},{qw:+.3f}  | '
                f'angle {ang:5.1f}deg')

    def _fmt_delta(self, arm):
        d = self.last_delta[arm]
        if d is None:
            return f'        Δ (no delta — /teleop/delta/{arm} 수신 없음)'
        eng = '✓' if d.get('engaged') else '·'
        dp = d.get('pos', [0.0, 0.0, 0.0])
        dr = d.get('rot', [0.0, 0.0, 0.0])
        r, pi, yw = quat_to_euler_deg(*rotvec_to_quat(dr))
        pmag = math.sqrt(sum(v * v for v in dp))
        amag = math.sqrt(sum(v * v for v in dr)) * RAD2DEG
        return (f'        Δ[eng {eng}] pos[m] x={dp[0]:+.3f} y={dp[1]:+.3f} z={dp[2]:+.3f}  | '
                f'rot[deg] r={r:+6.1f} p={pi:+6.1f} y={yw:+6.1f}  | '
                f'|Δp|={pmag:.3f}m |Δθ|={amag:5.1f}deg')

    def _print(self):
        body = '\n'.join(self._fmt(a) + '\n' + self._fmt_delta(a) for a in ARMS)
        if self.tty:
            sys.stdout.write('\033[H\033[J' + body + '\n')  # 화면 홈 + 클리어
            sys.stdout.flush()
        else:
            print(body, flush=True)


def main():
    rclpy.init()
    node = TrackerRead()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
