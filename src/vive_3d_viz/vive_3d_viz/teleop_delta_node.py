"""Vive Tracker -> 6DoF teleop delta -> ROS2 (DDS, cross-PC LAN).

Subscribes to the vive viz pose topics and, while a per-arm clutch ("engage")
is held, streams the relative 6DoF motion of the tracker w.r.t. the pose it had
at the engage instant. The Franka control PC applies this delta to the robot EE
pose it captured at the same instant.

Delta convention (anchor-local relative transform  dT = A^-1 . T(t)):
  A = tracker pose at engage = (p0, q0),  T(t) = current = (p, q)
    dp  = R0^T (p - p0)            # translation in the anchor's local frame [m]
    dq  = q0^-1 (x) q              # rotation in the anchor's local frame
    rot = axis-angle(dq)           # rotvec [rad], |rot| = angle, dir = axis

Output: std_msgs/String JSON on /teleop/delta/<arm> (left|right), e.g.
  {"arm":"right","stamp":1780999999.123,"engaged":true,"valid":true,
   "pos":[dx,dy,dz],"rot":[rx,ry,rz],
   "abs_pos":[x,y,z],"abs_quat":[qx,qy,qz,qw]}
pos/rot = anchor-local delta (see above). abs_pos/abs_quat = the latest tracker
pose in the vive_world frame (absolute), sent every tick regardless of engage so
the receiver can drive with the delta and use the absolute pose for
calibration/debug. When not engaged or the tracker pose is invalid/stale, pos/rot
are zeros so the receiver holds position.

Clutch: publish std_msgs/Bool to /teleop/engage/<arm> (true=engage, false=release).
  ros2 topic pub -1 /teleop/engage/right std_msgs/Bool "{data: true}"
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String


# ----------------------------- quaternion utils -----------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:  # q = [x, y, z, w]
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # Hamilton, [x,y,z,w]
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    if q[3] < 0.0:           # shortest path
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    v = q[:3]
    vn = float(np.linalg.norm(v))
    if vn < 1e-9:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vn, w)
    return (angle / vn) * v


# ------------------------------- per-arm state ------------------------------
class ArmState:
    def __init__(self) -> None:
        self.p: Optional[np.ndarray] = None      # latest position
        self.q: Optional[np.ndarray] = None      # latest orientation [x,y,z,w]
        self.last_rx: float = 0.0                # monotonic time of last pose
        self.valid_flag: bool = False            # /vive/<arm>/valid
        self.engaged: bool = False
        self.p0: Optional[np.ndarray] = None     # anchor position
        self.q0: Optional[np.ndarray] = None     # anchor orientation
        self.R0t: Optional[np.ndarray] = None    # cached R0^T


class TeleopDeltaNode(Node):
    def __init__(self) -> None:
        super().__init__('vive_teleop_delta')

        self.declare_parameter('arms', 'left,right')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('pos_scale', 1.0)
        self.declare_parameter('pose_timeout', 0.3)   # s; older -> invalid
        self.declare_parameter('reliable', True)      # delta pub QoS
        self.declare_parameter('auto_engage', False)  # engage on first valid pose

        self.arms: List[str] = [
            a.strip() for a in str(self.get_parameter('arms').value).split(',') if a.strip()
        ]
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.pos_scale = float(self.get_parameter('pos_scale').value)
        self.pose_timeout = float(self.get_parameter('pose_timeout').value)
        self.auto_engage = bool(self.get_parameter('auto_engage').value)
        reliable = bool(self.get_parameter('reliable').value)

        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,   # matches viz_node pose pub
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        delta_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.state: Dict[str, ArmState] = {}
        self.delta_pubs: Dict[str, rclpy.publisher.Publisher] = {}

        for arm in self.arms:
            st = ArmState()
            self.state[arm] = st
            self.create_subscription(
                PoseStamped, f'/vive/{arm}/pose',
                lambda msg, a=arm: self._on_pose(a, msg), pose_qos)
            self.create_subscription(
                Bool, f'/vive/{arm}/valid',
                lambda msg, a=arm: self._on_valid(a, msg), 1)
            self.create_subscription(
                Bool, f'/teleop/engage/{arm}',
                lambda msg, a=arm: self._on_engage(a, msg), 1)
            self.delta_pubs[arm] = self.create_publisher(
                String, f'/teleop/delta/{arm}', delta_qos)

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'teleop_delta up: arms={self.arms} rate={self.rate_hz:.0f}Hz '
            f'pos_scale={self.pos_scale} -> /teleop/delta/<arm> (JSON String). '
            f'Engage: ros2 topic pub -1 /teleop/engage/<arm> std_msgs/Bool "{{data: true}}"'
        )

    # ------------------------------ callbacks ------------------------------
    def _on_pose(self, arm: str, msg: PoseStamped) -> None:
        st = self.state[arm]
        st.p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        st.q = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                         msg.pose.orientation.z, msg.pose.orientation.w])
        st.last_rx = time.monotonic()
        if self.auto_engage and not st.engaged and self._pose_ok(st):
            self._engage(arm)

    def _on_valid(self, arm: str, msg: Bool) -> None:
        self.state[arm].valid_flag = bool(msg.data)

    def _on_engage(self, arm: str, msg: Bool) -> None:
        st = self.state[arm]
        if msg.data and not st.engaged:
            self._engage(arm)
        elif not msg.data and st.engaged:
            st.engaged = False
            self.get_logger().info(f'[{arm}] DISENGAGED')

    # ------------------------------- helpers -------------------------------
    def _pose_ok(self, st: ArmState) -> bool:
        return (st.p is not None and st.valid_flag
                and (time.monotonic() - st.last_rx) <= self.pose_timeout)

    def _engage(self, arm: str) -> None:
        st = self.state[arm]
        if not self._pose_ok(st):
            self.get_logger().warn(f'[{arm}] engage ignored: no valid tracker pose')
            return
        st.p0 = st.p.copy()
        st.q0 = quat_normalize(st.q.copy())
        st.R0t = quat_to_rotmat(st.q0).T
        st.engaged = True
        self.get_logger().info(f'[{arm}] ENGAGED (anchor captured)')

    def _tick(self) -> None:
        now = self.get_clock().now()
        stamp = now.nanoseconds * 1e-9
        for arm in self.arms:
            st = self.state[arm]
            ok = self._pose_ok(st)
            if st.engaged and ok and st.p0 is not None:
                dp = self.pos_scale * (st.R0t @ (st.p - st.p0))
                dq = quat_mul(quat_conj(st.q0), quat_normalize(st.q))
                rot = quat_to_rotvec(dq)
                pos_l = [float(v) for v in dp]
                rot_l = [float(v) for v in rot]
            else:
                pos_l = [0.0, 0.0, 0.0]
                rot_l = [0.0, 0.0, 0.0]
            if st.p is not None and st.q is not None:
                qn = quat_normalize(st.q)
                abs_pos_l = [float(v) for v in st.p]
                abs_quat_l = [float(v) for v in qn]
            else:
                abs_pos_l = [0.0, 0.0, 0.0]
                abs_quat_l = [0.0, 0.0, 0.0, 1.0]
            payload = {
                'arm': arm,
                'stamp': stamp,
                'engaged': bool(st.engaged),
                'valid': bool(ok),
                'pos': pos_l,
                'rot': rot_l,
                'abs_pos': abs_pos_l,
                'abs_quat': abs_quat_l,
            }
            self.delta_pubs[arm].publish(String(data=json.dumps(payload)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopDeltaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
