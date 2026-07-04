"""Vive Tracker -> 6DoF teleop delta -> ROS2, and -> Franka absolute EE target.

Subscribes to the vive viz pose topics and, while a per-arm clutch ("engage")
is held, streams the relative 6DoF motion of the tracker w.r.t. the pose it had
at the engage instant.

Two outputs:
  (a) std_msgs/String JSON on /teleop/delta/<arm> (left|right) — the raw delta,
      for local monitors / debug (see payload below).
  (b) geometry_msgs/PoseStamped on /franka_<r|l>/ee_target_world — the ABSOLUTE
      EE target the new control PC wants (see docs/VIVE_PC_HANDOFF.md). The delta
      is computed here (we generate it), so we convert to the absolute target in
      the same node instead of round-tripping the JSON through ROS2.

Delta convention (anchor-local relative transform  dT = A^-1 . T(t)):
  A = tracker pose at engage = (p0, q0),  T(t) = current = (p, q)
    dp  = R0^T (p - p0)            # translation in the anchor's local frame [m]
    dq  = q0^-1 (x) q              # rotation in the anchor's local frame
    rot = axis-angle(dq)           # rotvec [rad], |rot| = angle, dir = axis

Absolute EE target (control-PC contract). On the engage rising edge we also
capture the current real EE pose T_ee0 = (p_ee0, R_ee0) from /franka/ee_pose_<r|l>:
    R_rel = R_align . dR . R_align^T          # tracker rotation -> robot frame
    p_rel = ee_scale . (R_align . dp_raw)     # tracker translation -> robot frame [m]
    R_target = R_ee0 . R_rel
    p_target = p_ee0 + R_ee0 . p_rel
dp_raw = R0^T (p - p0) (the un-scaled anchor-local translation). R_align aligns the
"tracker held" orientation to the robot frame (per arm, decided once). While not
engaged or the tracker pose is invalid/stale, the EE target is NOT published — the
control PC holds its last target.

JSON payload on /teleop/delta/<arm> (unchanged):
  {"arm":"right","stamp":..,"engaged":true,"valid":true,
   "pos":[dx,dy,dz],"rot":[rx,ry,rz],"abs_pos":[..],"abs_quat":[..]}

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


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:  # -> [x, y, z, w]
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


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
        self.p: Optional[np.ndarray] = None      # latest tracker position
        self.q: Optional[np.ndarray] = None      # latest tracker orientation [x,y,z,w]
        self.last_rx: float = 0.0                # monotonic time of last pose
        self.valid_flag: bool = False            # /vive/<arm>/valid
        self.engaged: bool = False
        self.p0: Optional[np.ndarray] = None     # tracker anchor position
        self.q0: Optional[np.ndarray] = None     # tracker anchor orientation
        self.R0t: Optional[np.ndarray] = None    # cached R0^T
        # --- Franka EE (for absolute target) ---
        self.p_ee: Optional[np.ndarray] = None   # latest real EE position
        self.R_ee: Optional[np.ndarray] = None   # latest real EE rotation
        self.ee_rx: float = -1.0                 # monotonic time of last EE pose
        self.p_ee0: Optional[np.ndarray] = None  # EE anchor position (at engage)
        self.R_ee0: Optional[np.ndarray] = None  # EE anchor rotation (at engage)


class TeleopDeltaNode(Node):
    ARM_SUFFIX = {'right': 'r', 'left': 'l', 'r': 'r', 'l': 'l'}

    def __init__(self) -> None:
        super().__init__('vive_teleop_delta')

        self.declare_parameter('arms', 'left,right')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('pos_scale', 1.0)
        self.declare_parameter('pose_timeout', 0.3)   # s; older -> invalid
        self.declare_parameter('reliable', True)      # delta pub QoS
        self.declare_parameter('auto_engage', False)  # engage on first valid pose
        # --- absolute EE target (control-PC contract) ---
        self.declare_parameter('publish_ee_target', True)
        self.declare_parameter('ee_scale', 0.3)       # translation scale tracker->robot
        self.declare_parameter('ee_timeout', 0.5)     # s; EE pose older -> no anchor
        self.declare_parameter('r_align_right', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('r_align_left', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

        self.arms: List[str] = [
            a.strip() for a in str(self.get_parameter('arms').value).split(',') if a.strip()
        ]
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.pos_scale = float(self.get_parameter('pos_scale').value)
        self.pose_timeout = float(self.get_parameter('pose_timeout').value)
        self.auto_engage = bool(self.get_parameter('auto_engage').value)
        reliable = bool(self.get_parameter('reliable').value)
        self.publish_ee_target = bool(self.get_parameter('publish_ee_target').value)
        self.ee_scale = float(self.get_parameter('ee_scale').value)
        self.ee_timeout = float(self.get_parameter('ee_timeout').value)

        self.r_align: Dict[str, np.ndarray] = {}
        for arm in self.arms:
            key = 'r_align_right' if self.ARM_SUFFIX.get(arm) == 'r' else 'r_align_left'
            ra = list(self.get_parameter(key).value)
            if len(ra) != 9:
                raise ValueError(f'{key} must have 9 elements, got {len(ra)}')
            self.r_align[arm] = np.array(ra, dtype=float).reshape(3, 3)

        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,   # matches viz_node pose pub
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        ee_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,   # matches shm_state_publisher
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        delta_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        target_qos = QoSProfile(   # RELIABLE offer matches any subscriber
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.state: Dict[str, ArmState] = {}
        self.delta_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self.ee_target_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self.target_frame: Dict[str, str] = {}

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

            suffix = self.ARM_SUFFIX.get(arm)
            if self.publish_ee_target and suffix is not None:
                self.create_subscription(
                    PoseStamped, f'/franka/ee_pose_{suffix}',
                    lambda msg, a=arm: self._on_ee_pose(a, msg), ee_qos)
                self.ee_target_pubs[arm] = self.create_publisher(
                    PoseStamped, f'/franka_{suffix}/ee_target_world', target_qos)
                self.target_frame[arm] = f'fr3_link0_{suffix}'

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f'teleop_delta up: arms={self.arms} rate={self.rate_hz:.0f}Hz '
            f'pos_scale={self.pos_scale} ee_target={self.publish_ee_target} '
            f'ee_scale={self.ee_scale} -> /teleop/delta/<arm> (JSON) + '
            f'/franka_<r|l>/ee_target_world (PoseStamped). '
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
            st.p_ee0 = None
            st.R_ee0 = None
            self.get_logger().info(f'[{arm}] DISENGAGED')

    def _on_ee_pose(self, arm: str, msg: PoseStamped) -> None:
        st = self.state[arm]
        st.p_ee = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w])
        st.R_ee = quat_to_rotmat(quat_normalize(q))
        st.ee_rx = time.monotonic()

    # ------------------------------- helpers -------------------------------
    def _pose_ok(self, st: ArmState) -> bool:
        return (st.p is not None and st.valid_flag
                and (time.monotonic() - st.last_rx) <= self.pose_timeout)

    def _ee_ok(self, st: ArmState) -> bool:
        return (st.p_ee is not None and st.R_ee is not None
                and (time.monotonic() - st.ee_rx) <= self.ee_timeout)

    def _engage(self, arm: str) -> None:
        st = self.state[arm]
        if not self._pose_ok(st):
            self.get_logger().warn(f'[{arm}] engage ignored: no valid tracker pose')
            return
        st.p0 = st.p.copy()
        st.q0 = quat_normalize(st.q.copy())
        st.R0t = quat_to_rotmat(st.q0).T
        st.engaged = True
        # capture the real EE pose at the same instant as the absolute anchor
        if arm in self.ee_target_pubs:
            if self._ee_ok(st):
                st.p_ee0 = st.p_ee.copy()
                st.R_ee0 = st.R_ee.copy()
                self.get_logger().info(
                    f'[{arm}] ENGAGED (tracker + EE anchor @ {st.p_ee0.round(3)})')
            else:
                st.p_ee0 = None
                st.R_ee0 = None
                self.get_logger().warn(
                    f'[{arm}] ENGAGED but no fresh /franka/ee_pose_{self.ARM_SUFFIX[arm]}; '
                    f'EE target disabled until re-engage')
        else:
            self.get_logger().info(f'[{arm}] ENGAGED (anchor captured)')

    def _tick(self) -> None:
        now = self.get_clock().now()
        stamp = now.nanoseconds * 1e-9
        stamp_msg = now.to_msg()
        for arm in self.arms:
            st = self.state[arm]
            ok = self._pose_ok(st)
            if st.engaged and ok and st.p0 is not None:
                raw_dp = st.R0t @ (st.p - st.p0)              # un-scaled anchor-local
                dq = quat_mul(quat_conj(st.q0), quat_normalize(st.q))
                dp = self.pos_scale * raw_dp
                rot = quat_to_rotvec(dq)
                pos_l = [float(v) for v in dp]
                rot_l = [float(v) for v in rot]
                moving = True
            else:
                raw_dp = None
                dq = None
                pos_l = [0.0, 0.0, 0.0]
                rot_l = [0.0, 0.0, 0.0]
                moving = False
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

            # absolute EE target — only while engaged, valid, and EE-anchored
            if moving and arm in self.ee_target_pubs and st.p_ee0 is not None:
                dR = quat_to_rotmat(dq)
                R_align = self.r_align[arm]
                R_rel = R_align @ dR @ R_align.T
                p_rel = self.ee_scale * (R_align @ raw_dp)
                R_target = st.R_ee0 @ R_rel
                p_target = st.p_ee0 + st.R_ee0 @ p_rel
                q_target = rotmat_to_quat(R_target)
                out = PoseStamped()
                out.header.stamp = stamp_msg
                out.header.frame_id = self.target_frame[arm]
                out.pose.position.x = float(p_target[0])
                out.pose.position.y = float(p_target[1])
                out.pose.position.z = float(p_target[2])
                out.pose.orientation.x = float(q_target[0])
                out.pose.orientation.y = float(q_target[1])
                out.pose.orientation.z = float(q_target[2])
                out.pose.orientation.w = float(q_target[3])
                self.ee_target_pubs[arm].publish(out)


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
