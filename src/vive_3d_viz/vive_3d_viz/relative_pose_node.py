"""Vive Tracker engage-기준 상대 포즈 노드 (ROS2 Humble).

engage(클러치) 신호 True 엣지에서 World 기준 트래커 포즈 ``T_W_E`` 를 latch 하고,
이후 매 프레임 ``T_E_O = inv(T_W_E) @ T_W_O`` (Engage 에서 본 현재 트래커 포즈)를
PoseStamped + TF 로 발행한다. 수학 코어는 ROS 의존성 없는 ``relative_pose.py``.

컨벤션: ``T_A_B`` = A 프레임에서 본 B 의 포즈 (relative_pose.py 참고).

토픽:
  입력  <pose_topic>     geometry_msgs/PoseStamped  World 기준 트래커 절대포즈
        <engage_topic>   std_msgs/Bool              True=engage latch, False=해제
  출력  <output_topic>   geometry_msgs/PoseStamped  frame_id=<engage_frame>, = T_E_O
        <timeout_topic>  std_msgs/Bool              트래킹 로스(포즈 timeout) 여부
  TF    <world_frame> -> <engage_frame>            (static, latch 시 1회)
        <engage_frame> -> <relative_child_frame>   (dynamic, 매 프레임)

엣지케이스:
  - engage 전: 기본은 미발행(param ``publish_before_engage:=true`` 면 identity 발행)
  - 포즈 timeout(기본 0.1s): 마지막 유효 T_E_O 유지 + warning, timeout Bool=true
  - 재-engage(True 엣지 재발생): T_W_E 재latch
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from vive_3d_viz.relative_pose import (
    pose_to_matrix, matrix_to_pose, relative_pose, ensure_quat_continuity,
)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


class RelativePoseNode(Node):
    def __init__(self) -> None:
        super().__init__('vive_relative_pose')

        # ---- 파라미터 ----
        self.declare_parameter('pose_topic', '/vive/right/pose')
        self.declare_parameter('engage_topic', '/teleop/engage/right')
        self.declare_parameter('output_topic', '/vive/relative_pose')
        self.declare_parameter('timeout_topic', '/vive/relative_pose/timeout')
        self.declare_parameter('world_frame', 'vive_world')
        self.declare_parameter('engage_frame', 'engage_frame')
        self.declare_parameter('relative_child_frame', 'tracker_relative')
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('pose_timeout', 0.1)
        self.declare_parameter('publish_before_engage', False)

        gp = self.get_parameter
        self.pose_topic = str(gp('pose_topic').value)
        self.engage_topic = str(gp('engage_topic').value)
        self.output_topic = str(gp('output_topic').value)
        self.timeout_topic = str(gp('timeout_topic').value)
        self.world_frame = str(gp('world_frame').value)
        self.engage_frame = str(gp('engage_frame').value)
        self.relative_child_frame = str(gp('relative_child_frame').value)
        self.publish_rate = float(gp('publish_rate').value)
        self.pose_timeout = float(gp('pose_timeout').value)
        self.publish_before_engage = bool(gp('publish_before_engage').value)

        # ---- 상태 ----
        self._T_W_E: Optional[np.ndarray] = None   # latch 된 engage 변환 (4x4)
        self._engaged = False
        self._prev_engage_cmd = False              # True 엣지 검출용
        self._latest_p: Optional[np.ndarray] = None
        self._latest_q: Optional[np.ndarray] = None
        self._latest_rx: Optional[float] = None    # 최신 포즈 수신 monotonic time
        self._prev_q_rel: Optional[np.ndarray] = None   # 직전 출력 quat (연속성)
        self._last_pose_msg: Optional[PoseStamped] = None  # timeout 시 hold 용
        self._timeout_state: Optional[bool] = None  # 마지막으로 발행한 timeout 상태

        # ---- QoS: 입력 포즈는 BEST_EFFORT(viz pose pub과 매칭), 출력은 RELIABLE ----
        pose_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        out_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(PoseStamped, self.pose_topic, self._on_pose, pose_qos)
        self.create_subscription(Bool, self.engage_topic, self._on_engage, 1)
        self.pose_pub = self.create_publisher(PoseStamped, self.output_topic, out_qos)
        self.timeout_pub = self.create_publisher(Bool, self.timeout_topic, 1)
        self.tf_bc = TransformBroadcaster(self)
        self.static_bc = StaticTransformBroadcaster(self)

        self.timer = self.create_timer(1.0 / self.publish_rate, self._tick)
        self.get_logger().info(
            f'relative_pose up: pose<{self.pose_topic}> engage<{self.engage_topic}> '
            f'-> {self.output_topic} (frame={self.engage_frame}) @ {self.publish_rate:.0f}Hz'
        )

    # ------------------------------ 콜백 ------------------------------
    def _on_pose(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self._latest_p = np.array([p.x, p.y, p.z], dtype=float)
        self._latest_q = _normalize_quat(np.array([o.x, o.y, o.z, o.w], dtype=float))
        self._latest_rx = time.monotonic()

    def _on_engage(self, msg: Bool) -> None:
        cmd = bool(msg.data)
        rising = cmd and not self._prev_engage_cmd
        self._prev_engage_cmd = cmd
        if rising:
            self._latch_engage()              # True 엣지 -> (재)latch
        elif (not cmd) and self._engaged:
            self._engaged = False
            self.get_logger().info('DISENGAGED')

    # ------------------------------ 핵심 ------------------------------
    def _fresh(self) -> bool:
        return (self._latest_rx is not None
                and (time.monotonic() - self._latest_rx) <= self.pose_timeout)

    def _latch_engage(self) -> None:
        if self._latest_p is None or not self._fresh():
            self.get_logger().warn('engage ignored: 최신 트래커 포즈 없음 (latch 불가)')
            return
        self._T_W_E = pose_to_matrix(self._latest_p, self._latest_q)
        self._engaged = True
        self._prev_q_rel = None              # 연속성 리셋
        self._last_pose_msg = None
        self._broadcast_static_engage()
        self.get_logger().info('ENGAGED: T_W_E latch 완료 (engage_frame 고정)')

    def _tick(self) -> None:
        stamp = self.get_clock().now().to_msg()
        stale = not self._fresh()
        self._publish_timeout(stale)         # 트래킹 로스 신호 (engage 무관, 안전용)

        if not self._engaged or self._T_W_E is None:
            if self.publish_before_engage:
                self._publish_identity(stamp)
            return

        if stale:
            self.get_logger().warn('tracker pose timeout: 마지막 상대포즈 유지',
                                   throttle_duration_sec=1.0)
            if self._last_pose_msg is not None:
                self._last_pose_msg.header.stamp = stamp
                self.pose_pub.publish(self._last_pose_msg)
                self._broadcast_dynamic(self._last_pose_msg, stamp)
            return

        # 정상: T_E_O = inv(T_W_E) @ T_W_O
        T_W_O = pose_to_matrix(self._latest_p, self._latest_q)
        T_E_O = relative_pose(self._T_W_E, T_W_O)
        p, q = matrix_to_pose(T_E_O)
        q = ensure_quat_continuity(q, self._prev_q_rel)
        self._prev_q_rel = q

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.engage_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(p[0]), float(p[1]), float(p[2]))
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self.pose_pub.publish(ps)
        self._last_pose_msg = ps
        self._broadcast_dynamic(ps, stamp)

    # ------------------------------ TF / 보조 ------------------------------
    def _broadcast_static_engage(self) -> None:
        p, q = matrix_to_pose(self._T_W_E)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.engage_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(p[0]), float(p[1]), float(p[2]))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (
            float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        self.static_bc.sendTransform(t)

    def _broadcast_dynamic(self, ps: PoseStamped, stamp) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.engage_frame
        t.child_frame_id = self.relative_child_frame
        t.transform.translation.x = ps.pose.position.x
        t.transform.translation.y = ps.pose.position.y
        t.transform.translation.z = ps.pose.position.z
        t.transform.rotation = ps.pose.orientation
        self.tf_bc.sendTransform(t)

    def _publish_identity(self, stamp) -> None:
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.engage_frame
        ps.pose.orientation.w = 1.0
        self.pose_pub.publish(ps)

    def _publish_timeout(self, active: bool) -> None:
        active = bool(active)
        if self._timeout_state is None or active != self._timeout_state:
            self._timeout_state = active
            self.timeout_pub.publish(Bool(data=active))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RelativePoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
