"""
Mock Vive tracker publisher — no SteamVR required.

Publishes a slowly-orbiting PoseStamped so downstream Franka pipelines can be
developed/validated without the real hardware.
"""

from __future__ import annotations

import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class MockTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('vive_mock_tracker_node')

        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('frame_id', 'vive_world')
        self.declare_parameter('tracker_name', 'mock_tracker')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('center_xyz', [0.4, 0.0, 0.5])
        self.declare_parameter('orbit_radius', 0.1)
        self.declare_parameter('orbit_period_s', 6.0)

        rate = float(self.get_parameter('rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.tracker_name = str(self.get_parameter('tracker_name').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.center = list(self.get_parameter('center_xyz').value)
        self.radius = float(self.get_parameter('orbit_radius').value)
        self.period = float(self.get_parameter('orbit_period_s').value)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pose_pub = self.create_publisher(PoseStamped, f'/vive/{self.tracker_name}/pose', qos)
        self.valid_pub = self.create_publisher(Bool, f'/vive/{self.tracker_name}/valid', 1)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.start = self.get_clock().now()
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'Mock tracker publishing /vive/{self.tracker_name}/pose at {rate:.1f} Hz '
            f'(center={self.center}, r={self.radius}, T={self.period}s)'
        )

    def _tick(self) -> None:
        now = self.get_clock().now()
        t = (now.nanoseconds - self.start.nanoseconds) * 1e-9
        omega = 2.0 * math.pi / max(self.period, 1e-3)
        cx, cy, cz = self.center
        x = cx + self.radius * math.cos(omega * t)
        y = cy + self.radius * math.sin(omega * t)
        z = cz

        yaw = omega * t
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        stamp = now.to_msg()
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)

        self.valid_pub.publish(Bool(data=True))

        if self.tf_broadcaster is not None:
            tr = TransformStamped()
            tr.header.stamp = stamp
            tr.header.frame_id = self.frame_id
            tr.child_frame_id = f'vive_{self.tracker_name}'
            tr.transform.translation.x = x
            tr.transform.translation.y = y
            tr.transform.translation.z = z
            tr.transform.rotation.z = qz
            tr.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
