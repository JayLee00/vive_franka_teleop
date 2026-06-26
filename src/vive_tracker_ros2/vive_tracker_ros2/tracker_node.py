"""
HTC Vive Tracker → ROS2 publisher.

Reads pose from SteamVR via pyopenvr (wrapped by the vendored triad_openvr),
publishes `geometry_msgs/PoseStamped` per tracker and broadcasts TF.

Topics:
    /vive/<friendly_name>/pose       geometry_msgs/PoseStamped
    /vive/<friendly_name>/valid      std_msgs/Bool   (latched-ish, last value retained)

TF:
    parent = <vive_frame_id>  (default: "vive_world")
    child  = vive_<friendly_name>

Frame note:
    SteamVR world frame is right-handed, +Y up, +Z toward the user (room-scale origin).
    Hand-off to Franka base requires an external static transform (handeye calibration),
    which is NOT this node's responsibility.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('vive_tracker_node')

        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('frame_id', 'vive_world')
        # Comma-separated "SERIAL:friendly_name" pairs.
        # Example: "LHR-9AB9C3A4:right_hand,LHR-46A73216:left_hand"
        self.declare_parameter('tracker_name_map', '')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('require_trackers', True)

        self.rate_hz: float = float(self.get_parameter('rate_hz').value)
        self.frame_id: str = str(self.get_parameter('frame_id').value)
        self.publish_tf: bool = bool(self.get_parameter('publish_tf').value)
        require_trackers: bool = bool(self.get_parameter('require_trackers').value)

        raw_map = str(self.get_parameter('tracker_name_map').value or '')
        self.name_map: Dict[str, str] = {}
        for entry in raw_map.split(','):
            entry = entry.strip()
            if not entry:
                continue
            if ':' not in entry:
                self.get_logger().warn(f'Ignoring malformed tracker_name_map entry: {entry!r} (expected "SERIAL:name")')
                continue
            serial, friendly = entry.split(':', 1)
            self.name_map[serial.strip()] = friendly.strip()

        try:
            from . import _triad_openvr as triad_openvr
        except ImportError as e:
            self.get_logger().error(f'Failed to import triad_openvr: {e}')
            raise

        try:
            self.vr = triad_openvr.triad_openvr()
        except Exception as e:
            self.get_logger().error(
                f'triad_openvr init failed: {e}. '
                'Is SteamVR running and a Lighthouse + tracker visible?'
            )
            raise

        self.vr.print_discovered_objects()

        self._tracker_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._valid_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._friendly: Dict[str, str] = {}
        self._discover_trackers()

        if require_trackers and not self._tracker_pubs:
            self.get_logger().error(
                'No Vive trackers detected. Power them on, pair via SteamVR, '
                'and confirm green hexagons in the SteamVR overlay.'
            )
            raise RuntimeError('no trackers')

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self._last_event_poll = time.monotonic()

        self.get_logger().info(
            f'Publishing {len(self._tracker_pubs)} tracker(s) at {self.rate_hz:.1f} Hz '
            f'(frame_id="{self.frame_id}", tf={self.publish_tf})'
        )

    def _discover_trackers(self) -> None:
        for dev_name, device in self.vr.devices.items():
            if not dev_name.startswith('tracker_'):
                continue
            serial = device.get_serial()
            friendly = self.name_map.get(serial, dev_name)
            self._friendly[dev_name] = friendly

            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            topic_base = f'/vive/{friendly}'
            self._tracker_pubs[dev_name] = self.create_publisher(PoseStamped, f'{topic_base}/pose', qos)
            self._valid_pubs[dev_name] = self.create_publisher(Bool, f'{topic_base}/valid', 1)
            self.get_logger().info(f'  {dev_name} (serial={serial}) -> {topic_base}/pose')

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()

        if time.monotonic() - self._last_event_poll > 1.0:
            try:
                self.vr.poll_vr_events()
            except Exception as e:
                self.get_logger().warn(f'poll_vr_events failed: {e}', throttle_duration_sec=5.0)
            self._last_event_poll = time.monotonic()

        for dev_name, pub in self._tracker_pubs.items():
            device = self.vr.devices.get(dev_name)
            if device is None:
                continue
            pose7: Optional[list] = device.get_pose_quaternion()
            valid_msg = Bool()
            valid_msg.data = pose7 is not None
            self._valid_pubs[dev_name].publish(valid_msg)
            if pose7 is None:
                continue

            x, y, z, qw, qx, qy, qz = pose7

            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)
            msg.pose.orientation.x = float(qx)
            msg.pose.orientation.y = float(qy)
            msg.pose.orientation.z = float(qz)
            msg.pose.orientation.w = float(qw)
            pub.publish(msg)

            if self.tf_broadcaster is not None:
                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = self.frame_id
                t.child_frame_id = f'vive_{self._friendly[dev_name]}'
                t.transform.translation.x = float(x)
                t.transform.translation.y = float(y)
                t.transform.translation.z = float(z)
                t.transform.rotation.x = float(qx)
                t.transform.rotation.y = float(qy)
                t.transform.rotation.z = float(qz)
                t.transform.rotation.w = float(qw)
                self.tf_broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = TrackerNode()
    except Exception as e:
        print(f'[vive_tracker_node] startup failed: {e}', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
