"""Vive Tracker 3.0 + Lighthouse base stations → ROS2 3D visualization.

Reads pose from SteamVR via pyopenvr (vendored triad_openvr helper).
For every tracked device (trackers + tracking_references = base stations) publishes:
  - PoseStamped on /vive/<friendly>/pose
  - TF: parent vive_world → child vive_<friendly>
  - Marker on /vive/markers (MESH_RESOURCE-style shape + TEXT_VIEW_FACING label)

Frame note: SteamVR world is right-handed, +Y up. RViz default is +Z up; the included
RViz config rotates the view so the room looks natural. Frame data is unchanged.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


TRACKER_COLOR = ColorRGBA(r=0.20, g=0.80, b=0.30, a=0.95)
LIGHTHOUSE_COLOR = ColorRGBA(r=0.10, g=0.10, b=0.10, a=0.90)
LABEL_COLOR = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


class VizNode(Node):
    def __init__(self) -> None:
        super().__init__('vive_3d_viz_node')

        self.declare_parameter('rate_hz', 60.0)
        self.declare_parameter('frame_id', 'vive_world')
        self.declare_parameter('tracker_name_map', '')  # "SERIAL:friendly,SERIAL:friendly"
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('require_devices', True)

        self.rate_hz: float = float(self.get_parameter('rate_hz').value)
        self.frame_id: str = str(self.get_parameter('frame_id').value)
        self.publish_tf: bool = bool(self.get_parameter('publish_tf').value)
        require_devices: bool = bool(self.get_parameter('require_devices').value)

        raw_map = str(self.get_parameter('tracker_name_map').value or '')
        self.name_map: Dict[str, str] = {}
        for entry in raw_map.split(','):
            entry = entry.strip()
            if not entry or ':' not in entry:
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
                f'triad_openvr init failed: {e}. Is SteamVR running '
                'and at least one Lighthouse + tracker visible?'
            )
            raise

        self.vr.print_discovered_objects()

        self._pose_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._valid_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._friendly: Dict[str, str] = {}
        self._device_kind: Dict[str, str] = {}  # 'tracker' or 'lighthouse'
        self._discover()

        if require_devices and not self._pose_pubs:
            self.get_logger().error(
                'No trackers or base stations detected. Power them on, pair via SteamVR, '
                'and confirm in the SteamVR overlay.'
            )
            raise RuntimeError('no devices')

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
        )
        self.marker_pub = self.create_publisher(MarkerArray, '/vive/markers', marker_qos)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self._last_event_poll = time.monotonic()

        n_tr = sum(1 for k in self._device_kind.values() if k == 'tracker')
        n_lh = sum(1 for k in self._device_kind.values() if k == 'lighthouse')
        self.get_logger().info(
            f'Publishing {n_tr} tracker(s) + {n_lh} lighthouse(s) at {self.rate_hz:.1f} Hz '
            f'(frame_id="{self.frame_id}", tf={self.publish_tf})'
        )

    def _discover(self) -> None:
        for dev_name, device in self.vr.devices.items():
            if dev_name.startswith('tracker_'):
                kind = 'tracker'
            elif dev_name.startswith('tracking_reference_'):
                kind = 'lighthouse'
            else:
                continue  # skip HMD / controllers for this viz

            try:
                serial = device.get_serial()
            except Exception:
                serial = dev_name

            if kind == 'tracker':
                friendly = self.name_map.get(serial, dev_name)
            else:
                # base stations are typically "LHB-..." serials
                friendly = self.name_map.get(serial, dev_name)

            self._friendly[dev_name] = friendly
            self._device_kind[dev_name] = kind

            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            topic_base = f'/vive/{friendly}'
            self._pose_pubs[dev_name] = self.create_publisher(PoseStamped, f'{topic_base}/pose', qos)
            self._valid_pubs[dev_name] = self.create_publisher(Bool, f'{topic_base}/valid', 1)
            self.get_logger().info(
                f'  [{kind:10s}] {dev_name} (serial={serial}) -> {topic_base}/pose'
            )

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()

        if time.monotonic() - self._last_event_poll > 1.0:
            try:
                self.vr.poll_vr_events()
            except Exception as e:
                self.get_logger().warn(f'poll_vr_events failed: {e}', throttle_duration_sec=5.0)
            self._last_event_poll = time.monotonic()

        markers = MarkerArray()
        marker_id = 0

        for dev_name, pose_pub in self._pose_pubs.items():
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
            friendly = self._friendly[dev_name]
            kind = self._device_kind[dev_name]

            # PoseStamped
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            pose_pub.publish(ps)

            # TF
            if self.tf_broadcaster is not None:
                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = self.frame_id
                t.child_frame_id = f'vive_{friendly}'
                t.transform.translation.x = float(x)
                t.transform.translation.y = float(y)
                t.transform.translation.z = float(z)
                t.transform.rotation.x = float(qx)
                t.transform.rotation.y = float(qy)
                t.transform.rotation.z = float(qz)
                t.transform.rotation.w = float(qw)
                self.tf_broadcaster.sendTransform(t)

            # Markers (shape + label)
            shape, label = self._build_markers(
                kind=kind,
                friendly=friendly,
                stamp=now,
                pos=(float(x), float(y), float(z)),
                quat=(float(qx), float(qy), float(qz), float(qw)),
                marker_id=marker_id,
            )
            markers.markers.append(shape)
            markers.markers.append(label)
            marker_id += 2

        if markers.markers:
            self.marker_pub.publish(markers)

    def _build_markers(self, *, kind, friendly, stamp, pos, quat, marker_id):
        x, y, z = pos
        qx, qy, qz, qw = quat

        shape = Marker()
        shape.header.stamp = stamp
        shape.header.frame_id = self.frame_id
        shape.ns = f'vive_{kind}'
        shape.id = marker_id
        shape.action = Marker.ADD
        shape.pose.position.x = x
        shape.pose.position.y = y
        shape.pose.position.z = z
        shape.pose.orientation.x = qx
        shape.pose.orientation.y = qy
        shape.pose.orientation.z = qz
        shape.pose.orientation.w = qw

        if kind == 'lighthouse':
            # Vive Lighthouse 2.0 is ~7.7 x 7.7 x 8.6 cm, render slightly larger for visibility
            shape.type = Marker.CUBE
            shape.scale.x = 0.085
            shape.scale.y = 0.085
            shape.scale.z = 0.095
            shape.color = LIGHTHOUSE_COLOR
        else:
            # Vive Tracker 3.0 is a flat disk ~7 cm diameter, 1 cm thick
            shape.type = Marker.CYLINDER
            shape.scale.x = 0.072
            shape.scale.y = 0.072
            shape.scale.z = 0.015
            shape.color = TRACKER_COLOR

        label = Marker()
        label.header.stamp = stamp
        label.header.frame_id = self.frame_id
        label.ns = f'vive_{kind}_label'
        label.id = marker_id + 1
        label.action = Marker.ADD
        label.type = Marker.TEXT_VIEW_FACING
        label.pose.position.x = x
        label.pose.position.y = y
        label.pose.position.z = z + 0.10
        label.pose.orientation.w = 1.0
        label.scale.z = 0.05  # text height
        label.color = LABEL_COLOR
        label.text = friendly

        return shape, label


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = VizNode()
    except Exception as e:
        print(f'[vive_3d_viz_node] startup failed: {e}', file=sys.stderr)
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
