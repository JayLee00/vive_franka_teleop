"""
HTC Vive Tracker → ROS2 publisher using libsurvive (NO SteamVR required).

Requires:
  - PYTHONPATH=/home/js/Desktop/vive_franka_teleop/vendor/libsurvive/bindings/python
  - LD_LIBRARY_PATH=/home/js/Desktop/vive_franka_teleop/vendor/libsurvive/bin
  - SteamVR NOT running (libsurvive needs exclusive USB access)

Topics:
    /survive/<friendly>/pose       geometry_msgs/PoseStamped
    /survive/<friendly>/valid      std_msgs/Bool

TF:
    parent = <frame_id>           (default "survive_world")
    child  = survive_<friendly>

Frame:
    libsurvive world frame is defined by the lighthouse positions (right-handed).
    For Franka integration, apply a separate static handeye transform.

Quaternion order from libsurvive PoseData.Rot: (w, x, y, z).
ROS2 Quaternion: (x, y, z, w). Reorder accordingly.
"""

from __future__ import annotations

import sys
import time
import threading
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class SurviveTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('survive_tracker_node')

        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('frame_id', 'survive_world')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('tracker_name_map', '')   # "SERIAL:name,SERIAL2:name2"
        self.declare_parameter('survive_argv', '')        # extra argv to libsurvive (e.g. "--steam-path=/x")

        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)

        raw_map = str(self.get_parameter('tracker_name_map').value or '')
        self.name_map: Dict[str, str] = {}
        for entry in raw_map.split(','):
            entry = entry.strip()
            if not entry or ':' not in entry:
                continue
            serial, friendly = entry.split(':', 1)
            self.name_map[serial.strip()] = friendly.strip()

        try:
            import pysurvive
        except ImportError as e:
            self.get_logger().error(
                f'pysurvive import failed: {e}. '
                f'Set PYTHONPATH to <libsurvive>/bindings/python and LD_LIBRARY_PATH to <libsurvive>/bin.'
            )
            raise

        self._pysurvive = pysurvive

        # Build argv for SimpleContext
        argv = ['survive_tracker_node']
        extra = str(self.get_parameter('survive_argv').value or '').strip()
        if extra:
            argv.extend(extra.split())
        self.get_logger().info(f'libsurvive argv: {argv}')

        try:
            self.ctx = pysurvive.SimpleContext(argv)
        except Exception as e:
            self.get_logger().error(
                f'SimpleContext init failed: {e}. '
                'Check that SteamVR is NOT running and dongles/trackers are on USB.'
            )
            raise

        self._pose_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._valid_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._friendly: Dict[str, str] = {}
        self._discover()

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        self.get_logger().info(
            f'libsurvive bridge running. Discovered objects: {list(self._friendly.values())}'
        )

    def _decode_name(self, raw) -> str:
        if isinstance(raw, bytes):
            return raw.decode('utf-8', errors='replace')
        return str(raw)

    def _discover(self) -> None:
        for obj in self.ctx.Objects():
            name = self._decode_name(obj.Name())
            if not name:
                continue
            # libsurvive object names are like "T20", "HMD", "LH-XXXXXX", tracker serials, etc.
            # We treat anything that isn't a lighthouse (LHB / LH-) or HMD as a tracker.
            if name.startswith('LH') or name.upper().startswith('HMD'):
                self.get_logger().info(f'  skip non-tracker object: {name}')
                continue
            friendly = self.name_map.get(name, name)
            self._friendly[name] = friendly

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            topic_base = f'/survive/{friendly}'
            self._pose_pubs[name] = self.create_publisher(PoseStamped, f'{topic_base}/pose', qos)
            self._valid_pubs[name] = self.create_publisher(Bool, f'{topic_base}/valid', 1)
            self.get_logger().info(f'  tracker: {name}  -> {topic_base}/pose')

    def _poll_loop(self) -> None:
        """libsurvive's NextUpdated() blocks until an update arrives — drive it in a thread."""
        while self._running:
            try:
                if not self.ctx.Running():
                    self.get_logger().warn('libsurvive context stopped, exiting poll loop')
                    break
                updated = self.ctx.NextUpdated()
                if updated is None:
                    continue
                name = self._decode_name(updated.Name())
                if name not in self._pose_pubs:
                    # newly-arrived device — register on the fly
                    if name.startswith('LH') or name.upper().startswith('HMD'):
                        continue
                    self._friendly[name] = self.name_map.get(name, name)
                    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST, depth=1)
                    self._pose_pubs[name] = self.create_publisher(
                        PoseStamped, f'/survive/{self._friendly[name]}/pose', qos)
                    self._valid_pubs[name] = self.create_publisher(
                        Bool, f'/survive/{self._friendly[name]}/valid', 1)
                    self.get_logger().info(f'  late-arrival tracker: {name}')

                pose_obj = updated.Pose()
                pose_data = pose_obj[0]
                # pose_data.Pos: 3-tuple x,y,z (meters)
                # pose_data.Rot: 4-tuple w,x,y,z
                try:
                    px, py, pz = float(pose_data.Pos[0]), float(pose_data.Pos[1]), float(pose_data.Pos[2])
                    qw, qx, qy, qz = (float(pose_data.Rot[0]), float(pose_data.Rot[1]),
                                      float(pose_data.Rot[2]), float(pose_data.Rot[3]))
                except Exception as e:
                    self.get_logger().warn(f'pose unpack failed: {e}', throttle_duration_sec=5)
                    continue

                # libsurvive returns zero pose when not yet localized.
                # Treat (0,0,0)+identity as "not valid yet".
                valid = not (px == 0.0 and py == 0.0 and pz == 0.0 and qw == 1.0 and qx == qy == qz == 0.0)

                stamp = self.get_clock().now().to_msg()

                self._valid_pubs[name].publish(Bool(data=valid))
                if not valid:
                    continue

                ps = PoseStamped()
                ps.header.stamp = stamp
                ps.header.frame_id = self.frame_id
                ps.pose.position.x = px
                ps.pose.position.y = py
                ps.pose.position.z = pz
                ps.pose.orientation.x = qx
                ps.pose.orientation.y = qy
                ps.pose.orientation.z = qz
                ps.pose.orientation.w = qw
                self._pose_pubs[name].publish(ps)

                if self.tf_broadcaster is not None:
                    t = TransformStamped()
                    t.header.stamp = stamp
                    t.header.frame_id = self.frame_id
                    t.child_frame_id = f'survive_{self._friendly[name]}'
                    t.transform.translation.x = px
                    t.transform.translation.y = py
                    t.transform.translation.z = pz
                    t.transform.rotation.x = qx
                    t.transform.rotation.y = qy
                    t.transform.rotation.z = qz
                    t.transform.rotation.w = qw
                    self.tf_broadcaster.sendTransform(t)
            except Exception as e:
                self.get_logger().error(f'poll loop exception: {e}', throttle_duration_sec=5)
                time.sleep(0.1)

    def destroy_node(self) -> bool:
        self._running = False
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = SurviveTrackerNode()
    except Exception as e:
        print(f'[survive_tracker_node] startup failed: {e}', file=sys.stderr)
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
