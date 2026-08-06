#!/usr/bin/env python3
"""fruit-manip 의 3D 박스(/inhand/bbox_corners) → 깔끔한 과일 pose/크기 재발행.

fruit-manip(live_bbox_gui.py)은 8코너 OBB만 /inhand/bbox_corners(Float32MultiArray,
24=8x3, 카메라 광학 프레임, m)로 발행한다. 그 큰 파일을 안 건드리고, 여기서 8코너로부터
중심(위치)·주축(방향)·장단축(크기)을 PCA로 뽑아 레코더가 먹는 토픽으로 재발행:

    /fruit/pose  geometry_msgs/PoseStamped     position + orientation(quat, 박스 주축)
    /fruit/size  std_msgs/Float32MultiArray     [a, b, c]  전체축 길이 내림차순(장축..단축)[m]

프레임: 입력 corners 는 카메라 광학 프레임. header.frame_id 로 그대로 전달(로봇 base 변환은 소비측 tf2).
※ 더 정밀한 값(칼만/AprilTag 기반 center·quat, 캘리브 a/c)이 필요하면 live_bbox_gui.py 안에
  직접 퍼블리셔를 추가하는 게 낫다(777~782행). 이 브리지는 비침습 시작점.

실행:
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/fruit_pose_bridge.py
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

IN_TOPIC = "/inhand/bbox_corners"
FRAME_ID = "camera_color_optical_frame"   # corners header 없어서 fallback

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=1)


def rotmat_to_quat(R: np.ndarray):
    """3x3 회전행렬 -> 쿼터니언 [x,y,z,w]."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


class FruitPoseBridge(Node):
    def __init__(self):
        super().__init__("fruit_pose_bridge")
        self.pub_pose = self.create_publisher(PoseStamped, "/fruit/pose", QOS)
        self.pub_size = self.create_publisher(Float32MultiArray, "/fruit/size", QOS)
        self.create_subscription(Float32MultiArray, IN_TOPIC, self._on_box, QOS)
        self.get_logger().info(f"{IN_TOPIC} -> /fruit/pose + /fruit/size (PCA OBB)")

    def _on_box(self, msg: Float32MultiArray):
        if len(msg.data) < 24:
            return
        pts = np.asarray(msg.data[:24], dtype=np.float64).reshape(8, 3)
        center = pts.mean(axis=0)
        # PCA: 8코너의 주축 = 박스 축. SVD 로 축/extent 추출(코너 순서 무관).
        d = pts - center
        _, _, Vt = np.linalg.svd(d, full_matrices=False)
        axes = Vt                                   # (3,3) 주축(행)
        proj = d @ axes.T                            # 각 축으로 투영
        lengths = proj.max(axis=0) - proj.min(axis=0)  # 축별 전체 길이
        order = np.argsort(lengths)[::-1]            # 내림차순(장축..단축)
        lengths = lengths[order]
        R = axes[order].T                            # 열 = 정렬된 주축
        if np.linalg.det(R) < 0:                     # 오른손 좌표계 보정
            R[:, 2] = -R[:, 2]
        q = rotmat_to_quat(R)

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = FRAME_ID
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, center)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
        self.pub_pose.publish(ps)

        sz = Float32MultiArray()
        sz.data = [float(v) for v in lengths]        # [a, b, c] 장축..단축 [m]
        self.pub_size.publish(sz)


def main():
    rclpy.init()
    node = FruitPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
