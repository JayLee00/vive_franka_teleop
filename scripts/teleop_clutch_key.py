#!/usr/bin/env python3
"""키보드 클러치: space=정지(disengage), 1=재개(engage). 양팔 동시 제어.

실행:
  source /opt/ros/humble/setup.bash && source /home/js/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 /home/js/teleop_clutch_key.py
"""
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

ARMS = ['left', 'right']


class ClutchKey(Node):
    def __init__(self) -> None:
        super().__init__('teleop_clutch_key')
        self.pubs = {a: self.create_publisher(Bool, f'/teleop/engage/{a}', 1) for a in ARMS}

    def set_all(self, on: bool) -> None:
        for a in ARMS:
            self.pubs[a].publish(Bool(data=on))


def main() -> None:
    rclpy.init()
    node = ClutchKey()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print('[clutch] space=정지(disengage)   1=재개(engage)   q=종료')
    print('         시작 상태: 정지(안전). 1 누르면 현재 자세로 anchor 잡고 전송 시작.')
    node.set_all(False)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            c = sys.stdin.read(1)
            if c == ' ':
                node.set_all(False)
                print('STOP   (disengaged) - 로봇 홀드, 델타 0 전송')
            elif c == '1':
                node.set_all(True)
                print('RESUME (engaged)    - 현재 자세 anchor 재캡처, 델타 전송 재개')
            elif c in ('q', '\x03'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.set_all(False)  # 종료 시 안전하게 정지
        node.destroy_node()
        rclpy.shutdown()
        print('\n[clutch] 종료 (정지 상태로 빠짐)')


if __name__ == '__main__':
    main()
