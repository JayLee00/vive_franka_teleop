#!/usr/bin/env python3
"""오른손 핸드 제어 테스트 (서보 on/off + mode + 16 DOF 타겟).

제어PC 계약 (hand_target_receiver 가 모두 구독):
  /hand/cmd_servo_r   std_msgs/Bool    서보 on(true)/off(false)  ← 게이트 무관 항상 반영
  /hand/cmd_mode_r    std_msgs/Int32   1=position, 2=circular
  /hand/q_target_r    std_msgs/Float32MultiArray[16]  관절 타겟(엔코더 카운트), BEST_EFFORT
상태: /hand/mode_r  std_msgs/Int32MultiArray [mode, servo_on]

켜는 순서:  s(서보 on) -> m(mode 1) -> g 또는 0(타겟)
빼는 순서:  f(서보 off) 만.  (mode 는 건드리지 않음)

키:
  s = 서보 ON     f = 서보 OFF
  m = mode 1(position)
  g = TARGET_G 전송   i = TARGET_INIT 전송   0 = 전부 0 전송
  q = 종료 (서보 OFF 하고 나감, mode 는 안 건드림)

★ 스냅 주의: 시작하면 TARGET_INIT 을 계속 발행합니다(서보 off 라 무시됨).
  s 로 서보를 켜는 순간 손이 TARGET_INIT 자세로 갑니다 → TARGET_INIT 을 현재 손 자세 근처로
  두거나, 서보 켜기 전 손을 TARGET_INIT 근처에 두세요. (q_target 단위가 카운트라 joint_states
  라디안으로 현재자세 자동복사는 불가)

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
  python3 ~/Desktop/vive_franka_teleop/scripts/hand_target_test.py

⚠️ 실제 오른손을 움직입니다. 손 앞에서 / 비상정지 준비하고 실행하세요.
"""
import sys
import select
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Int32, Int32MultiArray, Float32MultiArray

SIDE = 'r'                              # 'r' 오른손 / 'l' 왼손
RATE_HZ = 30.0
MOVE_TIME = 2.0                         # 타겟 전환 보간 시간 [s]
POS_MODE = 1                            # 1=position
ZEROS = [0.0] * 16

# ── 관절 순서: hand_r_joint0 .. hand_r_joint15 (엔코더 카운트). 반드시 리밋 안에서! ──
# 시작 시 발행 (서보 켜는 순간 이 자세로 감 → 현재 손 근처로).
TARGET_INIT = [
    4096, -4096, -1000, 0,   # joint 0~3
    -500, 300, 300, 300,   # joint 4~7
    0.0, 300, 300, 300,   # joint 8~11
    500, 300, 300, 300,   # joint 12~15
]



# 'g' 눌렀을 때 발행.
TARGET_G = [
    4096, -4096, 2000, 2000,   # joint 0~3
    -1500, 2000, 2000, 2000,   # joint 4~7
    0.0, 2000, 2000, 2000,   # joint 8~11
    1500, 2000, 2000, 2000,   # joint 12~15
]


class HandTargetTest(Node):
    def __init__(self) -> None:
        super().__init__('hand_target_test')
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,   # q_target: receiver 에 맞춤
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,     # servo/mode: 확실히 전달
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Float32MultiArray, f'/hand/q_target_{SIDE}', be)
        self.pub_servo = self.create_publisher(Bool, f'/hand/cmd_servo_{SIDE}', rel)
        self.pub_mode = self.create_publisher(Int32, f'/hand/cmd_mode_{SIDE}', rel)
        self.create_subscription(Int32MultiArray, f'/hand/mode_{SIDE}', self._on_state, be)
        self.current = list(TARGET_INIT)   # 지금 발행 중인 값(보간 진행값)
        self.start = list(TARGET_INIT)     # 보간 시작값
        self.goal = list(TARGET_INIT)      # 보간 목표값
        self.t0 = time.monotonic()         # 보간 시작 시각
        self.last_state = None

    def _on_state(self, msg: Int32MultiArray) -> None:
        st = list(msg.data)
        if st != self.last_state:
            self.last_state = st
            if len(st) >= 2:
                self.get_logger().info(f'[state] mode={st[0]} servo_on={st[1]}')

    def publish(self) -> None:
        # start -> goal 로 MOVE_TIME 동안 smoothstep 보간 (시간 기반)
        a = 1.0 if MOVE_TIME <= 0 else min(1.0, (time.monotonic() - self.t0) / MOVE_TIME)
        s = a * a * (3.0 - 2.0 * a)                       # smoothstep (부드러운 가감속)
        self.current = [self.start[i] + (self.goal[i] - self.start[i]) * s for i in range(16)]
        msg = Float32MultiArray()
        msg.data = [float(v) for v in self.current]
        self.pub.publish(msg)

    def goto(self, vals, label: str) -> None:
        self.start = list(self.current)                   # 현재 발행값에서 출발
        self.goal = [float(v) for v in vals]
        self.t0 = time.monotonic()
        self.get_logger().info(f'{label} ({MOVE_TIME:.0f}s 보간): {[round(v, 1) for v in self.goal]}')

    def servo(self, on: bool) -> None:
        self.pub_servo.publish(Bool(data=on))
        self.get_logger().info(f'servo {"ON" if on else "OFF"} -> /hand/cmd_servo_{SIDE}')

    def set_mode(self, mode: int) -> None:
        self.pub_mode.publish(Int32(data=mode))
        self.get_logger().info(f'mode {mode} -> /hand/cmd_mode_{SIDE}')


def main() -> None:
    if len(TARGET_INIT) != 16 or len(TARGET_G) != 16:
        print('[hand] TARGET_INIT / TARGET_G 는 각각 16개여야 합니다. 중단.')
        return

    rclpy.init()
    node = HandTargetTest()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    period = 1.0 / RATE_HZ

    print(f'[hand] {SIDE}손 제어. /hand/q_target_{SIDE} {RATE_HZ:.0f}Hz 발행 (엔코더 카운트)')
    print('  s=서보ON  f=서보OFF  m=mode1  g=TARGET_G  i=TARGET_INIT  0=전부0  q=종료(서보OFF)')
    print(f'  g/i/0 는 현재값→목표 {MOVE_TIME:.0f}s 보간 이동 | 켜는순서 s->m->g/i | 빼는순서 f')
    node.goto(TARGET_INIT, 'START (TARGET_INIT)')
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)   # 상태(/hand/mode_r) 콜백 처리
            node.publish()
            r, _, _ = select.select([sys.stdin], [], [], period)
            if not r:
                continue
            c = sys.stdin.read(1)
            if c == 's':
                node.servo(True)
            elif c == 'f':
                node.servo(False)
            elif c == 'm':
                node.set_mode(POS_MODE)
            elif c in ('g', 'G'):
                node.goto(TARGET_G, 'G -> TARGET_G')
            elif c in ('i', 'I'):
                node.goto(TARGET_INIT, 'I -> TARGET_INIT')
            elif c == '0':
                node.goto(ZEROS, 'ZEROS')
            elif c in ('q', '\x03'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.servo(False)                # 종료 시 서보 OFF (mode 는 안건드림)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.0)
        node.destroy_node()
        rclpy.shutdown()
        print('\n[hand] 종료 (서보 OFF 발행)')


if __name__ == '__main__':
    main()
