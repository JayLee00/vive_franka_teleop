#!/usr/bin/env python3
"""3구 USB 풋스위치(PCsensor)로 텔레옵 클러치 + 핸드 제어.

페달 (기본 키: 왼쪽=a, 중간=b, 오른쪽=c, 각 페달이 Enter도 같이 보냄 → Enter 무시):
  왼쪽  = 클러치 STOP      → /teleop/engage/{left,right} = false (양팔 disengage)
  오른쪽 = 델타 EE 전송     → /teleop/engage/{left,right} = true  (양팔 engage, teleop_delta 가 ee_target 발행)
  중간  = 핸드 Init<->Grasp 토글 (시작 Init, 2초 smoothstep 보간)

핸드 자동 관리: 시작 시 mode 1 설정 + Init 발행 후 서보 ON (손이 Init 자세로 이동).
손 자세 값(TARGET_INIT/TARGET_G)은 hand_target_test.py 와 공유.

입력: /dev/input/event(PCsensor keyboard) 직접 read + EVIOCGRAB(페달 키가 다른 창에
안 새게 독점). 터미널 포커스/한영 상태와 무관하게 동작.

권한: event 노드는 root:input 이라 최초 1회
  sudo usermod -aG input $USER   # 후 재로그인, 또는 실행 셸에서  newgrp input

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
  python3 ~/Desktop/vive_franka_teleop/scripts/foot_pedal_teleop.py

⚠️ 실행하면 손이 Init 자세로 움직입니다(서보 자동 ON). 실행 전 손을 Init 근처에 두세요.
⚠️ teleop_clutch_key.py / hand_target_test.py 와 동시 실행 금지 (engage·q_target 발행 충돌).
"""
import os
import sys
import fcntl
import select
import struct
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hand_target_test import TARGET_INIT, TARGET_G   # 손 자세 단일 소스

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Int32, Int32MultiArray, Float32MultiArray

DEVICE = '/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd'
ARMS = ['left', 'right']
HAND_SIDE = 'r'
RATE_HZ = 50.0
MOVE_TIME = 2.0                     # 핸드 Init<->Grasp 보간 시간 [s]
POS_MODE = 1

# evdev input_event: struct timeval(long,long) + type(H) + code(H) + value(i)
EV_KEY = 1
KEY_A, KEY_B, KEY_C = 30, 48, 46   # 왼쪽 / 중간 / 오른쪽 페달
EVENT_FMT = 'llHHi'
EVENT_SIZE = struct.calcsize(EVENT_FMT)
EVIOCGRAB = 0x40044590             # 장치 독점(다른 창으로 키 안 샘)


class PedalCtrl(Node):
    def __init__(self) -> None:
        super().__init__('foot_pedal_teleop')
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_q = self.create_publisher(Float32MultiArray, f'/hand/q_target_{HAND_SIDE}', be)
        self.pub_servo = self.create_publisher(Bool, f'/hand/cmd_servo_{HAND_SIDE}', rel)
        self.pub_mode = self.create_publisher(Int32, f'/hand/cmd_mode_{HAND_SIDE}', rel)
        self.pub_engage = {a: self.create_publisher(Bool, f'/teleop/engage/{a}', rel) for a in ARMS}
        self.create_subscription(Int32MultiArray, f'/hand/mode_{HAND_SIDE}', self._on_state, be)

        self.current = list(TARGET_INIT)   # 발행 중 보간값
        self.start = list(TARGET_INIT)
        self.goal = list(TARGET_INIT)
        self.t0 = time.monotonic()
        self.grasping = False              # False=Init, True=Grasp
        self.last_state = None

    def _on_state(self, msg: Int32MultiArray) -> None:
        st = list(msg.data)
        if st != self.last_state:
            self.last_state = st
            if len(st) >= 2:
                self.get_logger().info(f'[hand state] mode={st[0]} servo_on={st[1]}')

    # --- hand q_target (보간 발행) ---
    def publish_q(self) -> None:
        a = 1.0 if MOVE_TIME <= 0 else min(1.0, (time.monotonic() - self.t0) / MOVE_TIME)
        s = a * a * (3.0 - 2.0 * a)
        self.current = [self.start[i] + (self.goal[i] - self.start[i]) * s for i in range(16)]
        msg = Float32MultiArray()
        msg.data = [float(v) for v in self.current]
        self.pub_q.publish(msg)

    def goto(self, vals, label: str) -> None:
        self.start = list(self.current)
        self.goal = [float(v) for v in vals]
        self.t0 = time.monotonic()
        self.get_logger().info(f'hand -> {label} ({MOVE_TIME:.0f}s)')

    def servo(self, on: bool) -> None:
        self.pub_servo.publish(Bool(data=on))
        self.get_logger().info(f'hand servo {"ON" if on else "OFF"}')

    def set_mode(self, mode: int) -> None:
        self.pub_mode.publish(Int32(data=mode))
        self.get_logger().info(f'hand mode {mode}')

    def engage(self, on: bool) -> None:
        for a in ARMS:
            self.pub_engage[a].publish(Bool(data=on))
        self.get_logger().info(f'clutch {"GO(engage)" if on else "STOP(disengage)"} -> {ARMS}')

    # --- 페달 동작 ---
    def on_left(self) -> None:   # 왼쪽 = STOP
        self.engage(False)

    def on_right(self) -> None:  # 오른쪽 = 델타 EE 전송
        self.engage(True)

    def on_mid(self) -> None:    # 중간 = Init<->Grasp 토글
        self.grasping = not self.grasping
        self.goto(TARGET_G if self.grasping else TARGET_INIT,
                  'GRASP' if self.grasping else 'INIT')


def open_pedal():
    try:
        fd = os.open(DEVICE, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError:
        user = os.environ.get('USER', '$USER')
        print(f'[pedal] 권한 없음: {DEVICE}\n'
              f'  최초 1회:  sudo usermod -aG input {user}   → 재로그인(또는 newgrp input)')
        return None
    except FileNotFoundError:
        print(f'[pedal] 장치 없음: {DEVICE}\n  풋스위치 USB 연결/재연결 확인')
        return None
    try:
        fcntl.ioctl(fd, EVIOCGRAB, 1)   # 독점 (다른 창으로 a/b/c 안 샘)
    except OSError as e:
        print(f'[pedal] EVIOCGRAB 실패(계속 진행, 키가 다른 창에도 갈 수 있음): {e}')
    return fd


def main() -> None:
    rclpy.init()
    node = PedalCtrl()
    fd = open_pedal()
    if fd is None:
        node.destroy_node()
        rclpy.shutdown()
        return

    # 핸드 자동 활성화: mode1 -> Init 먼저 실어보내기 -> 서보 ON
    node.set_mode(POS_MODE)
    node.goto(TARGET_INIT, 'START Init')
    for _ in range(10):
        node.publish_q()
        time.sleep(0.01)
    node.servo(True)

    print('[pedal] 왼쪽=클러치STOP  중간=핸드 Init/Grasp토글  오른쪽=델타EE전송  |  Ctrl-C 종료')
    period = 1.0 / RATE_HZ
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.publish_q()
            r, _, _ = select.select([fd], [], [], period)
            if not r:
                continue
            data = os.read(fd, EVENT_SIZE * 64)
            for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _, _, etype, code, val = struct.unpack(EVENT_FMT, data[off:off + EVENT_SIZE])
                if etype == EV_KEY and val == 1:      # 눌림(press)만
                    if code == KEY_A:
                        node.on_left()
                    elif code == KEY_C:
                        node.on_right()
                    elif code == KEY_B:
                        node.on_mid()
    except KeyboardInterrupt:
        pass
    finally:
        node.engage(False)     # 클러치 STOP
        node.servo(False)      # 서보 OFF
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.0)
        try:
            fcntl.ioctl(fd, EVIOCGRAB, 0)
        except OSError:
            pass
        os.close(fd)
        node.destroy_node()
        rclpy.shutdown()
        print('\n[pedal] 종료 (클러치 STOP, 서보 OFF)')


if __name__ == '__main__':
    main()
