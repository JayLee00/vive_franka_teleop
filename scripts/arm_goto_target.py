#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""프랑카 팔을 현재 관절 위치에서 절대 7관절 타겟까지 선형 보간으로 천천히 이동.

  현재 위치 : /franka/<side>/joint_states  sensor_msgs/JointState position[7]  (BEST_EFFORT)
  타겟 발행 : /franka/<side>/q_target      std_msgs/Float64MultiArray[7] [rad]  (BEST_EFFORT)
              → 제어 PC 의 arm_q_target_receiver 가 구독

동작: joint_states 로 현재 q0 수신 → q0→target 선형 보간을 --secs 동안 --rate-hz 로
발행 → --hold-secs 만큼 타겟 유지 → 종료(타겟은 그대로 두고 나감).

첫 명령은 측정값 q0 이라 제어기가 들고 있던 q_target 과 추종오차(보통 <0.01 rad)만큼
어긋난 채 시작한다. 그만큼은 첫 프레임에 계단으로 들어가지만 무시할 수준.

Ctrl+C 하면 그 시점 보간값에서 발행을 멈추고 나간다(팔은 그 자리에 선다).

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 ~/Desktop/vive_franka_teleop/scripts/arm_goto_target.py
"""
import argparse
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

NUM_JOINTS = 7

DEFAULT_TARGET = [-0.2866, 1.4185, 0.2677, -1.9216, 0.7769, 1.2157, 2.0401]


class ArmGoto(Node):
    def __init__(self, args):
        super().__init__('arm_goto_target')
        self.args = args
        self.target = np.asarray(args.target, dtype=np.float64)
        self.state_topic = f'/franka/{args.side}/joint_states'
        self.cmd_topic = f'/franka/{args.side}/q_target'

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self._q = None
        self.create_subscription(JointState, self.state_topic, self._on_state, qos)
        self.pub = self.create_publisher(Float64MultiArray, self.cmd_topic, qos)

    def _on_state(self, msg):
        if len(msg.position) >= NUM_JOINTS:
            self._q = np.asarray(msg.position[:NUM_JOINTS], dtype=np.float64)

    def wait_for_state(self, timeout_s=3.0):
        deadline = time.monotonic() + timeout_s
        while self._q is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._q

    def publish(self, q):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q]
        self.pub.publish(msg)


def main(argv=None):
    p = argparse.ArgumentParser(
        description='프랑카 팔을 현재 위치 → 7관절 절대 타겟으로 선형 보간 이동.')
    p.add_argument('--side', choices=('right', 'left'), default='right')
    p.add_argument('--target', type=float, nargs=NUM_JOINTS, default=list(DEFAULT_TARGET),
                   help='7관절 절대 타겟 [rad] (기본: 스크립트 상단 DEFAULT_TARGET)')
    p.add_argument('--secs', type=float, default=10.0, help='이동 시간 [s] (기본 10)')
    p.add_argument('--hold-secs', type=float, default=0.5,
                   help='도착 후 타겟 유지 시간 [s] (기본 0.5)')
    p.add_argument('--rate-hz', type=float, default=100.0, help='발행 주기 [Hz] (기본 100)')
    p.add_argument('-y', '--yes', action='store_true', help='확인 프롬프트 생략')
    args = p.parse_args(argv)

    if args.secs <= 0:
        p.error('--secs 는 0 보다 커야 한다 (계단 명령 방지)')

    rclpy.init()
    node = ArmGoto(args)
    try:
        q0 = node.wait_for_state()
        if q0 is None:
            print(f'[arm_goto] {node.state_topic} 수신 없음 — 제어 PC 스택(shm+nd) / '
                  f'ROS_DOMAIN_ID 확인', file=sys.stderr)
            return 1
        if node.pub.get_subscription_count() == 0:
            print(f'[arm_goto] 경고: {node.cmd_topic} 구독자 없음 '
                  f'(arm_q_target_receiver 떠 있나?)', file=sys.stderr)

        delta = node.target - q0
        vmax = float(np.max(np.abs(delta))) / args.secs
        print(f'[arm_goto] side={args.side}  {args.secs:.1f}s  {args.rate_hz:.0f}Hz'
              f'  → {node.cmd_topic}')
        print('  j      current      target       delta')
        for i in range(NUM_JOINTS):
            print(f'  {i}  {q0[i]:+10.4f}  {node.target[i]:+10.4f}  {delta[i]:+10.4f}')
        print(f'  max|delta| = {np.max(np.abs(delta)):.4f} rad, '
              f'max speed = {vmax:.4f} rad/s')

        if not args.yes:
            try:
                if input('이동할까? [y/N] ').strip().lower() not in ('y', 'yes'):
                    print('[arm_goto] 취소.')
                    return 0
            except (EOFError, KeyboardInterrupt):
                print('\n[arm_goto] 취소.')
                return 0

        dt = 1.0 / args.rate_hz
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            alpha = min(1.0, t / args.secs)
            node.publish(q0 + alpha * delta)
            rclpy.spin_once(node, timeout_sec=0.0)
            if t >= args.secs + args.hold_secs:
                break
            time.sleep(dt)

        # 도착 확인: 마지막으로 들어온 측정값과 타겟 비교.
        rclpy.spin_once(node, timeout_sec=0.2)
        err = node._q - node.target
        print(f'[arm_goto] 완료. 잔차 max|err| = {np.max(np.abs(err)):.4f} rad '
              f'({np.array2string(err, precision=4, floatmode="fixed")})')
        return 0
    except KeyboardInterrupt:
        print('\n[arm_goto] 중단 — 발행 멈춤(팔은 마지막 타겟에서 정지).')
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
