#!/usr/bin/env python3
"""3구 USB 풋스위치(PCsensor) → 통합 텔레옵 제어권 (vive 팔 / glove 손 / 둘 다) + 로깅 토글.

매핑 (모드 무관 고정):
  왼쪽  = STOP   → 팔 disengage + 손 disengage (홀드, 손 힘 안 풀림)
  오른쪽 = GO     → 팔 engage + 손 engage (스트리밍 시작)
  중간  = 로깅 S/E 토글 → /record/enable (true=에피소드 시작, false=저장)

모드별 발행 대상:
  vive  : /teleop/engage/{left,right}        (teleop_delta → 절대 EE 타겟 스트리밍)
  glove : /teleop/hand_engage/<side>         (glove_teleop → 핸드 q_target 스트리밍)
  both  : 위 둘 다 (한 페달로 팔+손 동시 제어권)

engage 는 TRANSIENT_LOCAL(latched): 이 페달을 먼저 띄워 STOP 을 깔아두면,
나중에 뜨는 glove_teleop / teleop_delta 가 마지막 상태를 받아 **뜨자마자 움직이지 않는다**.

입력: /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd 직접 read + EVIOCGRAB(독점).
      포커스/한영 무관. 페달은 한 프로세스만 잡을 수 있음(다른 페달 프로그램과 동시 실행 불가).
권한: 최초 1회  sudo usermod -aG input $USER  → 재로그인 (또는 newgrp input).

실행 (env 먼저):
  python3 scripts/foot_pedal.py --mode both          # 팔+손
  python3 scripts/foot_pedal.py --mode vive          # 팔만
  python3 scripts/foot_pedal.py --mode glove         # 손만
"""
import argparse
import fcntl
import os
import select
import struct
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool

DEVICE = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
ARMS = ["left", "right"]              # vive 팔 클러치 대상

# evdev input_event: struct timeval(long,long) + type(H) + code(H) + value(i)
EV_KEY = 1
KEY_A, KEY_B, KEY_C = 30, 48, 46      # 왼쪽 / 중간 / 오른쪽 페달
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)
EVIOCGRAB = 0x40044590


class FootPedal(Node):
    def __init__(self, mode: str, hand_side: str) -> None:
        super().__init__("foot_pedal")
        self.use_vive = mode in ("vive", "both")
        self.use_glove = mode in ("glove", "both")

        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        # latched: 늦게 뜬 구독자도 마지막 engage 상태를 받음 (VOLATILE 구독자와도 호환)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.pub_arm = ({a: self.create_publisher(Bool, f"/teleop/engage/{a}", latched)
                         for a in ARMS} if self.use_vive else {})
        self.pub_hand = (self.create_publisher(Bool, f"/teleop/hand_engage/{hand_side}", latched)
                         if self.use_glove else None)
        self.pub_record = self.create_publisher(Bool, "/record/enable", rel)

        self.recording = False
        self.engaged = None
        self._engage(False)            # 시작 상태: STOP (안전)

        tgt = []
        if self.use_vive:
            tgt.append(f"팔 /teleop/engage/{{{','.join(ARMS)}}}")
        if self.use_glove:
            tgt.append(f"손 /teleop/hand_engage/{hand_side}")
        self.get_logger().info(f"페달 준비 (mode={mode}) → {' + '.join(tgt)}")

    def _engage(self, on: bool) -> None:
        for pub in self.pub_arm.values():
            pub.publish(Bool(data=on))
        if self.pub_hand is not None:
            self.pub_hand.publish(Bool(data=on))
        if on != self.engaged:
            self.engaged = on
            self.get_logger().info("GO (engage, 스트리밍)" if on else "STOP (disengage, 홀드)")

    def on_left(self) -> None:      # STOP
        self._engage(False)

    def on_right(self) -> None:     # GO
        self._engage(True)

    def on_mid(self) -> None:       # 로깅 토글
        self.recording = not self.recording
        self.pub_record.publish(Bool(data=self.recording))
        self.get_logger().info("로깅 시작(S) — 에피소드 수집" if self.recording
                               else "로깅 종료(E) — 저장")


def open_pedal():
    try:
        fd = os.open(DEVICE, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError:
        user = os.environ.get("USER", "$USER")
        print(f"[pedal] 권한 없음: {DEVICE}\n"
              f"  최초 1회:  sudo usermod -aG input {user}   → 재로그인(또는 newgrp input)")
        return None
    except FileNotFoundError:
        print(f"[pedal] 장치 없음: {DEVICE}\n  풋스위치 USB 연결/재연결 확인")
        return None
    try:
        fcntl.ioctl(fd, EVIOCGRAB, 1)
    except OSError as e:
        print(f"[pedal] EVIOCGRAB 실패 — 다른 페달 프로그램이 잡고 있는지 확인: {e}")
    return fd


def main() -> None:
    ap = argparse.ArgumentParser(description="풋스위치 → 팔/손 텔레옵 제어권 + 로깅 토글")
    ap.add_argument("--mode", choices=["vive", "glove", "both"], default="both")
    ap.add_argument("--hand-side", choices=["right", "left"], default="right")
    args = ap.parse_args()

    rclpy.init()
    node = FootPedal(args.mode, args.hand_side)
    fd = open_pedal()
    if fd is None:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print("[pedal] 왼=STOP  오른=GO  중간=로깅 S/E토글  |  Ctrl-C 종료")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([fd], [], [], 0.05)
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
        node._engage(False)                            # 종료 시 안전하게 STOP
        if node.recording:
            node.pub_record.publish(Bool(data=False))  # 열려있던 에피소드 저장
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.01)
        try:
            fcntl.ioctl(fd, EVIOCGRAB, 0)
        except OSError:
            pass
        os.close(fd)
        node.destroy_node()
        rclpy.shutdown()
        print("\n[pedal] 종료 (STOP)")


if __name__ == "__main__":
    main()
