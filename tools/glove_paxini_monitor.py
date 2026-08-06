#!/usr/bin/env python3
"""글러브 16채널 + Paxini ft 를 ROS2 토픽으로 구독해 나란히 출력.

시리얼을 직접 열지 않으므로 glove_teleop.py / paxini_uart_node.py 가 포트를
배타적으로 잡고 있는 상태에서도 함께 띄울 수 있다. (glove_tactile_monitor.py 는
UART 를 직접 열기 때문에 러너와 동시 실행이 안 된다.)

구독:
    /glove/<side>/q_raw       Float32MultiArray[16]   글러브 엔코더 raw
    <prefix>/<side>/ft        Float32MultiArray[12]   4손가락 × (Fx,Fy,Fz)

사용:
    python3 tools/glove_paxini_monitor.py
    python3 tools/glove_paxini_monitor.py --side left --hz 4
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

NUM_CH = 16
SENSOR_NUM = 4
AXIS_NUM = 3

# 발행측(glove_teleop.py, paxini_uart_node.py)과 같아야 구독이 매칭된다.
# RELIABLE 로 두면 BEST_EFFORT 발행자와 붙지 않아 아무것도 안 들어온다.
SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


class GlovePaxiniMonitor(Node):
    def __init__(self, glove_topic: str, ft_topic: str, hz: float):
        super().__init__("glove_paxini_monitor")
        self.glove_topic = glove_topic
        self.ft_topic = ft_topic

        self.glove = None
        self.ft = None
        self._n_glove = 0
        self._n_ft = 0

        self.create_subscription(Float32MultiArray, glove_topic,
                                 self._on_glove, SENSOR_QOS)
        self.create_subscription(Float32MultiArray, ft_topic,
                                 self._on_ft, SENSOR_QOS)

        self._t0 = time.perf_counter()
        self._prev_t = self._t0
        self._prev_glove = 0
        self._prev_ft = 0
        self.create_timer(1.0 / hz, self._print)

    def _on_glove(self, msg: Float32MultiArray):
        self.glove = list(msg.data)
        self._n_glove += 1

    def _on_ft(self, msg: Float32MultiArray):
        self.ft = list(msg.data)
        self._n_ft += 1

    def _print(self):
        now = time.perf_counter()
        dt = max(1e-9, now - self._prev_t)
        ghz = (self._n_glove - self._prev_glove) / dt
        fhz = (self._n_ft - self._prev_ft) / dt
        self._prev_t, self._prev_glove, self._prev_ft = now, self._n_glove, self._n_ft

        print(f"\n[{now - self._t0:7.1f}s]  글러브 {ghz:6.1f}Hz   촉각 {fhz:6.1f}Hz")

        if self.glove is None:
            print(f"  글러브 … 수신 대기  ({self.glove_topic})")
        else:
            print("  글러브16: " + " ".join(f"{v:6.0f}" for v in self.glove[:NUM_CH]))

        if self.ft is None:
            print(f"  촉각   … 수신 대기  ({self.ft_topic})")
        else:
            d = self.ft
            blocks = [f"f{i}[{d[i*AXIS_NUM]:+7.2f},{d[i*AXIS_NUM+1]:+7.2f},"
                      f"{d[i*AXIS_NUM+2]:+7.2f}]"
                      for i in range(SENSOR_NUM) if len(d) >= (i + 1) * AXIS_NUM]
            print("  촉각 ft : " + "  ".join(blocks))


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(
        description="글러브16 + Paxini ft 토픽 모니터 (시리얼 안 건드림)")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--topic-prefix", default="/glove/paxini",
                    help="촉각 토픽 네임스페이스 (기본 /glove/paxini). "
                         "제어 PC 쪽 센서를 보려면 /paxini 로 준다.")
    ap.add_argument("--hz", type=float, default=2.0, help="화면 갱신 [Hz]")
    a = ap.parse_args()

    # 러너가 파이프/파일로 넘겨도 바로 보이게 (기본은 블록 버퍼링).
    sys.stdout.reconfigure(line_buffering=True)
    # 러너 cleanup 은 SIGTERM 을 보낸다. 그대로 두면 spin() 이 context 무효화
    # RCLError 로 터져 traceback 을 남기므로 SIGINT 와 같은 경로로 돌린다.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    prefix = "/" + a.topic_prefix.strip("/")
    rclpy.init()
    node = GlovePaxiniMonitor(f"/glove/{a.side}/q_raw",
                              f"{prefix}/{a.side}/ft", a.hz)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
