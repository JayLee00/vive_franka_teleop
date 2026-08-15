#!/usr/bin/env python3
"""Paxini 촉각센서 UART → ROS2 토픽 직접 발행 (SHM 우회).

제어 PC 는 `paxini_writer.py` 로 SHM(0x7951)에 쓰고 nd 가 토픽을 내보내지만,
글러브 PC 에는 `sysv_ipc` 도 `inc/shm.h` 도 없다. 이 노드는 SHM 을 건너뛰고
UART 에서 읽은 값을 제어 PC 와 **같은 토픽·같은 배열 크기**로 바로 발행한다.

발행 (record/ros2_hdf5_recorder.py 의 20/21번 필드와 동일):
    <prefix>/<side>/ft    Float32MultiArray[12]    4손가락 × (Fx,Fy,Fz)
    <prefix>/<side>/raw   Float32MultiArray[1524]  4손가락 × 127점 × (x,y,z)

prefix 는 `--topic-prefix` 로 정하고 기본은 `/paxini` 다. 로봇 핸드와 글러브에
**같은 Paxini 가 각각 달려 있어** 토픽 이름이 겹치므로, 이 PC(글러브 쪽)에서는
`--topic-prefix /glove/paxini` 로 갈라 발행한다. 제어 PC 는 기본값을 그대로 쓴다.

값은 `tools/paxini/` 모듈로 디코드한 그대로다 — 필터·스무딩 없음.
ft 는 `paxini_writer.py` 가 SHM Paxini_ft 에 쓰는 것과 동일한 배열
(`decoded_to_numpy_arrays()["ft"]`, 축 순서 Fx,Fy,Fz).
  ※ record/fruit_overlay.py 주석은 이 토픽을 손가락당 [Fz,Fx,Fy] 로 적어 두었다.
    디코더 원본 순서는 (Fx,Fy,Fz) 라 서로 다르다. 제어 PC 의 nd 가 재배열하는지는
    이 PC 에서 확인할 수 없어, 여기서는 paxini_writer 와 동일한 순서로 낸다.

사용:
    python3 tools/paxini_uart_node.py                      # right, /dev/ttyACM0
    python3 tools/paxini_uart_node.py --side left --port /dev/ttyACM1
    python3 tools/paxini_uart_node.py --no-calibrate       # 0점 보정 건너뜀

주의: /dev/ttyACM0 은 root:dialout 이다. dialout 그룹에 없으면
      `sudo chmod 666 /dev/ttyACM0` 또는 `sudo usermod -aG dialout $USER`(재로그인).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paxini"))

from tactile_uart import (  # noqa: E402
    TactileDecodeConfig,
    _pop_auto_push_frame,
    decode_auto_push_sample,
    decoded_to_numpy_arrays,
)
from Read_Single_Sensor_Hand import (  # noqa: E402
    BAUDRATE,
    disable_auto_push,
    enable_auto_push,
    get_device_version,
    make_timestamped_sample,
    open_sensor_serial,
    send_calibration,
)

SENSOR_NUM = 4
POINT_NUM = 127
AXIS_NUM = 3
FT_LEN = SENSOR_NUM * AXIS_NUM                 # 12
RAW_LEN = SENSOR_NUM * POINT_NUM * AXIS_NUM    # 1524

# record/ros2_hdf5_recorder.py 와 동일한 QoS
SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)

RECONNECT_SEC = 1.0


def _dims(*shape: tuple[str, int]) -> list[MultiArrayDimension]:
    """(label, size) 목록 → MultiArrayDimension 목록 (stride = 뒤쪽 크기 곱)."""
    out, sizes = [], [s for _, s in shape]
    for i, (label, size) in enumerate(shape):
        stride = 1
        for s in sizes[i:]:
            stride *= s
        out.append(MultiArrayDimension(label=label, size=size, stride=stride))
    return out


class PaxiniUartNode(Node):
    def __init__(self, side: str, port: str, calibrate: bool, print_hz: float,
                 topic_prefix: str = "/paxini"):
        super().__init__("paxini_uart")
        self.side = side
        self.port = port
        self.calibrate = calibrate

        # 로봇 핸드와 글러브에 같은 Paxini 가 달려 있어 토픽이 겹친다. 이 PC 는
        # 글러브 쪽을 읽으므로 prefix 로 갈라 준다(제어 PC 는 기본값 그대로).
        prefix = "/" + topic_prefix.strip("/")
        self.topic_ft = f"{prefix}/{side}/ft"
        self.topic_raw = f"{prefix}/{side}/raw"
        self.pub_ft = self.create_publisher(
            Float32MultiArray, self.topic_ft, SENSOR_QOS)
        self.pub_raw = self.create_publisher(
            Float32MultiArray, self.topic_raw, SENSOR_QOS)

        self._ft_dims = _dims(("sensor", SENSOR_NUM), ("axis", AXIS_NUM))
        self._raw_dims = _dims(("sensor", SENSOR_NUM), ("point", POINT_NUM),
                               ("axis", AXIS_NUM))

        self.cfg = TactileDecodeConfig()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._n = 0                # 발행 프레임 수
        self._bad = 0              # LRC 실패/에러코드로 버린 프레임 수
        self._err = None           # 마지막 오류 문자열
        self._tac_peak = np.zeros(SENSOR_NUM)   # 출력 주기 동안의 손가락별 최대 |압력|
        self._ft_last = np.zeros((SENSOR_NUM, AXIS_NUM))
        self._version = None

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        if print_hz > 0:
            self._prev_n = 0
            self._prev_bad = 0
            self._prev_t = time.perf_counter()
            self.create_timer(1.0 / print_hz, self._print_status)

    # ── 시리얼 ────────────────────────────────────────────────────────────
    def _open(self):
        """열고 정식 순서로 초기화: auto-push 끄기 → (보정) → auto-push 켜기.

        보정은 auto-push 를 켜기 *전에* 해야 응답(AA55)을 데이터 프레임(AA56)과
        헷갈리지 않는다. 보정 없이 켜면 센서가 error_code 0x11 과 0 페이로드를
        내보낸다(실측).
        """
        ser = open_sensor_serial(self.port, BAUDRATE, timeout=1.0)
        disable_auto_push(ser)
        time.sleep(0.4)
        ser.reset_input_buffer()
        if self._version is None:
            try:
                self._version = get_device_version(ser) or "?"
            except Exception:                                   # noqa: BLE001
                self._version = "?"
        if self.calibrate:
            ok = send_calibration(ser)
            self.get_logger().info(f"0점 보정 {'성공' if ok else '실패(계속 진행)'}")
            time.sleep(0.5)
            ser.reset_input_buffer()
        if not enable_auto_push(ser):
            # enable 응답은 이미 흐르는 데이터 프레임과 섞여 오판되기 쉽다.
            # 실제 스트림이 오는지는 아래 수신 루프가 판정하므로 경고만 남긴다.
            self.get_logger().warn("auto-push enable 응답 확인 실패 — 스트림으로 판정")
        return ser

    def _reader_loop(self):
        while not self._stop.is_set():
            ser = None
            try:
                ser = self._open()
                with self._lock:
                    self._err = None
                self.get_logger().info(
                    f"Paxini 연결: {self.port} (fw={self._version}) → "
                    f"{self.topic_ft}, {self.topic_raw}")
            except Exception as e:                              # noqa: BLE001
                with self._lock:
                    self._err = str(e)
                try:
                    if ser is not None and ser.is_open:
                        ser.close()
                except Exception:
                    pass
                time.sleep(RECONNECT_SEC)
                continue

            buf = bytearray()
            seq = 0
            try:
                while not self._stop.is_set():
                    waiting = ser.in_waiting
                    chunk = ser.read(waiting if waiting > 0 else 1)
                    if not chunk:
                        continue
                    recv_ns = time.monotonic_ns()
                    buf.extend(chunk)
                    latest = None
                    while True:            # 밀린 프레임은 버리고 최신만 (지연 누적 방지)
                        f = _pop_auto_push_frame(buf)
                        if f is None:
                            break
                        latest = f
                    if latest is None:
                        continue
                    sample = make_timestamped_sample(
                        seq, latest, read_start_mono_ns=recv_ns, read_end_mono_ns=recv_ns)
                    # LRC 실패/에러코드 프레임은 버린다. 센서 링크가 끊기면 payload 가
                    # 0xFF 로 채워져 오는데, 그대로 디코드하면 0xFF×0.1=25.5N 짜리
                    # 가짜 최대값이 토픽과 hdf5 기록까지 흘러든다.
                    parsed = sample.get("parsed") or {}
                    if not sample.get("lrc_ok") or parsed.get("error_code"):
                        with self._lock:
                            self._bad += 1
                        continue
                    decoded = decode_auto_push_sample(sample, self.cfg, decode_tactile=True)
                    arrays = decoded_to_numpy_arrays(decoded, self.cfg)
                    self._publish(arrays)
                    seq += 1
            except Exception as e:                              # noqa: BLE001
                with self._lock:
                    self._err = f"끊김: {e}"
                self.get_logger().error(f"Paxini 수신 오류: {e} — 재연결")
            finally:
                try:
                    if ser is not None and ser.is_open:
                        disable_auto_push(ser)
                        ser.close()
                except Exception:
                    pass

    # ── 발행 ──────────────────────────────────────────────────────────────
    def _publish(self, arrays: dict):
        ft = np.nan_to_num(arrays["ft"]).astype(np.float32)         # (4,3)
        tac = np.nan_to_num(arrays["tactile"]).astype(np.float32)   # (4,127,3)

        m_ft = Float32MultiArray()
        m_ft.layout.dim = self._ft_dims
        m_ft.data = ft.reshape(-1).tolist()
        self.pub_ft.publish(m_ft)

        m_raw = Float32MultiArray()
        m_raw.layout.dim = self._raw_dims
        m_raw.data = tac.reshape(-1).tolist()
        self.pub_raw.publish(m_raw)

        with self._lock:
            self._n += 1
            self._ft_last = ft
            peak = np.abs(tac).reshape(SENSOR_NUM, -1).max(axis=1)
            self._tac_peak = np.maximum(self._tac_peak, peak)

    def _print_status(self):
        now = time.perf_counter()
        with self._lock:
            n, err, peak, ft = self._n, self._err, self._tac_peak.copy(), self._ft_last
            bad = self._bad
            self._tac_peak[:] = 0.0
        hz = (n - self._prev_n) / max(1e-9, now - self._prev_t)
        self._prev_n, self._prev_t = n, now
        dropped, self._prev_bad = bad - self._prev_bad, bad

        if err:
            self.get_logger().warn(f"Paxini ✗ {err}")
            return
        if n == 0:
            # 프레임이 오는데 전부 버려지는 경우와 아예 안 오는 경우를 구분한다.
            if bad > 0:
                self.get_logger().warn(
                    f"Paxini 프레임 전부 폐기 ({bad}개) — LRC 실패/에러코드. "
                    f"센서 링크(배선·전원) 확인")
            else:
                self.get_logger().warn(f"Paxini 수신 없음 — {self.port} 확인")
            return
        peaks = "  ".join(f"f{i}={peak[i]:6.2f}" for i in range(SENSOR_NUM))
        fts = "  ".join(f"f{i}[{ft[i,0]:+6.2f},{ft[i,1]:+6.2f},{ft[i,2]:+6.2f}]"
                        for i in range(SENSOR_NUM))
        live = int((peak > 0).sum())
        self.get_logger().info(
            f"{hz:5.1f}Hz  반응센서 {live}/4"
            f"{f'  폐기 {dropped}' if dropped else ''}\n"
            f"        tac피크 {peaks}\n"
            f"        ft      {fts}")

    def shutdown(self):
        self._stop.set()
        self._reader.join(timeout=2.0)


def main():
    ap = argparse.ArgumentParser(description="Paxini 촉각 UART → ROS2 (/paxini/<side>/ft, /raw)")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false", default=True,
                    help="시작 시 0점 보정 건너뜀 (기본은 보정함)")
    ap.add_argument("--print-hz", type=float, default=1.0, help="상태 출력 [Hz], 0=끔")
    ap.add_argument("--quiet-driver", action="store_true", default=True,
                    help="paxini 드라이버의 LRC 경고 로그 억제")
    ap.add_argument("--topic-prefix", default="/paxini",
                    help="토픽 네임스페이스 (기본 /paxini). 글러브에 달린 센서는 "
                         "로봇 핸드 쪽과 겹치지 않게 /glove/paxini 로 준다.")
    a = ap.parse_args()

    if a.quiet_driver:
        logging.getLogger("Read_Single_Sensor_Hand").setLevel(logging.ERROR)

    rclpy.init()
    node = PaxiniUartNode(a.side, a.port, a.calibrate, a.print_hz, a.topic_prefix)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
