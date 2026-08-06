#!/usr/bin/env python3
"""글러브(16채널) + Paxini 촉각센서(4손가락) 동시 raw 모니터.

ROS/SHM 을 거치지 않고 두 UART 를 직접 읽어 나란히 찍는다. 배선·센서가
살아있는지, 접촉에 반응하는지 눈으로 확인하는 용도.

  글러브  /dev/ttyUSB0  115200   "j0,...,j15\\n" 정수 16개
  촉각    /dev/ttyACM0  921600   Paxini auto-push 프레임 (tools/paxini/ 모듈로 디코드)

사용:
    python3 tools/glove_tactile_monitor.py
    python3 tools/glove_tactile_monitor.py --calibrate      # 촉각 0점 보정 후 시작
    python3 tools/glove_tactile_monitor.py --no-glove       # 촉각만
    python3 tools/glove_tactile_monitor.py --sec 30         # 30초만 돌고 종료

주의:
  * /dev/ttyACM0 은 root:dialout 이다. js 가 dialout 그룹에 없으면 열리지 않는다:
        sudo usermod -aG dialout $USER   (재로그인 필요)   또는
        sudo chmod 666 /dev/ttyACM0      (임시, 재부팅/재연결 시 사라짐)
  * glove_teleop.py 가 돌고 있으면 글러브 포트를 exclusive 로 잡고 있어 열 수 없다.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import serial

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
    make_timestamped_sample,
    open_sensor_serial,
    send_calibration,
    start_auto_push_stream,
)

NUM_CH = 16
SENSOR_NUM = 4
GLOVE_BAUD = 115200


class Shared:
    """리더 스레드가 채우고 출력 스레드가 읽는 최신값 보관소."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.glove = None          # list[int] 16
        self.glove_n = 0
        self.glove_err = None
        self.ft = None             # (4,3) 센서가 내는 F/T
        self.res = None            # (4,3) 촉각 127점 합력
        self.tac_max = None        # (4,)  손가락별 최대 |압력|
        self.tac_n = 0
        self.tac_err = None


def glove_reader(sh: Shared, port: str):
    """글러브 CSV 리더 — 파싱만 하고 필터는 걸지 않는다."""
    while not sh.stop.is_set():
        try:
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout = port, GLOVE_BAUD, 0.2
            ser.dtr = False           # CH340 자동리셋 방지
            ser.rts = False
            try:
                ser.exclusive = True
            except (AttributeError, ValueError):
                pass
            ser.open()
            ser.reset_input_buffer()
            with sh.lock:
                sh.glove_err = None
        except Exception as e:                                  # noqa: BLE001
            with sh.lock:
                sh.glove_err = str(e)
            time.sleep(1.0)
            continue

        buf = b""
        try:
            while not sh.stop.is_set():
                c = ser.read(1)
                if not c:
                    continue
                n = ser.in_waiting
                if n:
                    c += ser.read(n)
                buf += c
                if b"\n" not in buf:
                    continue
                *lines, buf = buf.split(b"\n")
                if len(buf) > 4096:
                    buf = b""
                for ln in lines:                                # 최신 유효 프레임만
                    parts = ln.strip().split(b",")
                    if len(parts) != NUM_CH:
                        continue
                    try:
                        vals = [int(p) for p in parts]
                    except ValueError:
                        continue
                    with sh.lock:
                        sh.glove = vals
                        sh.glove_n += 1
        except Exception as e:                                  # noqa: BLE001
            with sh.lock:
                sh.glove_err = f"끊김: {e}"
        finally:
            try:
                ser.close()
            except Exception:
                pass


def tactile_reader(sh: Shared, port: str, calibrate: bool):
    """Paxini auto-push 스트림 리더 (tools/paxini 모듈 재사용)."""
    cfg = TactileDecodeConfig()
    while not sh.stop.is_set():
        ser = None
        try:
            ser = open_sensor_serial(port, BAUDRATE, timeout=1.0)
            if calibrate:
                send_calibration(ser)
            if not start_auto_push_stream(ser):
                raise RuntimeError("auto-push enable 실패")
            with sh.lock:
                sh.tac_err = None
        except Exception as e:                                  # noqa: BLE001
            with sh.lock:
                sh.tac_err = str(e)
            try:
                if ser is not None and ser.is_open:
                    ser.close()
            except Exception:
                pass
            time.sleep(1.0)
            continue

        buf = bytearray()
        seq = 0
        try:
            while not sh.stop.is_set():
                waiting = ser.in_waiting
                chunk = ser.read(waiting if waiting > 0 else 1)
                if not chunk:
                    continue
                recv_ns = time.monotonic_ns()
                buf.extend(chunk)
                latest = None
                while True:                                     # 밀린 프레임은 버리고 최신만
                    f = _pop_auto_push_frame(buf)
                    if f is None:
                        break
                    latest = f
                if latest is None:
                    continue
                sample = make_timestamped_sample(
                    seq, latest, read_start_mono_ns=recv_ns, read_end_mono_ns=recv_ns)
                decoded = decode_auto_push_sample(sample, cfg, decode_tactile=True)
                arrays = decoded_to_numpy_arrays(decoded, cfg)
                tac = np.nan_to_num(arrays["tactile"])          # (4,127,3)
                with sh.lock:
                    sh.ft = np.nan_to_num(arrays["ft"])         # (4,3)
                    sh.res = tac.sum(axis=1)                    # (4,3) 합력
                    sh.tac_max = np.abs(tac).reshape(SENSOR_NUM, -1).max(axis=1)
                    sh.tac_n += 1
                seq += 1
        except Exception as e:                                  # noqa: BLE001
            with sh.lock:
                sh.tac_err = f"끊김: {e}"
        finally:
            try:
                if ser is not None and ser.is_open:
                    disable_auto_push(ser)
                    ser.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="글러브 + Paxini 촉각 동시 raw 모니터")
    ap.add_argument("--glove-port", default="/dev/ttyUSB0")
    ap.add_argument("--tactile-port", default="/dev/ttyACM0")
    ap.add_argument("--no-glove", action="store_true")
    ap.add_argument("--no-tactile", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="촉각 0점 보정 후 시작")
    ap.add_argument("--hz", type=float, default=2.0, help="화면 갱신 [Hz]")
    ap.add_argument("--sec", type=float, default=0.0, help="이 시간 뒤 종료 [s], 0=무한")
    a = ap.parse_args()

    sh = Shared()
    threads = []
    if not a.no_glove:
        threads.append(threading.Thread(target=glove_reader,
                                        args=(sh, a.glove_port), daemon=True))
    if not a.no_tactile:
        threads.append(threading.Thread(target=tactile_reader,
                                        args=(sh, a.tactile_port, a.calibrate), daemon=True))
    for t in threads:
        t.start()

    t0 = time.perf_counter()
    prev = t0
    pg = pt = 0
    period = 1.0 / a.hz
    try:
        while True:
            time.sleep(period)
            now = time.perf_counter()
            dt = now - prev
            prev = now
            with sh.lock:
                g, gn, ge = sh.glove, sh.glove_n, sh.glove_err
                ft, res, tmax, tn, te = sh.ft, sh.res, sh.tac_max, sh.tac_n, sh.tac_err
            ghz, thz = (gn - pg) / dt, (tn - pt) / dt
            pg, pt = gn, tn

            print(f"\n[{now-t0:7.1f}s]  글러브 {ghz:6.1f}Hz   촉각 {thz:6.1f}Hz")
            if ge:
                print(f"  글러브 ✗ {ge}")
            elif g is None:
                print("  글러브 … 수신 대기")
            else:
                print("  글러브16: " + " ".join(f"{v:6d}" for v in g))

            if te:
                print(f"  촉각   ✗ {te}")
            elif ft is None:
                print("  촉각   … 수신 대기")
            else:
                print("  촉각 ft : " + "  ".join(
                    f"f{i}[{ft[i,0]:+7.2f},{ft[i,1]:+7.2f},{ft[i,2]:+7.2f}]"
                    for i in range(SENSOR_NUM)))
                print("  촉각합력: " + "  ".join(
                    f"f{i}[{res[i,0]:+7.1f},{res[i,1]:+7.1f},{res[i,2]:+7.1f}]"
                    for i in range(SENSOR_NUM)))
                print("  촉각최대: " + "  ".join(
                    f"f{i}={tmax[i]:7.2f}" for i in range(SENSOR_NUM)))

            if a.sec > 0 and now - t0 >= a.sec:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sh.stop.set()
        for t in threads:
            t.join(timeout=1.5)


if __name__ == "__main__":
    main()
