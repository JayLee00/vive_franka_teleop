#!/usr/bin/env python3
"""글러브 16관절 각도 라이브 모니터 — 관절 이름 + raw count + 도(deg).

UART 를 직접 읽으므로 ROS 를 안 켜도 된다 (pyserial 만 필요).
관절 이름·채널·클램프 한계는 tools/glove_teleop.py 의 JOINTS / HAND_LIMITS 와
같은 값을 여기에 복사해 뒀다. glove_teleop.py 는 import 하지 않는다 — 거기는
rclpy 를 top-level 로 import 해서, 이 모니터가 ROS 환경까지 요구하게 된다.

  글러브  /dev/ttyUSB0  500000 baud   "j0,...,j15\\n" 정수 16개
  단위    1 count = pi/8192 rad  →  4096 count = 90 deg

사용:
    python3 tools/glove_joint_monitor.py
    python3 tools/glove_joint_monitor.py --hz 5 --sec 30
    python3 tools/glove_joint_monitor.py --port /dev/ttyUSB1

주의:
  * glove_teleop.py 가 돌고 있으면 글러브 포트를 exclusive 로 잡고 있어 열 수 없다.
    그때는 tools/glove_paxini_monitor.py (ROS 토픽 구독) 를 쓴다.
  * 이 글러브는 USB 가 물리적으로 끊긴다(커널 urb -32). 끊기면 사유를 찍고 끝낸다.
"""
from __future__ import annotations

import argparse
import collections
import sys
import threading
import time

import serial

NUM_CH = 16
DEG_PER_COUNT = 180.0 / 8192.0        # 1 count = pi/8192 rad

# tools/glove_teleop.py JOINTS 의 (hand_idx, 이름, glove_ch) 부분.
NAMES = [
    "thumb_cmc_opposition", "thumb_cmc_abduction", "thumb_mcp", "thumb_ip",
    "index_mcp_abduction", "index_mcp_flexion", "index_pip", "index_dip",
    "middle_mcp_abduction", "middle_mcp_flexion", "middle_pip", "middle_dip",
    "ring_mcp_abduction", "ring_mcp_flexion", "ring_pip", "ring_dip",
]

# tools/glove_teleop.py HAND_LIMITS — 텔레옵이 최종 클램프하는 구간 [count].
LIMITS = {i: (0, 4096) for i in range(NUM_CH)}
LIMITS[1] = (-4096, 4096)
for _i in (4, 8, 12):
    LIMITS[_i] = (-1000, 1000)
for _i in (3, 7, 11, 15):
    LIMITS[_i] = (-2048, 4096)

WIN = 120                             # 통계 창 [프레임] ≈ 1.6 s @ 75 Hz
BAR_W = 18


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.hist = [collections.deque(maxlen=WIN) for _ in range(NUM_CH)]
        self.n = 0
        self.bad = 0
        self.err = None
        self.stop = False


def reader(sh: Shared, port: str, baud: str):
    try:
        ser = serial.Serial(port, baud, timeout=0.2)
    except (serial.SerialException, OSError) as e:
        with sh.lock:
            sh.err = f"포트 열기 실패: {e}"
        return

    buf = b""
    try:
        with ser:
            ser.reset_input_buffer()
            buf = ser.readline()          # 앞의 잘린 조각 버림
            buf = b""
            while not sh.stop:
                chunk = ser.read(max(1, ser.in_waiting))
                if not chunk:
                    continue
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    parts = ln.strip().split(b",")
                    if len(parts) != NUM_CH:
                        with sh.lock:
                            sh.bad += 1
                        continue
                    try:
                        vals = [int(p) for p in parts]
                    except ValueError:
                        with sh.lock:
                            sh.bad += 1
                        continue
                    with sh.lock:
                        sh.frame = vals
                        sh.n += 1
                        for ch, v in enumerate(vals):
                            sh.hist[ch].append(v)
    except (serial.SerialException, OSError) as e:
        with sh.lock:
            sh.err = f"USB 끊김: {e}  → journalctl -k | grep -E 'urb stopped|USB disconnect'"


def bar(v: int, lo: int, hi: int) -> str:
    frac = (v - lo) / (hi - lo) if hi > lo else 0.0
    filled = max(0, min(BAR_W, round(frac * BAR_W)))
    if frac < 0:
        return "<" + "·" * (BAR_W - 1)
    if frac > 1:
        return "·" * (BAR_W - 1) + ">"
    return "█" * filled + "·" * (BAR_W - filled)


def render(sh: Shared, port: str, baud: int, elapsed: float, hz: float) -> str:
    with sh.lock:
        frame = list(sh.frame) if sh.frame else None
        hist = [list(h) for h in sh.hist]
        n, bad, err = sh.n, sh.bad, sh.err

    out = [f"=== 글러브 16관절 각도  {port} @ {baud}  "
           f"{hz:6.1f} Hz  t={elapsed:6.1f}s  프레임 {n:,d} (불량 {bad}) ==="]
    if err:
        out.append(f"!! {err}")
    if frame is None:
        out.append("… 수신 대기 (글러브 전원 / baud 확인)")
        return "\n".join(out)

    out.append(f"{'ch':>2}  {'관절':<22} {'raw':>6} {'deg':>7}  "
               f"{'min~max':>13}  {'std':>5}  {'클램프구간':<20} 비고")
    for ch in range(NUM_CH):
        v = frame[ch]
        lo, hi = LIMITS[ch]
        h = hist[ch]
        mn, mx = (min(h), max(h)) if h else (v, v)
        if len(h) >= 2:
            mean = sum(h) / len(h)
            std = (sum((x - mean) ** 2 for x in h) / len(h)) ** 0.5
        else:
            std = 0.0

        flags = []
        if len(h) >= 20 and std == 0.0:
            flags.append("고정(센서 끊김?)")
        if v < lo or v > hi:
            flags.append(f"클램프됨(→{min(max(v, lo), hi)})")
        elif v in (lo, hi):
            flags.append("한계에 붙음")

        out.append(f"{ch:>2}  {NAMES[ch]:<22} {v:>6d} {v * DEG_PER_COUNT:>6.1f}° "
                   f"{mn:>6d}~{mx:<6d} {std:>5.1f}  "
                   f"{bar(v, lo, hi)} {'':1}{' '.join(flags)}")
    out.append(f"      1 count = pi/8192 rad  (4096 count = 90°)   "
               f"막대 = [{'채널별 클램프구간'}]   Ctrl-C 종료")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=500000,
                    help="글러브 펌웨어 Serial.begin() 과 일치해야 함")
    ap.add_argument("--hz", type=float, default=4.0, help="화면 갱신 [Hz]")
    ap.add_argument("--sec", type=float, default=0.0, help="0 = 무한")
    a = ap.parse_args()

    sh = Shared()
    th = threading.Thread(target=reader, args=(sh, a.port, a.baud), daemon=True)
    th.start()

    tty = sys.stdout.isatty()
    t0 = time.perf_counter()
    prev_t, prev_n = t0, 0
    try:
        while True:
            time.sleep(1.0 / a.hz)
            now = time.perf_counter()
            with sh.lock:
                n, err = sh.n, sh.err
            hz = (n - prev_n) / (now - prev_t) if now > prev_t else 0.0
            prev_t, prev_n = now, n

            text = render(sh, a.port, a.baud, now - t0, hz)
            if tty:
                sys.stdout.write("\033[H\033[J" + text + "\n")
            else:
                sys.stdout.write("\n" + text + "\n")
            sys.stdout.flush()

            if err and sh.frame is None:
                return 1
            if a.sec and now - t0 >= a.sec:
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        sh.stop = True
        th.join(timeout=0.5)


if __name__ == "__main__":
    sys.exit(main())
