#!/usr/bin/env python3
"""글러브 UART 순수 raw 리더 — 필터/스무딩/클램프/램프 전혀 없음.

글러브가 실제로 어떤 보레이트로, 몇 Hz로 쏘는지 측정하는 진단용 단독 스크립트.
teleop 코드에 의존하지 않는다 (pyserial 만 필요).

  --scan   보레이트 후보를 돌며 16필드 정수줄이 깨끗하게 나오는 곳을 찾음
  기본     확정 보레이트로 N초 읽어 실측 속도(B/s, Hz) + raw 16배열 + 채널별 통계

사용:
    python3 tools/glove_raw_uart.py --scan
    python3 tools/glove_raw_uart.py --baud 115200 --sec 10
    python3 tools/glove_raw_uart.py --sec 20 --show 0     # 통계만

주의: 포트를 exclusive 로 연다. glove_teleop.py 가 돌고 있으면 먼저 끄세요.
"""
import argparse
import statistics
import sys
import time

import serial

CANDIDATES = [115200, 230400, 250000, 460800, 500000, 921600,
              1000000, 1152000, 1500000, 2000000, 3000000]
NUM_CH = 16


def open_at(port, baud):
    """DTR/RTS 를 내린 상태로 연다 (CH340 은 DTR 토글에서 보드가 리셋된다)."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.05
    ser.dtr = False
    ser.rts = False
    try:
        ser.exclusive = True
    except (AttributeError, ValueError):
        pass
    ser.open()
    return ser


def drain(ser, sec):
    """sec초간 있는 그대로 읽어 raw 바이트 전부 반환. 파싱/필터 없음.

    read(1) 로 블로킹 → in_waiting 을 한 번에 흡입. readline() 은 1바이트씩
    syscall 을 날려 고속 스트림에서 뒤처진다.
    """
    time.sleep(0.15)
    ser.reset_input_buffer()
    chunks, stamps = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        c = ser.read(1)
        if not c:
            continue
        n = ser.in_waiting
        if n:
            c += ser.read(n)
        chunks.append(c)
        stamps.append(time.perf_counter())
    return b"".join(chunks), time.perf_counter() - t0, stamps


def parse_all(raw):
    """완전한 줄만 파싱 → (전체 줄 수, 16필드 정수 프레임 리스트)."""
    lines = raw.split(b"\n")[1:-1]        # 앞/뒤 잘린 조각 버림
    frames = []
    for ln in lines:
        parts = ln.strip().split(b",")
        if len(parts) != NUM_CH:
            continue
        try:
            frames.append([int(p) for p in parts])
        except ValueError:
            continue
    return len(lines), frames


def scan(port, sec):
    print(f"포트 {port} — 보레이트 스캔 (각 {sec}s, 필터 없음)\n")
    print(f"{'baud':>9} {'B/s':>10} {'line/s':>8} {'good16/s':>9}  sample")
    out = []
    for b in CANDIDATES:
        try:
            ser = open_at(port, b)
        except Exception as e:                       # noqa: BLE001 (진단 스크립트)
            print(f"{b:>9}  OPEN FAIL: {e}")
            if "lock" in str(e) or "Errno 11" in str(e) or "Errno 16" in str(e):
                print("\n=> 포트를 다른 프로세스가 잡고 있음. 그것부터 끄세요.")
                sys.exit(2)
            continue
        try:
            raw, dt, _ = drain(ser, sec)
        finally:
            ser.close()
        total, frames = parse_all(raw)
        out.append((b, len(frames) / dt, len(raw) / dt))
        print(f"{b:>9} {len(raw)/dt:>10.0f} {total/dt:>8.1f} {len(frames)/dt:>9.1f}"
              f"  {raw[:56]!r}")

    ok = [o for o in out if o[1] > 3]
    if not ok:
        print("\n16필드 정수줄이 안 나옴 — 위 sample 로 프레임 포맷 확인 필요")
        return None
    best = max(ok, key=lambda o: o[1])
    print(f"\n=> 확정: {best[0]} baud, {best[1]:.1f} good-line/s, {best[2]:.0f} B/s")
    print("   (스캔은 baud 마다 포트를 여닫아 보드가 리셋된다 → 여기 Hz 는 참고용."
          "\n    정확한 속도는 --baud 로 고정해서 다시 측정)")
    return best[0]


def measure(port, baud, sec, show):
    ser = open_at(port, baud)
    try:
        raw, dt, stamps = drain(ser, sec)
    finally:
        ser.close()

    total, frames = parse_all(raw)
    nl = raw.count(b"\n")
    gaps = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]

    print(f"\n=== raw 실측 ({port} @ {baud} baud, {dt:.2f}s, 필터 없음) ===")
    print(f"바이트      : {len(raw):>10,d}  →  {len(raw)/dt:>10,.0f} B/s "
          f"({len(raw)/dt*10/baud*100:.1f}% 회선 점유)")
    print(f"개행(\\n)    : {nl:>10,d}  →  {nl/dt:>10.1f} line/s")
    print(f"완전한 줄   : {total:>10,d}  →  {total/dt:>10.1f} line/s")
    print(f"16필드 정수 : {len(frames):>10,d}  →  {len(frames)/dt:>10.1f} Hz   "
          f"(유효율 {100*len(frames)/total if total else 0:.1f}%)")
    if total:
        print(f"줄 평균길이 : {len(raw)/total:>10.1f} B")
    if gaps:
        print(f"read 간격   : 평균 {statistics.mean(gaps):.2f} ms, "
              f"중앙 {statistics.median(gaps):.2f} ms, 최대 {max(gaps):.2f} ms")
    if frames:
        print(f"\n첫 raw 16배열: {frames[0]}")

    if show and frames:
        print(f"\n--- raw 16배열 앞 {show}줄 (가공 없음) ---")
        for f in frames[:show]:
            print(" ".join(f"{v:>6d}" for v in f))

    if len(frames) >= 2:
        print(f"\n--- 채널별 raw 통계 (n={len(frames)}, 필터 없음) ---")
        print(f"{'ch':>3} {'min':>7} {'max':>7} {'mean':>8} {'std':>7}  비고")
        for ch in range(NUM_CH):
            col = [f[ch] for f in frames]
            lo, hi = min(col), max(col)
            sd = statistics.pstdev(col)
            note = []
            if lo == hi:
                note.append("고정값(미사용?)")
            if lo < 0:
                note.append("음수 → HAND_LIMITS(0,4096) 에서 0 으로 클램프됨")
            if hi > 4096:
                note.append("4096 초과 → 클램프됨")
            print(f"{ch:>3} {lo:>7d} {hi:>7d} {statistics.mean(col):>8.1f} "
                  f"{sd:>7.1f}  {', '.join(note)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=0, help="0 = 스캔으로 자동 확정")
    ap.add_argument("--sec", type=float, default=10.0, help="측정 시간 [s]")
    ap.add_argument("--scan-sec", type=float, default=1.0)
    ap.add_argument("--scan", action="store_true", help="스캔만 하고 종료")
    ap.add_argument("--show", type=int, default=10, help="raw 16배열 출력 줄 수 (0=끔)")
    a = ap.parse_args()

    b = a.baud
    if a.scan or not b:
        b = scan(a.port, a.scan_sec)
        if a.scan or not b:
            sys.exit(0 if b else 1)
    measure(a.port, b, a.sec, a.show)
