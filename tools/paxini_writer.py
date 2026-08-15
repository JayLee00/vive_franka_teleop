#!/usr/bin/env python3
"""Paxini 촉각 센서 → SHMmsgs(key 0x7951) writer.

V1.0 프로젝트(paxini_driver)의 UART 프로토콜 모듈(tools/paxini/)을 재사용해
센서 스트림(~90Hz)을 파싱하고, 메인 SHMmsgs의 Paxini_tac / Paxini_ft /
Paxini_seq 필드에 직접 기록한다 (V1.0의 별도 세그먼트 0x3934 방식 대체).

쓰기 순서 (seqlock, inc/shm.h 주석과 동일):
    Paxini_seq[hand] = 홀수  →  데이터 기록  →  Paxini_seq[hand] = 짝수
독자는 seq가 홀수이거나 읽기 전후 값이 다르면 재시도한다.

실행 예:
    python3 tools/paxini_writer.py --hand r                # R손, 포트 자동탐색
    python3 tools/paxini_writer.py --hand l --port /dev/ttyACM1
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import sysv_ipc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paxini"))

from tactile_uart import (  # noqa: E402
    TactileDecodeConfig,
    _pop_auto_push_frame,
    decode_auto_push_sample,
    decoded_to_numpy_arrays,
    choose_serial_port,
)
from Read_Single_Sensor_Hand import (  # noqa: E402
    BAUDRATE,
    disable_auto_push,
    make_timestamped_sample,
    open_sensor_serial,
    send_calibration,
    start_auto_push_stream,
)

SHM_KEY = 0x7951
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENSOR_NUM = 4
POINT_NUM = 127
AXIS_NUM = 3
TAC_BYTES = SENSOR_NUM * POINT_NUM * AXIS_NUM * 4   # float32
FT_BYTES = SENSOR_NUM * AXIS_NUM * 4


def dump_shm_offsets() -> dict:
    """inc/shm.h에서 offsetof/sizeof를 g++로 추출 (하드코딩 방지)."""
    src = r"""
#include <cstdio>
#include <cstddef>
#include "shm.h"
int main() {
  printf("{\"size\":%zu,\"Paxini_tac\":%zu,\"Paxini_ft\":%zu,\"Paxini_seq\":%zu}\n",
         sizeof(SHMmsgs), offsetof(SHMmsgs, Paxini_tac),
         offsetof(SHMmsgs, Paxini_ft), offsetof(SHMmsgs, Paxini_seq));
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as td:
        cpp = os.path.join(td, "dump.cpp")
        exe = os.path.join(td, "dump")
        with open(cpp, "w") as f:
            f.write(src)
        subprocess.run(
            ["g++", "-std=c++17", "-I", os.path.join(PROJECT_ROOT, "inc"),
             "-Wno-invalid-offsetof", cpp, "-o", exe],
            check=True, capture_output=True)
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


class ShmPaxiniWriter:
    def __init__(self, hand_index: int):
        self.hand = hand_index
        off = dump_shm_offsets()
        self.size = off["size"]
        self.off_tac = off["Paxini_tac"] + hand_index * TAC_BYTES
        self.off_ft = off["Paxini_ft"] + hand_index * FT_BYTES
        self.off_seq = off["Paxini_seq"] + hand_index * 8
        self._seq = 0
        try:
            self.shm = sysv_ipc.SharedMemory(SHM_KEY)
        except sysv_ipc.ExistentialError:
            self.shm = sysv_ipc.SharedMemory(SHM_KEY, sysv_ipc.IPC_CREAT,
                                             mode=0o666, size=self.size)
        if self.shm.size < self.size:
            raise RuntimeError(
                f"기존 SHM 크기({self.shm.size})가 구조체({self.size})보다 작음 — "
                "구버전 세그먼트입니다. 전 프로세스 종료 후 `ipcrm -M 0x7951`로 제거하세요.")
        print(f"[paxini_writer] SHM attach OK (key=0x7951, size={self.shm.size}, "
              f"hand={self.hand}, tac_off={self.off_tac})", flush=True)

    def publish(self, tactile: np.ndarray, ft: np.ndarray) -> None:
        self._seq += 1                                             # 홀수 = 쓰는 중
        self.shm.write(struct.pack("<Q", self._seq), self.off_seq)
        self.shm.write(np.nan_to_num(tactile).astype("<f4").tobytes(), self.off_tac)
        self.shm.write(np.nan_to_num(ft).astype("<f4").tobytes(), self.off_ft)
        self._seq += 1                                             # 짝수 = 완료
        self.shm.write(struct.pack("<Q", self._seq), self.off_seq)


def run(hand_index: int, port: str | None, calibrate: bool, print_period: float) -> None:
    writer = ShmPaxiniWriter(hand_index)
    decode_cfg = TactileDecodeConfig()
    backoff = 0.5
    ser = None

    while True:
        # ── 시리얼 연결 (실패 시 백오프 재시도) ────────────────────────────
        try:
            resolved = choose_serial_port(port)
            ser = open_sensor_serial(resolved, BAUDRATE, timeout=1.0)
            if calibrate:
                ok = send_calibration(ser)
                print(f"[paxini_writer] calibration {'OK' if ok else 'FAILED (continuing)'}",
                      flush=True)
            if not start_auto_push_stream(ser):
                raise RuntimeError("auto-push enable 실패")
            print(f"[paxini_writer] streaming: port={resolved} hand={hand_index}", flush=True)
            backoff = 0.5
        except Exception as exc:
            print(f"[paxini_writer] connect 실패: {exc} — {backoff:.1f}s 후 재시도", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)
            continue

        # ── 수신 루프: 최신 프레임만 유지 → 디코드 → SHM 기록 ─────────────
        buf = bytearray()
        seq = 0
        next_print = 0.0
        try:
            while True:
                waiting = ser.in_waiting
                chunk = ser.read(waiting if waiting > 0 else 1)
                if not chunk:
                    continue
                recv_ns = time.monotonic_ns()
                buf.extend(chunk)
                latest = None
                while True:
                    frame = _pop_auto_push_frame(buf)
                    if frame is None:
                        break
                    latest = frame
                if latest is None:
                    continue

                sample = make_timestamped_sample(
                    seq, latest, read_start_mono_ns=recv_ns, read_end_mono_ns=recv_ns)
                decoded = decode_auto_push_sample(sample, decode_cfg, decode_tactile=True)
                arrays = decoded_to_numpy_arrays(decoded, decode_cfg)
                writer.publish(arrays["tactile"], arrays["ft"])
                seq += 1

                now = time.monotonic()
                if print_period > 0 and now >= next_print:
                    res = np.nan_to_num(arrays["tactile"]).sum(axis=1)  # (4,3) 합력
                    parts = " ".join(
                        f"f{i}=[{res[i,0]:+.2f},{res[i,1]:+.2f},{res[i,2]:+.2f}]"
                        for i in range(SENSOR_NUM))
                    print(f"[paxini hand{hand_index}] seq={seq} N: {parts}", flush=True)
                    next_print = now + print_period
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[paxini_writer] read 오류: {exc} — 재연결", flush=True)
        finally:
            try:
                if ser is not None and ser.is_open:
                    disable_auto_push(ser)
                    ser.close()
            except Exception:
                pass

    print("[paxini_writer] 종료", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paxini → SHMmsgs writer")
    ap.add_argument("--hand", choices=["r", "l"], required=True, help="r=hand0, l=hand1")
    # TODO(사용자 확인 필요): L손 Paxini 장착 여부와 포트 매핑(/dev/ttyACM0/1).
    # udev 규칙으로 by-id 심볼릭 링크를 만들어 --port에 지정하는 것을 권장.
    ap.add_argument("--port", default=None, help="시리얼 포트 (기본: 자동 탐색)")
    # ap.add_argument("--calibrate", action="store_true", help="시작 시 센서 보정 1회")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false", default=True, help="시작 시 센서 보정 건너뜀(기본 설정은 센서 보정 후 시작)")
    ap.add_argument("--print-period", type=float, default=2.0, help="합력 출력 주기 [s], 0=끔")
    args = ap.parse_args()

    run(0 if args.hand == "r" else 1, args.port, args.calibrate, args.print_period)


if __name__ == "__main__":
    main()
