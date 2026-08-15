#!/usr/bin/env python3
"""
UART 통신 설정, read command, raw sample 데이터 타입, LRC 계산, 프레임 파싱, 실제 수신 함수/class 관리
"""

from __future__ import annotations

import json, os, sys
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Generator, Literal

# In-repo bootstrap: paxini_shm and Read_Single_Sensor_Hand are siblings in this
# folder (paxini_driver/). Make them importable no matter which entry script
# imports us. Self-contained — no external workspace dependency.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paxini_shm import PaxiniShmWriter, PaxiniShmReader, PAXINI_SHM_KEY

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("Read_Single_Sensor_Hand").setLevel(logging.WARNING)

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover
    serial = None


FRAME_HEAD = b"\xAA\x55"
TX_FRAME_HEAD = b"\x55\xAA"
MIN_FRAME_LEN = 14

# Auto-push (AA56) data-frame header and a sanity bound on its total length.
# Used by the dedicated reader thread to frame the sensor's continuous stream.
AUTO_PUSH_FRAME_HEAD = b"\xAA\x56"
MAX_AUTO_PUSH_FRAME_LEN = 4096

ReadMode = Literal["chunk", "frame"]
CommunicationMethod = Literal["uart"]
TactileComponents = Literal["z", "xyz"]
FT_LABELS = ("Fx", "Fy", "Fz")

# -----------------------------------------------------------------------------
# Fixed GEN3 tactile auto-push payload layout
# -----------------------------------------------------------------------------
# Number of tactile sensor blocks contained in one auto-push payload.
SENSOR_COUNT = 4

# One sensor's resultant force block length in bytes:
# Fx(2 bytes) + Fy(2 bytes) + Fz(2 bytes).
FT_BYTES = 6

# Force conversion scale. The protocol sends force in 0.1 N units.
FT_SCALE = 0.1

# Number of tactile distribution points per sensor block.
TACTILE_COUNT = 127

# Tactile conversion scale. Each tactile component is sent in 0.1 N units.
TACTILE_SCALE = 0.1

# Tactile point format. "xyz" means each point has X/Y/Z components;
# "z" keeps only the normal component.
TACTILE_COMPONENTS: TactileComponents = "xyz"

# Default side-channel path used when another process, such as hdf5_logger_1k.py,
# needs to consume the latest PaXini FT sample without opening the serial port.
DEFAULT_PAXINI_FT_STATE_FILE = Path("/tmp/gen3_paxini_ft.json")
PAXINI_FT_PRINT_PERIOD_SEC = 2.0


@dataclass
class UARTReadCommand:
    device_id: int = 0x01
    func_code: int = 0x7B
    addr: int = 0x040E
    data_len: int = 0x20


@dataclass
class TactileUARTConfig:
    port: str
    baudrate: int = 115200
    timeout: float = 0.05
    read_size: int = 4096
    mode: ReadMode = "frame"
    communication: CommunicationMethod = "uart"
    poll_read: bool = False
    poll_interval_ms: float = 20.0
    read_command: UARTReadCommand = field(default_factory=UARTReadCommand)


@dataclass
class TactileDecodeConfig:
    sensor_count: int = SENSOR_COUNT
    sensor_stride: int | None = None
    payload_offset: int = 0
    ft_offset: int = 0
    ft_bytes: int = FT_BYTES
    ft_scale: float = FT_SCALE
    tactile_offset: int | None = None
    tactile_count: int = TACTILE_COUNT
    tactile_scale: float = TACTILE_SCALE
    tactile_components: TactileComponents = TACTILE_COMPONENTS

    @property
    def resolved_sensor_stride(self) -> int:
        if self.sensor_stride is not None:
            return self.sensor_stride
        return self.ft_bytes + self.tactile_count * 3

    @property
    def resolved_tactile_offset(self) -> int:
        if self.tactile_offset is not None:
            return self.tactile_offset
        return self.ft_offset + self.ft_bytes


@dataclass
class RawTactileSample:
    seq: int
    host_time_ns: int
    host_time_s: float
    monotonic_ns: int
    byte_count: int
    data: bytes
    data_hex: str
    mode: ReadMode
    lrc_ok: bool | None = None

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["data"] = self.data_hex
        return data


def lrc_cal(data: bytes) -> int:
    """Same LRC rule as UART Example.c: two's complement of byte sum."""
    return ((~sum(data) + 1) & 0xFF)


def build_read_request(command: UARTReadCommand) -> bytes:
    """Build one UART read request frame for the sensor."""
    payload = bytearray()
    payload.extend(TX_FRAME_HEAD)
    payload.extend((9).to_bytes(2, "little"))
    payload.append(command.device_id & 0xFF)
    payload.append(0x00)
    payload.append((command.func_code | 0x80) & 0xFF)
    payload.extend((command.addr & 0xFFFFFFFF).to_bytes(4, "little"))
    payload.extend((command.data_len & 0xFFFF).to_bytes(2, "little"))
    payload.append(lrc_cal(payload))
    return bytes(payload)


def write_sample_jsonl(fp: BinaryIO | None, sample: RawTactileSample) -> None:
    if fp is None:
        return
    line = json.dumps(sample.to_json_dict(), ensure_ascii=False).encode("utf-8") + b"\n"
    fp.write(line)
    fp.flush()


def signed_u8(value: int) -> int:
    return value if value <= 127 else value - 256


def scaled(value: int, scale: float) -> float:
    return round(value * scale, 3)


def parse_ft_values(payload: bytes, offset: int, scale: float) -> dict[str, Any]:
    raw_bytes: list[int | None] = []
    raw_hex: list[str | None] = []
    values: list[float | None] = []

    for axis_index in range(3):
        pos = offset + axis_index * 2
        if pos + 1 >= len(payload):
            raw_bytes.extend([None, None])
            raw_hex.append(None)
            values.append(None)
            continue

        low = payload[pos]
        high = payload[pos + 1]
        raw_bytes.extend([low, high])
        raw_hex.append(payload[pos:pos + 2].hex().upper())

        if axis_index < 2:
            raw = signed_u8(low)
        else:
            raw = low
        values.append(scaled(raw, scale))

    return {
        "labels": list(FT_LABELS),
        "values": values,
        "raw_bytes": raw_bytes,
        "raw_hex": raw_hex,
    }


def parse_tactile_values(
    payload: bytes,
    offset: int,
    count: int,
    scale: float,
    components: TactileComponents,
) -> list[float | list[float | None] | None]:
    values: list[float | list[float | None] | None] = []
    for index in range(count):
        pos = offset + index * 3
        if pos + 2 >= len(payload):
            values.append(None)
            continue

        x = scaled(signed_u8(payload[pos]), scale)
        y = scaled(signed_u8(payload[pos + 1]), scale)
        z = scaled(payload[pos + 2], scale)

        if components == "xyz":
            values.append([x, y, z])
        else:
            values.append(z)
    return values


def decode_payload(payload: bytes, config: TactileDecodeConfig,
                   decode_tactile: bool = True) -> dict[str, Any]:
    """Decode one auto-push payload.

    ``decode_tactile=False`` parses only the per-sensor resultant FT block
    (6 bytes each) and skips the 127-point distribution loop. This is the hot
    path for force control/SHM, where the 127x3 tactile field is not needed and
    parsing it every frame is the dominant per-frame cost.
    """
    ft_groups = []
    tactile_groups = []
    sensor_offsets = []

    for sensor_index in range(config.sensor_count):
        sensor_base = config.payload_offset + sensor_index * config.resolved_sensor_stride
        sensor_offsets.append(sensor_base)
        ft_groups.append(
            parse_ft_values(payload, sensor_base + config.ft_offset, config.ft_scale)
        )
        if decode_tactile:
            tactile_groups.append(
                parse_tactile_values(
                    payload,
                    sensor_base + config.resolved_tactile_offset,
                    config.tactile_count,
                    config.tactile_scale,
                    config.tactile_components,
                )
            )

    return {
        "ft_labels": list(FT_LABELS),
        "ft": ft_groups,
        "tactile": tactile_groups,
        "tactile_components": config.tactile_components,
        "payload_offset": config.payload_offset,
        "ft_offset": config.ft_offset,
        "tactile_offset": config.resolved_tactile_offset,
        "sensor_count": config.sensor_count,
        "sensor_stride": config.resolved_sensor_stride,
        "sensor_offsets": sensor_offsets,
        "expected_payload_len": config.sensor_count * config.resolved_sensor_stride,
        "actual_payload_len": len(payload),
    }


def decode_auto_push_sample(sample: dict[str, Any], config: TactileDecodeConfig,
                            decode_tactile: bool = True) -> dict[str, Any]:
    parsed = sample.get("parsed") or {}
    payload = parsed.get("valid_data")
    if not isinstance(payload, (bytes, bytearray)):
        payload = b""
    return decode_payload(bytes(payload), config, decode_tactile=decode_tactile)


def build_sync_record(
    sample: dict[str, Any],
    decoded: dict[str, Any],
    config: TactileDecodeConfig,
) -> dict[str, Any]:
    """Build the compact JSONL record used for sensor synchronization.

    The record intentionally excludes raw hex payloads and repeated schema
    fields. For cross-sensor alignment, use ``t_mono_ns`` as the primary time
    base because it comes from ``time.monotonic_ns()`` and is not affected by
    wall-clock changes.
    """
    parsed = sample.get("parsed") or {}
    lrc_ok = sample.get("lrc_ok")
    ft_values = [
        list(ft_info.get("values", []))
        for ft_info in decoded.get("ft", [])
    ]
    has_ft = any(
        any(value is not None for value in sensor_ft)
        for sensor_ft in ft_values
    )

    return {
        # Sequential sample index from this reader. Useful for detecting drops.
        "seq": int(sample.get("seq", 0)),

        # Primary synchronization timestamp in nanoseconds.
        # This is the midpoint of the host-side serial read window.
        "t_mono_ns": int(sample.get("t_mono_ns", sample.get("monotonic_ns", 0))),

        # Wall-clock timestamp in nanoseconds. Useful for human-readable logs,
        # but do not use it as the primary sync clock.
        "t_wall_ns": int(sample.get("t_wall_ns", sample.get("host_time_ns", 0))),

        # True when checksum passed and at least one FT block was parsed.
        # This stays usable while testing 1 sensor now and 4 sensors later.
        "valid": bool(lrc_ok) and has_ft,

        # Firmware/protocol error code from the auto-push frame, if present.
        "error_code": parsed.get("error_code"),

        # Resultant force per sensor:
        # shape = [sensor_count][Fx, Fy, Fz], unit = N.
        "ft": ft_values,

        # Tactile distribution data per sensor:
        # shape = [sensor_count][tactile_count][x, y, z] when components="xyz".
        # unit = N.
        "tactile": decoded.get("tactile", []),
    }


def tactile_resultant_ft(decoded: dict[str, Any], sensor_count: int = SENSOR_COUNT) -> np.ndarray:
    """Resultant force (합력) per sensor = vector sum of all tactile points.

    ``decoded["tactile"]`` has shape ``[sensor_count][tactile_count][x, y, z]``
    (each point already in N). This sums the 127 points of every sensor into one
    (Fx, Fy, Fz), returning an array of shape ``(sensor_count, 3)`` — each
    finger's resultant xyz force.
    """
    groups = decoded.get("tactile", [])
    out = np.zeros((sensor_count, 3), dtype=np.float64)
    for s in range(min(sensor_count, len(groups))):
        pts = groups[s]
        if not pts:
            continue
        arr = np.array(
            [p for p in pts
             if isinstance(p, (list, tuple)) and len(p) == 3 and None not in p],
            dtype=np.float64,
        )
        if arr.size:
            out[s] = arr.sum(axis=0)
    return out


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object atomically so a logger never reads a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def choose_serial_port(requested_port: str | None = None) -> str:
    """Return the requested port or the first likely USB/ACM serial port."""
    if requested_port:
        return requested_port
    if serial is None:
        raise RuntimeError("pyserial이 필요합니다. 설치 예: python -m pip install pyserial")

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found for PaXini sensor.")

    preferred = [
        port for port in ports
        if port.device.startswith("/dev/ttyACM") or port.device.startswith("/dev/ttyUSB")
    ]
    return (preferred[0] if preferred else ports[0]).device


class PaxiniFTSideChannelWriter:
    """Background PaXini reader that publishes latest FT values to JSON.

    Minimal use from another script:

        writer = PaxiniFTSideChannelWriter(state_file="/tmp/gen3_paxini_ft.json")
        writer.start()
        ...
        writer.stop()

    The written JSON follows ``build_sync_record(...)`` but excludes tactile
    arrays by default, because hdf5_logger_1k.py currently stores only FT.
    """

    def __init__(
        self,
        state_file: str | Path = DEFAULT_PAXINI_FT_STATE_FILE,
        port: str | None = None,
        print_period_sec: float = PAXINI_FT_PRINT_PERIOD_SEC,
        include_tactile: bool = False,
        decode_config: TactileDecodeConfig | None = None,
        shm_key: int | None = None,
        write_json: bool = True,
        json_rate_hz: float = 30.0,
        calibrate: bool = False,
        calib_cmd: str | None = None,
    ):
        self.state_file = Path(state_file)
        self.port = port
        self.print_period_sec = max(0.1, float(print_period_sec))
        self.include_tactile = bool(include_tactile)
        self.decode_config = decode_config or TactileDecodeConfig()
        # Optional one-shot sensor calibration sent right after the UART opens and
        # before auto-push streaming starts. Mirrors Hand_UI.py's "보정 시작".
        self.calibrate = bool(calibrate)
        self.calib_cmd = calib_cmd
        # Optional: also publish FT to a dedicated PaXini shared-memory segment.
        # SHM is the primary/low-latency path; JSON stays as a throttled fallback
        # (the realtime viewer reads JSON, and logger/sync-test use it if SHM is
        # absent). Set write_json=False to drop the JSON side-channel entirely.
        self.shm_key = shm_key
        self.write_json = bool(write_json)
        self._json_rate_hz = max(0.0, float(json_rate_hz))
        self._shm_writer = None

        # Two-thread design: a reader thread drains the UART and keeps only the
        # newest complete frame; a publisher thread decodes that frame, writes
        # SHM every time and JSON at a throttled rate. Decoupling them keeps the
        # UART buffer empty (minimal staleness) and lets throughput reach the
        # sensor's native rate instead of being gated by per-frame work.
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: tuple[int, int, bytes] | None = None   # (seq, recv_mono_ns, raw_frame)
        self._reader_thread: threading.Thread | None = None
        self._publisher_thread: threading.Thread | None = None
        self._ser = None
        self._make_timestamped_sample = None
        self._disable_auto_push = None

    def start(self) -> None:
        if self._reader_thread is not None:
            return
        try:
            from Read_Single_Sensor_Hand import (
                BAUDRATE,
                disable_auto_push,
                make_timestamped_sample,
                open_sensor_serial,
                send_calibration,
                start_auto_push_stream,
            )
        except Exception as exc:
            self._write_invalid(f"import failed: {exc}")
            print(f"[PaXini FT] disabled: {exc}", flush=True)
            return

        self._make_timestamped_sample = make_timestamped_sample
        self._disable_auto_push = disable_auto_push
        # 리더 스레드가 USB 끊김 후 재연결할 때 다시 쓸 수 있도록 헬퍼를 보관.
        self._baudrate = BAUDRATE
        self._open_sensor_serial = open_sensor_serial
        self._send_calibration = send_calibration
        self._start_auto_push_stream = start_auto_push_stream
        self._maybe_open_shm()
        try:
            self._arm_serial(verbose=True)
        except Exception as exc:
            self._write_invalid(str(exc))
            print(f"[PaXini FT] reader start failed: {exc}", flush=True)
            self._cleanup_serial()
            self._close_shm()
            return

        self._stop_event.clear()
        self._frame_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="PaxiniReader", daemon=True)
        self._publisher_thread = threading.Thread(
            target=self._publish_loop, name="PaxiniPublisher", daemon=True)
        self._reader_thread.start()
        self._publisher_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._frame_event.set()                # wake the publisher so it can exit
        for t in (self._publisher_thread, self._reader_thread):
            if t is not None:
                t.join(timeout=2.0)
        self._publisher_thread = None
        self._reader_thread = None
        self._cleanup_serial()
        self._close_shm()

    def _write_invalid(self, error: str) -> None:
        if self.write_json:
            atomic_write_json(
                self.state_file,
                {
                    "seq": -1,
                    "t_mono_ns": time.monotonic_ns(),
                    "t_wall_ns": time.time_ns(),
                    "valid": False,
                    "error_code": None,
                    "error": error,
                    "ft": [],
                },
            )
        if self._shm_writer is not None:
            try:
                self._shm_writer.publish_invalid()
            except Exception:
                pass

    def _maybe_open_shm(self) -> None:
        if self.shm_key is None:
            return
        try:
            from paxini_shm import PaxiniShmWriter
            self._shm_writer = PaxiniShmWriter(self.shm_key)
            self._shm_writer.attach_create()
            print(f"[PaXini FT] SHM publish enabled key={hex(self.shm_key)}", flush=True)
        except Exception as exc:
            self._shm_writer = None
            print(f"[PaXini FT] SHM publish disabled ({exc}); JSON only", flush=True)

    def _close_shm(self) -> None:
        if self._shm_writer is not None:
            try:
                self._shm_writer.close(remove=True)
            except Exception:
                pass
            self._shm_writer = None

    def _cleanup_serial(self) -> None:
        ser = self._ser
        if ser is not None and getattr(ser, "is_open", False):
            try:
                if self._disable_auto_push is not None:
                    self._disable_auto_push(ser)
            except Exception:
                pass
            try:
                ser.close()
            except Exception:
                pass
            print("[PaXini FT] UART closed", flush=True)
        self._ser = None

    def _arm_serial(self, verbose: bool = False) -> None:
        """Open the UART, calibrate, and enable auto-push. Raises on failure.

        start() 와 리더 스레드의 재연결 경로가 공유한다.
        """
        port = choose_serial_port(self.port)
        self.port = port
        self._ser = self._open_sensor_serial(port, self._baudrate, timeout=1.0)
        if verbose:
            print(f"[PaXini FT] UART open: port={self._ser.name} baudrate={self._baudrate}", flush=True)
        # One-shot calibration must run before auto-push: the AA55 response
        # would otherwise be buried in the AA56 auto-push stream.
        if self.calibrate:
            if verbose:
                print("[PaXini FT] calibrating sensor board ...", flush=True)
            ok = self._send_calibration(self._ser, self.calib_cmd) if self.calib_cmd \
                else self._send_calibration(self._ser)
            if verbose:
                print(f"[PaXini FT] calibration {'OK' if ok else 'FAILED (continuing)'}",
                      flush=True)
        if not self._start_auto_push_stream(self._ser):
            raise RuntimeError("auto-push enable failed")
        if verbose:
            print(f"[PaXini FT] auto-push enabled, state_file={self.state_file} "
                  f"json={'on@%.0fHz' % self._json_rate_hz if self.write_json else 'off'}", flush=True)

    def _reconnect_serial(self) -> bool:
        """USB 끊김 후 시리얼을 닫고 재연결(open+calibrate+enable)을 시도한다.

        성공하면 True. 중지 요청이 오면 False. 성공할 때까지 백오프로 재시도한다.
        """
        self._cleanup_serial()
        backoff = 0.5
        while not self._stop_event.is_set():
            try:
                self._arm_serial(verbose=False)
                print(f"[PaXini FT] reader reconnected: port={self.port}", flush=True)
                return True
            except Exception as exc:
                self._write_invalid(f"reconnect failed: {exc}")
                # 중지 신호에 빠르게 반응하도록 짧게 끊어서 대기
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 5.0)
        return False

    def _reader_loop(self) -> None:
        """Drain the UART and keep only the newest complete frame.

        Reads everything currently buffered, frames all complete AA56 packets,
        and stores just the last (freshest) one. Surplus older frames are
        intentionally dropped so the consumer never falls behind and staleness
        stays near one frame period.

        USB 끊김 등으로 read 가 실패하면 스레드를 끝내지 않고 재연결을 시도한다.
        """
        buf = bytearray()
        seq = 0
        try:
            while not self._stop_event.is_set():
                ser = self._ser
                try:
                    waiting = ser.in_waiting
                    # read(>=1) blocks up to the serial timeout when idle (no busy spin)
                    chunk = ser.read(waiting if waiting > 0 else 1)
                except Exception as exc:
                    # 일시적 USB 끊김/포트 오류 → 재연결 시도(영구히 죽지 않음)
                    print(f"[PaXini FT] reader read error: {exc} → reconnecting", flush=True)
                    buf.clear()
                    if not self._reconnect_serial():
                        break          # 중지 요청
                    continue
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
                with self._lock:
                    self._latest = (seq, recv_ns, latest)
                seq += 1
                self._frame_event.set()
        except Exception as exc:
            self._write_invalid(str(exc))
            print(f"[PaXini FT] reader stopped: {exc}", flush=True)
        finally:
            self._stop_event.set()
            self._frame_event.set()            # release the publisher

    def _publish_loop(self) -> None:
        """Decode the newest frame and publish to SHM (every frame) + JSON (throttled)."""
        next_print_time = 0.0
        next_json_time = 0.0
        json_period = 1.0 / self._json_rate_hz if self._json_rate_hz > 0 else 0.0
        while not self._stop_event.is_set():
            if not self._frame_event.wait(timeout=0.2):
                continue
            self._frame_event.clear()
            with self._lock:
                latest = self._latest
            if latest is None:
                continue
            seq, recv_ns, raw_frame = latest

            now = time.monotonic()
            do_print = now >= next_print_time
            # The 127x3 distribution is needed for the periodic print, when
            # include_tactile is set, OR when publishing to SHM (the SHM segment
            # now carries the full tactile distribution, not the resultant FT).
            need_tactile = self.include_tactile or do_print or self._shm_writer is not None

            sample = self._make_timestamped_sample(
                seq, raw_frame, read_start_mono_ns=recv_ns, read_end_mono_ns=recv_ns)
            decoded = decode_auto_push_sample(
                sample, self.decode_config, decode_tactile=need_tactile)
            record = build_sync_record(sample, decoded, self.decode_config)

            # SHM: publish the freshest tactile distribution (4,127,3) every frame.
            if self._shm_writer is not None:
                try:
                    self._shm_writer.publish(
                        seq=record.get("seq", seq),
                        t_mono_ns=record.get("t_mono_ns", time.monotonic_ns()),
                        t_wall_ns=record.get("t_wall_ns", time.time_ns()),
                        valid=bool(record.get("valid", True)),
                        error_code=record.get("error_code"),
                        tactile=record.get("tactile"),
                    )
                except Exception:
                    pass

            # JSON: throttled fallback (realtime viewer + logger/sync-test fallback).
            if self.write_json and now >= next_json_time:
                json_record = dict(record)
                if not self.include_tactile:
                    json_record.pop("tactile", None)
                atomic_write_json(self.state_file, json_record)
                next_json_time = now + json_period

            if do_print:
                # (4, 127, 3) tactile -> (4, 3) resultant force per finger
                res = tactile_resultant_ft(decoded, self.decode_config.sensor_count)
                parts = " ".join(
                    f"f{i}=[{res[i, 0]:+.2f},{res[i, 1]:+.2f},{res[i, 2]:+.2f}]"
                    for i in range(res.shape[0])
                )
                print(f"[PaXini] resultant(4x3) N: {parts}", flush=True)
                next_print_time = now + self.print_period_sec


def start_paxini_ft_side_channel(
    state_file: str | Path = DEFAULT_PAXINI_FT_STATE_FILE,
    port: str | None = None,
    print_period_sec: float = PAXINI_FT_PRINT_PERIOD_SEC,
    include_tactile: bool = False,
    decode_config: TactileDecodeConfig | None = None,
    shm_key: int | None = None,
    write_json: bool = True,
    json_rate_hz: float = 30.0,
    calibrate: bool = False,
    calib_cmd: str | None = None,
) -> PaxiniFTSideChannelWriter:
    """Create and start a PaXini FT side-channel writer.

    This is the simplest API for motion scripts:

        writer = start_paxini_ft_side_channel()
        ...
        writer.stop()

    Pass ``shm_key`` to additionally publish FT into a dedicated PaXini
    shared-memory segment (the primary low-latency path). The JSON file is a
    throttled fallback (``json_rate_hz``); set ``write_json=False`` to disable it.
    """
    writer = PaxiniFTSideChannelWriter(
        state_file=state_file,
        port=port,
        print_period_sec=print_period_sec,
        include_tactile=include_tactile,
        decode_config=decode_config,
        shm_key=shm_key,
        write_json=write_json,
        json_rate_hz=json_rate_hz,
        calibrate=calibrate,
        calib_cmd=calib_cmd,
    )
    writer.start()
    return writer


def decoded_to_numpy_arrays(decoded: dict[str, Any], config: TactileDecodeConfig) -> dict[str, np.ndarray]:
    """Convert decoded lists/dicts to fixed-shape arrays for external loggers."""
    ft = np.full((config.sensor_count, len(FT_LABELS)), np.nan, dtype=np.float32)
    tactile_component_count = 3 if config.tactile_components == "xyz" else 1
    tactile = np.full(
        (config.sensor_count, config.tactile_count, tactile_component_count),
        np.nan,
        dtype=np.float32,
    )

    for sensor_index, ft_info in enumerate(decoded.get("ft", [])):
        if sensor_index >= config.sensor_count:
            break
        for axis_index, value in enumerate(ft_info.get("values", [])):
            if axis_index >= len(FT_LABELS) or value is None:
                continue
            ft[sensor_index, axis_index] = value

    for sensor_index, tactile_values in enumerate(decoded.get("tactile", [])):
        if sensor_index >= config.sensor_count:
            break
        for point_index, value in enumerate(tactile_values):
            if point_index >= config.tactile_count or value is None:
                continue
            if config.tactile_components == "xyz":
                for component_index, component_value in enumerate(value):
                    if component_index < tactile_component_count and component_value is not None:
                        tactile[sensor_index, point_index, component_index] = component_value
            else:
                tactile[sensor_index, point_index, 0] = value

    return {"ft": ft, "tactile": tactile}


def _make_sample(seq: int, data: bytes, mode: ReadMode, lrc_ok: bool | None = None) -> RawTactileSample:
    wall_ns = time.time_ns()
    return RawTactileSample(
        seq=seq,
        host_time_ns=wall_ns,
        host_time_s=wall_ns / 1_000_000_000,
        monotonic_ns=time.monotonic_ns(),
        byte_count=len(data),
        data=data,
        data_hex=data.hex(" "),
        mode=mode,
        lrc_ok=lrc_ok,
    )


def _pop_frame(buffer: bytearray) -> tuple[bytes | None, bool | None]:
    head_index = buffer.find(FRAME_HEAD)
    if head_index < 0:
        if len(buffer) > 1:
            del buffer[:-1]
        return None, None

    if head_index > 0:
        del buffer[:head_index]

    if len(buffer) < MIN_FRAME_LEN:
        return None, None

    payload_len_field = buffer[2] | (buffer[3] << 8)
    total_len = 2 + 2 + payload_len_field + 1

    if total_len < MIN_FRAME_LEN:
        del buffer[:2]
        return None, False

    if len(buffer) < total_len:
        return None, None

    frame = bytes(buffer[:total_len])
    del buffer[:total_len]
    return frame, lrc_cal(frame[:-1]) == frame[-1]


def _pop_auto_push_frame(buffer: bytearray) -> bytes | None:
    """Pop one complete AA56 auto-push frame from a persistent buffer.

    Unlike ``read_protocol_frame`` (which allocates a fresh local buffer per call
    and discards any bytes past the first frame), this operates on a caller-owned
    ``bytearray`` so leftover/partial bytes survive across reads. The reader
    thread calls this in a loop to drain every buffered frame and keep the
    newest one, which both raises throughput and minimises staleness.

    Returns the frame bytes (and removes them from ``buffer``), or ``None`` when
    no complete frame is available yet.
    """
    head_index = buffer.find(AUTO_PUSH_FRAME_HEAD)
    if head_index < 0:
        # keep the last byte: it could be the first half of the 2-byte header
        if len(buffer) > 1:
            del buffer[:-1]
        return None
    if head_index > 0:
        del buffer[:head_index]
    if len(buffer) < 5:                       # need head(2)+reserved(1)+len(2)
        return None
    valid_frame_len = int.from_bytes(buffer[3:5], "little")
    total_len = 2 + 1 + 2 + valid_frame_len + 1
    if total_len < 7 or total_len > MAX_AUTO_PUSH_FRAME_LEN:
        del buffer[:2]                        # corrupt header, skip past it
        return None
    if len(buffer) < total_len:
        return None
    frame = bytes(buffer[:total_len])
    del buffer[:total_len]
    return frame


class TactileUARTReader:
    def __init__(self, config: TactileUARTConfig):
        if serial is None:
            raise RuntimeError("pyserial이 필요합니다. 설치 예: python -m pip install pyserial")

        if config.communication != "uart":
            raise ValueError(f"지원하지 않는 통신 방식입니다: {config.communication}")

        self.config = config
        self._serial: serial.Serial | None = None
        self._seq = 0
        self._frame_buffer = bytearray()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None

    def __enter__(self) -> "TactileUARTReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @property
    def read_request(self) -> bytes:
        return build_read_request(self.config.read_command)

    def open(self) -> None:
        if self.is_open:
            return

        self._serial = serial.Serial(
            self.config.port,
            self.config.baudrate,
            timeout=self.config.timeout,
        )

        if self.config.poll_read:
            self.start_polling()

    def close(self) -> None:
        self.stop_polling()
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def start_polling(self) -> None:
        if self._poll_thread is not None:
            return
        if self._serial is None:
            raise RuntimeError("UART 포트가 열려 있지 않습니다.")

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_polling(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None

    def _poll_loop(self) -> None:
        request = self.read_request
        interval_s = self.config.poll_interval_ms / 1000.0

        while not self._stop_event.is_set():
            if self._serial is not None:
                self._serial.write(request)
                self._serial.flush()
            self._stop_event.wait(interval_s)

    def read_sample(self) -> RawTactileSample:
        if self.config.mode == "frame":
            return self.read_frame()
        return self.read_chunk()

    def read_samples(self) -> Generator[RawTactileSample, None, None]:
        while True:
            yield self.read_sample()

    def read_chunk(self) -> RawTactileSample:
        if self._serial is None:
            raise RuntimeError("UART 포트가 열려 있지 않습니다.")

        while True:
            data = self._serial.read(self.config.read_size)
            if data:
                sample = _make_sample(self._seq, data, "chunk")
                self._seq += 1
                return sample

    def read_frame(self) -> RawTactileSample:
        if self._serial is None:
            raise RuntimeError("UART 포트가 열려 있지 않습니다.")

        while True:
            chunk = self._serial.read(self.config.read_size)
            if not chunk:
                continue

            self._frame_buffer.extend(chunk)
            while True:
                frame, lrc_ok = _pop_frame(self._frame_buffer)
                if frame is None:
                    break

                sample = _make_sample(self._seq, frame, "frame", lrc_ok)
                self._seq += 1
                return sample
