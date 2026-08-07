#!/usr/bin/env python3
"""TUMI 글러브 → KISTAR 핸드 텔레오퍼레이션 (시리얼 → ROS2 → SHM).

글러브 펌웨어가 이미 캘리브된 16 조인트 값을 내보내므로 **1:1 그대로** 핸드 타겟으로 쓴다.
    한 줄 형식:  "j0,j1,...,j15\\n"   (정수 16개, 콤마 구분, ~100Hz, 촉각 없음)

발행 토픽 2개:
    /glove/<side>/q_raw    Float32MultiArray[16]  글러브 엔코더 raw
                           → hand_target_receiver → SHM Glove_q_pos
                           → shm_state_publisher → /glove/<side>/joint_states
    /hand/<side>/q_target  Float32MultiArray[16]  핸드 관절 타겟 [count]
                           → hand_target_receiver → SHM Hand_q_tar → EtherCAT

실행 (터미널 3개):
    1) shm                                        # 루트, EtherCAT + SHM
    2) nd                                         # control_pc.launch.py
    3) glove                                      # = python3 tools/glove_teleop.py

    glove --dry-run    # 손 안 움직이고 글러브값/타겟만 확인 (튜닝용)
"""

from __future__ import annotations

import argparse
import glob
import os
import threading
import time

import rclpy
import serial
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int32

# ═══════════════════════════════════════════════════════════════════════════
#  사용자 설정 — 여기만 고치면 됩니다
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. 연결 ────────────────────────────────────────────────────────────────
PORT = "auto"                  # "auto"=CH340 자동 탐지(포트 바뀌어도 OK). 고정하려면 "/dev/ttyUSB0"
BAUD = 500000                  # ★ 글러브 펌웨어 Serial.begin() 과 반드시 일치 (현재 500000)
SIDE = "right"                 # "right"(=핸드0) | "left"(=핸드1)
                               # (발행 주기 설정 없음: 글러브에서 프레임이 오는 즉시 발행 = 장치 최대 속도 ~108Hz)

# ── 2. 안전 ────────────────────────────────────────────────────────────────
DRY_RUN = False                # True = 핸드 타겟 미발행 (값만 확인, 손 안 움직임)
AUTO_SERVO_ON = True           # 시작 시 cmd_mode=1(position) + cmd_servo=True 발행
SERVO_OFF_ON_EXIT = False      # 종료 시 서보 끔 (True면 손 힘 풀림 = 물체 떨굼 주의)
RAMP_SEC = 2.0                 # 현재 손자세 → 글러브 타겟으로 서서히 이동 [s]
                               #   ★ 프로세스 기동 후 "최초 1회만" 태운다.
                               #   USB 재연결이나 발판 재engage 에서는 다시 태우지 않는다
                               #   (재연결마다 램프를 다시 태우면 그때마다 2초간 손이 굼떠진다).
                               #   0 = 램프 없음(서보 켜는 순간 글러브 자세로 즉시 점프 — 주의)
STALE_SEC = 0.5                # (미사용 — _process 가 프레임 수신 시에만 호출되므로 불필요)
READ_ERR_TOLERANCE = 20        # 연속 read 오류 이 횟수까지는 포트를 닫지 않음(재오픈=보드리셋=공백 방지)
RECONNECT_SEC = 0.1            # USB 끊기면 이 주기로 재연결 시도 (0 = 재연결 안 함)
                               #   글러브 USB 가 물리적으로 잘 끊기므로(커널 urb -32) 짧게 잡아
                               #   끊김 비용을 최소화한다. 램프는 다시 태우지 않는다.

# ── 3. 표시 ────────────────────────────────────────────────────────────────
PRINT_HZ = 2.0                 # 상태 출력 주기 [Hz], 0 = 끔

# ── 4. 관절 매핑 표 ────────────────────────────────────────────────────────
# 기본은 전부 1:1 패스스루 (scale=1.0, offset=0).
#   hand_idx : 핸드 관절 번호 (0-3 엄지, 4-7 검지, 8-11 중지, 12-15 약지)
#   glove_ch : 쓸 글러브 채널. 보통 hand_idx와 동일.
#              (예: 약지를 중지에 묶고 싶으면 13~15의 채널을 9~11로 바꾸기)
#   scale    : 1.0=그대로, 1.1=조금 더 꽉, -1.0=방향 반전
#   offset   : 상수 보정 [count]
#   on       : False면 offset 값으로 고정 (그 관절 텔레옵 끔)
#
#   target = clamp(glove[glove_ch] * scale + offset)
#
#          hand_idx, 이름,      glove_ch, scale, offset, on
JOINTS = [
    (  0, "thumb_cmc_opposition",   0,      1.1,    0.0,  True ),
    (  1, "thumb_cmc_abduction",    1,      1.0,    0.0,  True ),
    (  2, "thumb_mcp",              2,      1.1,    0.0,  True ),
    (  3, "thumb_ip",               3,      1.1,    0.0,  True ),

    (  4, "index_mcp_abduction",    4,      1.0,    0.0,  True ),
    (  5, "index_mcp_flexion",      5,      1.1,    0.0,  True ),
    (  6, "index_pip",              6,      1.1,    0.0,  True ),
    (  7, "index_dip",              7,      1.1,    0.0,  True ),

    (  8, "middle_mcp_abduction",   8,      1.0,    0.0,  True ),
    (  9, "middle_mcp_flexion",     9,      1.1,    0.0,  True ),
    ( 10, "middle_pip",            10,      1.1,    0.0,  True ),
    ( 11, "middle_dip",            11,      1.1,    0.0,  True ),

    ( 12, "ring_mcp_abduction",    12,      1.0,    0.0,  True ),
    ( 13, "ring_mcp_flexion",      13,      1.1,    0.0,  True ),
    ( 14, "ring_pip",              14,      1.1,    0.0,  True ),
    ( 15, "ring_dip",              15,      1.1,    0.0,  True ),
]

# ── 5. 핸드 관절 하드리밋 [count] — 최종 클램프 ────────────────────────────
# 1 count = pi/8192 rad.
HAND_LIMITS = {i: (0, 4096) for i in range(16)}
HAND_LIMITS[1] = (-4096, 4096)                     # 엄지 외전
for _i in (4, 8, 12):                              # 검지/중지/약지 외전
    HAND_LIMITS[_i] = (-1000, 1000)
for _i in (3, 7, 11, 15):                          # 엄지 IP / 검지·중지·약지 DIP
    HAND_LIMITS[_i] = (-2048, 4096)                #   글러브가 음수로 나오는 구간 허용

# ═══════════════════════════════════════════════════════════════════════════
#  이하 구현
# ═══════════════════════════════════════════════════════════════════════════

NUM_CH = 16
SENSOR_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
# 발판 engage: 시작순서 무관하게 마지막 상태를 받도록 TRANSIENT_LOCAL
ENGAGE_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


GLOVE_USB_IDS = [(0x1A86, 0x7523)]   # CH340 (글러브 보드). 다른 칩 쓰면 여기에 (vid, pid) 추가


def find_glove_port():
    """글러브 시리얼 포트 자동 탐지 — 꽂은 USB 포트가 바뀌어도(ttyUSB0/1/2…) 알아서 찾는다.

    1) VID:PID 로 CH340 찾기  2) 없으면 /dev/ttyUSB*, /dev/ttyACM* 중 첫 번째
    없으면 None (아직 안 꽂힌 상태).
    """
    try:
        from serial.tools import list_ports
        cands = [p.device for p in list_ports.comports()
                 if (p.vid, p.pid) in GLOVE_USB_IDS]
        if cands:
            return sorted(cands)[0]
    except ImportError:
        pass
    nodes = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    return nodes[0] if nodes else None


def open_serial(port: str):
    """성공하면 Serial, 실패하면 None (미연결이어도 노드가 죽지 않게).

    DTR/RTS 를 내린 상태로 연다: CH340 보드는 DTR 토글에서 자동 리셋되는데,
    리셋되면 보드 부팅 동안 1~2초간 데이터가 끊긴다.
    """
    try:
        ser = serial.Serial()
        ser.port = port
        ser.baudrate = BAUD
        ser.timeout = 0.1
        ser.dtr = False          # 보드 자동리셋 방지
        ser.rts = False
        try:
            ser.exclusive = True  # 다른 프로세스가 같은 포트를 열어 데이터가 쪼개지는 것 방지
        except (AttributeError, ValueError):
            pass
        ser.open()
        ser.reset_input_buffer()
        time.sleep(0.05)         # 열린 직후 남은 조각만 버린다 (dtr=False 라 보드 리셋이 없어 길게 잘 필요 없음)
        ser.reset_input_buffer()
        return ser
    except (OSError, serial.SerialException):
        return None


def parse_line(line: str):
    """"j0,...,j15" → [int]*16 또는 None."""
    parts = line.strip().split(",")
    if len(parts) != NUM_CH:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def clamp(idx: int, value: float) -> float:
    lo, hi = HAND_LIMITS[idx]
    return max(lo, min(hi, value))


class GloveTeleop(Node):
    def __init__(self, port: str, side: str, dry_run: bool):
        super().__init__("glove_teleop")
        self.side = side
        self.dry_run = dry_run

        self.pub_glove = self.create_publisher(
            Float32MultiArray, f"/glove/{side}/q_raw", SENSOR_QOS)
        self.pub_target = self.create_publisher(
            Float32MultiArray, f"/hand/{side}/q_target", SENSOR_QOS)
        self.pub_servo = self.create_publisher(Bool, f"/hand/{side}/cmd_servo", 1)
        self.pub_mode = self.create_publisher(Int32, f"/hand/{side}/cmd_mode", 1)

        self.hand_q = None       # 현재 손 자세 [count]
        self.create_subscription(JointState, f"/hand/{side}/joint_states",
                                 self._on_hand_state, SENSOR_QOS)

        # 발판 제어권: engage=False 면 q_target 홀드(손 안 움직임). 발판 없으면 기본 engage.
        self.engaged = True
        self.create_subscription(Bool, f"/teleop/hand_engage/{side}",
                                 self._on_engage, ENGAGE_QOS)

        self.port_arg = port     # "auto" = 자동 탐지, 또는 명시 경로
        self.port = None
        self.ser = None
        if self._try_open():
            self.get_logger().info(f"글러브 시리얼 연결: {self.port}")
        else:
            p = find_glove_port() if port == "auto" else port
            if p is not None and os.path.exists(p):
                # 포트는 있는데 못 열었다 = 다른 glove_teleop 이 이미 점유(exclusive) 중일 가능성
                self.get_logger().error(
                    f"{p} 가 있는데 열 수 없습니다 — 다른 glove_teleop 이 이미 실행 중인지 확인하세요 "
                    f"(중복 실행 시 데이터가 쪼개져 끊김처럼 보입니다):  pgrep -af glove_teleop")
            else:
                self.get_logger().warn("글러브 미연결 — USB 꽂으면 자동 연결됩니다 (대기 중)")

        self.g_last = None        # 마지막으로 받은 글러브 raw 값 (필터 없음)
        self.last_target = None   # 직전 발행 타겟
        self.start_pose = None    # 램프 시작 자세
        self.t_start = None
        self.ramp_done = False    # 램프를 한 번 끝냈으면 재연결/재engage 에서 다시 태우지 않는다
        self.last_rx = 0.0
        self.t_reconnect = 0.0
        self.read_err = 0        # 연속 read 오류 횟수 (일시적 오류로 포트 닫지 않게)
        self.servo_sent = False

        # 시리얼 읽기 전용 스레드: 100Hz 타이머 안에서 블로킹 readline 을 돌리면
        # 타이머와 서로 밀려 수백 ms~초 단위 공백이 생긴다. 읽기를 분리해 항상 최신 프레임만 보관.
        # 진단용(GLOVE_DIAG=1): 내부 발행 간격 통계를 5초마다 로그
        self._diag = os.environ.get("GLOVE_DIAG") == "1"
        self._diag_t0 = time.time()
        self._pub_prev = None
        self._pub_n = 0
        self._pub_max = 0.0
        self._pub_gaps = 0

        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # 발행은 리더 스레드가 프레임마다 즉시 수행(최대 Hz). 타이머는 재연결 유지·상태표시만.
        # 재연결 감지 주기: USB 가 자주 끊기므로 짧게(0.1s) 돌려 복구 지연을 줄인다.
        self.create_timer(0.1, self._maintain)
        if PRINT_HZ > 0:
            self.create_timer(1.0 / PRINT_HZ, self._print_status)

    def _on_hand_state(self, msg: JointState):
        if len(msg.position) >= NUM_CH:
            self.hand_q = list(msg.position[:NUM_CH])

    def _on_engage(self, msg: Bool):
        new = bool(msg.data)
        if new != self.engaged:
            self.engaged = new
            self.get_logger().info(f"발판: 제어권 {'ENGAGE(스트리밍)' if new else 'DISENGAGE(홀드)'}")

    def _map_to_hand(self, g: list) -> list:
        """글러브 16채널 → 핸드 16타겟 (기본 1:1)."""
        out = [0.0] * NUM_CH
        for hand_idx, _name, ch, scale, offset, on in JOINTS:
            raw = g[ch] * scale + offset if on else offset
            out[hand_idx] = clamp(hand_idx, raw)
        return out

    def _try_open(self) -> bool:
        """포트를 (필요하면 다시 탐지해서) 연다. 성공 시 True."""
        p = find_glove_port() if self.port_arg == "auto" else self.port_arg
        if p is None:
            return False
        ser = open_serial(p)
        if ser is None:
            return False
        self.ser, self.port = ser, p
        return True

    def _reconnect(self, now: float) -> None:
        """USB 재열거 후 다시 붙는다(포트 번호가 바뀌어도 자동 탐지).

        ★ 램프를 다시 태우지 않는다. 이 글러브는 USB 가 물리적으로 자주 끊기는데
        (커널 `urb stopped: -32` → `USB disconnect`), 끊길 때마다 2초 램프를 다시 태우면
        손이 그때마다 2초씩 굼떠져 추종이 안 된다. 타겟은 last_target 에서 이어가므로
        끊김 비용은 "재연결에 걸린 시간" 뿐이다.
        """
        if RECONNECT_SEC <= 0 or now - self.t_reconnect < RECONNECT_SEC:
            return
        self.t_reconnect = now
        if not self._try_open():
            return
        self.get_logger().info(f"글러브 재연결됨: {self.port} — 램프 없이 이어서 추종")

    def _reader_loop(self):
        """전용 스레드: 블로킹 readline 으로 계속 읽어 최신 프레임만 보관 (검증된 순수 리더와 동일 구조).

        시리얼 I/O 를 ROS 타이머에서 완전히 분리하므로 타이머와 밀려 생기는 공백이 없다.
        포트가 없거나 오류면 여기서 삼키고, 장치 노드가 실제로 사라졌을 때만 재연결을 요청한다.
        """
        buf = b""
        while not self._stop.is_set():
            ser = self.ser
            if ser is None:
                buf = b""
                time.sleep(0.05)
                continue
            try:
                # 벌크 읽기: readline() 은 1바이트씩 읽어(syscall+GIL 쟁탈) 메인 스레드를 굶긴다.
                chunk = ser.read(1)                       # 블로킹(timeout 0.1)
                if chunk and ser.in_waiting:
                    chunk += ser.read(ser.in_waiting)     # 남은 건 한 번에
            except (OSError, serial.SerialException, TypeError):
                self.read_err += 1
                gone = self.port is not None and not os.path.exists(self.port)
                if gone or self.read_err >= READ_ERR_TOLERANCE:
                    self.get_logger().error(
                        f"글러브 시리얼 끊김 ({self.port}, 연속오류 {self.read_err}, "
                        f"노드소실={gone}) — 타겟 홀드하고 재연결 시도")
                    try:
                        ser.close()
                    except Exception:
                        pass
                    self.ser = None
                    self.read_err = 0
                time.sleep(0.01)
                continue
            if not chunk:
                continue
            buf += chunk
            if b"\n" not in buf:
                continue
            *lines, buf = buf.split(b"\n")            # 마지막 조각은 다음 루프로 이월
            if len(buf) > 4096:                       # 개행 없이 쓰레기만 쌓이는 경우 방지
                buf = b""
            got = None
            for ln in lines:                          # 최신 유효 프레임만 사용(지연 누적 방지)
                p = parse_line(ln.decode("utf-8", errors="ignore"))
                if p is not None:
                    got = p
            if got is None:
                continue
            self.read_err = 0
            # 읽은 즉시 이 스레드에서 바로 발행 (ROS 타이머를 거치지 않아 최대 Hz·최소 지연)
            self._process(got, time.time())

    def _maintain(self):
        """0.5초 주기: 포트가 없으면 재연결 시도 (발행은 리더 스레드가 담당)."""
        if self.ser is None:
            self._reconnect(time.time())

    def _process(self, got, now):
        """프레임 1개 처리(리더 스레드에서 호출): q_raw 발행 + EMA·램프·rate limit 후 q_target 발행."""
        # 내부 발행 간격 계측: 여기 공백이 없는데 토픽에만 공백이면 DDS/구독측 문제
        if self._diag:
            if self._pub_prev is not None:
                d = now - self._pub_prev
                self._pub_max = max(self._pub_max, d)
                if d > 0.2:
                    self._pub_gaps += 1
            self._pub_prev = now
            self._pub_n += 1
            if now - self._diag_t0 >= 5.0:
                self.get_logger().warn(
                    f"[DIAG] 내부발행 {self._pub_n/(now-self._diag_t0):.1f}Hz  "
                    f"최대간격 {self._pub_max*1000:.0f}ms  200ms초과 {self._pub_gaps}회")
                self._diag_t0, self._pub_n = now, 0
                self._pub_max, self._pub_gaps = 0.0, 0
        self.last_rx = now
        self.g_last = [float(v) for v in got]   # 필터 없음 — 받은 그대로
        m = Float32MultiArray()
        m.data = self.g_last
        self.pub_glove.publish(m)

        # (프레임을 방금 받았으므로 stale 검사는 불필요 — 수신이 끊기면 이 함수 자체가 호출되지 않아
        #  마지막 타겟이 그대로 유지된다. 제어 PC 가 hold 하므로 손은 힘이 풀리지 않는다.)
        if self.hand_q is None:                # 손 상태 미수신 → 램프 시작점 없음
            return
        if not self.engaged:                   # 발판 disengage → 타겟 홀드(제어 PC가 마지막값 유지)
            return                             #   ★ start_pose 를 지우지 않는다: 재engage 때 램프를
                                               #   다시 태우면 그때마다 2초간 손이 굼떠진다.
                                               #   타겟은 last_target 에서 그대로 이어진다.

        if self.start_pose is None and not self.ramp_done:
            # 첫 타겟 = 현재 손 자세 (서보 켜는 순간 튀지 않게)
            self.start_pose = list(self.hand_q)
            self.last_target = list(self.hand_q)
            self.t_start = now
            if not self.dry_run:
                self._publish(self.last_target)
                if AUTO_SERVO_ON and not self.servo_sent:
                    self.pub_mode.publish(Int32(data=1))      # 1 = position
                    self.pub_servo.publish(Bool(data=True))
                    self.servo_sent = True
                    self.get_logger().info("핸드 servo ON (mode=position)")
            self.get_logger().info(f"현재 자세에서 {RAMP_SEC:.1f}초 램프 시작")
            return

        mapped = self._map_to_hand(self.g_last)

        if not self.ramp_done:
            # 시작 자세 → 글러브 타겟 램프 (기동 후 최초 1회만)
            a = 1.0 if RAMP_SEC <= 0 else min(1.0, (now - self.t_start) / RAMP_SEC)
            target = [(1.0 - a) * s + a * t for s, t in zip(self.start_pose, mapped)]
            if a >= 1.0:
                self.ramp_done = True
                self.get_logger().info(
                    "램프 완료 — 이후 글러브 raw 그대로 통과 (EMA·rate limit 없음)")
        else:
            target = mapped        # ★ raw 그대로 통과: 필터도 주기당 변화량 제한도 없음

        # dry-run에서도 타겟은 계산한다 (확인용) — 발행만 안 함
        self.last_target = target
        if not self.dry_run:
            self._publish(target)

    def _publish(self, target: list):
        m = Float32MultiArray()
        m.data = [float(v) for v in target]
        self.pub_target.publish(m)

    def _print_status(self):
        if self.ser is None:
            self.get_logger().warn("글러브 끊김/미연결 — USB 꽂으면 자동 연결 (재시도 중)")
            return
        if self.g_last is None:
            self.get_logger().warn(f"글러브 수신 없음 — {self.port} 확인")
            return
        tag = "DRY" if self.dry_run else "RUN"
        if self.hand_q is None:
            g = " ".join(f"{v:6.0f}" for v in self.g_last[:8])
            self.get_logger().warn(
                f"[{tag}] glove[0:8] {g} | /hand/{self.side}/joint_states 대기 중 "
                "(shm + nd 떠 있는지 확인)")
            return
        # 16관절 전체를 손가락별 4줄로. 각 칸은 "글러브 raw→핸드 타겟" [count].
        # 글러브 채널은 JOINTS 표에서 끌어온다(관절을 다른 채널에 묶어도 맞게 나온다).
        # 발판 disengage 면 타겟이 아직 없다 — 그때도 글러브 raw 는 보여준다.
        # "!" = HAND_LIMITS 에 걸려 잘린 관절. 글러브가 움직여도 타겟이 한계값에
        #       고정되므로, raw 는 살아있는데 타겟만 안 바뀌는 상황을 여기서 바로 잡는다.
        src = {hand_idx: (ch, scale, offset, on)
               for hand_idx, _n, ch, scale, offset, on in JOINTS}
        rows = []
        for label, base in (("thumb", 0), ("index", 4), ("middle", 8), ("ring", 12)):
            cells = []
            for i in range(base, base + 4):
                ch, scale, offset, on = src[i]
                g = self.g_last[ch]
                pre = g * scale + offset if on else offset      # 클램프 전 값
                hit = "!" if clamp(i, pre) != pre else " "
                tgt = f"{self.last_target[i]:<6.0f}" if self.last_target else "  --  "
                cells.append(f"{g:>6.0f}→{tgt}{hit}")
            rows.append(f"  {label:<7}" + " ".join(cells))
        state = "engage" if self.engaged else "DISENGAGE(발판 대기, 타겟 홀드)"
        self.get_logger().info(
            f"[{tag}] {state}  glove→target [count] (1 count = pi/8192 rad, 4096=90°)  "
            "엄지=(cmc_opp,cmc_abd,mcp,ip) 나머지=(abd,flex,pip,dip)  !=클램프\n"
            + "\n".join(rows))

    def shutdown(self):
        self._stop.set()                      # 읽기 스레드 정지
        self._reader.join(timeout=1.0)
        if SERVO_OFF_ON_EXIT and self.servo_sent:
            self.pub_servo.publish(Bool(data=False))
            time.sleep(0.1)
            self.get_logger().info("핸드 servo OFF")
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except (OSError, serial.SerialException):
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="TUMI 글러브 → KISTAR 핸드 텔레옵")
    ap.add_argument("--port", default=PORT,
                    help=f"글러브 시리얼 포트 (기본 {PORT}=자동 탐지, 예: /dev/ttyUSB1)")
    ap.add_argument("--side", choices=["right", "left"], default=SIDE)
    ap.add_argument("--dry-run", action="store_true",
                    help="핸드 타겟 미발행 (글러브 raw만 SHM에 기록)")
    args = ap.parse_args()

    dry_run = args.dry_run or DRY_RUN
    if dry_run:
        print(f">>> DRY-RUN: /hand/{args.side}/q_target 발행 안 함 (손 안 움직임)\n")

    rclpy.init()
    node = GloveTeleop(args.port, args.side, dry_run)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass                      # Ctrl+C / SIGTERM — 정상 종료 경로
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
