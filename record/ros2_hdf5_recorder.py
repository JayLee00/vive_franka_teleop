#!/usr/bin/env python3
"""ROS2 구독형 HDF5 데이터 레코더 (글러브 텔레옵 과일조작 데모 수집).

이 PC(글러브/vive PC)엔 로봇 SHM이 없으므로 제어 PC가 발행하는 ROS2 토픽을 구독해
고정 주기(기본 100Hz)로 스냅샷 → 에피소드(Demo_N)로 HDF5에 저장한다.
스키마는 제어 PC의 hdf5_log_with_realsense_new.py(Demo_N/NN_name)를 따르고 과일 pose를 추가.

시작/종료(에피소드 경계)는 발판 노드(foot_pedal_glove.py)가 발행하는
    /record/enable  (std_msgs/Bool)   True=수집 시작(S), False=수집 종료 후 저장(E)
로 제어. (키보드 대신 발판. 토픽이라 어느 노드에서든 토글 가능.)

  S(rising edge)  -> 새 Demo_N 버퍼 시작
  E(falling edge) -> 버퍼를 Demo_N 그룹으로 저장, demo_idx += 1
  s 키            -> 현재 파일 저장하고 다음 에피소드부터 Demo_0 새 파일 (배치 분리용)
  Ctrl+C          -> 파일 닫고 종료 (저장된 데모는 유지)

배치(nominal / 낙하회복 등)를 다른 파일로 나눠 담고 싶을 때 프로세스를 재시작하지 않고
s 만 누르면 된다. 파일명은 생성 시각이므로 새 파일은 새 타임스탬프를 받는다.

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 record/ros2_hdf5_recorder.py            # 수집
  python3 record/ros2_hdf5_recorder.py --check    # 어떤 토픽이 들어오는지 3초 점검 후 종료
"""
from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Bool, Float32MultiArray, Float64, Int32MultiArray

# ── 설정 ────────────────────────────────────────────────────────────────────
SIDE = "right"
RATE_HZ = 100.0                       # 스냅샷 주기 (학습용 100Hz면 충분)
OUT_DIR = Path(__file__).resolve().parent / "logs"

# 각 필드: (HDF5 데이터셋 이름, 소스 토픽, 메시지 타입, 차원, 추출 함수)
#   추출 함수: 메시지 -> 길이 dim 의 1D float 리스트
def _js_pos(m):  return list(m.position)          # 16 or 7
def _js_eff(m):  return list(m.effort)            # 16 or 7 (모터 전류/토크)
def _js_vel(m):  return list(m.velocity)          # 7
def _js_tgt(m):  return list(m.position)          # 16 (target_joint_states)
def _fma(m):     return list(m.data)              # 가변
def _f64(m):     return [float(m.data)]
def _hand_mode(m): return [float(m.data[0])] if m.data else [0.0]
def _hand_servo(m): return [float(m.data[1])] if len(m.data) > 1 else [0.0]
def _tip_pos(m): return [v for p in m.poses for v in (p.position.x, p.position.y, p.position.z)]      # 4*3
def _tip_quat(m):return [v for p in m.poses for v in (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)]  # 4*4
def _ps_pos(m):  return [m.pose.position.x, m.pose.position.y, m.pose.position.z]                        # PoseStamped 위치
def _ps_quat(m): return [m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w]  # PoseStamped 방향

# name, topic, msg_type, dim, extractor  (번호 순 = hdf5_log_with_realsense_new.py + paxini 20번대 + 과일 30번대)
FIELDS = [
    ("01_hand_mode",             f"/hand/{SIDE}/mode",                Int32MultiArray,   1,    _hand_mode),
    ("02_hand_servo_on",         f"/hand/{SIDE}/mode",                Int32MultiArray,   1,    _hand_servo),
    ("03_hand_j_pos",            f"/hand/{SIDE}/joint_states",        JointState,        16,   _js_pos),
    ("04_hand_j_tar",            f"/hand/{SIDE}/q_target",            Float32MultiArray, 16,   _fma),
    ("05_hand_j_cur",            f"/hand/{SIDE}/joint_states",        JointState,        16,   _js_eff),
    ("06_hand_j_kin",            f"/hand/{SIDE}/kin",                 Float32MultiArray, 12,   _fma),
    ("07_hand_j_tac",            f"/hand/{SIDE}/tac_legacy",          Float32MultiArray, 4,    _fma),
    ("08_hand_tip_pos",          f"/hand/{SIDE}/fingertip_poses",     PoseArray,         12,   _tip_pos),
    ("09_hand_tip_quat",         f"/hand/{SIDE}/fingertip_poses",     PoseArray,         16,   _tip_quat),
    ("10_franka_Arm_j_pos",      "/franka/right/joint_states",        JointState,        7,    _js_pos),
    ("11_franka_Arm_j_tar",      "/franka/right/target_joint_states", JointState,        7,    _js_pos),
    ("12_franka_Arm_j_vel",      "/franka/right/joint_states",        JointState,        7,    _js_vel),
    ("13_franka_Arm_C_pos",      "/franka/right/ee_pose",             PoseStamped,       3,    _ps_pos),
    ("14_franka_Arm_j_tq",       "/franka/right/joint_states",        JointState,        7,    _js_eff),
    ("15_franka_Arm_speed_factor", "/franka/right/speed_factor",      Float64,           1,    _f64),
    ("16_glove_g_pos",           f"/glove/{SIDE}/q_raw",              Float32MultiArray, 16,   _fma),
    ("20_paxini_ft",             f"/paxini/{SIDE}/ft",                Float32MultiArray, 12,   _fma),
    ("21_paxini_raw",            f"/paxini/{SIDE}/raw",               Float32MultiArray, 1524, _fma),
    ("22_franka_Arm_C_quat",     "/franka/right/ee_pose",             PoseStamped,       4,    _ps_quat),
    ("23_franka_Arm_tar_pos",    "/franka/right/ee_target_world",     PoseStamped,       3,    _ps_pos),
    ("24_franka_Arm_tar_quat",   "/franka/right/ee_target_world",     PoseStamped,       4,    _ps_quat),
    ("30_fruit_pos",             "/fruit/pose",                       PoseStamped,       3,    _ps_pos),
    ("31_fruit_quat",            "/fruit/pose",                       PoseStamped,       4,    _ps_quat),
    ("32_fruit_size",            "/fruit/size",                       Float32MultiArray, 3,    _fma),
]

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


class Recorder(Node):
    def __init__(self, check: bool):
        super().__init__("ros2_hdf5_recorder")
        self.check = check
        self.latest: dict[str, np.ndarray] = {}   # 데이터셋 이름 -> 마지막 값
        self.seen: set[str] = set()               # 실제 수신된 데이터셋 이름
        self.dims = {name: dim for name, _, _, dim, _ in FIELDS}

        # 토픽별로 한 번만 구독(같은 토픽을 여러 필드가 공유). 콜백에서 관련 필드 모두 갱신.
        by_topic: dict[str, list] = {}
        for name, topic, typ, dim, fn in FIELDS:
            by_topic.setdefault((topic, typ), []).append((name, dim, fn))
        for (topic, typ), fields in by_topic.items():
            self.create_subscription(typ, topic,
                                     lambda m, fs=fields: self._on_msg(fs, m), SENSOR_QOS)

        # 로깅 시작/종료 트리거 (발판 노드가 발행)
        self.recording = False
        self.buffer: list[dict] = []
        self.demo_idx = 0
        self.t_demo_start = 0.0
        self.f = None
        self.out_path = None                      # 파일은 첫 에피소드 시작 때 생성
        self.roll_req = False                     # s 키 → 파일 롤오버 요청 (키 스레드가 세팅)
        self.create_subscription(Bool, "/record/enable", self._on_enable, 10)

        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.create_timer(0.1, self._service_keys)
        if check:
            self.create_timer(3.0, self._check_report)
            self.get_logger().info("── CHECK 모드: 3초간 수신 토픽 점검 후 종료 ──")
        else:
            self.get_logger().info(
                f"레코더 준비. 발판 중간(=/record/enable)으로 시작/종료.\n"
                f"    s = 현재 파일 저장하고 다음 에피소드부터 Demo_0 새 파일\n"
                f"    저장 폴더: {OUT_DIR}")

    def _on_msg(self, fields, msg):
        for name, dim, fn in fields:
            try:
                vals = fn(msg)
            except Exception:
                continue
            arr = np.zeros(dim, dtype=np.float32)
            n = min(dim, len(vals))
            arr[:n] = np.asarray(vals[:n], dtype=np.float32)
            self.latest[name] = arr
            self.seen.add(name)

    def _snapshot(self) -> dict:
        row = {}
        for name, dim in self.dims.items():
            row[name] = self.latest.get(name, np.zeros(dim, dtype=np.float32))
        return row

    def _tick(self):
        if not self.recording:
            return
        row = self._snapshot()
        now = time.perf_counter()
        row["18_real_time_demo"] = np.array(now - self.t_demo_start, dtype=np.float64)
        row["19_real_time_global"] = np.array(now, dtype=np.float64)
        self.buffer.append(row)

    def _open_file(self):
        """새 h5 생성. 파일명 = 생성 시각이라 롤오버마다 새 타임스탬프를 받는다."""
        import h5py
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # 초 단위 타임스탬프라 롤오버 직후 같은 초에 새 파일이 열리면 이전 파일을 덮어쓴다("w").
        stem = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_path = OUT_DIR / f"{stem}.h5"
        i = 2
        while self.out_path.exists():
            self.out_path = OUT_DIR / f"{stem}_{i}.h5"
            i += 1
        self.f = h5py.File(self.out_path, "w")
        self.f.attrs["rate_hz"] = RATE_HZ
        self.f.attrs["side"] = SIDE
        self.f.attrs["start_time"] = datetime.now().isoformat()
        self.get_logger().info(f"새 파일: {self.out_path.name}")

    def _close_file(self):
        """열려 있는 h5 를 마무리(n_demos 기록)하고 닫는다."""
        if self.f is None:
            return
        self.f.attrs["n_demos"] = self.demo_idx
        self.f.close()
        self.f = None
        print(f"\n저장 완료: {self.out_path.resolve()}  (데모 {self.demo_idx}개)")

    def _service_keys(self):
        """키 스레드가 올린 요청 처리 (h5 접근을 이 스레드로 모아 경쟁 방지)."""
        if not self.roll_req:
            return
        self.roll_req = False
        if self.recording:
            self.get_logger().warn("로깅 중 — 발판 중간으로 에피소드 먼저 종료. 파일 안 바꿈")
            return
        if self.f is None:
            self.get_logger().info("아직 열린 파일 없음 — 다음 에피소드가 새 파일로 시작됩니다")
            return
        self._close_file()
        self.demo_idx = 0
        self.get_logger().info("▶ 롤오버 완료. 다음 에피소드부터 Demo_0")

    def _on_enable(self, msg: Bool):
        if msg.data and not self.recording:      # S: 시작
            if self.f is None:
                self._open_file()
            self.buffer = []
            self.t_demo_start = time.perf_counter()
            self.recording = True
            miss = [n for n in self.dims if n not in self.seen]
            self.get_logger().info(f">>> 수집 시작 Demo_{self.demo_idx}"
                                   + (f"  (미수신: {miss})" if miss else ""))
        elif (not msg.data) and self.recording:   # E: 종료 후 저장
            self.recording = False
            self._save_demo()

    def _save_demo(self):
        if not self.buffer:
            self.get_logger().warn(f"<<< Demo_{self.demo_idx}: 수집된 step 없음 → 건너뜀")
            return
        import h5py
        n = len(self.buffer)
        grp = self.f.create_group(f"Demo_{self.demo_idx}")
        keys = list(self.buffer[0].keys())
        for k in keys:
            data = np.stack([b[k] for b in self.buffer])
            grp.create_dataset(k, data=data, compression="gzip")
        grp.attrs["n_samples"] = n
        self.f.flush()
        self.get_logger().info(f"<<< 저장 Demo_{self.demo_idx}  ({n} step, {n/RATE_HZ:.1f}s)")
        self.demo_idx += 1

    def _check_report(self):
        print("\n===== 수신 토픽 점검 =====")
        by_name: dict[str, list[tuple[str, int]]] = {}
        for name, topic, _, dim, _ in FIELDS:
            by_name.setdefault(name, []).append((topic, dim))
        for name in sorted(by_name, key=lambda n: int(n.split("_", 1)[0])):
            sources = by_name[name]
            dim = sources[0][1]
            topics = ", ".join(t for t, _ in sources)
            ok = "✓" if name in self.seen else "✗ (수신 없음)"
            got = len(self.latest[name]) if name in self.latest else 0
            print(f"  {ok:14s} {name:28s} <- {topics}  (기대 {dim}, 수신 {got})")
        print("=========================\n")
        raise KeyboardInterrupt

    def close(self):
        if self.recording:
            self.recording = False
            self._save_demo()
        self._close_file()


def _key_loop(node: Recorder):
    """엔터 없이 1글자씩 읽어 s 를 롤오버 요청으로 넘긴다 (터미널은 main 이 원복)."""
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            return
        if ch in ("s", "S"):
            node.roll_req = True


def main():
    ap = argparse.ArgumentParser(description="ROS2 구독형 HDF5 레코더 (글러브 데모 수집)")
    ap.add_argument("--check", action="store_true", help="3초간 수신 토픽 점검 후 종료")
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(check=args.check)

    # s 키 감지: cbreak 로 엔터 없이 1글자 읽기 (ISIG 는 유지되어 Ctrl+C 정상 동작).
    # 터미널 원복은 main 의 finally 에서 — 키 스레드는 daemon 이라 finally 가 안 돌 수 있다.
    old_tty = None
    if not args.check and sys.stdin.isatty():
        old_tty = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=_key_loop, args=(node,), daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if old_tty is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tty)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
