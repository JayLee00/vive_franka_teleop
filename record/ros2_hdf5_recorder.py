#!/usr/bin/env python3
"""ROS2 구독형 HDF5 데이터 레코더 (글러브 텔레옵 과일조작 데모 수집).

이 PC(글러브/vive PC)엔 로봇 SHM이 없으므로 제어 PC가 발행하는 ROS2 토픽을 구독해
고정 주기(기본 100Hz)로 스냅샷 → 에피소드(Demo_N)로 HDF5에 저장한다.
스키마는 제어 PC의 hdf5_log_with_realsense_new.py(Demo_N/NN_name)를 따르고 과일 pose를 추가.

RGB(--rgb-on)는 h5 안에 넣지 않고 옆 폴더에 JPEG 파일로 떨군다:
    record/logs/exp_20260806_143000.h5            <- 저차원 + 이미지 번호
    record/logs/exp_20260806_143000_image/Demo_0/000000.jpg, 000001.jpg, ...
h5 의 Demo_N/40_image[step] 이 그 스텝에 보이던 파일 번호다(-1 = 아직 프레임 없음).

시작/종료(에피소드 경계)는 발판 노드(foot_pedal_glove.py)가 발행하는
    /record/enable  (std_msgs/Bool)   True=수집 시작(S), False=수집 종료 후 저장(E)
로 제어. (키보드 대신 발판. 토픽이라 어느 노드에서든 토글 가능.)

  S(rising edge)  -> 새 Demo_N 버퍼 시작
  E(falling edge) -> 버퍼를 Demo_N 그룹으로 저장, demo_idx += 1
  k 키            -> 현재 파일 저장하고 다음 에피소드부터 Demo_0 새 파일 (배치 분리용)
  Ctrl+C          -> 파일 닫고 종료 (저장된 데모는 유지)

배치(nominal / 낙하회복 등)를 다른 파일로 나눠 담고 싶을 때 프로세스를 재시작하지 않고
k 만 누르면 된다. 파일명은 생성 시각이므로 새 파일은 새 타임스탬프를 받는다.
(S/E 는 발판의 에피소드 시작/종료라서 롤오버 키는 겹치지 않게 k 로 둔다.)

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
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Bool, Float32MultiArray, Float64, Int32, Int32MultiArray, String

# ── 설정 ────────────────────────────────────────────────────────────────────
SIDE = "right"
RATE_HZ = 100.0                       # 스냅샷 주기 (학습용 100Hz면 충분)
OUT_DIR = Path(__file__).resolve().parent / "logs"

# ── RGB(--rgb-on) ───────────────────────────────────────────────────────────
# 카메라가 이미 JPEG(CompressedImage)로 발행하므로 바이트를 그대로 .jpg 파일로 쓴다
# (재인코딩 없음 = CPU 0, 화질 손실 0). 실측: 640x480, 28.8Hz, 106KB/프레임.
# 파일로 빼두면 h5 가 가벼워서 저차원만 훑는 스크립트가 빨라지고, 이미지는 뷰어로 바로 열린다.
#
# ⚠️ 100Hz 스텝마다 프레임을 복사하면 41GB/시간이 된다. 그래서 고유 프레임만 한 번 저장하고
#    스텝별로는 파일 번호만 남긴다(중복 제거 3.47x) → --rgb-every 1 에서 약 12GB/시간.
#    (DROID 가 쓰는 방식. ALOHA 는 고정길이 패딩 + 매 스텝 복사라 용량을 더 쓴다.)
#    용량을 줄이려면 발행측 jpeg_quality 를 낮추는 게 가장 효과적:
#      q95(기본) 112KB→12GB/h   q85 64KB→6.8GB/h   q80 56KB→5.9GB/h
#      ros2 param set /front_cam/front .front.color.image_raw.jpeg_quality 80
RGB_TOPIC = "/front_cam/front/color/image_raw/compressed"
RGB_INFO_TOPIC = "/front_cam/front/color/camera_info"

# 각 필드: (HDF5 데이터셋 이름, 소스 토픽, 메시지 타입, 차원, 추출 함수)
#   추출 함수: 메시지 -> 길이 dim 의 1D float 리스트
def _js_pos(m):  return list(m.position)          # 16 or 7
def _js_eff(m):  return list(m.effort)            # 16 or 7 (모터 전류/토크)
def _js_vel(m):  return list(m.velocity)          # 7
def _js_tgt(m):  return list(m.position)          # 16 (target_joint_states)
def _fma(m):     return list(m.data)              # 가변
def _f64(m):     return [float(m.data)]          # 스칼라 (Float64 / Int32 공용)
def _hand_mode(m): return [float(m.data[0])] if m.data else [0.0]
def _hand_servo(m): return [float(m.data[1])] if len(m.data) > 1 else [0.0]
def _tip_pos(m): return [v for p in m.poses for v in (p.position.x, p.position.y, p.position.z)]      # 4*3
def _tip_quat(m):return [v for p in m.poses for v in (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)]  # 4*4
def _ps_pos(m):  return [m.pose.position.x, m.pose.position.y, m.pose.position.z]                        # PoseStamped 위치
def _ps_quat(m): return [m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w]  # PoseStamped 방향

# name, topic, msg_type, dim, extractor
#   번호 = 보기 편한 묶음 순서: 로봇핸드(01~11) → 로봇암(12~20) → 글러브(21~23) → 과일(30~34)
#          → 나머지(40_image, 41_rgb_time, 42_rgb_stamp) → 클락(50/51). h5 는 이름순으로 나열되므로
#          번호가 곧 표시 순서다. FIELDS 순서가 아니라 번호가 순서를 정한다.
#   ※ 40/41/42(이미지)와 50/51(클락)은 토픽 추출이 아니라 레코더가 만드는 값이라 여기 없다(_tick/_save_demo).
#   ※ 제어PC 스키마(hdf5_log_with_realsense_new.py)와 번호가 어긋난다 — 2026-08-06 이전 파일은 옛 번호.
FIELDS = [
    # ── 로봇 핸드 ──
    ("01_hand_mode",             f"/hand/{SIDE}/mode",                Int32MultiArray,   1,    _hand_mode),
    ("02_hand_servo_on",         f"/hand/{SIDE}/mode",                Int32MultiArray,   1,    _hand_servo),
    ("03_hand_j_pos",            f"/hand/{SIDE}/joint_states",        JointState,        16,   _js_pos),
    ("04_hand_j_tar",            f"/hand/{SIDE}/q_target",            Float32MultiArray, 16,   _fma),
    ("05_hand_j_cur",            f"/hand/{SIDE}/joint_states",        JointState,        16,   _js_eff),
    ("06_hand_j_kin",            f"/hand/{SIDE}/kin",                 Float32MultiArray, 12,   _fma),
    ("07_hand_j_tac",            f"/hand/{SIDE}/tac_legacy",          Float32MultiArray, 4,    _fma),
    ("08_hand_tip_pos",          f"/hand/{SIDE}/fingertip_poses",     PoseArray,         12,   _tip_pos),
    ("09_hand_tip_quat",         f"/hand/{SIDE}/fingertip_poses",     PoseArray,         16,   _tip_quat),
    ("10_hand_paxini_ft",        f"/paxini/{SIDE}/ft",                Float32MultiArray, 12,   _fma),
    ("11_hand_paxini_raw",       f"/paxini/{SIDE}/raw",               Float32MultiArray, 1524, _fma),
    # ── 로봇 암 ──
    ("12_franka_Arm_j_pos",      "/franka/right/joint_states",        JointState,        7,    _js_pos),
    ("13_franka_Arm_j_tar",      "/franka/right/target_joint_states", JointState,        7,    _js_pos),
    ("14_franka_Arm_j_vel",      "/franka/right/joint_states",        JointState,        7,    _js_vel),
    ("15_franka_Arm_j_tq",       "/franka/right/joint_states",        JointState,        7,    _js_eff),
    ("16_franka_Arm_C_pos",      "/franka/right/ee_pose",             PoseStamped,       3,    _ps_pos),
    ("17_franka_Arm_C_quat",     "/franka/right/ee_pose",             PoseStamped,       4,    _ps_quat),
    ("18_franka_Arm_tar_pos",    "/franka/right/ee_target_world",     PoseStamped,       3,    _ps_pos),
    ("19_franka_Arm_tar_quat",   "/franka/right/ee_target_world",     PoseStamped,       4,    _ps_quat),
    ("20_franka_Arm_speed_factor", "/franka/right/speed_factor",      Float64,           1,    _f64),
    # ── 글러브 ──
    ("21_glove_g_pos",           f"/glove/{SIDE}/q_raw",              Float32MultiArray, 16,   _fma),
    ("22_glove_paxini_ft",       f"/glove/paxini/{SIDE}/ft",          Float32MultiArray, 12,   _fma),
    ("23_glove_paxini_raw",      f"/glove/paxini/{SIDE}/raw",         Float32MultiArray, 1524, _fma),
    # ── 과일 ──
    ("30_fruit_pos",             "/fruit/pose",                       PoseStamped,       3,    _ps_pos),
    ("31_fruit_quat",            "/fruit/pose",                       PoseStamped,       4,    _ps_quat),
    ("32_fruit_size",            "/fruit/size",                       Float32MultiArray, 3,    _fma),
    ("33_fruit_type",            "/fruit/type",                       Int32,             1,    _f64),
    ("34_fruit_corners",         "/inhand/bbox_corners",              Float32MultiArray, 24,   _fma),   # 8코너 OBB (원천)
]
# /fruit/type_name(String) 은 float 배열에 못 담아 Demo_N attrs["fruit_type_name"] 으로 남긴다.
# /fruit/reset, /fruit/set_type 은 사람이 보내는 명령 토픽이라 상태 데이터가 아니라서 제외.

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


class Recorder(Node):
    def __init__(self, check: bool, rgb: bool = False,
                 rgb_topic: str = RGB_TOPIC, rgb_every: int = 1):
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
        self.img_root = None                      # <h5 stem>_image/ (JPEG 저장 폴더)
        self.roll_req = False                     # k 키 → 파일 롤오버 요청 (키 스레드가 세팅)
        self.create_subscription(Bool, "/record/enable", self._on_enable, 10)

        # 과일 종류 이름("lemon" 등)은 문자열이라 float 데이터셋에 못 넣는다 → Demo_N attrs 로.
        self.fruit_type_name = ""
        self.create_subscription(String, "/fruit/type_name",
                                 lambda m: setattr(self, "fruit_type_name", m.data), SENSOR_QOS)

        # ── RGB: 고유 프레임만 모으고 스텝별로는 인덱스만 기록 ──
        self.rgb = rgb
        self.rgb_every = max(1, rgb_every)        # N장마다 1장만 저장(용량 절감)
        self.rgb_frames: list[bytes] = []         # 이 에피소드의 고유 JPEG 프레임
        self.rgb_t = []                           # 수신 시각 (perf_counter, 51_real_time_global 과 동일 시계)
        self.rgb_stamp = []                       # 카메라 캡처 시각 (ROS header stamp)
        self._rgb_msg = None                      # 마지막 수신 프레임 (bytes, t_recv, stamp)
        self._rgb_seq = 0                         # 수신 카운터 (새 프레임 판별)
        self._rgb_stored_seq = -1                 # 마지막으로 저장한 seq
        self._rgb_idx = -1                        # 마지막으로 저장한 프레임의 인덱스
        self._rgb_n_recv = 0
        self.rgb_meta = {}                        # format/해상도/K
        if rgb:
            self.create_subscription(CompressedImage, rgb_topic, self._on_rgb, SENSOR_QOS)
            self.create_subscription(CameraInfo, RGB_INFO_TOPIC, self._on_caminfo, SENSOR_QOS)

        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.create_timer(0.1, self._service_keys)
        if check:
            self.create_timer(3.0, self._check_report)
            self.get_logger().info("── CHECK 모드: 3초간 수신 토픽 점검 후 종료 ──")
        else:
            self.get_logger().info(
                f"레코더 준비. 발판 중간(=/record/enable)으로 시작/종료.\n"
                f"    k = 현재 파일 저장하고 다음 에피소드부터 Demo_0 새 파일\n"
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

    def _on_rgb(self, msg: CompressedImage):
        """JPEG 바이트를 그대로 보관 (재인코딩 없음). 저장은 _tick 이 판단."""
        self._rgb_msg = (bytes(msg.data),
                         time.perf_counter(),
                         msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self._rgb_seq += 1
        self._rgb_n_recv += 1
        if "format" not in self.rgb_meta:
            self.rgb_meta["format"] = msg.format

    def _on_caminfo(self, msg: CameraInfo):
        if "fx" not in self.rgb_meta:
            self.rgb_meta.update(width=msg.width, height=msg.height,
                                 fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5])

    def _rgb_index_for_tick(self) -> int:
        """이 스텝이 가리킬 프레임 인덱스. 새 프레임이면 저장하고 그 인덱스를 반환."""
        if self._rgb_msg is None:
            return -1
        if self._rgb_seq != self._rgb_stored_seq:          # 새로 들어온 프레임
            self._rgb_stored_seq = self._rgb_seq
            if self._rgb_seq % self.rgb_every == 0:        # every=1 이면 전부 저장
                data, t_recv, stamp = self._rgb_msg
                self.rgb_frames.append(data)
                self.rgb_t.append(t_recv)
                self.rgb_stamp.append(stamp)
                self._rgb_idx = len(self.rgb_frames) - 1
        # 새 프레임이 없으면 직전 것을 가리킨다(hold) — 30Hz vs 100Hz 라 정상이지만,
        # 카메라가 죽으면 계속 얼어붙은 이미지를 가리키므로 오래되면 경고한다(조용한 오염 방지).
        age = time.perf_counter() - self._rgb_msg[1]
        if age > 0.5:
            self.get_logger().warn(
                f"RGB 프레임이 {age:.1f}초째 갱신 안 됨 — 카메라 확인 (이 구간은 같은 이미지)",
                throttle_duration_sec=2.0)
        return self._rgb_idx

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
        row["50_real_time_demo"] = np.array(now - self.t_demo_start, dtype=np.float64)
        row["51_real_time_global"] = np.array(now, dtype=np.float64)
        if self.rgb:
            row["40_image"] = np.array(self._rgb_index_for_tick(), dtype=np.int32)
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
        # 시계 앵커: 51_real_time_global 은 perf_counter(프로세스 기준 단조시계)라 이 값만으로는
        # 다른 프로세스·머신·rosbag 과 대조할 수 없다. 두 시계를 나란히 찍어 오프라인 변환이 되게 한다.
        #   wall_clock ≈ t0_unix + (perf - t0_perf)
        self.f.attrs["t0_perf"] = time.perf_counter()
        self.f.attrs["t0_unix"] = time.time()
        if self.rgb:
            # JPEG 폴더는 h5 와 같은 경로에 <h5 이름>_image (h5 를 옮기면 같이 옮겨야 한다)
            self.img_root = self.out_path.with_name(self.out_path.stem + "_image")
            # 카메라 메타를 파일에 남겨 나중에 재투영/디코딩에 쓸 수 있게 한다
            self.f.attrs["rgb"] = True
            self.f.attrs["rgb_topic"] = RGB_TOPIC
            self.f.attrs["rgb_every"] = self.rgb_every
            self.f.attrs["image_dir"] = self.img_root.name
            for k, v in self.rgb_meta.items():
                self.f.attrs[f"rgb_{k}"] = v
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
            self.rgb_frames, self.rgb_t, self.rgb_stamp = [], [], []
            # _rgb_msg 를 비우지 않으면 첫 스텝이 "발판 누르기 전" 프레임을 가리켜 임의로 낡은
            # 이미지가 들어간다. 새 프레임이 올 때까지 index=-1 로 두는 게 맞다.
            self._rgb_msg = None
            self._rgb_stored_seq, self._rgb_idx = -1, -1
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
        n = len(self.buffer)
        grp = self.f.create_group(f"Demo_{self.demo_idx}")
        keys = list(self.buffer[0].keys())
        for k in keys:
            data = np.stack([b[k] for b in self.buffer])
            # chunks 를 명시하지 않으면 h5py 가 알아서 고르는데, 시간축을 잘게 쪼개고 특징축을
            # 나눠버려서(예: (N,1524)->(938,24)) 한 스텝을 읽으려면 수십 chunk 를 풀어야 한다.
            # 실측: 자동 chunk 는 스텝당 2237us, chunks=(100,D) 는 61us → 37배 빠름(압축률은 73%).
            # 학습 DataLoader 의 랜덤 접근 속도를 좌우하는 부분이라 반드시 지정한다.
            chunks = (min(100, n),) + data.shape[1:] if data.ndim > 1 else (min(1000, n),)
            grp.create_dataset(k, data=data, chunks=chunks, compression="gzip",
                               compression_opts=4, shuffle=True)
        grp.attrs["n_samples"] = n
        grp.attrs["fruit_type_name"] = self.fruit_type_name   # 33_fruit_type 의 사람이 읽는 이름

        rgb_note = ""
        if self.rgb:
            # 이미지는 h5 밖 <h5 stem>_image/Demo_N/000000.jpg 로 나가고, h5 에는 번호만 남는다.
            # 40_image   : 스텝(N)별로 그 시점에 보이던 파일 번호 (-1 = 아직 프레임 없음) — _tick 이 채움
            # 41_rgb_time : 수신 시각(perf_counter) — 51_real_time_global 과 같은 시계라 바로 정렬 가능
            # 42_rgb_stamp: 카메라 캡처 시각(ROS header) — 전송 지연 확인용
            # (41/42 은 스텝이 아니라 "파일 번호" 축이다: 41_rgb_time[40_image[step]])
            demo_dir = self.img_root / f"Demo_{self.demo_idx}"
            demo_dir.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(self.rgb_frames):
                (demo_dir / f"{i:06d}.jpg").write_bytes(b)
            grp.create_dataset("41_rgb_time", data=np.asarray(self.rgb_t, dtype=np.float64))
            grp.create_dataset("42_rgb_stamp", data=np.asarray(self.rgb_stamp, dtype=np.float64))
            mb = sum(len(b) for b in self.rgb_frames) / 1e6
            grp.attrs["n_rgb_frames"] = len(self.rgb_frames)
            rgb_note = f", RGB {len(self.rgb_frames)}장 {mb:.1f}MB → {demo_dir.name}/"
        self.f.flush()
        self.get_logger().info(
            f"<<< 저장 Demo_{self.demo_idx}  ({n} step, {n/RATE_HZ:.1f}s{rgb_note})")
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
        if self.rgb:
            ok = "✓" if self._rgb_n_recv else "✗ (수신 없음)"
            m = self.rgb_meta
            info = (f"{m.get('width','?')}x{m.get('height','?')} {m.get('format','?')}"
                    if m else "메타 없음")
            print(f"  {ok:14s} {'40_image':28s} <- {RGB_TOPIC}")
            print(f"  {'':14s} {'':28s}    {self._rgb_n_recv}장 수신 / 3초, {info}")
        ok = "✓" if self.fruit_type_name else "✗ (수신 없음)"
        print(f"  {ok:14s} {'attrs fruit_type_name':28s} <- /fruit/type_name"
              f"  (수신 {self.fruit_type_name!r})")
        print("=========================\n")
        raise KeyboardInterrupt

    def close(self):
        if self.recording:
            self.recording = False
            self._save_demo()
        self._close_file()


def _key_loop(node: Recorder):
    """엔터 없이 1글자씩 읽어 k 를 롤오버 요청으로 넘긴다 (터미널은 main 이 원복)."""
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            return
        if ch in ("k", "K"):
            node.roll_req = True


def main():
    ap = argparse.ArgumentParser(description="ROS2 구독형 HDF5 레코더 (글러브 데모 수집)")
    ap.add_argument("--check", action="store_true", help="3초간 수신 토픽 점검 후 종료")
    ap.add_argument("--rgb-on", action="store_true",
                    help="RGB 도 저장. <h5 이름>_image/Demo_N/000000.jpg 로 나가고 "
                         "h5 에는 스텝별 파일 번호(40_image)만 남는다 (재인코딩 없음)")
    ap.add_argument("--rgb-topic", default=RGB_TOPIC, help=f"RGB 토픽 (기본 {RGB_TOPIC})")
    ap.add_argument("--rgb-every", type=int, default=1,
                    help="N장마다 1장만 저장(용량 절감). 1=전부(≈30Hz), 2=15Hz, 3=10Hz")
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(check=args.check, rgb=args.rgb_on,
                    rgb_topic=args.rgb_topic, rgb_every=args.rgb_every)

    # k 키 감지: cbreak 로 엔터 없이 1글자 읽기 (ISIG 는 유지되어 Ctrl+C 정상 동작).
    # 터미널 원복은 main 의 finally 에서 — 키 스레드는 daemon 이라 finally 가 안 돌 수 있다.
    old_tty = None
    if not args.check and sys.stdin.isatty():
        old_tty = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=_key_loop, args=(node,), daemon=True).start()
    elif not args.check:
        # nohup/백그라운드 등 tty 없이 띄우면 키를 읽을 수 없다 — 조용히 안 먹는 것보다 알려준다.
        node.get_logger().warn("tty 아님 → k 키 롤오버 사용 불가 (터미널에서 직접 실행하세요)")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C 시 rclpy 가 컨텍스트를 먼저 닫으면 KeyboardInterrupt 대신 후자가 올라온다.
        # 둘 다 잡아야 저장 직후 traceback 이 안 뜬다(파일은 finally 에서 이미 마무리됨).
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
