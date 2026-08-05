#!/usr/bin/env python3
"""Diffusion Policy 실기 배포 (ROS2) — 레몬 in-hand 회전, KISTAR 오른손 16관절.

═══════════════════════════════════════════════════════════════════════════
구독 (학습 obs 와 정확히 동일한 순서로 조립)
    /hand/right/joint_states   JointState          → 03_hand_j_pos   (16)
    /hand/right/kin            Float32MultiArray   → 06_hand_j_kin   (12)
    /paxini/right/ft           Float32MultiArray   → 20_paxini_ft    (12)
    /fruit/pose                PoseStamped         → 30_fruit_pos    (3)
    /fruit/size                Float32MultiArray   → 32_fruit_size   (3)
발행
    /hand/right/q_target       Float32MultiArray   (16, 절대 타겟 [count])
    /dp/debug                  Float64MultiArray   (진단)
═══════════════════════════════════════════════════════════════════════════

속도/청킹 조절 (전부 커맨드라인)
    --control_hz 20      정책 추론·액션 소비 주기 (학습은 20Hz. 바꾸면 시간축이 달라짐)
    --publish_hz 100     실제 타겟 발행 주기. 정책 타겟 사이를 선형 보간 → 스텝 없이 매끄럽게
    --exec_horizon 8     추론 1회당 실행할 액션 수 (작을수록 반응 빠름/추론 부하 큼)
    --obs_horizon 2      관찰 스텝 수 (체크포인트 값 기본)
    --pred_horizon 16    예측 길이 (체크포인트 값 기본)
    --temporal_ensemble  매 틱 추론 + 겹치는 예측 지수가중 평균 (ACT 방식, exec_horizon 무시)
    --ddim_steps 10      DDIM 추론 스텝 (적으면 빠르고 거칠다)

안전 가드
    관절 한계     체크포인트의 JOINT_LIMITS 로 클램프
    속도 한계     --max_rate_cps (기본: 학습 데이터 실측 p99.9 = 12000 count/s)
    시작 램프     --ramp_sec 동안 현재 손 자세 → 정책 타겟으로 서서히 (튐 방지)
    워치독        obs 가 --stale_sec 이상 낡으면 마지막 타겟 홀드 (새 타겟 발행 중단)
    인게이지      --enable_topic 이 True 여야 발행. 발판/키보드로 즉시 정지 가능
    중복 발행     시작 시 q_target 의 다른 퍼블리셔를 감지해 경고 (글러브 텔레옵과 동시 금지)
    --dry_run     계산만 하고 발행하지 않음

실행:
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 run.py --ckpt runs/dp_lemon/best.pt --dry_run          # 먼저 무발행 확인
  ros2 topic pub -1 /dp/enable std_msgs/Bool "{data: true}"      # 인게이지
  python3 run.py --ckpt runs/dp_lemon/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import torch

import rclpy
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray, Int32

import dp_config as C
from dp_data import Normalizer
from dp_model import DDPMScheduler, policy_from_ckpt

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=1)


# ══════════════════════════════════════════════════════════════════════════
class Runner(Node):
    def __init__(self, policy, sched, obs_norm, act_norm, cfg, args):
        super().__init__("dp_lemon_runner")
        self.policy, self.sched = policy, sched
        self.obs_norm, self.act_norm = obs_norm, act_norm
        self.cfg, self.args = cfg, args
        self.device = next(policy.parameters()).device
        side = args.side

        self.lock = threading.Lock()
        self.j_pos = np.zeros(16, np.float32)
        self.kin = np.zeros(12, np.float32)
        self.ft = np.zeros(12, np.float32)
        self.fruit_pos = np.zeros(3, np.float32)
        self.fruit_size = np.zeros(3, np.float32)
        self.last_fruit_ok = None          # 위생처리용 마지막 유효값 (pos,size)
        self.t_hand = 0.0
        self.t_fruit = 0.0
        self.enabled = not args.require_enable

        self.obs_buf = deque(maxlen=args.obs_horizon)
        self.act_buf = deque()             # 미소비 정책 액션 (각 (16,))
        self.te_acc: dict[int, tuple] = {} # temporal ensembling 누적
        self.policy_tick = 0

        self.prev_target = None            # 직전 발행 타겟 (보간·속도제한 기준)
        self.cur_target = None             # 지금 향하는 정책 타겟
        self.interp_i = 0
        self.n_interp = max(1, int(round(args.publish_hz / args.control_hz)))
        self.t_start = time.time()
        self.pub_cnt = 0
        self.infer_ms = deque(maxlen=50)
        self.n_clamp_j = 0
        self.n_clamp_v = 0
        self.n_stale = 0
        self.n_overrun = 0                 # 추론이 제어 주기를 넘긴 횟수
        self.n_starve = 0                  # 발행 시점에 새 정책 타겟이 없던 횟수
        self.n_fruit_stale = 0             # 과일 인식이 낡아 홀드한 횟수(정지 아님)
        self.fruit_ok = True
        self.next_target = None            # 정책 스레드 → 발행 스레드
        self.have_new = False
        self.interp_from = None
        self._plock = threading.Lock()     # 정책 타겟/보간 상태 보호
        self._stop = False
        self._policy_thread = None

        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, f"/hand/{side}/joint_states",
                                 self._cb_js, SENSOR_QOS, callback_group=cb)
        self.create_subscription(Float32MultiArray, f"/hand/{side}/kin",
                                 self._cb_kin, SENSOR_QOS, callback_group=cb)
        self.create_subscription(Float32MultiArray, f"/paxini/{side}/ft",
                                 self._cb_ft, SENSOR_QOS, callback_group=cb)
        self.create_subscription(PoseStamped, "/fruit/pose",
                                 self._cb_fpose, SENSOR_QOS, callback_group=cb)
        self.create_subscription(Float32MultiArray, "/fruit/size",
                                 self._cb_fsize, SENSOR_QOS, callback_group=cb)
        self.create_subscription(Bool, args.enable_topic,
                                 self._cb_enable, 10, callback_group=cb)

        self.pub_target = self.create_publisher(Float32MultiArray,
                                                f"/hand/{side}/q_target", SENSOR_QOS)
        self.pub_debug = self.create_publisher(Float64MultiArray, "/dp/debug", 10)
        self.pub_mode = self.create_publisher(Int32, f"/hand/{side}/cmd_mode", 1)
        self.pub_servo = self.create_publisher(Bool, f"/hand/{side}/cmd_servo", 1)
        self.servo_sent = False

        # 발행 타이머는 절대 겹치지 않게 (겹치면 콜백이 쌓여 지연이 눈덩이처럼 커진다)
        pub_cb = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / args.publish_hz, self._tick, callback_group=pub_cb)
        self.create_timer(2.0, self._conflict_check, callback_group=cb)

        self.jl = np.array(cfg["joint_limits"], dtype=np.float32)   # (16,2)
        self.max_delta_pub = args.max_rate_cps / args.publish_hz

        self._banner()
        self._policy_thread = threading.Thread(target=self._policy_loop, daemon=True)
        self._policy_thread.start()

    # ── 로그 ──────────────────────────────────────────────────────────────
    def _banner(self):
        a = self.args
        L = self.get_logger().info
        L("=" * 74)
        L("  Diffusion Policy Runner — 레몬 in-hand 회전")
        L(f"  ckpt          : {a.ckpt}")
        L(f"  obs/pred/exec : {a.obs_horizon} / {a.pred_horizon} / "
          f"{'TE(매 틱)' if a.temporal_ensemble else a.exec_horizon}")
        L(f"  control_hz    : {a.control_hz}  (학습 {self.cfg['ctrl_hz']:.0f}Hz)")
        L(f"  publish_hz    : {a.publish_hz}  → 정책 타겟 사이 {self.n_interp}틱 선형보간")
        L(f"  DDIM steps    : {a.ddim_steps}")
        L(f"  속도 한계     : {a.max_rate_cps:.0f} count/s "
          f"(발행 틱당 {self.max_delta_pub:.0f})")
        L(f"  램프          : {a.ramp_sec:.1f}s     워치독 : 손 {a.stale_sec:.2f}s / "
          f"과일 {a.fruit_stale_sec:.2f}s(홀드만)")
        L(f"  인게이지      : {a.enable_topic} "
          f"{'(필요)' if a.require_enable else '(무시 — 즉시 동작)'}")
        L(f"  발행          : {'DRY-RUN (발행 안 함)' if a.dry_run else f'/hand/{a.side}/q_target'}")
        if abs(a.control_hz - self.cfg["ctrl_hz"]) > 1e-6:
            L(f"  ⚠ control_hz({a.control_hz}) != 학습 {self.cfg['ctrl_hz']:.0f}Hz "
              f"→ 액션 시간축이 학습과 달라집니다")
        L("=" * 74)

    def _conflict_check(self):
        n = self.count_publishers(f"/hand/{self.args.side}/q_target")
        # 퍼블리셔 객체는 dry_run 에서도 만들므로 내 몫은 항상 1 이다.
        # (0 으로 두면 남이 없어도 오경보가 난다)
        mine = 1
        if n > mine:
            self.get_logger().error(
                f"⚠ /hand/{self.args.side}/q_target 퍼블리셔 {n}개 — "
                "글러브 텔레옵(glove_teleop.py)이 아직 떠 있으면 먼저 끄세요. "
                "동시 스트리밍은 손이 두 타겟 사이에서 튑니다.")

    # ── 콜백 ──────────────────────────────────────────────────────────────
    def _cb_js(self, m):
        with self.lock:
            v = np.asarray(m.position, np.float32)
            self.j_pos[:min(16, len(v))] = v[:16]
            self.t_hand = time.time()

    def _cb_kin(self, m):
        with self.lock:
            v = np.asarray(m.data, np.float32)
            self.kin[:min(12, len(v))] = v[:12]

    def _cb_ft(self, m):
        with self.lock:
            v = np.asarray(m.data, np.float32)
            self.ft[:min(12, len(v))] = v[:12]

    def _cb_fpose(self, m):
        with self.lock:
            self.fruit_pos[:] = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
            self.t_fruit = time.time()

    def _cb_fsize(self, m):
        with self.lock:
            v = np.asarray(m.data, np.float32)
            self.fruit_size[:min(3, len(v))] = v[:3]

    def _cb_enable(self, m):
        if m.data != self.enabled:
            self.get_logger().info(f"인게이지 {'ON' if m.data else 'OFF'}")
        self.enabled = bool(m.data)

    # ── obs 조립 (학습과 동일: 과일 이상치는 직전 유효값 홀드) ────────────
    def _make_obs(self, j_pos, kin, ft, fpos, fsize):
        ok = True
        n = float(np.linalg.norm(fpos))
        lo, hi = C.FRUIT_SIZE_RANGE
        if not (1e-9 < n <= C.FRUIT_POS_MAX_NORM) or not (lo <= fsize[0] <= hi):
            ok = False
        elif self.last_fruit_ok is not None:
            if np.linalg.norm(fpos - self.last_fruit_ok[0]) > C.FRUIT_POS_MAX_JUMP:
                ok = False
        if ok:
            self.last_fruit_ok = (fpos.copy(), fsize.copy())
        elif self.last_fruit_ok is not None:
            fpos, fsize = self.last_fruit_ok
        return np.concatenate([j_pos, kin, ft, fpos, fsize]).astype(np.float32), ok

    @torch.no_grad()
    def _infer(self):
        buf = list(self.obs_buf)
        while len(buf) < self.args.obs_horizon:
            buf.insert(0, buf[0])
        obs = np.stack(buf, 0)
        assert obs.shape == (self.args.obs_horizon, self.cfg["obs_dim"]), obs.shape
        x = torch.from_numpy(self.obs_norm.normalize(obs).astype(np.float32))
        t0 = time.perf_counter()
        a_n = self.policy.sample(x.unsqueeze(0).to(self.device), self.sched,
                                 self.args.pred_horizon, self.args.ddim_steps)
        self.infer_ms.append((time.perf_counter() - t0) * 1000)
        return self.act_norm.denormalize(a_n.squeeze(0).cpu().numpy())   # (T_pred,16)

    # ── 안전 가드 ─────────────────────────────────────────────────────────
    def _guard(self, target, ref):
        """관절 한계 + 속도 한계. ref = 직전 발행 타겟."""
        t = np.asarray(target, np.float32).copy()
        d = t - ref
        big = np.abs(d) > self.max_delta_pub
        if big.any():
            self.n_clamp_v += int(big.sum())
            t = ref + np.clip(d, -self.max_delta_pub, self.max_delta_pub)
        out = (t < self.jl[:, 0]) | (t > self.jl[:, 1])
        if out.any():
            self.n_clamp_j += int(out.sum())
        return np.clip(t, self.jl[:, 0], self.jl[:, 1]).astype(np.float32)

    # ── 정책 스레드: obs 샘플 + 추론 (발행 경로와 분리) ────────────────────
    #
    # 왜 스레드로 빼는가: 추론을 발행 타이머 콜백 안에서 하면, 추론이 발행 주기보다
    # 길어질 때 콜백이 밀려 쌓이고 (Reentrant 그룹이면 동시 실행까지 겹쳐) 지연이
    # 눈덩이처럼 커진다. 실기에서 11ms 추론이 272ms 로 불어났다. 발행은 절대 추론을
    # 기다리면 안 된다.
    #
    # 왜 control_hz 로 도는가: obs 히스토리 간격이 학습과 같아야 한다. 학습은 100Hz
    # 원본에서 stride 5 로 뽑아 T_obs 샘플 간격이 50ms 다. 발행 주기(100Hz)로 obs 를
    # 쌓으면 간격이 10ms 가 되어 시간축이 어긋난다.
    def _policy_loop(self):
        period = 1.0 / self.args.control_hz
        nxt = time.perf_counter()
        while not self._stop:
            nxt += period
            try:
                self._policy_step()
            except Exception as e:
                self.get_logger().error(f"정책 루프 오류: {e}")
            slack = nxt - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                self.n_overrun += 1
                nxt = time.perf_counter()      # 밀렸으면 기준 리셋(누적 지연 방지)

    def _policy_step(self):
        now = time.time()
        with self.lock:
            j_pos = self.j_pos.copy(); kin = self.kin.copy(); ft = self.ft.copy()
            fpos = self.fruit_pos.copy(); fsize = self.fruit_size.copy()
            t_hand, t_fruit = self.t_hand, self.t_fruit

        if t_hand == 0.0:
            return
        # 손 상태가 낡으면 정지(안전). 과일은 낡아도 직전 값을 홀드하고 계속 간다 —
        # 학습 데이터도 과일이 8~19Hz 로 들어와 100Hz 기록 사이를 홀드한 것이므로
        # 홀드가 오히려 학습 분포와 같다. 여기서 멈추면 8Hz 인식만으로 정책이 죽는다.
        if now - t_hand > self.args.stale_sec:
            self.n_stale += 1
            return                          # 새 타겟 안 만듦 → 발행은 마지막 값 유지
        if t_fruit > 0 and now - t_fruit > self.args.fruit_stale_sec:
            self.n_fruit_stale += 1

        obs, fruit_ok = self._make_obs(j_pos, kin, ft, fpos, fsize)
        self.fruit_ok = fruit_ok
        self.obs_buf.append(obs)            # 학습과 같은 control_hz 간격

        if self.args.temporal_ensemble:
            seq = self._infer()
            for k in range(self.args.pred_horizon):
                w = float(np.exp(-self.args.te_k * k))
                i = self.policy_tick + k
                s, ws = self.te_acc.get(i, (np.zeros(16), 0.0))
                self.te_acc[i] = (s + w * seq[k], ws + w)
            s, ws = self.te_acc.pop(self.policy_tick, (None, 0.0))
            nxt = (s / ws) if ws > 0 else seq[0]
            for i in list(self.te_acc):
                if i < self.policy_tick:
                    self.te_acc.pop(i, None)
        else:
            if not self.act_buf:
                seq = self._infer()
                for a in seq[:self.args.exec_horizon]:
                    self.act_buf.append(a.copy())
            nxt = self.act_buf.popleft()

        self.policy_tick += 1
        with self._plock:
            self.next_target = np.asarray(nxt, np.float32)
            self.have_new = True

    # ── 발행 틱: 보간 + 가드 + 발행만. 추론은 하지 않는다 ─────────────────
    def _tick(self):
        now = time.time()
        with self.lock:
            j_pos = self.j_pos.copy()
            t_hand = self.t_hand
        if t_hand == 0.0:
            self.get_logger().warn(
                f"/hand/{self.args.side}/joint_states 미수신 — 제어 PC 확인",
                throttle_duration_sec=3.0)
            return
        if self.prev_target is None:
            self.prev_target = j_pos.copy()      # 램프 시작점 = 현재 자세
            self.interp_from = j_pos.copy()

        with self._plock:
            if self.have_new:
                self.cur_target = self.next_target.copy()
                self.have_new = False
                self.interp_from = self.prev_target.copy()
                self.interp_i = 0
        if self.cur_target is None:
            self.n_starve += 1
            return

        self.interp_i += 1
        frac = min(1.0, self.interp_i / self.n_interp)
        target = self.interp_from + (self.cur_target - self.interp_from) * frac
        el = now - self.t_start
        if el < self.args.ramp_sec:                  # 시작 램프 (튐 방지)
            target = j_pos + (target - j_pos) * (el / self.args.ramp_sec)

        target = self._guard(target, self.prev_target)

        if self.enabled and not self.args.dry_run:
            if not self.servo_sent and self.args.servo_on:
                self.pub_mode.publish(Int32(data=1))
                self.pub_servo.publish(Bool(data=True))
                self.servo_sent = True
                self.get_logger().info("핸드 servo ON (mode=position)")
            m = Float32MultiArray(); m.data = [float(v) for v in target]
            self.pub_target.publish(m)
            self.pub_cnt += 1
        self.prev_target = target.copy()   # 속도 가드는 '실제로 보낸 값' 기준

        infer = float(np.mean(self.infer_ms)) if self.infer_ms else 0.0
        d = Float64MultiArray()
        d.data = [float(self.enabled), float(self.fruit_ok), infer,
                  float(len(self.act_buf)), float(np.abs(target - j_pos).max()),
                  float(self.n_clamp_j), float(self.n_clamp_v), float(self.n_stale),
                  float(self.n_overrun), float(self.n_starve)]
        self.pub_debug.publish(d)

        self.tick_cnt = getattr(self, "tick_cnt", 0) + 1
        if self.tick_cnt % int(self.args.publish_hz) == 0:
            self.get_logger().info(
                f"[{el:6.1f}s] en={int(self.enabled)} "
                f"fruit={'ok' if self.fruit_ok else 'HOLD'} "
                f"infer={infer:5.1f}ms buf={len(self.act_buf)} "
                f"|Δ|max={np.abs(target-j_pos).max():6.1f} "
                f"clamp(j/v)={self.n_clamp_j}/{self.n_clamp_v} "
                f"stale={self.n_stale}/{self.n_fruit_stale} overrun={self.n_overrun} "
                f"starve={self.n_starve} tgt[0:4]={np.round(target[:4],0)}")

    def shutdown(self):
        self._stop = True
        if self._policy_thread is not None:
            self._policy_thread.join(timeout=1.0)
        if self.args.hold_on_exit or self.args.dry_run:
            return
        self.get_logger().info("종료 — 마지막 타겟 유지 (서보는 끄지 않음)")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Diffusion Policy 실기 배포")
    ap.add_argument("--ckpt", default="runs/dp_lemon/best.pt")
    ap.add_argument("--weights", default="auto", choices=["auto", "ema", "raw"],
                    help="auto = 체크포인트가 기록한 더 좋은 쪽")
    ap.add_argument("--side", default="right")
    # 속도 / 청킹
    ap.add_argument("--control_hz", type=float, default=None, help="기본: 체크포인트 값")
    ap.add_argument("--publish_hz", type=float, default=100.0)
    ap.add_argument("--exec_horizon", type=int, default=None, help="기본: 체크포인트 값")
    ap.add_argument("--obs_horizon", type=int, default=None)
    ap.add_argument("--pred_horizon", type=int, default=None)
    ap.add_argument("--ddim_steps", type=int, default=None)
    ap.add_argument("--temporal_ensemble", action="store_true")
    ap.add_argument("--te_k", type=float, default=0.6, help="TE 지수가중 감쇠")
    # 안전
    ap.add_argument("--max_rate_cps", type=float, default=None,
                    help="관절 속도 상한 [count/s]. 기본: 체크포인트(학습 실측)")
    ap.add_argument("--ramp_sec", type=float, default=2.0)
    ap.add_argument("--stale_sec", type=float, default=0.3,
                    help="손 상태가 이보다 낡으면 새 타겟 생성 중단(안전)")
    ap.add_argument("--fruit_stale_sec", type=float, default=1.0,
                    help="과일 인식이 이보다 낡으면 카운트만(직전 값 홀드, 정지 아님)")
    ap.add_argument("--enable_topic", default="/dp/enable")
    ap.add_argument("--require_enable", type=int, default=1)
    ap.add_argument("--servo_on", type=int, default=1)
    ap.add_argument("--hold_on_exit", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU 강제 (지연 안정적)")
    args, ros_args = ap.parse_known_args()

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 체크포인트 없음: {args.ckpt}")
        return 1
    torch.set_num_threads(1)          # ROS 콜백과의 GIL 경쟁 줄이기
    torch.set_grad_enabled(False)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]

    # 체크포인트 값이 기본, 커맨드라인이 우선
    args.obs_horizon = args.obs_horizon or cfg["obs_horizon"]
    args.pred_horizon = args.pred_horizon or cfg["pred_horizon"]
    args.exec_horizon = args.exec_horizon or cfg["action_steps"]
    args.ddim_steps = args.ddim_steps or cfg.get("infer_steps", C.INFER_STEPS)
    args.control_hz = args.control_hz or cfg["ctrl_hz"]
    args.max_rate_cps = args.max_rate_cps or cfg.get("max_rate_cps", C.MAX_RATE_CPS)
    args.require_enable = bool(args.require_enable)
    args.servo_on = bool(args.servo_on)
    if args.exec_horizon > args.pred_horizon:
        print(f"[ERROR] exec_horizon({args.exec_horizon}) > pred_horizon({args.pred_horizon})")
        return 1

    if args.weights == "auto":
        args.weights = ck.get("best_metric", {}).get("prefer_weights", "ema")
    policy = policy_from_ckpt(ck, device, use_ema=(args.weights == "ema"))
    sched = DDPMScheduler(cfg["diff_steps"]).to(device)
    obs_norm = Normalizer(ck["obs_norm"]["mean"], ck["obs_norm"]["std"])
    act_norm = Normalizer(ck["act_norm"]["mean"], ck["act_norm"]["std"])
    print(f"[MODEL] {args.ckpt}  weights={args.weights}  epoch={ck.get('epoch')}  "
          f"best_val={ck.get('best_val', float('nan')):.5f}  device={device}")
    print(f"[NORM ] 체크포인트 내장 통계 사용 (obs {cfg['obs_dim']}dim / act {cfg['action_dim']}dim)")

    rclpy.init(args=ros_args or None)
    node = Runner(policy, sched, obs_norm, act_norm, cfg, args)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
