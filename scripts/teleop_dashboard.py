#!/usr/bin/env python3
"""텔레옵 통합 대시보드 — vive / franka EE / 글러브 / 핸드 / paxini 를 한 화면에.

구독 전용 노드다. 아무것도 발행하지 않으므로 텔레옵 동작에 개입하지 않는다.
통신 부하를 안 주려고:
  - 모든 구독이 BEST_EFFORT + depth 1 (최신 것만, 밀리면 그냥 버림)
  - 콜백은 값 저장만 (포맷/출력 없음)
  - 화면은 별도 타이머로 RENDER_HZ 회만, 커서를 올려 같은 자리에 숫자만 덮어씀
    (스크롤이 없어 터미널 I/O 도 최소)

실행 (env 먼저):
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  export FASTRTPS_DEFAULT_PROFILES_FILE=~/Desktop/vive_franka_teleop/config/fastdds_lan_only.xml
  python3 scripts/teleop_dashboard.py            # Ctrl-C 로 종료 (텔레옵은 계속 돎)
  python3 scripts/teleop_dashboard.py --hz 5     # 갱신 느리게
"""
import argparse
import json
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, String

HAND_SIDE = 'right'
STALE = 0.5          # s, 이보다 오래되면 stale 표시
NJ = 16
DASH = '--'          # 값 없음 표시 (f-string 중첩 따옴표 회피용)

C_OK, C_WARN, C_BAD, C_DIM, C_HEAD, C_RST = (
    '\033[32m', '\033[33m', '\033[31m', '\033[90m', '\033[36m', '\033[0m')


def age_mark(t, now):
    """(표시문자열, 색) — 수신 없음/오래됨/정상."""
    if t < 0:
        return '  --  ', C_DIM
    a = now - t
    if a > STALE:
        return f'{a:5.1f}s', C_BAD
    return f'{a*1000:4.0f}ms', C_OK


class Dash(Node):
    def __init__(self, hz):
        super().__init__('teleop_dashboard')
        q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST, depth=1)

        # 값 저장소 (콜백은 여기 쓰기만)
        self.vive = {s: {'p': None, 'q': None, 't': -1.0, 'valid': False} for s in ('right', 'left')}
        self.delta = {s: {'d': None, 't': -1.0} for s in ('right', 'left')}
        self.ee = {s: {'p': None, 't': -1.0} for s in ('right', 'left')}
        self.tgt = {s: {'p': None, 't': -1.0} for s in ('right', 'left')}
        self.glove = {'v': None, 't': -1.0}
        self.hand_tgt = {'v': None, 't': -1.0}
        self.hand_cur = {'v': None, 't': -1.0}
        self.paxini = {'v': None, 't': -1.0}
        self.engage = {'arm': None, 'hand': None}
        self.recording = None

        for s in ('right', 'left'):
            self.create_subscription(PoseStamped, f'/vive/{s}/pose',
                                     lambda m, x=s: self._pose(self.vive[x], m), q)
            self.create_subscription(Bool, f'/vive/{s}/valid',
                                     lambda m, x=s: self.vive[x].update(valid=bool(m.data)), q)
            self.create_subscription(PoseStamped, f'/franka/{s}/ee_pose',
                                     lambda m, x=s: self._pose(self.ee[x], m), q)
            self.create_subscription(PoseStamped, f'/franka/{s}/ee_target_world',
                                     lambda m, x=s: self._pose(self.tgt[x], m), q)
            self.create_subscription(String, f'/teleop/delta/{s}',
                                     lambda m, x=s: self._delta(x, m), q)
            self.create_subscription(Bool, f'/teleop/engage/{s}',
                                     lambda m: self.engage.update(arm=bool(m.data)), q)

        self.create_subscription(Float32MultiArray, f'/glove/{HAND_SIDE}/q_raw',
                                 lambda m: self._arr(self.glove, m), q)
        self.create_subscription(Float32MultiArray, f'/hand/{HAND_SIDE}/q_target',
                                 lambda m: self._arr(self.hand_tgt, m), q)
        self.create_subscription(JointState, f'/hand/{HAND_SIDE}/joint_states',
                                 self._hand_state, q)
        self.create_subscription(Float32MultiArray, f'/glove/paxini/{HAND_SIDE}/ft',
                                 lambda m: self._arr(self.paxini, m), q)
        self.create_subscription(Bool, f'/teleop/hand_engage/{HAND_SIDE}',
                                 lambda m: self.engage.update(hand=bool(m.data)), q)
        self.create_subscription(Bool, '/record/enable',
                                 lambda m: setattr(self, 'recording', bool(m.data)), q)

        self.first = True
        self.nlines = 0
        self.create_timer(1.0 / hz, self._render)

    # ---- 콜백: 저장만 ----
    def _pose(self, slot, m):
        slot['p'] = (m.pose.position.x, m.pose.position.y, m.pose.position.z)
        slot['t'] = time.monotonic()

    def _arr(self, slot, m):
        slot['v'] = list(m.data)
        slot['t'] = time.monotonic()

    def _hand_state(self, m):
        if len(m.position) >= NJ:
            self.hand_cur['v'] = list(m.position[:NJ])
            self.hand_cur['t'] = time.monotonic()

    def _delta(self, side, m):
        try:
            self.delta[side]['d'] = json.loads(m.data)
            self.delta[side]['t'] = time.monotonic()
        except (ValueError, TypeError):
            pass

    # ---- 화면 ----
    def _render(self):
        now = time.monotonic()
        L = []
        A = L.append

        def onoff(v, on='ON', off='off'):
            if v is None:
                return f'{C_DIM}  -- {C_RST}'
            return f'{C_OK}{on:>4}{C_RST}' if v else f'{C_WARN}{off:>4}{C_RST}'

        A(f'{C_HEAD}══ TELEOP DASHBOARD ══{C_RST}  '
          f'페달 팔:{onoff(self.engage["arm"], "GO", "STOP")} '
          f'손:{onoff(self.engage["hand"], "GO", "STOP")}  '
          f'REC:{onoff(self.recording, "REC", "off")}   '
          f'{C_DIM}왼=STOP 오른=GO 중간=REC | Ctrl-C=대시보드만 종료{C_RST}')

        # --- VIVE ---
        A(f'{C_HEAD}VIVE   {C_RST}   valid   age        pos[m]                 Δpos[m] (engage 기준)      eng')
        for s in ('right', 'left'):
            v = self.vive[s]
            am, ac = age_mark(v['t'], now)
            p = v['p']
            ps = f'{p[0]:+7.3f}{p[1]:+8.3f}{p[2]:+8.3f}' if p else f'{C_DIM}     --      --      --{C_RST}'
            d = self.delta[s]['d']
            if d:
                dp = d.get('pos', [0, 0, 0])
                ds = f'{dp[0]:+7.3f}{dp[1]:+8.3f}{dp[2]:+8.3f}'
                eng = f'{C_OK} eng{C_RST}' if d.get('engaged') else f'{C_DIM}  · {C_RST}'
            else:
                ds, eng = f'{C_DIM}     --      --      --{C_RST}', f'{C_DIM}  · {C_RST}'
            vf = f'{C_OK}  ok {C_RST}' if v['valid'] else f'{C_BAD} BAD {C_RST}'
            A(f'  {s:<6}  {vf}  {ac}{am}{C_RST}  {ps}   {ds}  {eng}')

        # --- FRANKA EE ---
        A(f'{C_HEAD}FRANKA {C_RST}     age        실제 EE[m]                  보내는 TARGET[m]')
        for s in ('right', 'left'):
            e, t = self.ee[s], self.tgt[s]
            am, ac = age_mark(e['t'], now)
            tm, tc = age_mark(t['t'], now)
            es = (f'{e["p"][0]:+7.3f}{e["p"][1]:+8.3f}{e["p"][2]:+8.3f}' if e['p']
                  else f'{C_DIM}     --      --      --{C_RST}')
            ts = (f'{t["p"][0]:+7.3f}{t["p"][1]:+8.3f}{t["p"][2]:+8.3f}' if t['p']
                  else f'{C_DIM}     --      --      --{C_RST}')
            A(f'  {s:<6}  {ac}{am}{C_RST}  {es}    {tc}{tm}{C_RST} {ts}')

        # --- GLOVE / HAND (16관절, 8개씩 2줄) ---
        gm, gc = age_mark(self.glove['t'], now)
        hm, hc = age_mark(self.hand_cur['t'], now)
        A(f'{C_HEAD}HAND {HAND_SIDE}{C_RST}  glove {gc}{gm}{C_RST}   상태 {hc}{hm}{C_RST}   '
          f'{C_DIM}(count){C_RST}')
        for blk in (0, 8):
            A(f'  {C_DIM}idx  {"".join(f"{i:>7}" for i in range(blk, blk+8))}{C_RST}')
            for name, slot in (('glove', self.glove), ('tgt  ', self.hand_tgt), ('now  ', self.hand_cur)):
                v = slot['v']
                if v and len(v) >= blk + 8:
                    A(f'  {name}{"".join(f"{v[i]:>7.0f}" for i in range(blk, blk+8))}')
                else:
                    A(f'  {name}{C_DIM}{"".join(f"{DASH:>7}" for _ in range(8))}{C_RST}')

        # --- PAXINI (DexFIT) ---
        pm, pc = age_mark(self.paxini['t'], now)
        A(f'{C_HEAD}PAXINI {C_RST}{pc}{pm}{C_RST}  {C_DIM}(글러브 촉각, 손가락별 Fx Fy Fz){C_RST}')
        v = self.paxini['v']
        for i, fn in enumerate(('엄지', '검지', '중지', '약지')):
            if v and len(v) >= 3 * i + 3:
                fx, fy, fz = v[3*i], v[3*i+1], v[3*i+2]
                hot = C_OK if abs(fz) > 0.05 else C_DIM
                A(f'  {fn}  Fx{fx:+7.2f}  Fy{fy:+7.2f}  {hot}Fz{fz:+7.2f}{C_RST}')
            else:
                A(f'  {fn}  {C_DIM}Fx     --  Fy     --  Fz     --{C_RST}')

        out = ''.join(ln + '\033[K\n' for ln in L)
        if not self.first and self.nlines:
            out = f'\033[{self.nlines}F' + out
        sys.stdout.write(out)
        sys.stdout.flush()
        self.first = False
        self.nlines = len(L)


def main():
    ap = argparse.ArgumentParser(description='텔레옵 통합 대시보드 (구독 전용)')
    ap.add_argument('--hz', type=float, default=10.0, help='화면 갱신 [Hz] (기본 10)')
    args = ap.parse_args()

    rclpy.init()
    node = Dash(args.hz)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass                      # Ctrl-C / SIGTERM(stop) — 정상 종료 경로
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print()


if __name__ == '__main__':
    main()
