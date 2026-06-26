#!/usr/bin/env python3
"""델타 시각화: /teleop/delta/<arm> 의 델타를 그래프 + 3D 경로로 라이브 표시.

화면 구성:
  - 왼쪽 3D: engage anchor 원점(0,0,0) 기준으로 트래커가 그린 위치 경로(궤적).
             pos=[dx,dy,dz] 가 곧 anchor 기준 좌표이므로 그대로 점을 이으면 경로가 됨.
  - 오른쪽 위: 위치 델타 dx,dy,dz [m] 시계열
  - 오른쪽 아래: 회전 델타 rotvec rx,ry,rz [rad] 시계열
engage(클러치 1) 후 트래커를 움직이면 경로가 그려지고, 재engage 하면 경로 리셋.
disengage/invalid 구간은 기록 안 함(경로가 0으로 튀지 않게).

실행:
  source /opt/ros/humble/setup.bash && source ~/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 ~/Desktop/vive_franka_teleop/scripts/delta_viz.py [arm]   # arm 기본 right
  python3 ~/Desktop/vive_franka_teleop/scripts/delta_viz.py right --selftest  # GUI 없이 수신확인
"""
import sys
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

try:
    import ujson as json  # noqa
except ImportError:
    import json

MAXLEN = 6000      # 보관 샘플 수 (100Hz면 60초)
WIN_SEC = 20.0     # 시계열 표시 창 [s]


class DeltaCollector(Node):
    def __init__(self, arm):
        super().__init__('delta_viz')
        self.arm = arm
        self.lock = threading.Lock()
        self.t = deque(maxlen=MAXLEN)
        self.dx = deque(maxlen=MAXLEN)
        self.dy = deque(maxlen=MAXLEN)
        self.dz = deque(maxlen=MAXLEN)
        self.rx = deque(maxlen=MAXLEN)
        self.ry = deque(maxlen=MAXLEN)
        self.rz = deque(maxlen=MAXLEN)
        self.engaged = False
        self.valid = False
        self.msg_count = 0
        self._t0 = None
        self._prev_engaged = False
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, f'/teleop/delta/{arm}', self._cb, qos)

    def _clear(self):
        for d in (self.t, self.dx, self.dy, self.dz, self.rx, self.ry, self.rz):
            d.clear()
        self._t0 = None

    def _cb(self, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        eng = bool(d.get('engaged', False))
        val = bool(d.get('valid', False))
        pos = d.get('pos', [0, 0, 0])
        rot = d.get('rot', [0, 0, 0])
        stamp = float(d.get('stamp', 0.0))
        with self.lock:
            self.msg_count += 1
            self.engaged, self.valid = eng, val
            # engage 상승엣지 → 경로 리셋(새 anchor)
            if eng and not self._prev_engaged:
                self._clear()
            self._prev_engaged = eng
            if eng and val:
                if self._t0 is None:
                    self._t0 = stamp
                self.t.append(stamp - self._t0)
                self.dx.append(float(pos[0])); self.dy.append(float(pos[1])); self.dz.append(float(pos[2]))
                self.rx.append(float(rot[0])); self.ry.append(float(rot[1])); self.rz.append(float(rot[2]))

    def snapshot(self):
        with self.lock:
            return (list(self.t), list(self.dx), list(self.dy), list(self.dz),
                    list(self.rx), list(self.ry), list(self.rz),
                    self.engaged, self.valid, self.msg_count)


def run_gui(node, arm):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(13, 7))
    fig.canvas.manager.set_window_title(f'Delta Viz [{arm}]')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    axp = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, 1])

    ax3d.set_title('Path (pos delta from engage origin)')
    ax3d.set_xlabel('dx [m]'); ax3d.set_ylabel('dy [m]'); ax3d.set_zlabel('dz [m]')
    (path3d,) = ax3d.plot([], [], [], lw=1.6, color='tab:blue')
    (cur3d,) = ax3d.plot([], [], [], 'o', color='tab:red', ms=7)
    ax3d.plot([0], [0], [0], '^', color='k', ms=9)  # 원점(engage)

    axp.set_title('Position delta [m]'); axp.grid(alpha=0.3)
    (lpx,) = axp.plot([], [], label='dx', color='tab:red')
    (lpy,) = axp.plot([], [], label='dy', color='tab:green')
    (lpz,) = axp.plot([], [], label='dz', color='tab:blue')
    axp.legend(loc='upper left', ncol=3, fontsize=8)

    axr.set_title('Rotation delta rotvec [rad]'); axr.grid(alpha=0.3)
    axr.set_xlabel('t [s] since engage')
    (lrx,) = axr.plot([], [], label='rx', color='tab:red')
    (lry,) = axr.plot([], [], label='ry', color='tab:green')
    (lrz,) = axr.plot([], [], label='rz', color='tab:blue')
    axr.legend(loc='upper left', ncol=3, fontsize=8)

    def set_3d_limits(xs, ys, zs):
        if not xs:
            lim = 0.1
            cx = cy = cz = 0.0
        else:
            allv = xs + ys + zs + [0.0]
            r = max(0.05, max(abs(v) for v in allv))
            lim = r * 1.1
            cx = cy = cz = 0.0
        ax3d.set_xlim(cx - lim, cx + lim)
        ax3d.set_ylim(cy - lim, cy + lim)
        ax3d.set_zlim(cz - lim, cz + lim)
        try:
            ax3d.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    def update(_frame):
        t, dx, dy, dz, rx, ry, rz, eng, val, n = node.snapshot()
        path3d.set_data_3d(dx, dy, dz)
        if dx:
            cur3d.set_data_3d([dx[-1]], [dy[-1]], [dz[-1]])
        else:
            cur3d.set_data_3d([], [], [])
        set_3d_limits(dx, dy, dz)

        lpx.set_data(t, dx); lpy.set_data(t, dy); lpz.set_data(t, dz)
        lrx.set_data(t, rx); lry.set_data(t, ry); lrz.set_data(t, rz)
        if t:
            t1 = t[-1]
            t0 = max(0.0, t1 - WIN_SEC)
            for ax in (axp, axr):
                ax.set_xlim(t0, max(t1, t0 + 1e-3))
            axp.relim(); axp.autoscale_view(scaley=True, scalex=False)
            axr.relim(); axr.autoscale_view(scaley=True, scalex=False)
        flag = 'ENGAGED' if eng else 'STANDBY'
        col = 'green' if (eng and val) else 'gray'
        mag = (dx[-1] ** 2 + dy[-1] ** 2 + dz[-1] ** 2) ** 0.5 if dx else 0.0
        fig.suptitle(f'[{arm}]  {flag}  valid={val}   |p|={mag:.3f} m   pts={len(dx)}   msgs={n}',
                     color=col, fontsize=11)
        return path3d, cur3d, lpx, lpy, lpz, lrx, lry, lrz

    set_3d_limits([], [], [])
    _anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def main():
    argv = [a for a in sys.argv[1:]]
    selftest = '--selftest' in argv
    pos = [a for a in argv if not a.startswith('-')]
    arm = pos[0] if pos else 'right'

    rclpy.init()
    node = DeltaCollector(arm)

    def _spin():
        try:
            rclpy.spin(node)
        except Exception:
            pass

    spin = threading.Thread(target=_spin, daemon=True)
    spin.start()

    def _cleanup():
        rclpy.shutdown()
        spin.join(timeout=1.0)
        try:
            node.destroy_node()
        except Exception:
            pass

    if selftest:
        import time
        time.sleep(3.0)
        t, dx, *_ , eng, val, n = node.snapshot()
        print(f'[selftest] arm={arm} msgs={n} engaged_samples={len(dx)} engaged={eng} valid={val}')
        _cleanup()
        return

    try:
        run_gui(node, arm)
    finally:
        _cleanup()


if __name__ == '__main__':
    main()
