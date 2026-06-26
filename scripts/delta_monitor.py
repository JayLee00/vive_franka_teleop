#!/usr/bin/env python3
"""실시간 델타 비교 모니터: anchored(절댓값) vs per-step(증분) 동시 표시.

데이터원: /teleop/delta/<arm> (std_msgs/String JSON, ~100Hz). 양팔(left,right).

세 가지를 나란히 보여준다:
  ABS   = engage anchor 기준 누적 이동  (= 송신 pos/rot 그대로, A 방식)
  INC   = 직전 메시지 대비 증분          (consecutive 차분, B 방식이 보낼 값)
  WORLD = vive_world 기준 트래커 절대 포즈 (abs_pos/abs_quat, 참고용)

A(anchored) vs B(step증분) 어느 쪽이 좋은지 실제 데이터로 보려고 만든 도구.
- 손을 멈추면: ABS는 값이 그대로 유지(누적 위치 고정), INC는 ~0 으로 떨어짐.
- 노이즈/지연이 있으면: INC가 0 근처에서 튀고, 그게 B에서는 영구 누적된다.
- 손을 engage 위치로 되돌리면: ABS가 0 으로 복귀(A의 1:1 대응). B는 드리프트로 안 맞음.

실행 (랩 표준 env, 송신측/제어측 어느 PC에서나 같은 DOMAIN 9이면 됨):
  source /opt/ros/humble/setup.bash && source /home/js/franka_ros2_ws/install/setup.bash
  export ROS_DOMAIN_ID=9 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0
  python3 ~/Desktop/vive_franka_teleop/scripts/delta_monitor.py            # left,right
  python3 ~/Desktop/vive_franka_teleop/scripts/delta_monitor.py right      # 특정 팔만
"""
import json
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

RAD2DEG = 180.0 / math.pi


# --------------------------- quaternion / rotvec utils ---------------------------
def rotvec_to_quat(rv):
    """rotvec(axis-angle) [rx,ry,rz] -> 단위 쿼터니언 [x,y,z,w]."""
    ang = math.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
    if ang < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    s = math.sin(ang / 2.0) / ang
    return [rv[0] * s, rv[1] * s, rv[2] * s, math.cos(ang / 2.0)]


def quat_conj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def quat_mul(a, b):  # Hamilton, [x,y,z,w]
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_angle_deg(q):
    """단위쿼터니언 회전각 [deg] (shortest path)."""
    w = abs(max(-1.0, min(1.0, q[3])))
    vn = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    return 2.0 * math.atan2(vn, w) * RAD2DEG


def quat_to_euler_deg(q):
    """ZYX(roll-pitch-yaw) Euler [deg]: roll=x축, pitch=y축, yaw=z축. (tracker_read.py와 동일)"""
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG


# --------------------------------- per-arm state ---------------------------------
class ArmDisp:
    def __init__(self):
        self.have = False
        self.engaged = False
        self.valid = False
        # ABS (anchored, 송신값 그대로)
        self.abs_pos = [0.0, 0.0, 0.0]
        self.abs_rotvec = [0.0, 0.0, 0.0]
        # INC (직전 메시지 대비)
        self.inc_pos = [0.0, 0.0, 0.0]
        self.inc_ang_deg = 0.0
        self.lin_vel = 0.0   # m/s
        self.ang_vel = 0.0   # deg/s
        # WORLD (vive_world 절대 트래커 포즈)
        self.world_pos = [0.0, 0.0, 0.0]
        self.world_quat = [0.0, 0.0, 0.0, 1.0]
        # 직전값 (증분 계산용)
        self._prev_pos = None
        self._prev_rotvec = None
        self._prev_stamp = None
        self.rate_hz = 0.0   # 송신 수신율 (EMA)


class DeltaMonitor(Node):
    def __init__(self, arms):
        super().__init__('delta_monitor')
        self.arms = arms
        self.disp = {a: ArmDisp() for a in arms}
        # BEST_EFFORT 구독: RELIABLE/ BEST_EFFORT 송신 둘 다와 호환.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        for a in arms:
            self.create_subscription(String, f'/teleop/delta/{a}',
                                     lambda m, arm=a: self._cb(arm, m), qos)
        self.tty = sys.stdout.isatty()
        self.create_timer(1.0 / 15.0, self._render)  # 15 Hz 화면 갱신

    def _cb(self, arm, msg):
        try:
            d = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        st = self.disp[arm]
        st.have = True
        st.engaged = bool(d.get('engaged', False))
        st.valid = bool(d.get('valid', False))
        pos = [float(v) for v in d.get('pos', [0, 0, 0])]
        rotvec = [float(v) for v in d.get('rot', [0, 0, 0])]
        stamp = float(d.get('stamp', 0.0))
        st.abs_pos = pos
        st.abs_rotvec = rotvec
        st.world_pos = [float(v) for v in d.get('abs_pos', [0, 0, 0])]
        st.world_quat = [float(v) for v in d.get('abs_quat', [0, 0, 0, 1])]

        # 송신 수신율 EMA
        if st._prev_stamp is not None:
            dt = stamp - st._prev_stamp
            if dt > 1e-6:
                inst = 1.0 / dt
                st.rate_hz = inst if st.rate_hz == 0.0 else 0.9 * st.rate_hz + 0.1 * inst

        # 증분: 직전·현재 모두 engaged&valid 일 때만 (disengage 점프 배제)
        if (st.engaged and st.valid and st._prev_pos is not None
                and st._prev_rotvec is not None):
            st.inc_pos = [pos[i] - st._prev_pos[i] for i in range(3)]
            dq = quat_mul(quat_conj(rotvec_to_quat(st._prev_rotvec)),
                          rotvec_to_quat(rotvec))
            st.inc_ang_deg = quat_angle_deg(dq)
            dt = (stamp - st._prev_stamp) if st._prev_stamp is not None else 0.0
            if dt > 1e-6:
                ip = math.sqrt(sum(v * v for v in st.inc_pos))
                st.lin_vel = ip / dt
                st.ang_vel = st.inc_ang_deg / dt
        else:
            st.inc_pos = [0.0, 0.0, 0.0]
            st.inc_ang_deg = 0.0
            st.lin_vel = 0.0
            st.ang_vel = 0.0

        if st.engaged and st.valid:
            st._prev_pos = pos
            st._prev_rotvec = rotvec
            st._prev_stamp = stamp
        else:
            st._prev_pos = None       # disengage 후 재engage 시 깨끗하게 다시 시작
            st._prev_rotvec = None
            st._prev_stamp = stamp

    # --------------------------------- 렌더 ---------------------------------
    def _block(self, arm):
        st = self.disp[arm]
        head = f'== {arm.upper():5s} =='
        if not st.have:
            return f'{head}  (no data — /teleop/delta/{arm} 수신 없음)'
        eng = '✓' if st.engaged else '·'
        val = '✓' if st.valid else '·'
        flag = 'ENGAGED' if st.engaged else ('STANDBY' if st.have else '?')
        ap = st.abs_pos
        ar = st.abs_rotvec
        amag = math.sqrt(sum(v * v for v in ap))
        aang = math.sqrt(sum(v * v for v in ar)) * RAD2DEG
        # engage-프레임 회전을 축별 RPY로 분해 (rotvec -> quat -> ZYX Euler)
        roll, pitch, yaw = quat_to_euler_deg(rotvec_to_quat(ar))
        ip = st.inc_pos
        imag_mm = math.sqrt(sum(v * v for v in ip)) * 1000.0
        wp = st.world_pos
        wq = st.world_quat
        lines = [
            f'{head}  engaged {eng}  valid {val}  [{flag}]  in {st.rate_hz:5.1f}Hz',
            f'  ABS pos[m]   x={ap[0]:+.4f} y={ap[1]:+.4f} z={ap[2]:+.4f}   |p|={amag:.4f} m',
            f'  ABS rot[deg] roll(x)={roll:+7.2f} pitch(y)={pitch:+7.2f} yaw(z)={yaw:+7.2f}'
            f'   |θ|={aang:6.2f}°',
            f'  INC  Δpos[mm] x={ip[0]*1e3:+7.3f} y={ip[1]*1e3:+7.3f} z={ip[2]*1e3:+7.3f}'
            f'   |Δp|={imag_mm:6.3f}mm  |Δθ|={st.inc_ang_deg:5.3f}°'
            f'   ({st.lin_vel:5.3f}m/s {st.ang_vel:5.1f}°/s)',
            f'  WORLD pos[m] x={wp[0]:+.3f} y={wp[1]:+.3f} z={wp[2]:+.3f}'
            f'   quat {wq[0]:+.3f},{wq[1]:+.3f},{wq[2]:+.3f},{wq[3]:+.3f}',
        ]
        return '\n'.join(lines)

    def _render(self):
        header = (
            'Vive Teleop  Δ Monitor   (Ctrl-C 종료)\n'
            ' ABS=engage anchor 기준 누적(A, 송신값)   '
            'INC=직전 메시지 대비 증분(B)   WORLD=vive_world 절대\n'
            ' rot RPY=engage프레임 ZYX 축별 각도(roll=x,pitch=y,yaw=z)   '
            '|θ|=등가 단일축 총 회전각(성분 합 아님)\n'
            '-' * 92
        )
        body = '\n\n'.join(self._block(a) for a in self.arms)
        out = header + '\n' + body + '\n'
        if self.tty:
            sys.stdout.write('\033[H\033[J' + out)
            sys.stdout.flush()
        else:
            print(body, flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    arms = []
    for a in (args[0].split(',') if args else ['left', 'right']):
        a = a.strip()
        if a:
            arms.append(a)
    rclpy.init()
    node = DeltaMonitor(arms)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
