#!/usr/bin/env python3
"""대칭 스냅 + 쿼터니언 부호 연속성 검증 (카메라·서버 불필요)."""
import sys
sys.path.insert(0, "/home/js/Desktop/vive_franka_teleop/foundation_pose")
import numpy as np
import rclpy
import fp_ros_node as F

FAIL = []
def chk(name, ok, extra=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not ok: FAIL.append(name)

def rot(axis, th):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def ang(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2, -1, 1)))

class A:  # argparse 대용
    pass

rclpy.init()
a = A()
for k, v in dict(server="127.0.0.1:1", ns="/t", color_topic="/c", depth_topic="/d",
                 info_topic="/i", frame_id="f", roi_frac=(.5,.5,.9,.9),
                 depth_band=(0.15,1.2), near_slab=.05, min_mask_px=400,
                 diameter=.07, abc=[.07,.055,.055], click=False, auto_reset=True,
                 check_tol=.15, check_win=6, lost_patience=30, seg_hz=0,
                 reseg_every=30, size_source="cad", sym_steps=36).items():
    setattr(a, k, v)
n = F.FoundationPoseNode(a)
n.mesh_extents = np.array([0.070, 0.055, 0.055])      # 장축 = 축0

sym = n._sym_rots()
chk("대칭 집합 생성", sym is not None and len(sym) == 36, f"{len(sym) if sym else 0}개")

# 장축(축0) 둘레 회전은 대칭이어야 한다 → 스냅이 원래대로 되돌려야
Rp = rot([0.3, 1.0, 0.2], 0.7)                        # 임의의 직전 자세
n.last_pose = np.eye(4); n.last_pose[:3,:3] = Rp

for deg in (40, 120, 200, 350):
    T = np.eye(4); T[:3,:3] = Rp @ rot([1,0,0], np.deg2rad(deg))   # 대칭 동등물
    out = n._snap_to_prev(T)
    before, after = ang(Rp, T[:3,:3]), ang(Rp, out[:3,:3])
    chk(f"장축 {deg}° 회전 → 직전 자세로 스냅", after < 6.0,
        f"{before:.0f}° → {after:.1f}°")

# 대칭이 아닌 회전(장축에 수직)은 보존되어야 한다 — 진짜 자세 변화를 지우면 안 됨
T = np.eye(4); T[:3,:3] = Rp @ rot([0,1,0], np.deg2rad(35))
out = n._snap_to_prev(T)
chk("수직축 35° 는 지우지 않음", ang(T[:3,:3], out[:3,:3]) < 1.0,
    f"변화 {ang(T[:3,:3], out[:3,:3]):.1f}°")

# 쿼터니언 부호 연속성
from std_msgs.msg import Header
def H():                      # 진짜 ROS Header 를 쓴다
    return Header(frame_id="f")
n.size = [.07,.055,.055]
T1 = np.eye(4); T1[:3,:3] = Rp
n._publish(T1, H())
q1 = n._q_prev.copy()
n._publish(T1, H())                                    # 같은 자세 재발행
q2 = n._q_prev.copy()
chk("같은 자세 → 부호 동일", float(q1 @ q2) > 0.99, f"dot={float(q1@q2):.3f}")
n._q_prev = -q1                                        # 부호가 뒤집힌 상태로 강제
n._publish(T1, H())
chk("부호 뒤집힘 자동 교정", float(n._q_prev @ (-q1)) > 0.99)

n._stop.set(); n.destroy_node(); rclpy.shutdown()
print("\n" + (f"실패 {len(FAIL)}: {FAIL}" if FAIL else "전부 PASS"))
sys.exit(1 if FAIL else 0)
