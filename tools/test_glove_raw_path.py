#!/usr/bin/env python3
"""고친 glove_teleop 의 raw 통과 / 램프-1회 로직 검증 (시리얼 포트 불필요)."""
import sys, time
sys.path.insert(0, "tools")

import rclpy
from sensor_msgs.msg import JointState
import glove_teleop as G

FAIL = []

def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        FAIL.append(name)

rclpy.init()
node = G.GloveTeleop("/dev/definitely_not_a_port", "right", dry_run=True)

# 현재 손 자세 = 전부 0 (램프 시작점)
js = JointState()
js.position = [0.0] * 16
node._on_hand_state(js)

RAW = [1062, -2254, 1731, 511, -462, 3036, 1282, 84,
       1000, 0, 0, -95, 1000, 483, 0, -370]
mapped = node._map_to_hand([float(v) for v in RAW])

t = 1000.0
node._process(RAW, t)                      # 1회차: start_pose 잡고 return
check("첫 프레임에 start_pose = 현재 손자세", node.start_pose == [0.0] * 16)
check("램프 아직 안 끝남", node.ramp_done is False)

node._process(RAW, t + 0.5)                # 램프 25% 지점
q = node.last_target
check("램프 중에는 mapped 에 아직 못 도달", q != mapped, f"ch0={q[0]:.0f} vs {mapped[0]:.0f}")
check("램프 중 값이 0과 mapped 사이", 0 < q[0] < mapped[0], f"ch0={q[0]:.1f}")

node._process(RAW, t + G.RAMP_SEC + 0.01)  # 램프 종료
check("램프 완료 플래그", node.ramp_done is True)
check("램프 끝나면 target == mapped (raw 그대로)", node.last_target == mapped,
      f"ch0={node.last_target[0]:.0f}")

# ── 핵심: MAX_STEP 이 사라졌는지 = 큰 점프가 한 프레임에 그대로 반영되는지
BIG = [4000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
big_mapped = node._map_to_hand([float(v) for v in BIG])
node._process(BIG, t + G.RAMP_SEC + 0.02)
check("한 프레임에 큰 점프 그대로 통과 (rate limit 없음)",
      node.last_target == big_mapped,
      f"ch0 {node.last_target[0]:.0f} == {big_mapped[0]:.0f}")

# ── EMA 가 사라졌는지: 같은 값 1프레임만 넣어도 즉시 그 값
node._process(RAW, t + G.RAMP_SEC + 0.03)
check("1프레임만에 즉시 해당 값 (EMA 없음)", node.last_target == mapped,
      f"ch0={node.last_target[0]:.0f}")
check("q_raw 는 받은 정수 그대로", node.g_last == [float(v) for v in RAW])

# ── 재연결이 램프를 다시 태우지 않는지
node._reconnect(t + 100.0)                 # 포트 없어서 실패하지만 상태를 건드리면 안 됨
check("재연결 시도가 ramp_done 을 되돌리지 않음", node.ramp_done is True)
check("재연결 시도가 start_pose 를 지우지 않음", node.start_pose is not None)

# ── 발판 disengage → 재engage 가 램프를 다시 태우지 않는지
node.engaged = False
node._process(RAW, t + 101.0)
check("disengage 중 start_pose 유지", node.start_pose is not None)
node.engaged = True
before = list(node.last_target)
node._process(BIG, t + 102.0)
check("재engage 후 램프 재시작 없이 즉시 추종",
      node.ramp_done is True and node.last_target == big_mapped,
      f"ch0 {before[0]:.0f} -> {node.last_target[0]:.0f}")

node.shutdown(); node.destroy_node(); rclpy.shutdown()
print("\n" + (f"실패 {len(FAIL)}건: {FAIL}" if FAIL else "전부 PASS"))
sys.exit(1 if FAIL else 0)
