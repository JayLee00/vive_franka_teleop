#!/usr/bin/env python3
"""수집한 HDF5 에피소드 품질 점검 (수집 중간중간 돌려서 오염 데이터 조기 발견).

ros2_hdf5_recorder.py 는 토픽이 죽어도 마지막 값을 계속 홀드하므로,
인식 노드가 중간에 죽으면 fruit 채널이 '조용히 상수로 고정'된 채 저장된다.
그 상태로 20 에피소드를 모으면 통째로 버려야 하므로 매 에피소드 후 확인한다.

점검 항목 (Demo 별):
  길이/주기        n_samples, 지속시간, 실효 Hz
  얼어붙은 채널     에피소드 내내 std==0  → 토픽 끊김 의심
  전부 0 채널       한 번도 수신 안 됨
  action           04_hand_j_tar: 움직인 관절 수, 정지(무동작) 구간 비율
  촉각             06_hand_j_kin(paxini ft): 접촉에 반응했는지
  과일             30_fruit_pos 이동량, 32_fruit_size 평균(레몬 ~0.04~0.09m 기대)

실행:
  python3 record/verify_demo.py record/logs/exp_20260805_143000.h5
  python3 record/verify_demo.py record/logs/exp_*.h5 --last   # 마지막 데모만
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

FRUIT_SIZE_RANGE = (0.03, 0.12)   # 레몬 축 길이 상식 범위 [m]
IDLE_EPS = 2.0                    # action 변화 이하면 '정지'로 간주 [count/step]
FT_KEY = "20_paxini_ft"           # 촉각(접촉 반응) — 미끄러짐 감지의 근거
# 상수/미수신이 정상인 채널: 팔 고정 데모라 franka 는 상수, mode/servo_on 도 상수
CONST_OK = ("01_hand_mode", "02_hand_servo_on")


def check_demo(name: str, g: h5py.Group, rate_hz: float) -> list[str]:
    """Demo 하나 점검. 반환: 경고 문자열 목록(빈 리스트면 정상)."""
    warn: list[str] = []
    n = g["04_hand_j_tar"].shape[0]
    dur = float(g["18_real_time_demo"][-1]) if "18_real_time_demo" in g else n / rate_hz
    eff_hz = n / dur if dur > 0 else 0.0
    print(f"\n── {name}:  {n} step, {dur:.1f}s, 실효 {eff_hz:.1f}Hz")
    if dur < 10.0:
        warn.append(f"에피소드가 짧음({dur:.1f}s)")
    if eff_hz < rate_hz * 0.9:
        warn.append(f"샘플링 저하({eff_hz:.0f}Hz < {rate_hz:.0f}Hz) — 레코더 부하/드롭 의심")

    # 채널별 상태
    frozen, allzero = [], []
    for k in sorted(g.keys()):
        if "franka" in k or k.startswith(CONST_OK):   # 팔 고정 데모 → 상수가 정상
            continue
        a = np.asarray(g[k][()], dtype=np.float64).reshape(n, -1)
        if not np.any(a):
            allzero.append(k)
        elif float(a.std(axis=0).max()) == 0.0:
            frozen.append(k)
    if allzero:
        warn.append(f"전부 0(미수신): {allzero}")
    if frozen:
        warn.append(f"얼어붙음(값 고정): {frozen}")

    # action 품질
    tar = np.asarray(g["04_hand_j_tar"][()], dtype=np.float64).reshape(n, -1)
    d = np.abs(np.diff(tar, axis=0)).max(axis=1)
    moved = int((tar.std(axis=0) > 5.0).sum())
    idle = float((d < IDLE_EPS).mean())
    print(f"   action  움직인 관절 {moved}/16,  정지 구간 {idle*100:.0f}%,  "
          f"최대 슬루 {d.max():.0f} count/step")
    if moved < 3:
        warn.append(f"action 이 거의 안 움직임(관절 {moved}개) — engage 안 된 상태로 로깅?")
    if idle > 0.5:
        warn.append(f"정지 구간 {idle*100:.0f}% — 시연 밀도 낮음")

    # 촉각 (접촉 반응)
    if FT_KEY in g:
        ft = np.asarray(g[FT_KEY][()], dtype=np.float64).reshape(n, -1)
        print(f"   촉각 ft  |max| {np.abs(ft).max():.3f},  채널 std 최대 {ft.std(axis=0).max():.4f}")
        if np.abs(ft).max() < 1e-6:
            warn.append(f"{FT_KEY} 가 항상 0 — 촉각 없이 학습하는 셈")

    # 과일
    if "30_fruit_pos" in g:
        p = np.asarray(g["30_fruit_pos"][()], dtype=np.float64).reshape(n, 3)
        travel = float(np.linalg.norm(p - p[0], axis=1).max())
        print(f"   fruit_pos  평균 {np.round(p.mean(0), 3)} m,  최대 이동 {travel*1000:.0f} mm")
        if travel < 0.002:
            warn.append(f"fruit_pos 거의 안 변함({travel*1000:.1f}mm) — 인식 끊김/오검출 의심")
    if "32_fruit_size" in g:
        s = np.asarray(g["32_fruit_size"][()], dtype=np.float64).reshape(n, 3)
        a_b = s[:, :2]
        print(f"   fruit_size a/b/c 평균 {np.round(s.mean(0), 4)} m,  "
              f"a 변동 {a_b[:, 0].std()*1000:.1f}mm  b 변동 {a_b[:, 1].std()*1000:.1f}mm")
        lo, hi = FRUIT_SIZE_RANGE
        if not (lo <= s[:, 0].mean() <= hi):
            warn.append(f"fruit_size 장축 평균 {s[:,0].mean():.3f}m — 상식 범위({lo}~{hi}m) 밖")

    return warn


def main() -> int:
    ap = argparse.ArgumentParser(description="수집 HDF5 에피소드 품질 점검")
    ap.add_argument("path", help="exp_*.h5")
    ap.add_argument("--last", action="store_true", help="마지막 Demo 만 점검")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        rate = float(f.attrs.get("rate_hz", 100.0))
        demos = sorted((k for k in f.keys() if k.startswith("Demo_")),
                       key=lambda s: int(s.split("_")[1]))
        if args.last:
            demos = demos[-1:]
        print(f"{args.path}  |  rate {rate:.0f}Hz  |  Demo {len(f.keys())}개")
        bad = {}
        for name in demos:
            w = check_demo(name, f[name], rate)
            if w:
                bad[name] = w
                for m in w:
                    print(f"   ⚠ {m}")
            else:
                print("   ✓ 이상 없음")

    print(f"\n{'='*60}")
    if bad:
        print(f"  경고 있는 Demo {len(bad)}/{len(demos)}: {list(bad)}")
        print("  ▶ 원인 해결 후 해당 에피소드는 재수집 권장.")
    else:
        print(f"  Demo {len(demos)}개 전부 정상. ▶ 계속 수집.")
    print(f"{'='*60}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
