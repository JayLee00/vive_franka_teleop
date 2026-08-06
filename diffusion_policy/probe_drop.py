#!/usr/bin/env python3
"""낙하 에피소드에서는 촉각·과일이 정보를 갖는가? — 분석 전용 (학습에 쓰지 않음).

정상 에피소드에서는 촉각·과일이 액션 잔차를 거의 설명하지 못했다(probe.py, 최대 +0.02).
가설: 그 센서가 '정답 액션을 바꾸는' 상황은 미끄러짐/회복인데, 그 구간이 학습셋에서
낙하로 분류돼 통째로 빠졌기 때문이다.

노트가 학습 제외로 지정한 낙하 데모(031811:Demo_5, 035756:Demo_0)에서 같은 프로브를
돌려 이 가설을 검증한다. **학습에는 절대 쓰지 않는다** — 이 스크립트는 읽고 재기만 한다.

  python3 probe_drop.py
"""
from __future__ import annotations

import json

import numpy as np

import dp_config as C
from dp_data import load_demos
from probe import ridge_r2

DROPPED = {
    "exp_20260807_031811_lemon_wo_object.h5": {5: None},
    "exp_20260807_035756_lemon_wo_object_2.h5": {0: None},
}
K = 8      # 0.4s 뒤


def build(ds, ha, hb, stride):
    X, R = [], []
    for d in ds:
        t = np.arange(0, d.n_used - K * stride)
        X.append(d.obs[t])
        R.append(d.act[t + K * stride] - d.obs[t, ha:hb])
    return np.concatenate(X).astype(np.float64), np.concatenate(R).astype(np.float64)


def main():
    stride = C.DS_STRIDE
    ha, hb = next((a, b) for n, a, b in C.OBS_GROUPS if n == "03_hand_j_pos")

    normal, _ = load_demos(verbose=False)
    dropped, _ = load_demos(clamp_spec=DROPPED, verbose=False)
    print("=" * 92)
    print(f"  낙하 에피소드 프로브 (분석 전용, 학습 미사용)")
    print(f"  정상 {len(normal)}개 / 낙하 {len(dropped)}개: "
          f"{[d.name.split(':')[1] for d in dropped]}")
    print("=" * 92)

    Xn, Rn = build(normal, ha, hb, stride)
    Xd, Rd = build(dropped, ha, hb, stride)
    print(f"  정상 {len(Xn):,} 프레임 / 낙하 {len(Xd):,} 프레임\n")

    groups = C.OBS_GROUPS
    hand = [(n, a, b) for n, a, b in groups if n == "03_hand_j_pos"]
    other = [(n, a, b) for n, a, b in groups if n != "03_hand_j_pos"]

    out = {}
    for tag, (Xtr, Rtr, Xte, Rte) in {
        "정상→정상 (5-fold 대신 절반분할)": (Xn[::2], Rn[::2], Xn[1::2], Rn[1::2]),
        "정상→낙하 (분포 이동)": (Xn, Rn, Xd, Rd),
        "낙하→낙하 (절반분할)": (Xd[::2], Rd[::2], Xd[1::2], Rd[1::2]),
    }.items():
        cols_h = np.concatenate([np.arange(a, b) for _, a, b in hand])
        r2h, _ = ridge_r2(Xtr[:, cols_h], Rtr, Xte[:, cols_h], Rte)
        row = {"hand_only": r2h}
        line = f"  {tag:<28s} 손만 R2={r2h:7.4f}"
        for n, a, b in other:
            cols = np.concatenate([cols_h, np.arange(a, b)])
            r2, _ = ridge_r2(Xtr[:, cols], Rtr, Xte[:, cols], Rte)
            row[n] = r2 - r2h
            line += f"   {n.split('_', 1)[1][:9]}:{r2 - r2h:+.4f}"
        out[tag] = row
        print(line)

    print("\n  해석: '낙하→낙하' 에서 촉각·과일의 증가분이 '정상→정상' 보다 크면,")
    print("        그 센서는 미끄러짐 구간에서만 정보를 갖는다는 뜻이다.")
    print("=" * 92)
    json.dump(out, open("runs/probe_drop.json", "w"), indent=2, ensure_ascii=False)
    print("  저장: runs/probe_drop.json")


if __name__ == "__main__":
    main()
