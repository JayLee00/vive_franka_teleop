#!/usr/bin/env python3
"""선형 프로브 — 촉각·과일이 액션에 대한 정보를 **원리적으로** 갖고 있는가?

ablation 은 "학습된 정책이 그 센서를 쓰는가"를 재고, 이건 "쓸 정보가 데이터에
있기는 한가"를 잰다. 둘은 다르다:
    정보 있음 + 정책이 안 씀  → 모델·목적함수 문제 (파라미터화·증강으로 고칠 수 있다)
    정보 자체가 없음          → 데이터 문제 (그 센서가 액션을 바꾸는 상황을 안 모았다)

방법: ridge 회귀로 obs 부분집합 → 액션(그리고 액션 잔차)을 예측하고 홀드아웃 R2 를 본다.
잔차 = action - 현재 손 자세. '자세 베끼기'로 설명되는 성분을 뺀 나머지라서,
여기서 촉각·과일이 R2 를 못 올리면 그 센서엔 (선형적으로는) 정보가 없다는 뜻이다.

  python3 probe.py --ckpt runs/v2_base/best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import dp_config as C
from dp_data import load_demos, split_demos


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    P = Xte @ W
    ss_res = ((P - Yte) ** 2).sum()
    ss_tot = ((Yte - Ytr.mean(0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot), float(np.abs(P - Yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--horizon_step", type=int, default=8,
                    help="몇 스텝 뒤 액션을 예측할지 (20Hz 격자, 8 = 0.4s 뒤)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    stride = cfg["ds_stride"]
    k = args.horizon_step

    demos, _ = load_demos(verbose=False)
    name2d = {d.name: d for d in demos}
    held = [name2d[n] for n in ck["held_out_demos"] if n in name2d]
    if not held:
        _, held = split_demos(demos, cfg.get("n_held_out", 3), cfg["seed"])
    train = [d for d in demos if d.name not in {h.name for h in held}]

    ha, hb = next((a, b) for n, a, b in C.OBS_GROUPS if n == "03_hand_j_pos")

    def build(ds):
        X, Y, R = [], [], []
        for d in ds:
            n = d.n_used
            t = np.arange(0, n - k * stride)
            X.append(d.obs[t])
            Y.append(d.act[t + k * stride])
            R.append(d.act[t + k * stride] - d.obs[t, ha:hb])
        return (np.concatenate(X).astype(np.float64),
                np.concatenate(Y).astype(np.float64),
                np.concatenate(R).astype(np.float64))

    Xtr, Ytr, Rtr = build(train)
    Xte, Yte, Rte = build(held)
    print("=" * 92)
    print(f"  선형 프로브 — obs(t) → action(t+{k}스텝={k*stride/100:.2f}s)")
    print(f"  train {len(Xtr):,} / holdout {len(Xte):,} 프레임")
    print("=" * 92)

    groups = C.OBS_GROUPS
    hand = [(n, a, b) for n, a, b in groups if n == "03_hand_j_pos"]
    other = [(n, a, b) for n, a, b in groups if n != "03_hand_j_pos"]

    sets = {"전체 obs": groups, "손 관절만": hand, "손 제외 전부": other}
    for n, a, b in other:
        sets[f"손 + {n}"] = hand + [(n, a, b)]

    out = {}
    print(f"  {'입력':<26s} {'절대 액션 R2':>13s} {'잔차 R2':>10s} {'잔차 MAE':>10s}")
    print("  " + "-" * 88)
    for name, gs in sets.items():
        cols = np.concatenate([np.arange(a, b) for _, a, b in gs])
        r2a, _ = ridge_r2(Xtr[:, cols], Ytr, Xte[:, cols], Yte)
        r2r, maer = ridge_r2(Xtr[:, cols], Rtr, Xte[:, cols], Rte)
        out[name] = {"r2_abs": r2a, "r2_resid": r2r, "mae_resid": maer}
        print(f"  {name:<26s} {r2a:13.4f} {r2r:10.4f} {maer:10.1f}")

    base = out["손 관절만"]["r2_resid"]
    print("  " + "-" * 88)
    print("  손 관절만 대비 잔차 R2 증가분 (= 그 센서가 추가로 설명하는 몫):")
    for n, _a, _b in other:
        d = out[f"손 + {n}"]["r2_resid"] - base
        print(f"    {n:<24s} {d:+.4f}")
    print("=" * 92)
    p = os.path.join(os.path.dirname(args.ckpt), "probe.json")
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"  저장: {p}")


if __name__ == "__main__":
    main()
