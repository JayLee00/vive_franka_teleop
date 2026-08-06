#!/usr/bin/env python3
"""'현재 손 자세를 그대로 베끼기' 기준선 — 정책이 이걸 이기는지 본다.

action 은 q_target 이고 obs 에 q_pos 가 들어 있다. 손이 타겟을 추종하므로
q_target[t+k] ≈ q_pos[t] 가 이미 매우 강한 해다. 정책이 이 기준선을 못 이기면
"태스크를 배웠다"고 말할 수 없다 — 관측을 그대로 복사한 것뿐이다.

기준선 3개를 홀드아웃에서 같은 격자로 잰다:
  copy_pose   a[t+k] = q_pos[t]                (현재 자세 유지)
  copy_last   a[t+k] = a[t-1] (=직전 타겟)      실기에선 못 쓰지만 상한 참고용
  mean_action a[t+k] = 학습셋 평균              (정보 0)

  python3 copy_baseline.py --ckpt runs/v2_base/best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import dp_config as C
from dp_data import load_demos, split_demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--action_steps", type=int, default=C.ACTION_STEPS)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    T_obs, T_pred = cfg["obs_horizon"], cfg["pred_horizon"]
    stride = cfg["ds_stride"]
    back = (T_obs - 1) * stride
    E = args.action_steps

    demos, _ = load_demos(verbose=False)
    name2d = {d.name: d for d in demos}
    held = [name2d[n] for n in ck["held_out_demos"] if n in name2d]
    if not held:
        _, held = split_demos(demos, cfg.get("n_held_out", 3), cfg["seed"])

    # obs 안에서 손 관절각 구간
    a0, b0 = next((a, b) for n, a, b in C.OBS_GROUPS if n == "03_hand_j_pos")
    act_off = np.arange(T_pred) * stride

    P = {"copy_pose": [], "copy_last": [], "mean_action": []}
    G = []
    train_demos = [d for d in demos if d not in held]
    amean = np.concatenate([d.act for d in train_demos]).mean(0)

    for d in held:
        t = back
        while t + (T_pred - 1) * stride <= d.n_used - 1:
            idxs = (t + act_off)[:E]
            G.append(d.act[idxs])
            P["copy_pose"].append(np.repeat(d.obs[t, a0:b0][None], len(idxs), 0))
            prev = d.act[max(0, t - 1)]
            P["copy_last"].append(np.repeat(prev[None], len(idxs), 0))
            P["mean_action"].append(np.repeat(amean[None], len(idxs), 0))
            t += E * stride
    G = np.concatenate(G)

    print("=" * 84)
    print(f"  기준선 (홀드아웃 {len(held)}개, E={E}, 같은 격자)")
    print("=" * 84)
    out = {}
    for k, v in P.items():
        Pk = np.concatenate(v)
        mse = float(((Pk - G) ** 2).mean())
        mae = float(np.abs(Pk - G).mean())
        r2 = 1.0 - mse / float(((G - G.mean(0)) ** 2).mean())
        out[k] = {"mae": mae, "r2": r2}
        print(f"  {k:<14s} MAE={mae:8.1f} count ({mae*90/4096:5.2f}deg)  R2={r2:7.4f}")

    ev = os.path.join(os.path.dirname(args.ckpt), "eval", "eval_summary.json")
    if os.path.exists(ev):
        s = json.load(open(ev))
        best = min(s["variants"].items(), key=lambda kv: kv[1]["mae_count"])
        pm = best[1]["mae_count"]
        print("  " + "-" * 80)
        print(f"  {'정책(' + best[0] + ')':<14s} MAE={pm:8.1f} count "
              f"({pm*90/4096:5.2f}deg)  R2={best[1]['r2']:7.4f}")
        cp = out["copy_pose"]["mae"]
        print(f"\n  → 정책이 'copy_pose' 대비 {100*(1-pm/cp):+.1f}% "
              f"({'이김' if pm < cp else '못 이김'})")
        out["policy"] = {"mae": pm, "r2": best[1]["r2"], "weights": best[0],
                         "vs_copy_pose_pct": 100 * (1 - pm / cp)}
    print("=" * 84)
    p = os.path.join(os.path.dirname(args.ckpt), "baselines.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"  저장: {p}")


if __name__ == "__main__":
    main()
