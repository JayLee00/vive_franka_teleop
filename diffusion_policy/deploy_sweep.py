#!/usr/bin/env python3
"""배포 설정 비교 — exec_horizon · temporal ensembling · DDIM 스텝을 홀드아웃에서 재본다.

`error_vs_horizon` 플롯에서 보이듯 실행 지평 내 뒤쪽 스텝의 오차가 앞쪽의 3배 이상이다.
따라서 exec_horizon 을 줄이면 정확도가 크게 오르는데 추론 부하가 그만큼 늘어난다.
어디까지 줄일 수 있는지는 지연(bench_latency.py)과 함께 봐야 결정된다.

  python3 deploy_sweep.py --ckpt runs/dp_lemon_final/best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import dp_config as C
from dp_data import load_demos, split_demos
from dp_model import DDPMScheduler
from eval_rollout import load_ckpt, rollout_demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default=None, choices=[None, "raw", "ema"])
    ap.add_argument("--exec_horizons", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--ddim_steps", nargs="*", type=int, default=[10])
    ap.add_argument("--te_k", type=float, default=0.6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, cfg, ck = load_ckpt(args.ckpt, device, use_ema=False)
    w = args.weights or ck.get("best_metric", {}).get("prefer_weights", "ema")
    policy, obs_norm, act_norm, cfg, _ = load_ckpt(args.ckpt, device, use_ema=(w == "ema"))
    sched = DDPMScheduler(cfg["diff_steps"]).to(device)

    demos, _ = load_demos(verbose=False)
    name2d = {d.name: d for d in demos}
    held = [name2d[n] for n in ck["held_out_demos"] if n in name2d]
    if not held:
        _, held = split_demos(demos, cfg.get("n_held_out", C.N_HELD_OUT), cfg["seed"])

    is_bc = cfg.get("algo") == "bc_mlp"
    steps_list = [1] if is_bc else args.ddim_steps
    rows = []

    def run(label, E, steps, te):
        P, G = [], []
        for d in held:
            p, g, _, _ = rollout_demo(policy, sched, obs_norm, act_norm, d, cfg,
                                      device, E, steps, temporal_ensemble=te,
                                      te_k=args.te_k)
            P.append(p); G.append(g)
        P = np.concatenate(P); G = np.concatenate(G)
        mae = float(np.abs(P - G).mean())
        # 매끄러움: 연속 액션 차분의 평균 크기 (GT 대비 얼마나 지터가 큰가)
        jit_p = float(np.abs(np.diff(P, axis=0)).mean())
        jit_g = float(np.abs(np.diff(G, axis=0)).mean())
        r = {"label": label, "exec_horizon": E, "ddim_steps": steps,
             "temporal_ensemble": te, "mae_count": mae, "mae_deg": mae * 90 / 4096,
             "rmse_count": float(np.sqrt(((P - G) ** 2).mean())),
             "r2": 1.0 - float(((P - G) ** 2).mean()) / float(((G - G.mean(0)) ** 2).mean()),
             "jitter_pred": jit_p, "jitter_gt": jit_g,
             "jitter_ratio": jit_p / max(1e-9, jit_g),
             "infer_per_sec": (cfg["ctrl_hz"] if te else cfg["ctrl_hz"] / E)}
        rows.append(r)
        print(f"  {label:<26s} MAE={mae:7.1f}c ({r['mae_deg']:5.2f}deg) "
              f"R2={r['r2']:6.3f}  지터 x{r['jitter_ratio']:4.2f}  "
              f"추론 {r['infer_per_sec']:5.1f}/s")

    print("=" * 100)
    print(f"  배포 설정 비교   ckpt={args.ckpt}  weights={w}  algo={cfg.get('algo')}")
    print(f"  홀드아웃 {len(held)}개 · 제어 {cfg['ctrl_hz']:.0f}Hz · T_pred={cfg['pred_horizon']}")
    print("=" * 100)
    for steps in steps_list:
        for E in args.exec_horizons:
            if E > cfg["pred_horizon"]:
                continue
            run(f"chunk E={E:<2d} DDIM={steps}", E, steps, False)
        run(f"temporal-ens DDIM={steps}", 1, steps, True)

    best = min(rows, key=lambda r: r["mae_count"])
    print("-" * 100)
    print(f"  최적: {best['label']}  MAE={best['mae_count']:.1f}c "
          f"({best['mae_deg']:.2f}deg), 초당 추론 {best['infer_per_sec']:.1f}회")
    print("=" * 100)
    out = os.path.join(os.path.dirname(args.ckpt), "deploy_sweep.json")
    json.dump({"weights": w, "rows": rows, "best": best}, open(out, "w"), indent=2)
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
