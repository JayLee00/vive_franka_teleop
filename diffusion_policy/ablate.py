#!/usr/bin/env python3
"""센서 기여도 측정 — "정책이 hand state 만 외운 건 아닌가?" 를 숫자로 답한다.

홀드아웃 open-loop rollout 을 돌리되, 추론 시 특정 센서군을 **정규화 공간의 0**
(= 학습 평균 = '정보 없음')으로 가려 MAE 가 얼마나 나빠지는지 잰다.

  가렸을 때 MAE 가 크게 오른다  → 정책이 그 센서를 실제로 쓴다
  거의 안 오른다               → 그 센서는 사실상 무시되고 있다

두 방향을 모두 본다:
  leave-one-out   그 군 하나만 가림 → 그 군의 고유 기여
  only-this       그 군만 남기고 나머지 전부 가림 → 그 군만으로 얼마나 되나

  python3 ablate.py --ckpt runs/v2_base/best.pt
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


def run(policy, sched, obs_norm, act_norm, held, cfg, device, steps, infer,
        zero_slices):
    P, G = [], []
    for d in held:
        p, g, _, _ = rollout_demo(policy, sched, obs_norm, act_norm, d, cfg,
                                  device, steps, infer, zero_slices=zero_slices)
        P.append(p); G.append(g)
    P = np.concatenate(P); G = np.concatenate(G)
    mse = float(((P - G) ** 2).mean())
    return {"mae": float(np.abs(P - G).mean()),
            "r2": 1.0 - mse / float(((G - G.mean(0)) ** 2).mean())}


def main():
    ap = argparse.ArgumentParser(description="센서군 기여도(ablation) 측정")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--action_steps", type=int, default=C.ACTION_STEPS)
    ap.add_argument("--infer_steps", type=int, default=C.INFER_STEPS)
    ap.add_argument("--weights", default=None, choices=[None, "raw", "ema"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, cfg, ck = load_ckpt(args.ckpt, device, use_ema=False)
    w = args.weights or ck.get("best_metric", {}).get("prefer_weights", "ema")
    policy, obs_norm, act_norm, cfg, _ = load_ckpt(args.ckpt, device,
                                                   use_ema=(w == "ema"))
    sched = DDPMScheduler(cfg["diff_steps"]).to(device)

    demos, _ = load_demos(verbose=False)
    name2d = {d.name: d for d in demos}
    held = [name2d[n] for n in ck["held_out_demos"] if n in name2d]
    if not held:
        _, held = split_demos(demos, cfg.get("n_held_out", C.N_HELD_OUT), cfg["seed"])

    groups = [(n, a, b) for n, a, b in
              cfg.get("obs_groups", None) or C.OBS_GROUPS]

    print("=" * 96)
    print(f"  센서군 기여도   ckpt={args.ckpt}  weights={w}  "
          f"E={args.action_steps} DDIM={args.infer_steps}")
    print(f"  홀드아웃 {len(held)}개 · obs {cfg['obs_dim']}차원")
    print("=" * 96)

    base = run(policy, sched, obs_norm, act_norm, held, cfg, device,
               args.action_steps, args.infer_steps, None)
    print(f"  {'전부 사용 (기준)':<28s} MAE={base['mae']:8.1f}  R2={base['r2']:7.4f}")
    print("  " + "-" * 92)

    out = {"ckpt": args.ckpt, "weights": w, "baseline": base,
           "leave_one_out": {}, "only_this": {}}

    print("  [leave-one-out] 이 군만 가림 → MAE 상승폭이 그 군의 기여")
    for n, a, b in groups:
        r = run(policy, sched, obs_norm, act_norm, held, cfg, device,
                args.action_steps, args.infer_steps, [(a, b)])
        d = r["mae"] - base["mae"]
        out["leave_one_out"][n] = {**r, "delta_mae": d,
                                   "delta_pct": 100 * d / base["mae"]}
        print(f"    {n:<24s} MAE={r['mae']:8.1f}  R2={r['r2']:7.4f}  "
              f"Δ={d:+8.1f} ({100*d/base['mae']:+6.1f}%)")

    print("  " + "-" * 92)
    print("  [only-this] 이 군만 남기고 전부 가림 → 그 군만으로 낼 수 있는 성능")
    for n, a, b in groups:
        zs = [(x, y) for m, x, y in groups if m != n]
        r = run(policy, sched, obs_norm, act_norm, held, cfg, device,
                args.action_steps, args.infer_steps, zs)
        out["only_this"][n] = r
        print(f"    {n:<24s} MAE={r['mae']:8.1f}  R2={r['r2']:7.4f}")

    allz = run(policy, sched, obs_norm, act_norm, held, cfg, device,
               args.action_steps, args.infer_steps,
               [(a, b) for _, a, b in groups])
    out["all_masked"] = allz
    print("  " + "-" * 92)
    print(f"  {'전부 가림 (하한)':<28s} MAE={allz['mae']:8.1f}  R2={allz['r2']:7.4f}")
    print("=" * 96)

    p = os.path.join(os.path.dirname(args.ckpt), "ablation.json")
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"  저장: {p}")


if __name__ == "__main__":
    main()
