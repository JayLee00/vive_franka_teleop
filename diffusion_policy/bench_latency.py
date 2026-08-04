#!/usr/bin/env python3
"""추론 지연 측정 — 배포 설정(제어 Hz · 청킹 · temporal ensembling)이 가능한지 판단.

배치 1 (실기와 동일) 로 DDIM 추론 1회 시간을 재고, 예산과 비교한다.
    action chunking (exec_horizon=E) : 추론이 E 틱마다 1회 → 예산 = E / control_hz
    temporal ensembling             : 매 틱 1회       → 예산 = 1 / control_hz

  python3 bench_latency.py --ckpt runs/dp_lemon/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import dp_config as C
from dp_model import DDPMScheduler, policy_from_ckpt


def bench(policy, sched, obs_dim, obs_h, pred_h, steps, device, n=60):
    x = torch.randn(1, obs_h, obs_dim, device=device)
    for _ in range(8):                                   # warmup
        policy.sample(x, sched, pred_h, steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        policy.sample(x, sched, pred_h, steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--control_hz", type=float, default=None)
    ap.add_argument("--exec_horizon", type=int, default=None)
    ap.add_argument("--steps", nargs="*", type=int, default=[4, 6, 10, 20])
    args = ap.parse_args()

    rows = []
    for dev_name in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        device = torch.device(dev_name)
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ck["config"]
        policy = policy_from_ckpt(ck, device, use_ema=True)
        sched = DDPMScheduler(cfg["diff_steps"]).to(device)
        hz = args.control_hz or cfg["ctrl_hz"]
        E = args.exec_horizon or cfg["action_steps"]
        is_bc = cfg.get("algo") == "bc_mlp"
        for s in ([1] if is_bc else args.steps):
            r = bench(policy, sched, cfg["obs_dim"], cfg["obs_horizon"],
                      cfg["pred_horizon"], s, device)
            r.update(device=dev_name, ddim_steps=s,
                     budget_chunk_ms=1000.0 * E / hz, budget_te_ms=1000.0 / hz)
            r["ok_chunk"] = r["p95_ms"] < r["budget_chunk_ms"]
            r["ok_te"] = r["p95_ms"] < r["budget_te_ms"]
            rows.append(r)

    print("=" * 96)
    print(f"  추론 지연 (배치1)   ckpt={os.path.basename(args.ckpt)}")
    print("=" * 96)
    print(f"  {'dev':>5s} {'DDIM':>5s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'max':>8s}"
          f" | {'청킹예산':>9s} {'ok':>3s} | {'TE예산':>8s} {'ok':>3s}")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r['device']:>5s} {r['ddim_steps']:>5d} {r['mean_ms']:>8.2f} "
              f"{r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f} {r['max_ms']:>8.2f} | "
              f"{r['budget_chunk_ms']:>9.1f} {'O' if r['ok_chunk'] else 'X':>3s} | "
              f"{r['budget_te_ms']:>8.1f} {'O' if r['ok_te'] else 'X':>3s}")
    print("=" * 96)
    out = os.path.join(os.path.dirname(args.ckpt), "latency.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()
