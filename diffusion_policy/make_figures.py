#!/usr/bin/env python3
"""결과 요약 그림 — ablation / 기준선 / 프로브 / 낙하구간을 한 장으로.

  python3 make_figures.py --run runs/v2_base
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHORT = {"03_hand_j_pos": "hand joints\n(16)", "06_hand_j_kin": "hand F/T\n(12)",
         "10_hand_paxini_ft": "paxini\n(12)", "30_fruit_pos": "fruit pos\n(3)",
         "31_fruit_quat": "fruit quat\n(4)", "32_fruit_size": "fruit size\n(3)"}
C_HAND, C_OTHER, C_BASE = "#c0392b", "#7f8c8d", "#2980b9"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v2_base")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = args.run
    abl = json.load(open(os.path.join(R, "ablation.json")))
    base = json.load(open(os.path.join(R, "baselines.json")))
    prb = json.load(open(os.path.join(R, "probe.json")))
    drop = json.load(open("runs/probe_drop.json"))

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    b = abl["baseline"]["mae"]

    # ── A: leave-one-out ────────────────────────────────────────────────
    a = ax[0, 0]
    ks = list(abl["leave_one_out"].keys())
    v = [abl["leave_one_out"][k]["mae"] for k in ks]
    cols = [C_HAND if k == "03_hand_j_pos" else C_OTHER for k in ks]
    bars = a.bar(range(len(ks)), v, color=cols)
    a.axhline(b, color="k", ls="--", lw=1.5, label=f"all sensors = {b:.0f}")
    for i, (bar, k) in enumerate(zip(bars, ks)):
        d = abl["leave_one_out"][k]["delta_pct"]
        a.text(i, bar.get_height() + 8, f"{d:+.1f}%", ha="center",
               fontsize=10, fontweight="bold" if abs(d) > 10 else "normal")
    a.set_xticks(range(len(ks)))
    a.set_xticklabels([SHORT[k] for k in ks], fontsize=9)
    a.set_ylabel("held-out MAE [count]")
    a.set_title("A. Mask ONE sensor group\n"
                "(higher = policy relies on it)", fontweight="bold")
    a.legend(); a.grid(alpha=0.3, axis="y")

    # ── B: only-this ────────────────────────────────────────────────────
    a = ax[0, 1]
    v = [abl["only_this"][k]["mae"] for k in ks]
    cols = [C_HAND if k == "03_hand_j_pos" else C_OTHER for k in ks]
    a.bar(range(len(ks)), v, color=cols)
    a.axhline(b, color="k", ls="--", lw=1.5, label=f"all sensors = {b:.0f}")
    a.axhline(abl["all_masked"]["mae"], color="r", ls=":", lw=1.5,
              label=f"nothing = {abl['all_masked']['mae']:.0f}")
    a.set_xticks(range(len(ks)))
    a.set_xticklabels([SHORT[k] for k in ks], fontsize=9)
    a.set_ylabel("held-out MAE [count]")
    a.set_title("B. Keep ONLY this group\n"
                "(hand joints alone ≈ everything)", fontweight="bold")
    a.legend(); a.grid(alpha=0.3, axis="y")

    # ── C: 기준선 + 변형 ────────────────────────────────────────────────
    a = ax[1, 0]
    names, vals, cs = [], [], []
    for k, lab in [("mean_action", "mean action"), ("copy_pose", "copy pose"),
                   ("copy_last", "copy last target")]:
        names.append(lab); vals.append(base[k]["mae"]); cs.append("#bdc3c7")
    for lab, p in [("policy\n(all sensors)", "v2_base"),
                   ("mod-dropout 0.2", "v2_moddrop02"),
                   ("hand masked 50%", "v2_handdrop05"),
                   ("no hand state", "v2_nohand")]:
        f = os.path.join("runs", p, "eval", "eval_summary.json")
        if os.path.exists(f):
            s = json.load(open(f))
            m = min(x["mae_count"] for x in s["variants"].values())
            names.append(lab); vals.append(m)
            cs.append(C_BASE if p == "v2_base" else "#95a5a6")
    bars = a.barh(range(len(names)), vals, color=cs)
    for i, bar in enumerate(bars):
        a.text(bar.get_width() + 6, i, f"{vals[i]:.0f}", va="center", fontsize=9)
    a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=9)
    a.invert_yaxis()
    a.set_xlabel("held-out MAE [count]  (lower = better)")
    a.set_title("C. Policy vs trivial baselines vs variants",
                fontweight="bold")
    a.grid(alpha=0.3, axis="x")

    # ── D: 프로브 — 정상 vs 낙하 ────────────────────────────────────────
    a = ax[1, 1]
    other = [k for k in ks if k != "03_hand_j_pos"]
    kn = "정상→정상 (5-fold 대신 절반분할)"
    kd = "낙하→낙하 (절반분할)"
    nv = [drop[kn][k] for k in other]
    dv = [drop[kd][k] for k in other]
    x = np.arange(len(other))
    a.bar(x - 0.2, nv, 0.4, label="normal episodes", color="#95a5a6")
    a.bar(x + 0.2, dv, 0.4, label="DROPPED episodes", color="#e67e22")
    for i, (n_, d_) in enumerate(zip(nv, dv)):
        if n_ > 1e-4:
            a.text(i + 0.2, d_ + 0.002, f"x{d_/n_:.0f}", ha="center",
                   fontsize=10, fontweight="bold", color="#d35400")
    a.set_xticks(x); a.set_xticklabels([SHORT[k] for k in other], fontsize=9)
    a.set_ylabel("residual R² gained over hand-joints-only")
    a.set_title("D. Do the sensors carry information?\n"
                "only in the episodes that were excluded", fontweight="bold")
    a.axhline(0, color="k", lw=0.8)
    a.set_ylim(0, max(dv) * 1.28)
    a.legend(loc="upper left", framealpha=0.9)
    a.grid(alpha=0.3, axis="y")

    fig.suptitle("Lemon in-hand rotation — what the policy actually uses "
                 f"(24 demos, 6.6 min, held-out 3)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = args.out or os.path.join(R, "eval", "summary_analysis.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
