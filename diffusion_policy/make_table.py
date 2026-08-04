#!/usr/bin/env python3
"""runs/sweep/*/ 결과를 모아 마크다운 비교표를 만든다 (BASELINES.md 에 붙이는 용).

  python3 make_table.py                       # 콘솔 출력
  python3 make_table.py --write BASELINES.md   # <!-- SWEEP_TABLE --> 자리에 삽입
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = {
    "base": "DP U-Net, 참조 레시피 (정규화 없음)",
    "noise10": "+ obs 노이즈 σ=0.1",
    "noise30": "+ obs 노이즈 σ=0.3",
    "drop_wd": "+ dropout 0.2, wd 1e-2",
    "noise30_drop_wd": "+ 노이즈 0.3 + dropout 0.2 + wd 1e-2",
    "small_noise30": "+ 노이즈 0.3, 채널 32/64/128, cond 128",
    "dit_noise30": "DiT 디노이저 + 노이즈 0.3",
    "bc_noise30": "BC MLP (확산 없음) + 노이즈 0.3",
    "dit": "DiT 디노이저",
    "bc": "BC MLP (확산 없음)",
    "lr3e4": "lr 3e-4",
    "oh4": "T_obs=4 (200ms 히스토리)",
    "oh8": "T_obs=8 (400ms)",
    "oh16": "T_obs=16 (800ms)",
    "oh8_dit": "T_obs=8 + DiT",
    "oh8_bc": "T_obs=8 + BC MLP",
}


def collect(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, root, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        ep = os.path.join(d, "eval", "eval_summary.json")
        tr = os.path.join(d, "train_result.json")
        if not (os.path.exists(ep) and os.path.exists(tr)):
            continue
        ev = json.load(open(ep))
        t = json.load(open(tr))
        if not ev.get("variants"):
            continue
        w, v = min(ev["variants"].items(), key=lambda kv: kv[1]["mae_count"])
        rows.append({
            "name": name, "label": LABEL.get(name, name),
            "mae": v["mae_count"], "deg": v["mae_deg"], "rmse": v["rmse_count"],
            "r2": v["r2"], "impr": v["improve_vs_baseline_pct"], "w": w,
            "epochs": t.get("epochs_done"), "params": None,
            "mae_raw": ev["variants"].get("raw", {}).get("mae_count"),
            "mae_ema": ev["variants"].get("ema", {}).get("mae_count"),
        })
    rows.sort(key=lambda r: r["mae"])
    return rows


def md(rows):
    if not rows:
        return "_(결과 없음)_\n"
    base = next((r for r in rows if r["name"] == "base"), rows[0])
    out = []
    out.append("| # | 설정 | 변경점 | MAE [count] | MAE [deg] | RMSE | R² | vs base | 가중치 |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        rel = 100.0 * (r["mae"] / base["mae"] - 1.0)
        out.append(f"| {i} | `{r['name']}` | {r['label']} | **{r['mae']:.1f}** | "
                   f"{r['deg']:.2f} | {r['rmse']:.0f} | {r['r2']:.3f} | "
                   f"{rel:+.1f}% | {r['w']} |")
    out.append("")
    out.append(f"기준선(항상 데모 평균 액션) MAE = "
               f"{base['mae'] / (1 - base['impr']/100):.0f} count. "
               f"1위는 그 대비 {rows[0]['impr']:.0f}% 개선.")
    out.append("")
    out.append("- **MAE [count]**: 홀드아웃 3 데모 open-loop, 16관절 평균 절대오차. "
               "1 count = π/8192 rad → 4096 count = 90°.")
    out.append("- **가중치**: raw 와 EMA 중 더 좋았던 쪽. `run.py --weights auto` 가 "
               "체크포인트에 기록된 이 값을 따른다.")
    out.append("- 홀드아웃이 3 데모뿐이라 **10% 안쪽 차이는 유의미하지 않다**고 봐야 한다.")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/sweep")
    ap.add_argument("--write", default=None, help="이 파일의 <!-- SWEEP_TABLE --> 를 교체")
    args = ap.parse_args()
    rows = collect(args.root)
    table = md(rows)
    print(table)
    if args.write:
        p = os.path.join(HERE, args.write)
        s = open(p).read()
        mark = "<!-- SWEEP_TABLE -->"
        if mark not in s:
            print(f"[WARN] {args.write} 에 {mark} 없음 — 건너뜀")
            return
        open(p, "w").write(s.replace(mark, table.rstrip()))
        print(f"[OK] {args.write} 갱신")


if __name__ == "__main__":
    main()
