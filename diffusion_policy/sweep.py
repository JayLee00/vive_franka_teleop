#!/usr/bin/env python3
"""설정 스윕 — 학습 1회가 ~수 분이라, 어떤 기법을 쓸지 '의견' 대신 '측정'으로 정한다.

smoke test 에서 드러난 사실: 참조 레시피(정규화 없음)는 epoch 17 에서 val 최저를 찍고
그 뒤 급격히 과적합한다(9분치 데이터 + 2.5M 파라미터). 그래서 비교 대상은
  - 증강(obs 노이즈)  - dropout+weight decay  - 모델 축소
  - 디노이저 종류(U-Net vs DiT)  - 확산이 필요한지(BC 기준선)

각 설정을 학습 → best.pt 로 홀드아웃 open-loop rollout → MAE(count) 로 순위.
결과: sweep_results.json + 콘솔 비교표.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

# name -> train.py 추가 인자
CONFIGS: dict[str, list[str]] = {
    "base":            [],
    "noise10":         ["--obs_noise", "0.10"],
    "noise30":         ["--obs_noise", "0.30"],
    "drop_wd":         ["--dropout", "0.2", "--weight_decay", "1e-2"],
    "noise30_drop_wd": ["--obs_noise", "0.30", "--dropout", "0.2",
                        "--weight_decay", "1e-2"],
    "small_noise30":   ["--obs_noise", "0.30", "--channels", "32,64,128",
                        "--cond_dim", "128"],
    "dit_noise30":     ["--algo", "dp_dit", "--obs_noise", "0.30"],
    "bc_noise30":      ["--algo", "bc_mlp", "--obs_noise", "0.30"],

    # ── 2라운드: 관찰 히스토리 길이 ────────────────────────────────────────
    # obs 에 과일 각도가 없어 회전 위상을 모른다(POMDP). T_obs=2 는 100ms 뿐이라
    # 게이트 위상을 담을 수 없다. 히스토리를 늘리면 위상을 부분적으로 복구할 수 있다.
    "oh4":             ["--obs_horizon", "4"],     # 200ms
    "oh8":             ["--obs_horizon", "8"],     # 400ms ~ 게이트 위상 1개
    "oh16":            ["--obs_horizon", "16"],    # 800ms ~ 게이트 사이클 1개
    "oh8_dit":         ["--obs_horizon", "8", "--algo", "dp_dit"],
    "oh8_bc":          ["--obs_horizon", "8", "--algo", "bc_mlp"],

    # 1라운드에서 obs 노이즈가 해롭다고 나왔으므로, DiT/BC 는 노이즈 없이 다시 본다
    # (noise30 과 묶으면 알고리즘 효과와 증강 효과가 섞여 비교가 안 된다)
    "dit":             ["--algo", "dp_dit"],
    "bc":              ["--algo", "bc_mlp"],
    # base 가 lr 1e-4 에서 ep125 쯤 정체했다 → 더 큰 lr 로 같은 예산 안에서 더 갈까
    "lr3e4":           ["--lr", "3e-4"],
}


def run(cmd, log_path):
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=HERE).returncode


def one(name, extra, args):
    out = os.path.join(HERE, args.root, name)
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "sweep.log")
    t0 = time.time()
    rc = run([sys.executable, "train.py", "--mode", "full",
              "--out_dir", os.path.join(args.root, name),
              "--epochs", str(args.epochs), "--batch", str(args.batch),
              "--workers", "2", "--ckpt_every", "100000",
              "--val_action_every", str(args.val_action_every),
              "--resume", "none"] + extra, log)
    if rc != 0:
        return name, {"status": f"train rc={rc}", "log": log}
    rc = run([sys.executable, "eval_rollout.py", "--ckpt",
              os.path.join(args.root, name, "best.pt")], log)
    if rc != 0:
        return name, {"status": f"eval rc={rc}", "log": log}
    ev = json.load(open(os.path.join(out, "eval", "eval_summary.json")))
    tr = json.load(open(os.path.join(out, "train_result.json")))
    best = min(ev["variants"].items(), key=lambda kv: kv[1]["mae_count"])
    return name, {
        "status": "ok", "extra": " ".join(extra) or "(참조 레시피)",
        "minutes": (time.time() - t0) / 60.0,
        "best_weights": best[0],
        "rollout_mae_count": best[1]["mae_count"],
        "rollout_mae_deg": best[1]["mae_deg"],
        "rollout_rmse_count": best[1]["rmse_count"],
        "rollout_r2": best[1]["r2"],
        "improve_vs_baseline_pct": best[1]["improve_vs_baseline_pct"],
        "val_action_mae_count": tr.get("best_val_action_mae_count"),
        "epochs_done": tr.get("epochs_done"),
        "mae_raw": ev["variants"].get("raw", {}).get("mae_count"),
        "mae_ema": ev["variants"].get("ema", {}).get("mae_count"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/sweep")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val_action_every", type=int, default=5)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    todo = {k: v for k, v in CONFIGS.items() if not args.only or k in args.only}
    os.makedirs(os.path.join(HERE, args.root), exist_ok=True)
    print(f"[SWEEP] {len(todo)} 설정 x {args.epochs} epoch, 동시 {args.parallel}개")
    for k, v in todo.items():
        print(f"   {k:18s} {' '.join(v) or '(참조 레시피)'}")

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for name, res in ex.map(lambda kv: one(kv[0], kv[1], args), todo.items()):
            results[name] = res
            print(f"[DONE] {name:18s} {res.get('status')} "
                  f"MAE={res.get('rollout_mae_count', float('nan')):.1f}c "
                  f"({res.get('minutes', 0):.1f}분)", flush=True)

    ok = {k: v for k, v in results.items() if v["status"] == "ok"}
    order = sorted(ok, key=lambda k: ok[k]["rollout_mae_count"])
    print("\n" + "=" * 112)
    print(f"  스윕 결과 — 홀드아웃 open-loop MAE 기준  (총 {(time.time()-t0)/60:.1f}분)")
    print("=" * 112)
    print(f"  {'설정':<18s} {'변경점':<42s} {'MAE[c]':>8s} {'MAE[deg]':>9s} "
          f"{'R2':>7s} {'가중치':>6s} {'ep':>4s}")
    print("  " + "-" * 108)
    for k in order:
        r = ok[k]
        print(f"  {k:<18s} {r['extra'][:42]:<42s} {r['rollout_mae_count']:>8.1f} "
              f"{r['rollout_mae_deg']:>9.2f} {r['rollout_r2']:>7.4f} "
              f"{r['best_weights']:>6s} {r['epochs_done']:>4d}")
    for k, v in results.items():
        if v["status"] != "ok":
            print(f"  {k:<18s} 실패: {v['status']}  ({v.get('log')})")
    print("=" * 112)
    if order:
        print(f"  1위: {order[0]}  MAE={ok[order[0]]['rollout_mae_count']:.1f} count "
              f"({ok[order[0]]['rollout_mae_deg']:.2f}deg), "
              f"기준선 대비 {ok[order[0]]['improve_vs_baseline_pct']:.1f}% 개선")
    with open(os.path.join(HERE, args.root, "sweep_results.json"), "w") as fh:
        json.dump({"ranking": order, "results": results,
                   "epochs": args.epochs}, fh, indent=2, ensure_ascii=False)
    print(f"  저장: {args.root}/sweep_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
