#!/usr/bin/env python3
"""홀드아웃 데모에서 open-loop rollout 평가 — 16관절 예측 vs GT.

**무엇을 재는가 / 무엇을 못 재는가**
시뮬레이터가 없으므로 관찰(obs)은 항상 GT 궤적에서 읽는다. 즉 이 지표는
"사람 시연과 같은 상태에 놓였을 때 정책이 같은 액션을 내는가"(= 액션 예측 정확도)를
재며, 실제 태스크 성공률(closed-loop)은 재지 못한다. 배포 전 마지막 관문은 실기 테스트다.

배포와 동일한 receding-horizon 으로 굴린다:
    t 에서 DDIM 추론 → T_pred 개 액션 → 앞 T_exec 개만 실행(=평가) → t += T_exec*stride
--temporal_ensemble 을 주면 겹치는 예측을 지수가중 평균(ACT 방식)해서 비교한다.

출력 (out_dir/eval/):
    eval_summary.json      전체·데모별·관절별 지표
    per_joint_mae.png      관절별 MAE
    error_vs_horizon.png   실행 지평 내 스텝별 오차 증가
    rollout_<demo>.png     GT vs 예측 시계열 (대표 관절)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dp_config as C
from dp_data import Normalizer, load_demos, split_demos
from dp_model import DDPMScheduler, policy_from_ckpt

PLOT_JOINTS = [2, 6, 10, 14]      # 엄지 MCP / 검지·중지·약지 MCP flexion


def load_ckpt(path, device, use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    policy = policy_from_ckpt(ck, device, use_ema=use_ema)
    obs_norm = Normalizer(ck["obs_norm"]["mean"], ck["obs_norm"]["std"])
    act_norm = Normalizer(ck["act_norm"]["mean"], ck["act_norm"]["std"])
    return policy, obs_norm, act_norm, cfg, ck


@torch.no_grad()
def rollout_demo(policy, sched, obs_norm, act_norm, demo, cfg,
                 device, action_steps, infer_steps, temporal_ensemble=False,
                 te_k=0.6, seed=0):
    """반환: (pred (M,16), gt (M,16), t_idx (M,), step_err (T_exec,))"""
    T_obs, T_pred = cfg["obs_horizon"], cfg["pred_horizon"]
    stride = cfg["ds_stride"]
    back = (T_obs - 1) * stride
    N = demo.n_used

    g = torch.Generator(device=device).manual_seed(seed)
    obs_off = np.array([-(T_obs - 1 - k) * stride for k in range(T_obs)])
    act_off = np.arange(T_pred) * stride

    # temporal ensembling 용 누적 버퍼 (원본 인덱스 → 가중 예측 합/가중치 합)
    acc = {}
    preds, gts, tidx = [], [], []
    step_err_sum = np.zeros(action_steps)
    step_err_cnt = np.zeros(action_steps)

    t = back
    while t + (T_pred - 1) * stride <= N - 1:
        obs = demo.obs[t + obs_off]                                # (T_obs,46)
        obs_n = torch.from_numpy(obs_norm.normalize(obs).astype(np.float32))
        a_n = policy.sample(obs_n.unsqueeze(0).to(device), sched, T_pred,
                            infer_steps, generator=g)
        a = act_norm.denormalize(a_n.squeeze(0).cpu().numpy())     # (T_pred,16)

        idxs = t + act_off
        gt_full = demo.act[idxs]
        if temporal_ensemble:
            for k in range(T_pred):
                w = float(np.exp(-te_k * k))
                i = int(idxs[k])
                s, ws = acc.get(i, (np.zeros(16), 0.0))
                acc[i] = (s + w * a[k], ws + w)
        else:
            for k in range(min(action_steps, T_pred)):
                e = np.abs(a[k] - gt_full[k]).mean()
                step_err_sum[k] += e
                step_err_cnt[k] += 1
            preds.append(a[:action_steps])
            gts.append(gt_full[:action_steps])
            tidx.append(idxs[:action_steps])
        t += action_steps * stride

    if temporal_ensemble:
        keys = sorted(acc)
        pred = np.stack([acc[i][0] / acc[i][1] for i in keys])
        gt = np.stack([demo.act[i] for i in keys])
        return pred, gt, np.array(keys), np.full(action_steps, np.nan)

    pred = np.concatenate(preds, 0)
    gt = np.concatenate(gts, 0)
    ti = np.concatenate(tidx, 0)
    step_err = step_err_sum / np.maximum(1, step_err_cnt)
    return pred, gt, ti, step_err


def main():
    ap = argparse.ArgumentParser(description="홀드아웃 open-loop rollout 평가")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=None, help="기본: ckpt 폴더/eval")
    ap.add_argument("--action_steps", type=int, default=C.ACTION_STEPS)
    ap.add_argument("--infer_steps", type=int, default=C.INFER_STEPS)
    ap.add_argument("--weights", default="both", choices=["raw", "ema", "both"])
    ap.add_argument("--temporal_ensemble", action="store_true")
    ap.add_argument("--keep_dead", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = args.out_dir or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, _, cfg, ck = load_ckpt(args.ckpt, device, use_ema=False)
    held_names = ck["held_out_demos"]
    demos, _ = load_demos(keep_dead=args.keep_dead, verbose=False)
    name2d = {d.name: d for d in demos}
    held = [name2d[n] for n in held_names if n in name2d]
    if not held:
        # 체크포인트에 이름이 없으면 같은 시드로 재분할
        _, held = split_demos(demos, cfg.get("n_held_out", C.N_HELD_OUT), cfg["seed"])
    sched = DDPMScheduler(cfg["diff_steps"]).to(device)

    print("=" * 100)
    print(f"  open-loop rollout 평가   ckpt={args.ckpt}")
    print(f"  T_obs={cfg['obs_horizon']} T_pred={cfg['pred_horizon']} "
          f"T_exec={args.action_steps} DDIM={args.infer_steps} "
          f"stride={cfg['ds_stride']} ({cfg['ctrl_hz']:.0f}Hz)")
    print(f"  홀드아웃 {len(held)}개: {[d.name for d in held]}")
    print("=" * 100)

    variants = (["raw", "ema"] if args.weights == "both" else [args.weights])
    summary = {"ckpt": args.ckpt, "config": {k: v for k, v in cfg.items()
                                             if k not in ("joint_limits", "obs_spec")},
               "action_steps": args.action_steps, "infer_steps": args.infer_steps,
               "temporal_ensemble": bool(args.temporal_ensemble),
               "variants": {}}

    for vi, var in enumerate(variants):
        if var == "ema" and not ck.get("ema"):
            print("[SKIP] ema 가중치 없음")
            continue
        policy, obs_norm, act_norm, cfg, _ = load_ckpt(args.ckpt, device,
                                                       use_ema=(var == "ema"))
        per_demo, all_p, all_g = {}, [], []
        step_errs = []
        for d in held:
            p, gt, ti, se = rollout_demo(policy, sched, obs_norm, act_norm, d, cfg,
                                         device, args.action_steps, args.infer_steps,
                                         args.temporal_ensemble)
            mae = float(np.abs(p - gt).mean())
            rmse = float(np.sqrt(((p - gt) ** 2).mean()))
            # 기준선: "현재 자세 유지"(action = 마지막 관측 joint_pos) 대비 개선율
            per_demo[d.name] = {"n_pred": int(len(p)), "mae_count": mae,
                                "rmse_count": rmse,
                                "mae_deg": mae * 90.0 / 4096.0}
            all_p.append(p); all_g.append(gt); step_errs.append(se)
            print(f"  [{var}] {d.name:38s} n={len(p):5d}  "
                  f"MAE={mae:8.2f} count ({mae*90/4096:5.2f}deg)  RMSE={rmse:8.2f}")

        P = np.concatenate(all_p, 0); G = np.concatenate(all_g, 0)
        mae = float(np.abs(P - G).mean()); rmse = float(np.sqrt(((P - G) ** 2).mean()))
        mse = float(((P - G) ** 2).mean())
        pj_mae = np.abs(P - G).mean(0)
        # 기준선 2개: (a) 전체 평균 액션 예측, (b) 학습셋 상수
        base_mean = float(np.abs(G - G.mean(0)).mean())
        r2 = 1.0 - mse / float(((G - G.mean(0)) ** 2).mean())
        se = np.nanmean(np.stack(step_errs), 0) if not args.temporal_ensemble else None

        print(f"  [{var}] 전체  MSE={mse:10.1f}  MAE={mae:7.2f} count "
              f"({mae*90/4096:.2f}deg)  RMSE={rmse:7.2f}  R2={r2:.4f}")
        print(f"  [{var}] 기준선(항상 평균액션) MAE={base_mean:7.2f} → "
              f"개선 {100*(1-mae/base_mean):.1f}%")

        summary["variants"][var] = {
            "mse_count2": mse, "mae_count": mae, "rmse_count": rmse,
            "mae_deg": mae * 90.0 / 4096.0, "r2": r2,
            "baseline_mean_action_mae": base_mean,
            "improve_vs_baseline_pct": 100 * (1 - mae / base_mean),
            "per_joint_mae_count": pj_mae.tolist(),
            "per_demo": per_demo,
            "err_vs_exec_step": (se.tolist() if se is not None else None),
        }

        # ── 플롯: 관절별 MAE ──
        plt.figure(figsize=(10, 3.2))
        plt.bar(np.arange(16) + (0.35 * vi - 0.17), pj_mae, width=0.35, label=var)
        plt.xticks(range(16), [f"j{i}" for i in range(16)])
        plt.ylabel("MAE [count]"); plt.title("Per-joint MAE (held-out open-loop)")
        plt.grid(alpha=0.3, axis="y"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out, f"per_joint_mae_{var}{args.tag}.png"), dpi=110)
        plt.close()

        # ── 플롯: 지평 스텝별 오차 ──
        if se is not None:
            plt.figure(figsize=(5, 3.2))
            plt.plot(np.arange(1, len(se) + 1), se, "o-")
            plt.xlabel("step within exec horizon (1=immediate)"); plt.ylabel("MAE [count]")
            plt.title(f"Error vs exec step ({var})"); plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"error_vs_horizon_{var}{args.tag}.png"), dpi=110)
            plt.close()

        # ── 플롯: 데모별 시계열 ──
        for d in held:
            p, gt, ti, _ = rollout_demo(policy, sched, obs_norm, act_norm, d, cfg,
                                        device, args.action_steps, args.infer_steps,
                                        args.temporal_ensemble)
            fig, axes = plt.subplots(len(PLOT_JOINTS), 1, figsize=(11, 2.0 * len(PLOT_JOINTS)),
                                     sharex=True)
            for ax, j in zip(np.atleast_1d(axes), PLOT_JOINTS):
                ax.plot(ti / C.SRC_HZ, gt[:, j], lw=1.4, label="GT")
                ax.plot(ti / C.SRC_HZ, p[:, j], lw=1.0, alpha=0.85, label="pred")
                ax.set_ylabel(f"j{j} [count]"); ax.grid(alpha=0.3)
            np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2)
            np.atleast_1d(axes)[-1].set_xlabel("time [s]")
            fig.suptitle(f"{d.name}  ({var})  MAE={np.abs(p-gt).mean():.1f} count")
            fig.tight_layout()
            safe = d.name.replace(":", "_").replace(".h5", "")
            fig.savefig(os.path.join(out, f"rollout_{safe}_{var}{args.tag}.png"), dpi=110)
            plt.close(fig)

    with open(os.path.join(out, f"eval_summary{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("=" * 100)
    print(f"  저장: {out}/eval_summary{args.tag}.json + png")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
