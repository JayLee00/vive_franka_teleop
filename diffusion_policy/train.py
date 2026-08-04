#!/usr/bin/env python3
"""Diffusion Policy 학습 — 레몬 in-hand 회전 (KISTAR Hand 16관절).

모드 4개:
  preflight : 클램프 요약표 + shape assert + NaN 체크 + 전체 ETA 만 출력하고 종료
  overfit   : 배치 1개를 (x0, noise, t) 고정으로 과적합 → 파이프라인이 학습 가능한지 증명
  smoke     : N분만 돌려 실제 처리량/메모리/체크포인트 저장 경로 검증
  full      : 본 학습

사용:
  python3 train.py --mode preflight
  python3 train.py --mode overfit
  python3 train.py --mode smoke --smoke_minutes 5
  python3 train.py --mode full --epochs 600 --batch 256 --out_dir runs/dp_lemon

체크포인트(3종 + 사고 대비):
  best.pt    val loss 최저
  last.pt    매 에폭 (auto-resume 대상)
  ep####.pt  --ckpt_every 에폭마다
  crash.pt   예외/SIGTERM 시
각 체크포인트에 정규화 통계(train split 전용)와 전체 설정이 함께 들어간다 →
배포 시 run.py 가 체크포인트만 읽어 동일 전처리를 재현.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dp_config as C
from dp_data import Normalizer, WindowDataset, load_demos, split_demos
from dp_model import DDPMScheduler, EMA, build_policy

_STOP = {"flag": False, "why": ""}


def _on_signal(signum, frame):
    _STOP["flag"] = True
    _STOP["why"] = f"signal {signum}"
    print(f"\n[SIGNAL] {signum} 수신 → 이번 에폭 끝에 저장하고 종료", flush=True)


class Tee:
    """stdout 을 콘솔 + 파일 양쪽으로."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fmt_eta(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


# ══════════════════════════════════════════════════════════════════════════
# 검증용 고정 노이즈 val loss
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_val_action_mae(policy, sched, ds, act_norm, device, n_samples,
                        infer_steps, pred_h, seed=7):
    """val 윈도우를 균등 샘플해 DDIM 으로 실제 액션을 뽑고 count 단위 MAE 를 낸다.

    노이즈 MSE(val_loss)는 알고리즘마다 스케일이 달라 비교가 안 되고, 우리가 실제로
    아끼는 값도 아니다. 이 지표가 배포 성능에 가장 가깝고 algo 간 비교도 가능하다.
    """
    policy.eval()
    n = min(n_samples, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    obs = np.stack([ds[int(i)][0] for i in idx])
    act = np.stack([ds[int(i)][1] for i in idx])
    tot, cnt = 0.0, 0
    for s0 in range(0, n, 256):
        o = torch.from_numpy(obs[s0:s0 + 256]).to(device)
        g = torch.Generator(device=device).manual_seed(seed + s0)
        a = policy.sample(o, sched, pred_h, infer_steps, generator=g)
        pa = act_norm.denormalize(a.cpu().numpy())
        ga = act_norm.denormalize(act[s0:s0 + 256])
        tot += float(np.abs(pa - ga).sum()); cnt += pa.size
    return tot / max(1, cnt)


@torch.no_grad()
def eval_val_loss(policy, sched, loader, device, diff_steps, seed=1234):
    """val 은 배치마다 노이즈/타임스텝을 시드로 고정 → 에폭 간 비교 가능한 곡선."""
    policy.eval()
    tot, n = 0.0, 0
    for bi, (obs, act) in enumerate(loader):
        obs = obs.to(device, non_blocking=True)
        act = act.to(device, non_blocking=True)
        B = obs.shape[0]
        if getattr(policy, "denoiser_kind", "") == "bc":
            l = F.mse_loss(policy.predict(obs), act)
        else:
            g = torch.Generator(device="cpu").manual_seed(seed * 100003 + bi)
            t = torch.randint(0, diff_steps, (B,), generator=g)
            noise = torch.randn(act.shape, generator=g).to(device)
            x_t = sched.add_noise(act, noise, t.to(device))
            l = F.mse_loss(policy(x_t, t.to(device), obs), noise)
        tot += l.item() * B
        n += B
    return tot / max(1, n)


# ══════════════════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════════════════
def make_ckpt(policy, ema, opt, sched_lr, epoch, gstep, best_val,
              obs_norm, act_norm, args, train_names, held_names, history):
    return {
        "format": 1,
        "policy": policy.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": opt.state_dict(),
        "lr_sched": sched_lr.state_dict(),
        "epoch": epoch,
        "global_step": gstep,
        "best_val": best_val,
        "obs_norm": obs_norm.state(),
        "act_norm": act_norm.state(),
        "history": history,
        "train_demos": train_names,
        "held_out_demos": held_names,
        "config": {
            "obs_spec": C.OBS_SPEC, "obs_dim": C.OBS_DIM,
            "action_key": C.ACTION_KEY, "action_dim": C.ACTION_DIM,
            "obs_horizon": args.obs_horizon, "pred_horizon": args.pred_horizon,
            "action_steps": C.ACTION_STEPS, "diff_steps": args.diff_steps,
            "infer_steps": C.INFER_STEPS,
            "ds_stride": args.ds_stride, "src_hz": C.SRC_HZ,
            "ctrl_hz": C.SRC_HZ / args.ds_stride,
            "joint_limits": C.JOINT_LIMITS, "max_rate_cps": C.MAX_RATE_CPS,
            "seed": args.seed, "n_held_out": args.n_held_out,
            "algo": args.algo,
            "model": {
                "cond_dim": args.cond_dim,
                "channels": [int(x) for x in args.channels.split(",")],
                "dit_d_model": args.dit_d_model, "dit_n_head": args.dit_n_head,
                "dit_n_layer": args.dit_n_layer, "dropout": args.dropout,
            },
            "train": {"obs_noise": args.obs_noise, "lr": args.lr,
                      "weight_decay": args.weight_decay, "batch": args.batch,
                      "epochs": args.epochs, "ema_decay": args.ema_decay},
        },
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def save_ckpt(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)          # 저장 중 죽어도 기존 파일이 안 깨지게


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Diffusion Policy 학습 (레몬 in-hand 회전)")
    ap.add_argument("--mode", choices=["preflight", "overfit", "smoke", "full"],
                    default="full")
    ap.add_argument("--out_dir", default="runs/dp_lemon")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--obs_horizon", type=int, default=C.OBS_HORIZON)
    ap.add_argument("--pred_horizon", type=int, default=C.PRED_HORIZON)
    ap.add_argument("--diff_steps", type=int, default=C.DIFF_STEPS)
    ap.add_argument("--ds_stride", type=int, default=C.DS_STRIDE)
    ap.add_argument("--n_held_out", type=int, default=C.N_HELD_OUT)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    # 알고리즘 / 정규화 / 모델 크기
    ap.add_argument("--algo", default="dp_unet",
                    choices=["dp_unet", "dp_dit", "bc_mlp"])
    ap.add_argument("--obs_noise", type=float, default=0.0,
                    help="정규화된 obs 에 더하는 가우시안 노이즈 sigma (증강)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--cond_dim", type=int, default=256)
    ap.add_argument("--channels", default="64,128,256")
    ap.add_argument("--dit_d_model", type=int, default=256)
    ap.add_argument("--dit_n_head", type=int, default=4)
    ap.add_argument("--dit_n_layer", type=int, default=4)
    ap.add_argument("--val_action_every", type=int, default=5,
                    help="N 에폭마다 액션공간 MAE 측정 (best 선택 기준)")
    ap.add_argument("--val_action_n", type=int, default=768)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=0, help="0 = early stop 없음")
    ap.add_argument("--smoke_minutes", type=float, default=5.0)
    ap.add_argument("--overfit_steps", type=int, default=3000)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--keep_dead", action="store_true",
                    help="정지 데모(act_std<5)도 포함")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.out_dir, "train.log"))
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    print("\n" + "#" * 108)
    print(f"#  Diffusion Policy 학습  mode={args.mode}  시작 {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("#" * 108)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] torch={torch.__version__} device={device} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    # ── 데이터 ────────────────────────────────────────────────────────────
    demos, table = load_demos(keep_dead=args.keep_dead)
    train_demos, held_demos = split_demos(demos, args.n_held_out, args.seed)
    print(f"\n[SPLIT] train {len(train_demos)} demos | held-out {len(held_demos)} demos "
          f"(seed={args.seed})")
    for d in held_demos:
        print(f"   held-out: {d.name}  ({d.n_used} steps, {d.n_used/100:.1f}s)")

    train_raw = WindowDataset(train_demos, args.obs_horizon, args.pred_horizon,
                              args.ds_stride)
    held_raw = WindowDataset(held_demos, args.obs_horizon, args.pred_horizon,
                             args.ds_stride)

    # 정규화: train split 프레임만
    obs_norm = Normalizer.fit([d.obs for d in train_demos])
    act_norm = Normalizer.fit([d.act for d in train_demos])
    obs_norm.save(os.path.join(args.out_dir, "obs_norm.npz"))
    act_norm.save(os.path.join(args.out_dir, "act_norm.npz"))
    print(f"\n[NORM] train split 전용 통계 (obs {C.OBS_DIM}dim, act {C.ACTION_DIM}dim)")
    print(f"   obs  mean[:6]={np.round(obs_norm.mean[:6],3)}  std[:6]={np.round(obs_norm.std[:6],3)}")
    print(f"   act  mean[:6]={np.round(act_norm.mean[:6],1)}  std[:6]={np.round(act_norm.std[:6],1)}")
    if float(obs_norm.std.min()) < 1e-4:
        bad = np.flatnonzero(obs_norm.std < 1e-4)
        print(f"   ⚠ 분산 거의 0인 obs 차원 {bad.tolist()} (상수 채널 — 정보 없음)")

    train_ds = WindowDataset(train_demos, args.obs_horizon, args.pred_horizon,
                             args.ds_stride, obs_norm, act_norm)
    val_ds = WindowDataset(held_demos, args.obs_horizon, args.pred_horizon,
                           args.ds_stride, obs_norm, act_norm)

    # ── shape / NaN assert ────────────────────────────────────────────────
    o, a = train_ds[0]
    assert o.shape == (args.obs_horizon, C.OBS_DIM), f"obs shape {o.shape}"
    assert a.shape == (args.pred_horizon, C.ACTION_DIM), f"act shape {a.shape}"
    print(f"\n[SHAPE] obs {o.shape} (T_obs x obs_dim)   act {a.shape} (T_pred x action_dim)")
    print(f"[WINDOW] train {len(train_ds):,}  held-out {len(val_ds):,}  "
          f"(stride={args.ds_stride} → 실효 {C.SRC_HZ/args.ds_stride:.0f}Hz, "
          f"지평 {args.pred_horizon*args.ds_stride/C.SRC_HZ:.2f}s)")
    n_check = min(2000, len(train_ds))
    idxs = np.linspace(0, len(train_ds) - 1, n_check).astype(int)
    bad = 0
    for i in idxs:
        oo, aa = train_ds[int(i)]
        if not (np.isfinite(oo).all() and np.isfinite(aa).all()):
            bad += 1
    print(f"[NAN] 표본 {n_check:,} 윈도우 검사 → 비정상 {bad}개")
    assert bad == 0, "NaN/Inf 윈도우 발견"

    if args.mode == "preflight":
        pass  # 아래에서 ETA 만 계산

    # ── 모델 ──────────────────────────────────────────────────────────────
    chans = tuple(int(x) for x in args.channels.split(","))
    mkw = dict(algo=args.algo, cond_dim=args.cond_dim, channels=chans,
               dropout=args.dropout, pred_horizon=args.pred_horizon,
               dit_d_model=args.dit_d_model, dit_n_head=args.dit_n_head,
               dit_n_layer=args.dit_n_layer)
    policy, n_params = build_policy(C.OBS_DIM, args.obs_horizon, C.ACTION_DIM,
                                    device, **mkw)
    sched = DDPMScheduler(args.diff_steps).to(device)
    ema = EMA(policy, args.ema_decay)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs), eta_min=1e-6)
    print(f"[MODEL] algo={args.algo} params={n_params:,}  DDPM T={args.diff_steps}  "
          f"DDIM infer={C.INFER_STEPS}  dropout={args.dropout} obs_noise={args.obs_noise}")

    # ══════════════════════════════════════════════════════════════════
    # overfit 모드: (x0, noise, t) 고정 배치 하나로 학습 가능성 증명
    # ══════════════════════════════════════════════════════════════════
    if args.mode == "overfit":
        print("\n" + "=" * 108)
        print("  [OVERFIT] 배치 1개 · 노이즈/타임스텝 고정 → loss 가 0 으로 떨어져야 정상")
        print("=" * 108)
        loader = DataLoader(train_ds, batch_size=min(64, args.batch), shuffle=True,
                            num_workers=0)
        obs, act = next(iter(loader))
        obs, act = obs.to(device), act.to(device)
        g = torch.Generator(device="cpu").manual_seed(0)
        t_fix = torch.randint(0, args.diff_steps, (obs.shape[0],), generator=g).to(device)
        noise_fix = torch.randn(act.shape, generator=g).to(device)
        x_t = sched.add_noise(act, noise_fix, t_fix)
        t0 = time.time()
        loss = float("nan")
        for step in range(1, args.overfit_steps + 1):
            policy.train()
            if args.algo == "bc_mlp":
                loss = F.mse_loss(policy.predict(obs), act)
            else:
                loss = F.mse_loss(policy(x_t, t_fix, obs), noise_fix)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if step % 250 == 0 or step == 1:
                print(f"   step {step:5d}/{args.overfit_steps}  loss={loss.item():.6f}")
            if loss.item() < 1e-4:
                print(f"   step {step:5d}  loss={loss.item():.2e} < 1e-4 → 조기 통과")
                break
        ok = loss.item() < 1e-3
        print(f"\n  최종 loss={loss.item():.3e}  ({time.time()-t0:.0f}s)  "
              f"→ {'PASS (파이프라인 학습 가능)' if ok else 'FAIL (구조/데이터 확인 필요)'}")
        print("=" * 108)
        json.dump({"mode": "overfit", "final_loss": loss.item(), "pass": ok},
                  open(os.path.join(args.out_dir, "overfit_result.json"), "w"), indent=2)
        return 0 if ok else 1

    # ── 로더 ──────────────────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    steps_per_epoch = len(train_loader)

    # ── ETA 측정 (짧게 돌려 처리량 측정) ────────────────────────────────
    print("\n[ETA] 처리량 측정 중 (20 step)...")
    policy.train()
    it = iter(train_loader)
    for _ in range(3):                      # warmup
        obs, act = next(it)
        obs, act = obs.to(device), act.to(device)
        loss = policy.compute_loss(obs, act, sched, args.diff_steps)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_meas = 20
    for _ in range(n_meas):
        try:
            obs, act = next(it)
        except StopIteration:
            it = iter(train_loader); obs, act = next(it)
        obs, act = obs.to(device), act.to(device)
        loss = policy.compute_loss(obs, act, sched, args.diff_steps)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.time() - t0) / n_meas
    epoch_sec = per_step * steps_per_epoch
    val_sec = epoch_sec * len(val_loader) / max(1, steps_per_epoch) * 0.35
    total_sec = (epoch_sec + val_sec) * args.epochs
    print(f"[ETA] {per_step*1000:.1f} ms/step · {steps_per_epoch} step/epoch "
          f"→ {epoch_sec+val_sec:.1f} s/epoch")
    print(f"[ETA] {args.epochs} epoch 전체 ≈ {fmt_eta(total_sec)} "
          f"(예상 종료 {datetime.now()+timedelta(seconds=total_sec):%m-%d %H:%M})")
    if device.type == "cuda":
        print(f"[ETA] GPU mem {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

    if args.mode == "preflight":
        print("\n[PREFLIGHT] 검증 완료 — 학습은 하지 않음")
        json.dump({"per_step_ms": per_step * 1000, "steps_per_epoch": steps_per_epoch,
                   "epoch_sec": epoch_sec + val_sec, "epochs": args.epochs,
                   "eta_sec": total_sec, "train_windows": len(train_ds),
                   "val_windows": len(val_ds),
                   "train_demos": [d.name for d in train_demos],
                   "held_out_demos": [d.name for d in held_demos]},
                  open(os.path.join(args.out_dir, "preflight.json"), "w"), indent=2)
        return 0

    # 측정으로 흐트러진 상태를 초기화 (모델/옵티마이저 새로)
    set_seed(args.seed)
    policy, _ = build_policy(C.OBS_DIM, args.obs_horizon, C.ACTION_DIM, device, **mkw)
    ema = EMA(policy, args.ema_decay)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs), eta_min=1e-6)

    # ── auto-resume ───────────────────────────────────────────────────────
    start_epoch, gstep, best_val = 1, 0, float("inf")
    history = []
    last_path = os.path.join(args.out_dir, "last.pt")
    if args.resume == "auto" and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"])
        if ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        lr_sched.load_state_dict(ck["lr_sched"])
        start_epoch = ck["epoch"] + 1
        gstep = ck["global_step"]
        best_val = ck["best_val"]
        history = ck.get("history", [])
        r = ck.get("rng") or {}
        try:
            if r.get("torch") is not None:
                torch.set_rng_state(r["torch"].cpu() if hasattr(r["torch"], "cpu") else r["torch"])
            if r.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s
                                              for s in r["cuda"]])
            if r.get("numpy"):
                np.random.set_state(r["numpy"])
            if r.get("python"):
                random.setstate(r["python"])
        except Exception as e:
            print(f"[RESUME] RNG 복원 실패(무해): {e}")
        print(f"\n[RESUME] {last_path} → epoch {start_epoch} 부터 재개 "
              f"(best_val={best_val:.5f})")

    if args.mode == "smoke":
        print(f"\n[SMOKE] {args.smoke_minutes:.1f}분만 학습하고 종료")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as fh:
            fh.write("epoch,global_step,train_loss,val_loss,val_loss_ema,"
                     "val_mae_raw,val_mae_ema,lr,sec\n")

    print("\n" + "=" * 108)
    print(f"  학습 시작  epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"train_win={len(train_ds):,} val_win={len(val_ds):,}")
    print("=" * 108)

    t_start = time.time()
    deadline = t_start + args.smoke_minutes * 60 if args.mode == "smoke" else None
    stop_reason = "정상 종료"
    bad_epochs = 0

    def _save_all(epoch, tag=None):
        ck = make_ckpt(policy, ema, opt, lr_sched, epoch, gstep, best_val,
                       obs_norm, act_norm, args,
                       [d.name for d in train_demos], [d.name for d in held_demos],
                       history)
        save_ckpt(ck, last_path)
        if tag:
            save_ckpt(ck, os.path.join(args.out_dir, tag))
        return ck

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            te = time.time()
            policy.train()
            tot, n = 0.0, 0
            for obs, act in train_loader:
                obs = obs.to(device, non_blocking=True)
                act = act.to(device, non_blocking=True)
                B = obs.shape[0]
                if args.obs_noise > 0:      # 증강: 정규화 공간에서 obs 에 노이즈
                    obs = obs + torch.randn_like(obs) * args.obs_noise
                loss = policy.compute_loss(obs, act, sched, args.diff_steps)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"epoch {epoch}: loss={loss.item()}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                ema.update(policy)
                tot += loss.item() * B
                n += B
                gstep += 1
            train_loss = tot / max(1, n)

            val_loss = eval_val_loss(policy, sched, val_loader, device, args.diff_steps)
            val_ema = eval_val_loss(ema.shadow, sched, val_loader, device, args.diff_steps)

            # 액션공간 MAE(count) — algo 간 비교 가능하고 배포 성능에 가장 가까운 지표.
            # 이게 있으면 best 체크포인트 선택 기준으로 쓴다.
            mae_raw = mae_ema = float("nan")
            if args.val_action_every and (epoch % args.val_action_every == 0
                                          or epoch == start_epoch):
                mae_raw = eval_val_action_mae(policy, sched, val_ds, act_norm, device,
                                              args.val_action_n, C.INFER_STEPS,
                                              args.pred_horizon)
                mae_ema = eval_val_action_mae(ema.shadow, sched, val_ds, act_norm,
                                              device, args.val_action_n, C.INFER_STEPS,
                                              args.pred_horizon)
            lr_sched.step()
            dt = time.time() - te
            history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                            "val_ema": val_ema, "mae_raw": mae_raw, "mae_ema": mae_ema})
            with open(csv_path, "a") as fh:
                fh.write(f"{epoch},{gstep},{train_loss:.6f},{val_loss:.6f},"
                         f"{val_ema:.6f},{mae_raw:.3f},{mae_ema:.3f},"
                         f"{lr_sched.get_last_lr()[0]:.3e},{dt:.2f}\n")

            if np.isfinite(mae_raw) or np.isfinite(mae_ema):
                score = float(np.nanmin([mae_raw, mae_ema]))
                best_is_ema = bool(np.nanargmin([mae_raw, mae_ema]))
            else:
                score = float("inf")        # 측정 안 한 에폭은 best 갱신 대상 아님
                best_is_ema = False
            improved = score < best_val
            if improved:
                best_val = score
                bad_epochs = 0
            else:
                bad_epochs += 1

            if epoch % 10 == 0 or epoch == start_epoch or improved:
                done = epoch - start_epoch + 1
                left = (args.epochs - epoch) * (time.time() - t_start) / max(1, done)
                mae_s = ("" if not np.isfinite(mae_raw)
                         else f" mae={mae_raw:6.1f}/{mae_ema:6.1f}c")
                print(f"[{epoch:4d}/{args.epochs}] train={train_loss:.5f} "
                      f"val={val_loss:.5f} val_ema={val_ema:.5f}{mae_s} "
                      f"lr={lr_sched.get_last_lr()[0]:.2e} {dt:.1f}s "
                      f"ETA {fmt_eta(left)}{'  ★' if improved else ''}")

            _save_all(epoch, tag=f"ep{epoch:04d}.pt" if epoch % args.ckpt_every == 0 else None)
            if improved:
                ck = make_ckpt(policy, ema, opt, lr_sched, epoch, gstep, best_val,
                               obs_norm, act_norm, args,
                               [d.name for d in train_demos],
                               [d.name for d in held_demos], history)
                ck["best_metric"] = {"val_action_mae_count": best_val,
                                     "prefer_weights": "ema" if best_is_ema else "raw"}
                save_ckpt(ck, os.path.join(args.out_dir, "best.pt"))

            if _STOP["flag"]:
                stop_reason = f"중단 요청 ({_STOP['why']})"
                break
            if deadline and time.time() > deadline:
                stop_reason = f"smoke {args.smoke_minutes}분 도달"
                break
            if args.patience and bad_epochs >= args.patience:
                stop_reason = f"early stop (patience {args.patience})"
                break

    except BaseException as e:
        print(f"\n[CRASH] {type(e).__name__}: {e}")
        try:
            ck = make_ckpt(policy, ema, opt, lr_sched, locals().get("epoch", start_epoch),
                           gstep, best_val, obs_norm, act_norm, args,
                           [d.name for d in train_demos],
                           [d.name for d in held_demos], history)
            save_ckpt(ck, os.path.join(args.out_dir, "crash.pt"))
            print(f"[CRASH] 상태 저장: {args.out_dir}/crash.pt")
        except Exception as e2:
            print(f"[CRASH] 저장 실패: {e2}")
        raise

    total = time.time() - t_start
    print("\n" + "=" * 108)
    print(f"  [DONE] {stop_reason}  경과 {fmt_eta(total)}  "
          f"best val_action_MAE={best_val:.2f} count ({best_val*90/4096:.3f}deg)")
    print(f"  체크포인트: {args.out_dir}/best.pt  last.pt  ep*.pt")
    print(f"  정규화    : {args.out_dir}/obs_norm.npz  act_norm.npz (체크포인트에도 포함)")
    print(f"  로그      : {args.out_dir}/train.log  metrics.csv")
    print("=" * 108)
    json.dump({"stop_reason": stop_reason, "best_val_action_mae_count": best_val,
               "algo": args.algo, "obs_noise": args.obs_noise,
               "dropout": args.dropout, "lr": args.lr,
               "weight_decay": args.weight_decay,
               "epochs_done": history[-1]["epoch"] if history else 0,
               "elapsed_sec": total},
              open(os.path.join(args.out_dir, "train_result.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
