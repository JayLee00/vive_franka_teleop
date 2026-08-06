#!/usr/bin/env python3
"""학습 루프.

    python train.py --config configs/01_anchor.yaml --data /path/to/logs
    python train.py --config configs/01_anchor.yaml --mode preflight   # 데이터·shape·ETA 만
    python train.py --config configs/01_anchor.yaml --mode smoke       # 몇 스텝만
    python train.py --config configs/01_anchor.yaml --override model.fusion=gated

refer/train.py 의 규약을 유지한다(검증된 것들이라 바꿀 이유가 없다):
  · **best 는 노이즈 MSE 가 아니라 홀드아웃 액션 MAE[count] 로 고른다.**
    알고리즘 간 비교가 되고 배포 성능에 가깝다. flow/bc/diffusion 을 같은 자로 잰다.
  · 정규화 통계를 **체크포인트 안에** 넣는다 → 배포 시 전처리 불일치가 구조적으로 불가능.
  · last.pt 자동 재개, 예외·SIGTERM 시 crash.pt.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vtdp.config import dump_config, load_config, timing_table      # noqa: E402
from vtdp.data import Normalizer, build_datasets                    # noqa: E402
from vtdp.policy import build_policy                                # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
class EMA:
    """지수이동평균. warmup 램프로 초기 스텝의 과도한 관성을 피한다.

    decay = min(max_decay, (1+step)/(10+step))
    """

    def __init__(self, model: torch.nn.Module, max_decay: float = 0.9999):
        self.max_decay = max_decay
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @property
    def decay(self) -> float:
        return min(self.max_decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                s.copy_(m)
        self.step += 1

    def state_dict(self):
        return {"step": self.step, "max_decay": self.max_decay,
                "shadow": self.shadow.state_dict()}

    def load_state_dict(self, sd):
        self.step = sd["step"]
        self.max_decay = sd.get("max_decay", self.max_decay)
        self.shadow.load_state_dict(sd["shadow"])


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def to_device(batch: dict, dev) -> dict:
    out = {"obs": {k: v.to(dev, non_blocking=True) for k, v in batch["obs"].items()},
           "action": batch["action"].to(dev, non_blocking=True)}
    if batch.get("mask"):          # docs/SHAPES.md 2.1 — 버리면 계약이 조용히 깨진다
        out["mask"] = {k: v.to(dev, non_blocking=True) for k, v in batch["mask"].items()}
    return out


def flatten_lstms(model: torch.nn.Module) -> None:
    """`.to(device)`·`deepcopy` 후 LSTM 가중치를 다시 연속 메모리로 모은다.

    안 하면 호출마다 cuDNN 이 재압축하며 경고를 뿜는다(EMA shadow 가 특히 그렇다).
    """
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.flatten_parameters()


def build_param_groups(policy, lr: float, backbone_scale: float):
    """사전학습 backbone 은 낮은 LR 로 따로 묶는다.

    ImageNet/DINOv2 가중치를 head 와 같은 3e-4 로 밀면 첫 수백 스텝에 사전학습 feature 가
    지워진다(우리가 그걸 쓰려고 불러온 건데). LoRA 파라미터는 **새로 초기화된 것**이라
    backbone 이 아니라 일반 그룹에 넣는다.
    """
    pre, rest = [], []
    for n, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        (pre if (".backbone." in n and "lora_" not in n) else rest).append(p)
    groups = [{"params": rest, "lr": lr, "name": "policy"}]
    if pre:
        groups.append({"params": pre, "lr": lr * backbone_scale, "name": "backbone"})
    return groups, pre, rest


# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_action_mae(policy, loader, act_norm: Normalizer, dev,
                    max_batches: int | None = None, infer_steps: int | None = None,
                    seed: int = 7) -> dict:
    """홀드아웃 액션 MAE[count]. **best 선택 기준.**

    노이즈 MSE 가 아니라 실제로 액션을 뽑아 원 단위로 재기 때문에
    diffusion/flow/bc 를 같은 자로 비교할 수 있고 배포 성능에 가깝다.

    ⚠️ 샘플링 노이즈를 **배치별로 시드 고정**한다(refer/train.py 와 동일). 안 하면
    에폭마다 지표가 흔들려서, 10% 안쪽 차이가 유의미하지 않은 데이터에서
    운 좋은 에폭이 best 로 뽑힌다.
    """
    was_training = policy.training
    policy.eval()
    errs, n = [], 0
    mean = torch.as_tensor(act_norm.mean, device=dev)
    std = torch.as_tensor(act_norm.std, device=dev)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = to_device(batch, dev)
        g = torch.Generator(device=dev).manual_seed(seed + i)
        pred = policy.sample(b["obs"], generator=g, infer_steps=infer_steps)
        pred = pred * std + mean
        gt = b["action"] * std + mean
        errs.append((pred - gt).abs().mean(dim=(1, 2)).cpu())
        n += pred.shape[0]
    policy.train(was_training)
    if not errs:
        return {"mae": float("nan"), "n": 0}
    e = torch.cat(errs)
    return {"mae": float(e.mean()), "mae_std": float(e.std()) if len(e) > 1 else 0.0, "n": n}


def save_ckpt(path, policy, ema, opt, sched, cfg, obs_norm, act_norm,
              epoch, gstep, best, prefer_weights="raw"):
    """체크포인트를 **원자적으로** 쓴다 (refer/train.py 와 동일).

    142MB 를 그대로 덮어쓰다 죽으면 last.pt/best.pt 가 깨져 재개도 배포도 못 한다.
    `lr_sched` 와 `best_metric.prefer_weights` 는 refer/ 규약이다 —
    앞은 재개 시 LR 스케줄을 이어받기 위해, 뒤는 `run.py --weights auto` 가 읽는다.
    """
    obj = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "policy": policy.state_dict(),
        "ema": ema.state_dict() if ema else None,
        "optimizer": opt.state_dict() if opt else None,
        "lr_sched": sched.state_dict() if sched else None,
        "obs_norm": {k: v.state() for k, v in obs_norm.items()},
        "act_norm": act_norm.state(),
        "epoch": epoch, "gstep": gstep, "best_mae": best,
        "best_metric": {"val_action_mae_count": best, "prefer_weights": prefer_weights},
    }
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(description="visuo-tactile diffusion policy 학습")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help="HDF5 폴더 (config 의 data.root 를 덮어씀)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--mode", choices=["preflight", "smoke", "full"], default="full")
    ap.add_argument("--override", nargs="*", default=[], help="a.b=c 형식")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--smoke_steps", type=int, default=20)
    ap.add_argument("--resume", default="auto", help="auto | none | <경로>")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None,
                    help="N 에폭마다 홀드아웃 MAE 측정 (기본 train.eval_every)")
    ap.add_argument("--eval_n", type=int, default=None,
                    help="MAE 측정에 쓸 홀드아웃 윈도우 수 (기본 train.eval_n)")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if args.data:
        cfg["data"]["root"] = args.data
    seed = args.seed if args.seed is not None else int(cfg["data"]["seed"])
    torch.manual_seed(seed); np.random.seed(seed)

    out_dir = Path(args.out_dir or f"runs/{Path(args.config).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    print("=" * 96)
    print(f"  config : {cfg['_config_path']}")
    print(f"  상속   : {' ← '.join(reversed(cfg['_base_chain']))}")
    print(f"  출력   : {out_dir}   device: {dev}   seed: {seed}")
    print("=" * 96)
    print(timing_table(cfg))
    print()

    # ── 데이터 ─────────────────────────────────────────────────────────────
    train_ds, val_ds, obs_norm, act_norm, _ = build_datasets(cfg)
    tr = cfg["train"]
    bs = int(tr["batch_size"])
    nw = args.num_workers
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=(dev.type == "cuda"), drop_last=True,
                          persistent_workers=nw > 0)

    # 홀드아웃 윈도우는 100Hz 시작점이라 거의 같은 창이 수천 개 겹친다. 전부 재면
    # eval 이 에폭 시간의 절반을 먹으면서 지표는 사실상 같다 → 균등 추출로 고정한다
    # (refer/train.py 의 --val_action_n 768 과 같은 규약).
    eval_n = int(args.eval_n if args.eval_n is not None else tr.get("eval_n", 768))
    eval_idx = np.unique(np.linspace(0, len(val_ds) - 1,
                                     min(eval_n, len(val_ds))).astype(int)).tolist()
    val_ld = DataLoader(Subset(val_ds, eval_idx), batch_size=bs, shuffle=False,
                        num_workers=nw, pin_memory=(dev.type == "cuda"),
                        persistent_workers=nw > 0)
    print(f"\n  train 윈도우 {len(train_ds):,} ({len(train_ds.demos)} 데모) | "
          f"holdout {len(val_ds):,} ({len(val_ds.demos)} 데모) → MAE 측정 {len(eval_idx):,} 개")
    for ds_, tag in ((train_ds, "train"), (val_ds, "holdout")):
        if getattr(ds_, "n_rgb_skipped", 0):
            print(f"  [RGB] {tag}: 첫 프레임 이전 구간 {ds_.n_rgb_skipped:,} 윈도우 제외 "
                  f"(미래 프레임 누출 방지)")

    # ── 모델 ───────────────────────────────────────────────────────────────
    policy = build_policy(cfg).to(dev)
    flatten_lstms(policy)
    print()
    print(policy.summary())

    # ── shape 실검증: 실제 배치 하나로 계약을 확인한다 ─────────────────────
    batch = to_device(next(iter(train_ld)), dev)
    for k, v in batch["obs"].items():
        s = cfg["obs_spec"][k]
        want = (bs, s["horizon"]) + (tuple(s["shape"]) if s["kind"] == "vision"
                                     else (int(s["shape"]),))
        assert tuple(v.shape) == want, f"obs[{k}] {tuple(v.shape)} != {want}"
    assert tuple(batch["action"].shape) == (bs, cfg["action"]["pred_horizon"],
                                            cfg["action"]["dim"])
    loss0 = policy.compute_loss(batch)
    assert torch.isfinite(loss0), "첫 배치 loss 가 NaN/Inf"
    print(f"\n  ✅ shape·loss 검증 통과 (loss={loss0.item():.4f})")

    if args.mode == "preflight":
        steps_per_epoch = len(train_ds) // bs
        t0 = time.perf_counter()
        for _ in range(3):
            policy.compute_loss(batch).backward()
        dt = (time.perf_counter() - t0) / 3
        eta = dt * steps_per_epoch * int(tr["epochs"])
        print(f"\n  스텝당 {dt*1000:.0f}ms · epoch 당 {steps_per_epoch} step · "
              f"{tr['epochs']} epoch ETA ≈ {eta/3600:.1f}시간")
        print("\n  preflight 통과 — 실제 학습은 --mode full")
        train_ds.close(); val_ds.close()
        return 0

    # ── 옵티마이저 ─────────────────────────────────────────────────────────
    epochs = 1 if args.mode == "smoke" else int(tr["epochs"])
    steps_per_epoch = max(1, len(train_ds) // bs)
    total_steps = steps_per_epoch * epochs
    groups, pre_p, rest_p = build_param_groups(
        policy, float(tr["lr"]), float(tr.get("lr_backbone_scale", 0.1)))
    opt = torch.optim.AdamW(groups, weight_decay=float(tr["weight_decay"]),
                            betas=(0.95, 0.999))
    if pre_p:
        print(f"  [LR] 사전학습 backbone {sum(p.numel() for p in pre_p)/1e6:.2f}M "
              f"@ {float(tr['lr'])*float(tr.get('lr_backbone_scale', 0.1)):.2e} · "
              f"그 외 {sum(p.numel() for p in rest_p)/1e6:.2f}M @ {float(tr['lr']):.2e}")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, int(tr["warmup_steps"]), total_steps))
    ema = EMA(policy, float(tr["ema_max_decay"])) if tr.get("use_ema", True) else None
    if ema:
        flatten_lstms(ema.shadow)

    eval_every = int(args.eval_every if args.eval_every is not None
                     else tr.get("eval_every", 5))
    eval_seed = int(tr.get("eval_seed", 7))

    start_epoch, gstep, best = 0, 0, float("inf")
    prefer = "raw"
    resume_path = (out_dir / "last.pt") if args.resume == "auto" else (
        None if args.resume == "none" else Path(args.resume))
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location=dev, weights_only=False)
        policy.load_state_dict(ck["policy"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        if ck.get("lr_sched"):
            sched.load_state_dict(ck["lr_sched"])     # 없으면 LR 이 warmup 부터 다시 시작한다
        else:
            print("  ⚠️ 체크포인트에 lr_sched 가 없다 (구버전) — LR 스케줄이 처음부터 다시 간다")
        prefer = (ck.get("best_metric") or {}).get("prefer_weights", "raw")
        start_epoch, gstep, best = ck["epoch"] + 1, ck["gstep"], ck["best_mae"]
        flatten_lstms(policy)
        if ema:
            flatten_lstms(ema.shadow)
        print(f"\n  ▶ 재개: {resume_path} (epoch {start_epoch}, best MAE {best:.1f}, "
              f"lr {sched.get_last_lr()[0]:.2e})")

    (out_dir / "config.yaml").write_text(dump_config(cfg), encoding="utf-8")
    log_path = out_dir / "log.jsonl"

    stop = {"flag": False}

    def _sigterm(*_):
        if not stop["flag"]:
            print("\n  SIGTERM 수신 — 현재 epoch 후 저장하고 종료한다")
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sigterm)

    def _mute_worker_watchdog():
        """torch 의 SIGCHLD 워치독을 끈다. **종료 경로에서만** 부른다.

        SIGTERM 은 보통 프로세스 **그룹**에 간다(timeout·kill·scancel 전부) → DataLoader
        worker 도 같이 죽는다. torch 는 worker 사망을 SIGCHLD 핸들러로 즉시 예외로
        바꾸는데, 그게 `torch.save` 의 pickling 도중에 터지면 체크포인트를 못 남긴다.
        종료 중에는 worker 사망이 더 이상 알려야 할 사고가 아니다.
        (시그널 핸들러 **안에서** signal.signal 을 부르면 pending 신호와 race 가 나서
         `OSError: Signal 17 ignored due to race condition` 이 뜬다 → 반드시 밖에서 부른다.)
        """
        if hasattr(signal, "SIGCHLD"):
            try:
                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            except (OSError, ValueError):
                pass

    # ── 학습 ───────────────────────────────────────────────────────────────
    print(f"\n{'epoch':>6s}{'train':>10s}{'MAE[cnt]':>11s}{'MAE[deg]':>10s}"
          f"{'가중치':>7s}{'lr':>10s}{'초':>7s}")
    print("  " + "-" * 60)
    try:
        for epoch in range(start_epoch, epochs):
            t0, tot, nb = time.perf_counter(), 0.0, 0
            try:
                for i, raw in enumerate(train_ld):
                    if args.mode == "smoke" and i >= args.smoke_steps:
                        break
                    b = to_device(raw, dev)
                    loss = policy.compute_loss(b)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    opt.step(); sched.step()
                    if ema:
                        ema.update(policy)
                    tot += loss.item(); nb += 1; gstep += 1
            except RuntimeError as e:
                # SIGTERM 은 보통 프로세스 **그룹**에 간다(timeout·kill·scancel 전부).
                # 그러면 DataLoader worker 가 먼저 죽고 부모는 여기서 터진다 —
                # 우아하게 끝내려던 게 crash 경로로 떨어지므로, 신호를 받은 상태면
                # 정상 종료로 취급해 지금까지 돈 만큼 저장하고 나간다.
                if not (stop["flag"] and "DataLoader worker" in str(e)):
                    raise
                print(f"  (worker 가 같은 SIGTERM 으로 종료됨 — {nb} step 까지 반영하고 저장한다)")
            if stop["flag"]:
                _mute_worker_watchdog()      # 이 뒤로는 저장만 남았다

            train_loss = tot / max(1, nb)
            # 매 에폭 재면 eval 이 에폭 시간의 절반을 먹는다 (실측 56%). refer/ 처럼 간격을 둔다.
            # SIGTERM 을 받은 뒤에는 재지 않는다 — val loader 의 worker 도 같은 신호로 죽어 있다.
            do_eval = (args.mode == "smoke" or epoch % eval_every == 0
                       or epoch == epochs - 1) and not stop["flag"]
            nvb = 2 if args.mode == "smoke" else None
            r_raw = r_ema = {"mae": float("nan")}
            use_ema, mae = False, float("inf")
            if do_eval:
                try:
                    r_raw = eval_action_mae(policy, val_ld, act_norm, dev,
                                            max_batches=nvb, seed=eval_seed)
                    r_ema = (eval_action_mae(ema.shadow, val_ld, act_norm, dev,
                                             max_batches=nvb, seed=eval_seed)
                             if ema else {"mae": float("inf")})
                    use_ema = r_ema["mae"] < r_raw["mae"]
                    mae = min(r_raw["mae"], r_ema["mae"])
                except RuntimeError as e:
                    if "DataLoader worker" not in str(e):
                        raise
                    print("  (val worker 종료 — 이 에폭 MAE 는 건너뛴다)")
                    do_eval = False
            dt = time.perf_counter() - t0

            # 1 count = pi/8192 rad → deg
            mae_s = f"{mae:>11.1f}{mae * 180.0 / 8192.0:>10.2f}" if do_eval else f"{'-':>11s}{'-':>10s}"
            print(f"{epoch:>6d}{train_loss:>10.4f}{mae_s}"
                  f"{('ema' if use_ema else 'raw') if do_eval else '-':>7s}"
                  f"{sched.get_last_lr()[0]:>10.2e}{dt:>7.1f}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "gstep": gstep, "train_loss": train_loss,
                                    "mae_raw": r_raw["mae"], "mae_ema": r_ema["mae"],
                                    "weights": ("ema" if use_ema else "raw") if do_eval else None,
                                    "lr": sched.get_last_lr()[0], "sec": dt}) + "\n")

            if mae < best:
                best, prefer = mae, "ema" if use_ema else "raw"
                save_ckpt(out_dir / "best.pt", policy, ema, opt, sched, cfg,
                          obs_norm, act_norm, epoch, gstep, best, prefer)
            save_ckpt(out_dir / "last.pt", policy, ema, opt, sched, cfg,
                      obs_norm, act_norm, epoch, gstep, best, prefer)
            if stop["flag"]:
                break

    except KeyboardInterrupt:
        print("\n  Ctrl-C — crash.pt 저장")
        save_ckpt(out_dir / "crash.pt", policy, ema, opt, sched, cfg,
                  obs_norm, act_norm, -1, gstep, best, prefer)
        return 130
    except Exception:
        save_ckpt(out_dir / "crash.pt", policy, ema, opt, sched, cfg,
                  obs_norm, act_norm, -1, gstep, best, prefer)
        raise
    finally:
        train_ds.close(); val_ds.close()
        for p in out_dir.glob("*.pt.tmp"):          # 중단된 저장의 잔여물
            try:
                p.unlink()
            except OSError:
                pass

    print("\n" + "=" * 96)
    print(f"  best 홀드아웃 액션 MAE = {best:.1f} count ({best*180/8192:.2f} deg)  "
          f"가중치 = {prefer}")
    print(f"  체크포인트: {out_dir}/best.pt  (정규화 통계 + prefer_weights 포함)")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
