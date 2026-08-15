#!/usr/bin/env python3
"""조합 곱집합 shape 테스트 — 데이터도 GPU 도 없이 돈다.

docs/SHAPES.md 의 불변식 5개를 전부 강제한다:
  1. 인코더 출력은 (B, n_tokens, D)
  2. fusion 출력은 (tokens (B,N,D), pooled (B,C))
  3. head.sample() 은 (B, T_p, A)
  4. compute_loss() 는 0-dim scalar 이고 backward 가 통한다
  5. 모달리티를 빼도 1~4 가 성립한다

**연구실 가기 전에 이게 다 통과해야 한다.** 여기서 걸리는 버그는 전부
'실제 데이터와 무관한 버그'라 여기서 잡는 게 압도적으로 싸다.

    python -m pytest tests/test_shapes.py -q          # pytest 있으면
    python tests/test_shapes.py                       # 없어도 그냥 돈다
"""
from __future__ import annotations

import itertools
import sys
import traceback
from pathlib import Path

import torch

# Windows 콘솔 기본이 cp949 라 한글/기호 출력에서 죽는다. Linux 에서는 무해.
for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtdp import VTDPolicy, available, describe   # noqa: E402

B = 2
ACTION_DIM = 16
PRED_HORIZON = 16
D_MODEL = 128          # 테스트는 작게 — 조합 수가 많다
IMG = (3, 96, 96)      # 96 = 32*3 이라 resnet stride-32 후 3x3

# ── 모달리티 정의 (실제 하드웨어 값에 맞춤) ──────────────────────────────────
STATE = dict(kind="state", shape=28, horizon=2)          # hand_j_pos(16) + kin(12)
TACTILE = dict(kind="tactile", shape=12, horizon=8)      # paxini_ft, 100Hz 라 더 길게
VISION = dict(kind="vision", shape=IMG, horizon=2)

STATE_ENCODERS = ["mlp"]
TACTILE_ENCODERS = ["mlp", "conv1d", "lstm", "transformer"]
VISION_ENCODERS = ["smallcnn", "resnet18", "dinov2"]
FUSIONS = ["concat", "passthrough", "cross_attn", "gated"]
DENOISERS = ["unet1d", "dit", "transformer"]
HEADS = ["diffusion", "flow", "bc"]


def make_obs(spec: dict) -> dict:
    out = {}
    for name, s in spec.items():
        if s["kind"] == "vision":
            out[name] = torch.rand(B, s["horizon"], *s["shape"])
        else:
            out[name] = torch.randn(B, s["horizon"], s["shape"])
    return out


def check(policy: VTDPolicy, spec: dict) -> str:
    obs = make_obs(spec)

    # 불변식 1·2 — 인코더/fusion 출력
    for name, enc in policy.encoders.items():
        z = enc(obs[name])
        assert z.ndim == 3, f"{name}: 출력이 3차원이 아니다 {tuple(z.shape)}"
        assert z.shape == (B, enc.n_tokens, policy.d_model), \
            f"{name}: {tuple(z.shape)} != {(B, enc.n_tokens, policy.d_model)}"

    tokens, pooled = policy(obs)
    assert tokens.shape == (B, policy.fusion.n_cond_tokens, policy.d_model), \
        f"fusion tokens {tuple(tokens.shape)}"
    assert pooled.shape == (B, policy.fusion.cond_dim), f"fusion pooled {tuple(pooled.shape)}"

    # 불변식 4 — loss 는 scalar 이고 backward 가 통한다
    action = torch.randn(B, PRED_HORIZON, ACTION_DIM)
    loss = policy.compute_loss({"obs": obs, "action": action})
    assert loss.ndim == 0, f"loss 가 scalar 가 아니다: {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"loss 가 NaN/Inf: {loss}"
    loss.backward()
    n_grad = sum(1 for p in policy.parameters() if p.requires_grad and p.grad is not None)
    assert n_grad > 0, "gradient 가 하나도 안 흘렀다"

    # 불변식 3 — sample
    policy.eval()
    with torch.no_grad():
        act = policy.sample(obs, infer_steps=2)
    assert act.shape == (B, PRED_HORIZON, ACTION_DIM), f"sample {tuple(act.shape)}"
    assert torch.isfinite(act).all(), "sample 에 NaN/Inf"
    policy.train()

    return f"{policy.n_params()/1e6:6.2f}M  loss={loss.item():.4f}"


def build_and_check(spec: dict, fusion: str, denoiser: str, head: str,
                    fusion_kwargs: dict | None = None, modality_dropout: float = 0.0) -> str:
    p = VTDPolicy(obs_spec=spec, action_dim=ACTION_DIM, pred_horizon=PRED_HORIZON,
                  d_model=D_MODEL, cond_dim=D_MODEL,
                  fusion=fusion, fusion_kwargs=fusion_kwargs,
                  denoiser=denoiser, head=head,
                  denoiser_kwargs={"channels": (32, 64, 128)} if denoiser == "unet1d" else
                                  {"n_layer": 2, "n_head": 4},
                  modality_dropout=modality_dropout)
    return check(p, spec)


# ══════════════════════════════════════════════════════════════════════════
def run() -> int:
    torch.manual_seed(0)
    n_ok = n_fail = 0
    failures: list[tuple[str, str]] = []

    def attempt(label: str, fn):
        nonlocal n_ok, n_fail
        try:
            info = fn()
            print(f"  ✅ {label:62s} {info}")
            n_ok += 1
        except Exception as e:
            print(f"  ❌ {label:62s} {type(e).__name__}: {e}")
            failures.append((label, traceback.format_exc()))
            n_fail += 1

    print("등록된 모듈:")
    print(describe())

    # ── A. 모달리티 조합 (불변식 5) ─────────────────────────────────────────
    print("\n[A] 모달리티 조합 — concat + unet1d + diffusion")
    combos = {
        "state only":               {"state": {**STATE, "encoder": "mlp"}},
        "tactile only":             {"tactile": {**TACTILE, "encoder": "conv1d"}},
        "rgb only":                 {"rgb": {**VISION, "encoder": "smallcnn"}},
        "state+tactile":            {"state": {**STATE, "encoder": "mlp"},
                                     "tactile": {**TACTILE, "encoder": "conv1d"}},
        "state+rgb":                {"state": {**STATE, "encoder": "mlp"},
                                     "rgb": {**VISION, "encoder": "smallcnn"}},
        "rgb+tactile":              {"rgb": {**VISION, "encoder": "smallcnn"},
                                     "tactile": {**TACTILE, "encoder": "conv1d"}},
        "state+rgb+tactile":        {"state": {**STATE, "encoder": "mlp"},
                                     "rgb": {**VISION, "encoder": "smallcnn"},
                                     "tactile": {**TACTILE, "encoder": "conv1d"}},
    }
    for label, spec in combos.items():
        attempt(label, lambda s=spec: build_and_check(s, "concat", "unet1d", "diffusion"))

    # ── B. 인코더 스윕 ─────────────────────────────────────────────────────
    print("\n[B] 인코더 스윕 — concat + unet1d + diffusion")
    for e in STATE_ENCODERS:
        spec = {"state": {**STATE, "encoder": e}, "tactile": {**TACTILE, "encoder": "mlp"}}
        attempt(f"state.{e}", lambda s=spec: build_and_check(s, "concat", "unet1d", "diffusion"))
    for e in TACTILE_ENCODERS:
        spec = {"state": {**STATE, "encoder": "mlp"}, "tactile": {**TACTILE, "encoder": e}}
        attempt(f"tactile.{e}", lambda s=spec: build_and_check(s, "concat", "unet1d", "diffusion"))
    for e in VISION_ENCODERS:
        v = dict(VISION)
        if e == "dinov2":
            v["shape"] = (3, 224, 224)        # patch 14 의 배수여야 한다
        spec = {"state": {**STATE, "encoder": "mlp"}, "rgb": {**v, "encoder": e}}
        attempt(f"vision.{e}", lambda s=spec: build_and_check(s, "concat", "unet1d", "diffusion"))
    # patch 토큰 (cross-attn 용)
    spec = {"state": {**STATE, "encoder": "mlp"},
            "rgb": {**VISION, "encoder": "resnet18",
                    "encoder_kwargs": {"tokens": "patch", "pretrained": False}}}
    attempt("vision.resnet18(patch)",
            lambda s=spec: build_and_check(s, "cross_attn", "transformer", "diffusion"))

    # ── C. fusion × denoiser × head 곱집합 ─────────────────────────────────
    print("\n[C] fusion × denoiser × head")
    full = {"state": {**STATE, "encoder": "mlp"},
            "rgb": {**VISION, "encoder": "smallcnn"},
            "tactile": {**TACTILE, "encoder": "conv1d"}}
    for fu, de, hd in itertools.product(FUSIONS, DENOISERS, HEADS):
        if hd == "bc" and de != "unet1d":
            continue                                  # bc 는 디노이저를 안 쓴다 — 1회만
        fk = {"gate_on": "tactile"} if fu == "gated" else None
        attempt(f"{fu} + {de} + {hd}",
                lambda f=fu, d=de, h=hd, k=fk: build_and_check(full, f, d, h, fusion_kwargs=k))

    # ── D. modality dropout ────────────────────────────────────────────────
    print("\n[D] modality dropout (p=0.2) — 학습 시 랜덤 마스킹")
    attempt("dropout p=0.2",
            lambda: build_and_check(full, "concat", "unet1d", "diffusion", modality_dropout=0.2))
    attempt("dropout p=1.0 (전부 가려도 최소 1개는 살아야)",
            lambda: build_and_check(full, "concat", "unet1d", "diffusion", modality_dropout=1.0))

    # ── E. 실패해야 하는 것들 ──────────────────────────────────────────────
    print("\n[E] 오류를 제대로 내는가 (실패가 정상)")

    def expect_raise(fn, exc):
        try:
            fn()
        except exc:
            return "정상적으로 거부됨"
        raise AssertionError(f"{exc.__name__} 가 안 났다")

    attempt("없는 인코더 이름",
            lambda: expect_raise(
                lambda: build_and_check({"state": {**STATE, "encoder": "nope"}},
                                        "concat", "unet1d", "diffusion"), KeyError))
    attempt("taxel_cnn 레이아웃 불일치",
            lambda: expect_raise(
                lambda: build_and_check(
                    {"tactile": {**TACTILE, "encoder": "taxel_cnn"}},
                    "concat", "unet1d", "diffusion"), (ValueError, TypeError)))
    attempt("gated gate_on 이 없는 모달리티",
            lambda: expect_raise(
                lambda: build_and_check({"state": {**STATE, "encoder": "mlp"}},
                                        "gated", "unet1d", "diffusion",
                                        fusion_kwargs={"gate_on": "tactile"}), KeyError))
    attempt("dinov2 입력이 14의 배수가 아님",
            lambda: expect_raise(
                lambda: build_and_check(
                    {"rgb": {**VISION, "encoder": "dinov2"}},   # 96 은 14로 안 나뉜다
                    "concat", "unet1d", "diffusion"), ValueError))

    # ── 결과 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  통과 {n_ok} / 실패 {n_fail}")
    print("=" * 80)
    if failures:
        print("\n실패 상세:")
        for label, tb in failures:
            print(f"\n──── {label} ────\n{tb}")
    return 1 if n_fail else 0


# pytest 로도 돌게
def test_all_shapes():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
