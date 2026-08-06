#!/usr/bin/env python3
"""configs/*.yaml 이 전부 로드·검증되고 실제로 조립·학습스텝까지 도는지 확인한다.

    python tests/test_configs.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vtdp.config import CONFIG_DIR, load_config, timing_table   # noqa: E402
from vtdp.policy import build_policy                     # noqa: E402

B = 2


def fake_batch(cfg: dict) -> dict:
    obs = {}
    for name, s in cfg["obs_spec"].items():
        if s["kind"] == "vision":
            obs[name] = torch.rand(B, s["horizon"], *s["shape"])
        else:
            obs[name] = torch.randn(B, s["horizon"], s["shape"])
    return {"obs": obs,
            "action": torch.randn(B, cfg["action"]["pred_horizon"], cfg["action"]["dim"])}


def run() -> int:
    files = sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.name.startswith("_"))
    if not files:
        print("configs/*.yaml 이 없다")
        return 1

    n_ok = n_fail = 0
    for path in files:
        try:
            cfg = load_config(path)
            policy = build_policy(cfg)
            batch = fake_batch(cfg)

            loss = policy.compute_loss(batch)
            assert loss.ndim == 0 and torch.isfinite(loss), f"loss={loss}"
            loss.backward()

            # 실제 optimizer 스텝까지 — 학습이 물리적으로 도는지 확인
            opt = torch.optim.AdamW(policy.parameters(), lr=float(cfg["train"]["lr"]))
            opt.step(); opt.zero_grad()

            policy.eval()
            with torch.no_grad():
                act = policy.sample(batch["obs"], infer_steps=2)
            assert act.shape == batch["action"].shape, f"sample {tuple(act.shape)}"

            mods = ", ".join(sorted(cfg["obs_spec"]))
            chain = " ← ".join(reversed(cfg["_base_chain"]))
            print(f"  ✅ {path.name:26s} [{mods:24s}] "
                  f"{policy.n_params()/1e6:6.2f}M  loss={loss.item():.4f}")
            print(f"     {'':26s} 상속: {chain}")
            for ln in timing_table(cfg).splitlines()[1:]:
                print(f"     {ln}")
            n_ok += 1
        except Exception as e:
            print(f"  ❌ {path.name:26s} {type(e).__name__}: {e}")
            traceback.print_exc()
            n_fail += 1

    # ── 기준선 파라미터 고정 ──────────────────────────────────────────────
    # refer/dp_unet 과 파라미터가 정확히 같아야 한다. 이게 깨지면 모든 비교의 원점이
    # 사라지는데 지금까지는 테스트가 강제하지 않아 사람이 기억해야 했다.
    print("\n[기준선 재현] refer/ 와 파라미터가 같아야 한다")
    REFER_DP_UNET = 2_469_200
    try:
        n = build_policy(load_config(CONFIG_DIR / "10_baseline_refer.yaml")).n_params()
        assert n == REFER_DP_UNET, f"{n:,} != refer dp_unet {REFER_DP_UNET:,}"
        print(f"  ✅ {'10_baseline_refer':26s} {n:,} = refer/dp_unet")
        n_ok += 1
    except Exception as e:
        print(f"  ❌ {'10_baseline_refer':26s} {type(e).__name__}: {e}")
        n_fail += 1

    # ── vision backbone: frozen 과 정규화가 짝지어지는지 ──────────────────
    # frozen 인데 BN→GN 치환을 하면 사전학습 통계가 버려져 feature 가 거의 상수가 된다
    # (실측: 이미지-의존 성분 72.8% → 15.3%). frozen 이면 BN 유지 + eval 고정이어야 한다.
    print("\n[vision] frozen 이면 BN 유지 · 파인튜닝이면 GN 치환")
    try:
        import torch.nn as nn

        from vtdp.encoders import VisionResNet18
        fz = VisionResNet18((3, 224, 224), 2, d_model=64, pretrained=False, frozen=True).train()
        ft = VisionResNet18((3, 224, 224), 2, d_model=64, pretrained=False, frozen=False).train()
        cnt = lambda m, t: sum(isinstance(x, t) for x in m.modules())    # noqa: E731
        assert fz.norm_kind == "bn" and cnt(fz, nn.BatchNorm2d) > 0 and cnt(fz, nn.GroupNorm) == 0
        assert not fz.backbone.training, "frozen backbone 이 train 모드다 (BN stat 이 갱신된다)"
        assert ft.norm_kind == "gn" and cnt(ft, nn.BatchNorm2d) == 0 and cnt(ft, nn.GroupNorm) > 0
        assert ft.backbone.training, "파인튜닝 backbone 이 eval 모드다"
        print(f"  ✅ {'norm=auto':26s} frozen→bn(eval 고정) {cnt(fz, nn.BatchNorm2d)}개 · "
              f"파인튜닝→gn {cnt(ft, nn.GroupNorm)}개")
        n_ok += 1
    except Exception as e:
        print(f"  ❌ {'norm=auto':26s} {type(e).__name__}: {e}")
        n_fail += 1

    # ── 촉각 인코더가 손가락 identity 와 시간축을 남기는지 ────────────────
    print("\n[촉각] 부위 identity · 시간축 보존")
    try:
        from vtdp.encoders import TactileLSTM, TactileTransformer
        for cls, kw in ((TactileLSTM, dict(n_part=4, return_seq=True)),
                        (TactileTransformer, dict(n_part=4, use_cls=False, n_layer=1))):
            enc = cls(in_dim=12, horizon=4, d_model=32, **kw).eval()
            assert enc.n_tokens == 16, f"{cls.__name__}: n_tokens={enc.n_tokens} != 4손가락x4스텝"
            with torch.no_grad():
                base = enc(torch.zeros(1, 4, 12))
                x = torch.zeros(1, 4, 12); x[..., 0:3] = 3.0        # 손가락 0 만 자극
                delta = (enc(x) - base).abs().mean(-1)[0]
            assert delta.max() > 1e-3, f"{cls.__name__}: 자극에 반응이 없다"
            # 손가락별 응답이 균일하면 identity 가 사라진 것이다
            spread = float(delta.max() / delta.mean())
            assert spread > 1.5, (f"{cls.__name__}: 토큰 응답이 균일하다(spread={spread:.2f}) "
                                 f"— 손가락 identity 가 첫 Linear 에서 섞였다")
            print(f"  ✅ {cls.__name__:26s} n_tokens=16, 손가락0 자극 응답 편차 {spread:.1f}x")
            n_ok += 1
        # in_dim 이 n_part 로 안 나뉘면 생성 시점에 거부해야 한다
        try:
            TactileLSTM(in_dim=13, horizon=2, d_model=32, n_part=4)
            print(f"  ❌ {'n_part 불일치':26s} 통과해버렸다")
            n_fail += 1
        except ValueError:
            print(f"  ✅ {'n_part 불일치':26s} 생성 시점에 거부됨")
            n_ok += 1
    except Exception as e:
        print(f"  ❌ {'촉각 부위 토큰':26s} {type(e).__name__}: {e}")
        traceback.print_exc()
        n_fail += 1

    # 검증이 실제로 오류를 잡는지 — 통과하면 안 되는 것들
    print("\n[검증 로직 확인] 아래는 거부되어야 정상")
    bad_cases = [
        ("없는 fusion 이름", ["model.fusion=nope"]),
        ("exec > pred horizon", ["action.exec_horizon=99"]),
        ("modality_dropout 범위 밖", ["model.modality_dropout=1.5"]),
        ("없는 override 경로", ["model.nonexistent=1"]),
    ]
    for label, ov in bad_cases:
        try:
            load_config(files[0], overrides=ov)
            print(f"  ❌ {label:26s} 통과해버렸다 (거부됐어야 함)")
            n_fail += 1
        except (ValueError, KeyError) as e:
            msg = str(e).splitlines()[0][:70]
            print(f"  ✅ {label:26s} 거부됨 — {msg}")
            n_ok += 1

    print("\n" + "=" * 80)
    print(f"  통과 {n_ok} / 실패 {n_fail}")
    print("=" * 80)
    return 1 if n_fail else 0


def test_configs():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
