#!/usr/bin/env python3
"""로더 검증 — 합성 HDF5(레코더와 동일 형식) 위에서 돈다.

여기서 보는 건 shape 이 아니라 **의미**다:
  · 주파수 비율이 실제로 올바른 원본 인덱스를 뽑는가 (눈으로 안 보이는 부분)
  · 정규화 통계가 train split 프레임만으로 나왔는가 (누출)
  · 상수 채널이 std=1 로 처리되는가 (1e-6 폭발 방지)
  · RGB 가 없는 데이터에서 **명확한 메시지로** 실패하는가
  · 체크포인트만으로 정책과 정규화를 복원할 수 있는가

    python tests/test_data.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vtdp.config import load_config                                   # noqa: E402
from vtdp.data import (Normalizer, VTWindowDataset, build_datasets,   # noqa: E402
                       load_demos, split_demos)
from vtdp.policy import build_policy                                  # noqa: E402


def make_data(tmp: Path, with_rgb: bool = True, demos: int = 4, steps: int = 700) -> Path:
    out = tmp / ("rgb" if with_rgb else "norgb")
    cmd = [sys.executable, str(ROOT / "tools" / "make_fake_h5.py"),
           "--out", str(out), "--files", "1", "--demos", str(demos), "--steps", str(steps)]
    if not with_rgb:
        cmd.append("--no-rgb")
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def run() -> int:
    n_ok = n_fail = 0
    fails = []

    def attempt(label, fn):
        nonlocal n_ok, n_fail
        try:
            info = fn() or ""
            print(f"  ✅ {label:56s} {info}")
            n_ok += 1
        except Exception as e:
            print(f"  ❌ {label:56s} {type(e).__name__}: {e}")
            fails.append((label, traceback.format_exc()))
            n_fail += 1

    tmp = Path(tempfile.mkdtemp(prefix="vtdp_test_"))
    print(f"임시 폴더: {tmp}\n")
    root = make_data(tmp, with_rgb=True)
    root_norgb = make_data(tmp, with_rgb=False, demos=2, steps=400)

    cfg = load_config(ROOT / "configs" / "08_freq_1to3.yaml")
    cfg["data"]["root"] = str(root)
    cfg["data"]["n_held_out"] = 1

    # ── 1. 주파수 비율이 올바른 원본 인덱스를 뽑는가 ────────────────────────
    print("[1] 주파수 비율 → 실제 샘플 인덱스")

    def check_offsets():
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        ds = VTWindowDataset(demos, cfg["obs_spec"], cfg["action"],
                             default_stride=cfg["data"]["ds_stride"])
        got = {k: v.tolist() for k, v in ds.offsets.items()}
        want = {"rgb": [-6, 0], "state": [-6, 0], "tactile": [-6, -4, -2, 0]}
        assert got == want, f"offsets {got} != {want}"
        # lookback 이 모든 모달리티에서 같아야 한다(= 주파수만 바뀐 것)
        spans = {k: v[0] for k, v in got.items()}
        assert len(set(spans.values())) == 1, f"lookback 불일치: {spans}"
        return f"tactile offsets={want['tactile']} (50Hz), lookback 60ms 일치"

    attempt("08_freq_1to3 의 offset 이 1:3 인가", check_offsets)

    def check_values_match_source():
        """뽑힌 텐서가 **원본 배열의 그 인덱스**와 실제로 같은 값인지 대조."""
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        ds = VTWindowDataset(demos, cfg["obs_spec"], cfg["action"],
                             default_stride=cfg["data"]["ds_stride"])
        i = len(ds) // 2
        di, t = ds.index[i]
        item = ds[i]
        for name in ("state", "tactile"):
            offs = ds.offsets[name]
            expect = demos[di].lowdim[name][t + offs]
            assert np.allclose(item["obs"][name], expect), f"{name} 값 불일치"
        expect_a = demos[di].action[t + ds.act_offsets]
        assert np.allclose(item["action"], expect_a), "action 값 불일치"
        return f"윈도우 {i} (demo {di}, t={t}) 원본과 일치"

    attempt("뽑힌 값이 원본 인덱스의 값과 같은가", check_values_match_source)

    # ── 2. 정규화 ──────────────────────────────────────────────────────────
    print("\n[2] 정규화")

    def check_no_leak():
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        tr, hd = split_demos(demos, 1, 42)
        nz = Normalizer.fit([d.lowdim["state"] for d in tr])
        nz_all = Normalizer.fit([d.lowdim["state"] for d in demos])
        assert not np.allclose(nz.mean, nz_all.mean), \
            "train-only 통계가 전체 통계와 같다 — 홀드아웃이 섞였을 수 있다"
        assert len(tr) + len(hd) == len(demos) and not (set(id(d) for d in tr) &
                                                       set(id(d) for d in hd))
        return f"train {len(tr)} / holdout {len(hd)} 데모, 통계 분리됨"

    attempt("통계가 train split 프레임만으로 나왔는가", check_no_leak)

    def check_const_channel():
        x = np.random.randn(500, 4).astype(np.float32)
        x[:, 2] = 7.0                                  # 상수 채널
        nz = Normalizer.fit([x])
        assert nz.const_idx == [2], f"상수 채널 탐지 실패: {nz.const_idx}"
        assert nz.std[2] == 1.0, f"상수 채널 std={nz.std[2]} (1.0 이어야)"
        # 배포 때 그 채널이 조금 움직여도 폭발하지 않아야 한다
        y = x[:1].copy(); y[0, 2] = 7.01
        out = nz.normalize(y)
        assert abs(out[0, 2]) < 1.0, f"상수 채널이 폭발했다: {out[0,2]}"
        return "std=1 고정, 0.01 변화 → 정규화값 0.01 (1e-6 clip 이면 10^4)"

    attempt("상수 채널이 std=1 로 처리되는가", check_const_channel)

    def check_roundtrip():
        x = np.random.randn(100, 8).astype(np.float32) * 30 + 5
        nz = Normalizer.fit([x])
        assert np.allclose(nz.denormalize(nz.normalize(x)), x, atol=1e-3)
        return "normalize → denormalize 복원"

    attempt("정규화 왕복이 값을 보존하는가", check_roundtrip)

    # ── 3. RGB ─────────────────────────────────────────────────────────────
    print("\n[3] RGB")

    def check_rgb_shape():
        tr, va, on, an, _ = build_datasets(cfg, verbose=False)
        item = tr[0]
        s = cfg["obs_spec"]["rgb"]
        want = (s["horizon"], *s["shape"])
        assert item["obs"]["rgb"].shape == want, f"{item['obs']['rgb'].shape} != {want}"
        v = item["obs"]["rgb"]
        assert v.min() >= 0.0 and v.max() <= 1.0, f"[0,1] 범위 밖: [{v.min()}, {v.max()}]"
        assert v.std() > 1e-3, "디코드된 이미지가 상수다 — 디코드 실패 의심"
        tr.close(); va.close()
        return f"{want}, 범위 [{v.min():.2f}, {v.max():.2f}]"

    attempt("JPEG 디코드 → (T,3,224,224) [0,1]", check_rgb_shape)

    def check_norgb_error():
        c = dict(cfg); c = load_config(ROOT / "configs" / "08_freq_1to3.yaml")
        c["data"]["root"] = str(root_norgb)
        c["data"]["n_held_out"] = 1
        try:
            build_datasets(c, verbose=False)
        except KeyError as e:
            msg = str(e)
            assert "40_rgb_jpeg" in msg and "rgb-on" in msg, f"메시지가 불친절: {msg}"
            return "40_rgb_jpeg 없음 + --rgb-on 안내"
        raise AssertionError("RGB 없는 데이터인데 통과했다")

    attempt("RGB 없는 데이터가 명확히 실패하는가", check_norgb_error)

    # ── 4. 학습 경로 ───────────────────────────────────────────────────────
    print("\n[4] 학습 경로")

    def check_train_step():
        tr, va, on, an, _ = build_datasets(cfg, verbose=False)
        policy = build_policy(cfg)
        from torch.utils.data import DataLoader
        ld = DataLoader(tr, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(ld))
        loss = policy.compute_loss(batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        opt.step()
        tr.close(); va.close()
        return f"loss={loss.item():.4f}, backward+step 통과"

    attempt("실데이터 형식으로 학습 스텝이 도는가", check_train_step)

    def check_ckpt_roundtrip():
        """체크포인트만으로 정책과 정규화를 복원할 수 있어야 한다(배포 요건)."""
        tr, va, on, an, _ = build_datasets(cfg, verbose=False)
        policy = build_policy(cfg)
        p = tmp / "ck.pt"
        torch.save({"config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                    "policy": policy.state_dict(),
                    "obs_norm": {k: v.state() for k, v in on.items()},
                    "act_norm": an.state()}, p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        p2 = build_policy(ck["config"])
        p2.load_state_dict(ck["policy"])
        on2 = {k: Normalizer.from_state(v) for k, v in ck["obs_norm"].items()}
        assert set(on2) == set(on)
        for k in on:
            assert np.allclose(on2[k].mean, on[k].mean)
        item = tr[0]
        obs = {k: torch.as_tensor(v)[None] for k, v in item["obs"].items()}
        p2.eval()
        with torch.no_grad():
            a = p2.sample(obs, infer_steps=2)
        assert a.shape == (1, cfg["action"]["pred_horizon"], cfg["action"]["dim"])
        tr.close(); va.close()
        return f"복원 후 sample {tuple(a.shape)}"

    attempt("체크포인트만으로 정책+정규화 복원", check_ckpt_roundtrip)

    # ── 5. 윈도우 경계 ─────────────────────────────────────────────────────
    print("\n[5] 윈도우 경계")

    def check_bounds():
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        ds = VTWindowDataset(demos, cfg["obs_spec"], cfg["action"],
                             default_stride=cfg["data"]["ds_stride"])
        for di, t in (ds.index[0], ds.index[-1]):
            d = demos[di]
            for name, offs in ds.offsets.items():
                assert (t + offs).min() >= 0, f"{name} 인덱스가 음수"
            assert (t + ds.act_offsets).max() <= d.n - 1, "action 인덱스가 범위 밖"
        # 전 윈도우를 훑어 범위 위반이 없는지
        for di, t in ds.index:
            d = demos[di]
            assert t - ds.back >= 0 and t + ds.fwd <= d.n - 1
        return f"{len(ds):,} 윈도우 전부 범위 안 (back={ds.back}, fwd={ds.fwd})"

    attempt("모든 윈도우가 배열 범위 안인가", check_bounds)

    def check_too_short():
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        for d in demos:
            d.n = 10                                    # 일부러 너무 짧게
        try:
            VTWindowDataset(demos, cfg["obs_spec"], cfg["action"], default_stride=5)
        except RuntimeError as e:
            assert "너무 짧다" in str(e)
            return "필요 최소 길이를 알려주며 거부"
        raise AssertionError("짧은 데모인데 통과했다")

    attempt("데모가 너무 짧으면 이유를 알려주는가", check_too_short)

    def check_rgb_no_future_leak():
        """41_rgb_index == -1 구간(카메라가 늦게 켜짐)이 미래 프레임을 끌어오면 안 된다.

        실데이터에는 데모 시작에 -1 이 있다(레코더가 첫 프레임 전까지 -1 을 쓴다).
        합성 데이터는 -1 을 안 만들기 때문에 여기서 직접 심어 확인한다.
        """
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        d = demos[0]
        d.rgb_index = d.rgb_index.copy()
        n_dead = 40
        d.rgb_index[:n_dead] = -1
        ds = VTWindowDataset([d], cfg["obs_spec"], cfg["action"], default_stride=5)
        back = ds.back
        # 모든 윈도우의 모든 관측 시각에 이미 유효 프레임이 있어야 한다
        for di, t in ds.index:
            for off in ds.offsets["rgb"]:
                assert d.rgb_index[t + off] >= 0, \
                    f"t={t} off={off} 에서 rgb_index=-1 인데 윈도우에 남았다 (미래 프레임을 끌어온다)"
        assert ds.index[0][1] >= n_dead + back, \
            f"첫 윈도우 t={ds.index[0][1]} 가 -1 구간({n_dead}) + lookback({back}) 보다 이르다"
        assert ds.n_rgb_skipped > 0, "제외된 윈도우 수가 기록되지 않았다"
        ds.close()
        return (f"-1 구간 {n_dead}step → 윈도우 {ds.n_rgb_skipped}개 제외, "
                f"첫 t={ds.index[0][1]} (≥{n_dead + back})")

    attempt("RGB 없는 시작 구간이 미래 프레임을 끌어오지 않는가", check_rgb_no_future_leak)

    def check_n_held_out_zero():
        demos, _ = load_demos(str(root), cfg["obs_spec"], cfg["action"]["key"], verbose=False)
        try:
            split_demos(demos, 0, 42)
        except ValueError as e:
            assert "n_held_out" in str(e), f"메시지가 불친절: {e}"
            return "이유를 설명하며 거부 (예전엔 max() 빈 시퀀스 에러)"
        raise AssertionError("n_held_out=0 인데 통과했다")

    attempt("n_held_out=0 이 명확히 거부되는가", check_n_held_out_zero)

    print("\n" + "=" * 78)
    print(f"  통과 {n_ok} / 실패 {n_fail}")
    print("=" * 78)
    for label, tb in fails:
        print(f"\n──── {label} ────\n{tb}")
    return 1 if n_fail else 0


def test_data():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
