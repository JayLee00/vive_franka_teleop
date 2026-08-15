#!/usr/bin/env python3
"""레코더 레이아웃과 **동일한** 합성 HDF5 를 만든다.

목적: 실데이터 없이 로더·학습 루프를 끝까지 검증한다. 여기서 통과하면 연구실에서
남는 실패는 '진짜 데이터 고유의 문제'뿐이라 원인 추적이 훨씬 쉽다.

refer/ros2_hdf5_recorder.py 의 저장 형식을 그대로 따른다:
  Demo_i/<키>            (n, dim) float32
  Demo_i/40_rgb_jpeg     (n_frames,) vlen uint8  — 고유 JPEG 프레임만
  Demo_i/41_rgb_index    (n,) int32              — 스텝 → 프레임 인덱스 (-1 = 없음)
  Demo_i/42_rgb_time     (n_frames,) float64
  Demo_i/43_rgb_stamp    (n_frames,) float64
  파일 attrs: rgb_fx/fy/cx/cy/width/height, rate_hz, ...

사용:
    python tools/make_fake_h5.py --out /tmp/fake --demos 6 --steps 1200
    python tools/make_fake_h5.py --out /tmp/fake_norgb --no-rgb   # RGB 없는 경우 재현
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

import h5py
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vtdp.data import KEY_DIMS                                        # noqa: E402

RATE_HZ = 100.0
IMG_W, IMG_H = 160, 120


def _smooth(n: int, dim: int, rng, scale: float = 1.0, hz: float = 0.7) -> np.ndarray:
    """저주파 사인 합 — 관절궤적처럼 부드러운 신호."""
    t = np.arange(n) / RATE_HZ
    out = np.zeros((n, dim), np.float32)
    for d in range(dim):
        for k in range(3):
            f = hz * (k + 1) * rng.uniform(0.6, 1.4)
            out[:, d] += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / (k + 1)
    return (out * scale).astype(np.float32)


def make_demo(g: h5py.Group, n: int, rng, with_rgb: bool, rgb_every: int = 3) -> int:
    # ── 손 관절: 부드러운 궤적 + 주기적 게이팅 ─────────────────────────────
    j_pos = 2048 + _smooth(n, 16, rng, scale=600.0)
    j_tar = j_pos + _smooth(n, 16, rng, scale=80.0, hz=1.3)     # 타겟은 현재값 근처
    kin = _smooth(n, 12, rng, scale=0.3)

    # ── 촉각: 접촉 이벤트가 있는 구간에서만 크게 튄다 ──────────────────────
    contact = (np.sin(2 * np.pi * 0.5 * np.arange(n) / RATE_HZ) > 0.2).astype(np.float32)
    ft = (_smooth(n, 12, rng, scale=0.4, hz=3.0) * contact[:, None]
          + rng.normal(0, 0.02, (n, 12))).astype(np.float32)
    raw = (np.repeat(ft, KEY_DIMS["21_paxini_raw"] // 12 + 1, axis=1)[:, :KEY_DIMS["21_paxini_raw"]]
           + rng.normal(0, 0.01, (n, KEY_DIMS["21_paxini_raw"]))).astype(np.float32)

    # ── 과일: 위생처리 임계 안에 들되 오검출 프레임을 일부러 섞는다 ────────
    fruit_pos = np.stack([0.05 + 0.01 * np.sin(np.arange(n) / 90.0),
                          0.02 + 0.01 * np.cos(np.arange(n) / 70.0),
                          0.30 + 0.01 * np.sin(np.arange(n) / 50.0)], 1).astype(np.float32)
    bad = rng.random(n) < 0.02
    fruit_pos[bad] += rng.normal(0, 0.5, (int(bad.sum()), 3))       # 튄 프레임
    fruit_size = np.tile(np.array([0.055, 0.042, 0.042], np.float32), (n, 1))
    fruit_size += rng.normal(0, 0.0005, (n, 3)).astype(np.float32)
    fruit_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], np.float32), (n, 1))  # 항등 = 각도 정보 없음

    fixed = {
        "03_hand_j_pos": j_pos, "04_hand_j_tar": j_tar, "06_hand_j_kin": kin,
        "20_paxini_ft": ft, "21_paxini_raw": raw,
        "30_fruit_pos": fruit_pos, "31_fruit_quat": fruit_quat, "32_fruit_size": fruit_size,
    }
    for k, dim in KEY_DIMS.items():
        data = fixed.get(k)
        if data is None:
            data = _smooth(n, dim, rng, scale=0.5)
        g.create_dataset(k, data=data.astype(np.float32),
                         compression="gzip", compression_opts=1)

    g.create_dataset("18_real_time_demo", data=np.arange(n) / RATE_HZ)
    g.create_dataset("19_real_time_global", data=1000.0 + np.arange(n) / RATE_HZ)

    if not with_rgb:
        return 0

    # ── RGB: 고유 프레임만 저장하고 스텝은 인덱스로 가리킨다 ───────────────
    from PIL import Image
    n_frames = max(1, n // rgb_every)
    frames, idx = [], np.zeros(n, np.int32)
    for fi in range(n_frames):
        a = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        a[..., 0] = np.linspace(0, 255, IMG_W, dtype=np.uint8)[None, :]
        a[..., 1] = np.uint8((fi * 7) % 256)
        cy, cx = IMG_H // 2, int(IMG_W / 2 + 30 * np.sin(fi / 12.0))
        yy, xx = np.ogrid[:IMG_H, :IMG_W]
        a[(yy - cy) ** 2 + (xx - cx) ** 2 < 18 ** 2] = (240, 220, 60)   # '레몬'
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=80)
        frames.append(np.frombuffer(buf.getvalue(), np.uint8))
    for i in range(n):
        idx[i] = min(i // rgb_every, n_frames - 1)

    vlen = h5py.vlen_dtype(np.uint8)
    ds = g.create_dataset("40_rgb_jpeg", (n_frames,), dtype=vlen)
    for i, b in enumerate(frames):
        ds[i] = b
    g.create_dataset("41_rgb_index", data=idx)
    g.create_dataset("42_rgb_time", data=1000.0 + np.arange(n_frames) * rgb_every / RATE_HZ)
    g.create_dataset("43_rgb_stamp", data=1000.0 + np.arange(n_frames) * rgb_every / RATE_HZ)
    g.attrs["n_rgb_frames"] = n_frames
    return n_frames


def main() -> int:
    ap = argparse.ArgumentParser(description="레코더 형식과 동일한 합성 HDF5 생성")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--demos", type=int, default=4, help="파일당 데모 수")
    ap.add_argument("--steps", type=int, default=1200, help="데모당 step (@100Hz)")
    ap.add_argument("--no-rgb", action="store_true", help="RGB 없이 (--rgb-on 안 준 경우 재현)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total_frames = 0
    for fi in range(args.files):
        path = os.path.join(args.out, f"exp_2026080{fi+1}_000000.h5")
        with h5py.File(path, "w") as f:
            f.attrs["rate_hz"] = RATE_HZ
            f.attrs["side"] = "right"
            f.attrs["start_time"] = datetime.now().isoformat()
            f.attrs["t0_perf"] = 1000.0
            f.attrs["t0_unix"] = 1785000000.0
            if not args.no_rgb:
                f.attrs["rgb"] = True
                f.attrs["rgb_topic"] = "/camera/camera/color/image_raw/compressed"
                f.attrs["rgb_every"] = 1
                f.attrs["rgb_format"] = "jpeg"
                f.attrs["rgb_width"], f.attrs["rgb_height"] = IMG_W, IMG_H
                f.attrs["rgb_fx"], f.attrs["rgb_fy"] = 190.0, 190.0
                f.attrs["rgb_cx"], f.attrs["rgb_cy"] = IMG_W / 2, IMG_H / 2
            for di in range(args.demos):
                n = args.steps + int(rng.integers(-100, 100))
                g = f.create_group(f"Demo_{di}")
                total_frames += make_demo(g, n, rng, not args.no_rgb)
            f.attrs["n_demos"] = args.demos
        print(f"  생성 {path}  데모 {args.demos}개")

    print(f"\n총 {args.files * args.demos} 데모 · "
          f"{args.files * args.demos * args.steps / 6000:.1f}분 @100Hz"
          + ("" if args.no_rgb else f" · RGB {total_frames:,} 프레임"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
