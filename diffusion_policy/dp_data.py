#!/usr/bin/env python3
"""데이터 로딩 — HDF5 데모 → 클램프 → 위생처리 → 슬라이딩 윈도우 → 데모 단위 분할.

설계 선택 3가지:

1) **데모 단위 분할** (윈도우 단위 아님)
   같은 데모에서 뽑은 윈도우는 서로 크게 겹치므로, 윈도우를 랜덤 분할하면 val 이
   train 을 거의 그대로 보게 되어 val loss 가 낙관적으로 나온다. 홀드아웃 데모를
   따로 떼어 open-loop rollout 까지 그 데모로 평가한다.

2) **dilated 윈도우 + dense start**
   원본 100Hz 에서 PRED_HORIZON=16 은 0.16초라 의미가 없다. 윈도우 내부만 stride 5
   로 샘플해 0.8초 지평을 만들고, 윈도우 시작점은 100Hz 매 스텝으로 둔다.
   → 20Hz 로 다운샘플하면서도 5가지 위상을 모두 학습에 쓴다(윈도우 5배).

3) **정규화 통계는 train split 에서만**
   val/홀드아웃 프레임은 mean/std 계산에 넣지 않는다(누출 방지). 계산된 통계는
   npz 와 체크포인트 양쪽에 저장해 배포 시 같은 값을 쓴다.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from dp_config import (ACTION_DIM, ACTION_KEY, CLAMP_SPEC, DATA_ROOT,
                       DS_STRIDE, FRUIT_POS_MAX_JUMP, FRUIT_POS_MAX_NORM,
                       FRUIT_SIZE_RANGE, OBS_DIM, OBS_KEYS, OBS_SPEC,
                       PRED_HORIZON)

DEAD_ACTION_STD = 5.0     # 관절별 action std 최대가 이 미만이면 '정지 데모'


# ══════════════════════════════════════════════════════════════════════════
# 위생 처리
# ══════════════════════════════════════════════════════════════════════════
def sanitize_fruit(pos: np.ndarray, size: np.ndarray, quat: np.ndarray = None):
    """과일 오검출 프레임을 직전 유효값으로 홀드.

    위치가 못 믿을 프레임이면 같은 프레임의 방향도 못 믿는다 → quat 도 같이 홀드한다.
    반환: (pos, size, quat, 고친 프레임 수).  quat 가 None 이면 None 을 그대로 돌려준다.
    """
    pos = pos.copy()
    size = size.copy()
    quat = None if quat is None else quat.copy()
    n = len(pos)
    bad = np.zeros(n, dtype=bool)

    norm = np.linalg.norm(pos, axis=1)
    bad |= norm > FRUIT_POS_MAX_NORM
    bad |= norm < 1e-9                                    # 미수신(0,0,0)
    lo, hi = FRUIT_SIZE_RANGE
    bad |= (size[:, 0] < lo) | (size[:, 0] > hi)

    # 프레임간 점프: 직전 유효 프레임 기준으로 순차 판정
    last = None
    for i in range(n):
        if not bad[i] and last is not None:
            if np.linalg.norm(pos[i] - pos[last]) > FRUIT_POS_MAX_JUMP:
                bad[i] = True
        if not bad[i]:
            last = i

    # 홀드(앞이 없으면 뒤에서 당겨옴)
    fixed = int(bad.sum())
    if fixed:
        good = np.flatnonzero(~bad)
        if len(good) == 0:
            return pos, size, quat, fixed                  # 전부 불량 → 그대로
        for i in np.flatnonzero(bad):
            prev = good[good < i]
            src = prev[-1] if len(prev) else good[0]
            pos[i] = pos[src]
            size[i] = size[src]
            if quat is not None:
                quat[i] = quat[src]
    return pos, size, quat, fixed


# ══════════════════════════════════════════════════════════════════════════
# 데모 로딩
# ══════════════════════════════════════════════════════════════════════════
class Demo:
    """클램프·위생처리까지 끝난 하나의 에피소드."""

    __slots__ = ("name", "obs", "act", "n_raw", "n_used", "fruit_fixed")

    def __init__(self, name, obs, act, n_raw, fruit_fixed):
        self.name = name
        self.obs = obs            # (n_used, OBS_DIM) float32
        self.act = act            # (n_used, ACTION_DIM) float32
        self.n_raw = n_raw
        self.n_used = len(obs)
        self.fruit_fixed = fruit_fixed


def load_demos(data_root: str = DATA_ROOT,
               clamp_spec: dict = None,
               keep_dead: bool = False,
               verbose: bool = True) -> tuple[list[Demo], list[dict]]:
    """CLAMP_SPEC 대로 데모를 읽어 Demo 리스트와 요약 테이블(행 dict)을 만든다."""
    clamp_spec = clamp_spec or CLAMP_SPEC
    demos: list[Demo] = []
    table: list[dict] = []

    for fname, spec in clamp_spec.items():
        path = os.path.join(data_root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            for idx in sorted(spec):
                key = f"Demo_{idx}"
                if key not in f:
                    raise KeyError(f"{fname}:{key} 없음")
                g = f[key]
                n_raw = g[ACTION_KEY].shape[0]
                clamp = spec[idx]

                row = {"file": fname, "demo": idx, "n_raw": n_raw,
                       "note": clamp if clamp is not None else "full",
                       "n_used": 0, "seconds": 0.0, "fruit_fixed": 0,
                       "status": "", "act_std": 0.0}

                if clamp == "DELETE":
                    row["status"] = "삭제(노트)"
                    table.append(row)
                    continue

                n_use = n_raw if clamp in (None, "DEAD") else int(clamp)
                if n_use > n_raw:
                    raise ValueError(f"{fname}:{key} 클램프 {n_use} > N {n_raw}")

                act = np.asarray(g[ACTION_KEY][:n_use], dtype=np.float32)
                act_std = float(act.std(axis=0).max())
                row["act_std"] = act_std

                if clamp == "DEAD" or act_std < DEAD_ACTION_STD:
                    row["status"] = f"제외(정지, act_std={act_std:.2f})"
                    if not keep_dead:
                        table.append(row)
                        continue
                    row["status"] += " [keep_dead]"

                parts = []
                for k, d in OBS_SPEC:
                    a = np.asarray(g[k][:n_use], dtype=np.float32)
                    if a.ndim == 1:
                        a = a[:, None]
                    if a.shape[1] != d:
                        raise ValueError(f"{fname}:{key}:{k} dim {a.shape[1]} != {d}")
                    parts.append(a)
                pos_i = OBS_KEYS.index("30_fruit_pos")
                size_i = OBS_KEYS.index("32_fruit_size")
                q_i = OBS_KEYS.index("31_fruit_quat") if "31_fruit_quat" in OBS_KEYS else None
                parts[pos_i], parts[size_i], _q, fixed = sanitize_fruit(
                    parts[pos_i], parts[size_i],
                    parts[q_i] if q_i is not None else None)
                if q_i is not None:
                    parts[q_i] = _q

                obs = np.concatenate(parts, axis=1)
                assert obs.shape == (n_use, OBS_DIM), obs.shape
                assert act.shape == (n_use, ACTION_DIM), act.shape
                if not (np.isfinite(obs).all() and np.isfinite(act).all()):
                    raise ValueError(f"{fname}:{key} NaN/Inf")

                row.update(n_used=n_use, seconds=n_use / 100.0, fruit_fixed=fixed,
                           status=row["status"] or "사용")
                table.append(row)
                demos.append(Demo(f"{fname}:{key}", obs, act, n_raw, fixed))

    if verbose:
        print_clamp_table(table)
    if not demos:
        raise RuntimeError("사용 가능한 데모가 없음")
    return demos, table


def print_clamp_table(table: list[dict]):
    print("=" * 108)
    print("  데모별 클램프 요약")
    print("=" * 108)
    print(f"  {'파일':<26s} {'demo':>5s} {'N_raw':>7s} {'노트':>7s} {'사용':>7s} "
          f"{'초':>6s} {'act_std':>8s} {'fruit보정':>9s}  상태")
    print("  " + "-" * 104)
    tot_used = tot_fixed = 0
    n_ok = 0
    for r in table:
        note = r["note"] if r["note"] != "full" else "full"
        print(f"  {r['file']:<26s} {r['demo']:>5d} {r['n_raw']:>7d} {str(note):>7s} "
              f"{r['n_used']:>7d} {r['seconds']:>6.1f} {r['act_std']:>8.2f} "
              f"{r['fruit_fixed']:>9d}  {r['status']}")
        if r["status"].startswith("사용"):
            n_ok += 1
            tot_used += r["n_used"]
            tot_fixed += r["fruit_fixed"]
    print("  " + "-" * 104)
    print(f"  사용 데모 {n_ok}개 | {tot_used:,} steps @100Hz = {tot_used/100:.0f}s "
          f"= {tot_used/6000:.1f}분 | 과일 보정 프레임 {tot_fixed:,} "
          f"({100*tot_fixed/max(1,tot_used):.2f}%)")
    print("=" * 108)


# ══════════════════════════════════════════════════════════════════════════
# 정규화
# ══════════════════════════════════════════════════════════════════════════
class Normalizer:
    """per-feature z-score. 통계는 반드시 train split 프레임만으로 계산한다."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, arrays: list[np.ndarray], eps: float = 1e-6):
        cat = np.concatenate(arrays, axis=0)
        return cls(cat.mean(0), cat.std(0).clip(eps))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["std"])

    def state(self):
        return {"mean": self.mean, "std": self.std}


# ══════════════════════════════════════════════════════════════════════════
# 윈도우 데이터셋
# ══════════════════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    """dilated 슬라이딩 윈도우.

    시작점 t (원본 100Hz 인덱스) 하나에 대해
        obs  = obs[t - (T_obs-1-k)*stride]  k=0..T_obs-1     → (T_obs,  OBS_DIM)
        act  = act[t + k*stride]            k=0..T_pred-1    → (T_pred, ACTION_DIM)
    유효 t : (T_obs-1)*stride  ≤  t  ≤  N-1 - (T_pred-1)*stride
    """

    def __init__(self, demos: list[Demo], obs_horizon: int, pred_horizon: int,
                 stride: int = DS_STRIDE,
                 obs_norm: Normalizer = None, act_norm: Normalizer = None):
        self.demos = demos
        self.T_obs = obs_horizon
        self.T_pred = pred_horizon
        self.stride = stride
        self.obs_norm = obs_norm
        self.act_norm = act_norm

        back = (obs_horizon - 1) * stride
        fwd = (pred_horizon - 1) * stride
        self.index: list[tuple[int, int]] = []
        for di, d in enumerate(demos):
            t_lo, t_hi = back, d.n_used - 1 - fwd
            if t_hi < t_lo:
                continue
            self.index.extend((di, t) for t in range(t_lo, t_hi + 1))
        if not self.index:
            raise RuntimeError(
                f"윈도우 0개: 데모가 너무 짧다 (필요 최소 길이 {back+fwd+1} steps)")

        self._obs_off = np.array([-(obs_horizon - 1 - k) * stride
                                  for k in range(obs_horizon)], dtype=np.int64)
        self._act_off = np.arange(pred_horizon, dtype=np.int64) * stride

    def __len__(self):
        return len(self.index)

    def raw(self, i):
        di, t = self.index[i]
        d = self.demos[di]
        return d.obs[t + self._obs_off], d.act[t + self._act_off]

    def __getitem__(self, i):
        obs, act = self.raw(i)
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize(obs)
            act = self.act_norm.normalize(act)
        return obs.astype(np.float32), act.astype(np.float32)


def split_demos(demos: list[Demo], n_held_out: int, seed: int
                ) -> tuple[list[Demo], list[Demo]]:
    """데모 단위 분할. 시드 고정이라 재실행 시 같은 홀드아웃."""
    order = np.random.RandomState(seed).permutation(len(demos))
    held = [demos[i] for i in order[:n_held_out]]
    train = [demos[i] for i in order[n_held_out:]]
    if not train:
        raise ValueError("train 데모가 0개")
    return train, held
