#!/usr/bin/env python3
"""HDF5 → 모달리티별 윈도우 데이터셋.

refer/dp_data.py 의 설계 3가지를 그대로 유지한다(검증된 것들이라 바꿀 이유가 없다):
  1. **데모 단위 분할** — 윈도우 단위로 나누면 val 이 train 을 거의 그대로 봐서 낙관적이 된다
  2. **dilated 윈도우 + dense start** — 윈도우 내부만 stride 를 걸고 시작점은 100Hz 매 스텝
  3. **정규화 통계는 train split 프레임만** — 누출 방지

여기서 새로 하는 것:
  4. **모달리티마다 stride/horizon 이 다르다** — 주파수 비율 축이 여기서 실현된다
  5. **모달리티 그룹별 정규화** — 그리고 상수 채널을 std=1 로 처리한다(아래 주의 참조)
  6. **RGB** — 40_rgb_jpeg(고유 프레임) + 41_rgb_index(스텝→프레임) 간접 참조를 푼다

⚠️ **상수 채널 처리가 refer/ 와 다르다.** refer/ 의 `std.clip(1e-6)` 은 train 에서
   상수인 채널을 만나면 std=1e-6 이 되어, 배포 때 그 채널이 조금만 움직여도
   정규화 값이 10^6 배로 폭발한다. 여기서는 std<eps 인 채널을 **std=1 로 두고 이름을 기록**한다.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import h5py
import numpy as np
from torch.utils.data import Dataset

# ── 레코더가 쓰는 키의 차원 (refer/ros2_hdf5_recorder.py FIELDS 와 일치) ──────
KEY_DIMS: dict[str, int] = {
    "01_hand_mode": 1, "02_hand_servo_on": 1, "03_hand_j_pos": 16,
    "04_hand_j_tar": 16, "05_hand_j_cur": 16, "06_hand_j_kin": 12,
    "07_hand_j_tac": 4, "08_hand_tip_pos": 12, "09_hand_tip_quat": 16,
    "10_franka_Arm_j_pos": 7, "11_franka_Arm_j_tar": 7, "12_franka_Arm_j_vel": 7,
    "13_franka_Arm_C_pos": 3, "14_franka_Arm_j_tq": 7, "15_franka_Arm_speed_factor": 1,
    "16_glove_g_pos": 16, "20_paxini_ft": 12, "21_paxini_raw": 1524,
    "22_franka_Arm_C_quat": 4, "23_franka_Arm_tar_pos": 3, "24_franka_Arm_tar_quat": 4,
    "30_fruit_pos": 3, "31_fruit_quat": 4, "32_fruit_size": 3,
    # ── 2026-08-07 수집분: 레코더가 순서·이름을 바꿨다. 이름 매칭이라 순서는 무관하고
    #    개명된 것만 더한다. 옛 이름은 옛 h5 를 위해 남겨 둔다.
    "10_hand_paxini_ft": 12, "11_hand_paxini_raw": 1524,
    "12_franka_Arm_j_pos_v2": 7, "17_franka_Arm_C_quat_v2": 4,
    "22_glove_paxini_ft": 12, "23_glove_paxini_raw": 1524,
    "33_fruit_type": 1, "34_fruit_corners": 24,
}
RGB_JPEG_KEY, RGB_INDEX_KEY = "40_rgb_jpeg", "41_rgb_index"

# 과일 오검출 위생 처리 임계값 (refer/dp_config.py 와 동일)
FRUIT_POS_MAX_NORM = 0.60
FRUIT_POS_MAX_JUMP = 0.10
FRUIT_SIZE_RANGE = (0.02, 0.15)
DEAD_ACTION_STD = 5.0


# ══════════════════════════════════════════════════════════════════════════
# 위생 처리
# ══════════════════════════════════════════════════════════════════════════
def sanitize_fruit(pos: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """과일 오검출 프레임을 직전 유효값으로 홀드. refer/dp_data.py 이식."""
    pos, size = pos.copy(), size.copy()
    n = len(pos)
    bad = np.zeros(n, dtype=bool)
    norm = np.linalg.norm(pos, axis=1)
    bad |= norm > FRUIT_POS_MAX_NORM
    bad |= norm < 1e-9
    lo, hi = FRUIT_SIZE_RANGE
    bad |= (size[:, 0] < lo) | (size[:, 0] > hi)

    last = None
    for i in range(n):
        if not bad[i] and last is not None:
            if np.linalg.norm(pos[i] - pos[last]) > FRUIT_POS_MAX_JUMP:
                bad[i] = True
        if not bad[i]:
            last = i

    fixed = int(bad.sum())
    if fixed:
        good = np.flatnonzero(~bad)
        if len(good) == 0:
            return pos, size, fixed
        for i in np.flatnonzero(bad):
            prev = good[good < i]
            src = prev[-1] if len(prev) else good[0]
            pos[i], size[i] = pos[src], size[src]
    return pos, size, fixed


# ══════════════════════════════════════════════════════════════════════════
# 데모
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Demo:
    """클램프·위생처리까지 끝난 에피소드 하나.

    저차원 신호는 전부 RAM 에 올린다(54k step x ~60 float = 13MB 수준이라 무시 가능).
    RGB 는 파일에 두고 필요할 때만 디코드한다.
    """
    name: str
    path: str
    group: str
    n: int
    lowdim: dict[str, np.ndarray]          # 모달리티 이름 -> (n, D)
    action: np.ndarray                     # (n, A)
    rgb_index: np.ndarray | None = None    # (n,) int32, -1 = 프레임 없음
    n_rgb_frames: int = 0
    fruit_fixed: int = 0
    meta: dict = field(default_factory=dict)


def _concat_keys(g: h5py.Group, keys: list[str], n: int) -> np.ndarray:
    parts = []
    for k in keys:
        if k not in g:
            raise KeyError(f"{g.name}: 키 {k!r} 가 없다. 있는 키: {sorted(g.keys())[:20]}")
        a = np.asarray(g[k][:n], dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        want = KEY_DIMS.get(k)
        if want is not None and a.shape[1] != want:
            raise ValueError(f"{g.name}:{k} 차원 {a.shape[1]} != 기대 {want}")
        parts.append(a)
    return np.concatenate(parts, axis=1)


def load_demos(data_root: str, obs_spec: dict, action_key: str,
               clamp_spec: dict | None = None, keep_dead: bool = False,
               verbose: bool = True) -> tuple[list[Demo], list[dict]]:
    """HDF5 들을 읽어 Demo 리스트를 만든다.

    `clamp_spec` 이 None 이면 폴더의 모든 *.h5 의 모든 Demo_* 를 전부 쓴다.
    dict 면 refer/dp_config.CLAMP_SPEC 형식({파일: {데모idx: None|int|"DEAD"|"DELETE"}}).
    """
    files: list[str] = []
    if os.path.isdir(data_root):
        files = sorted(os.path.join(data_root, f)
                       for f in os.listdir(data_root) if f.endswith(".h5"))
    elif os.path.exists(data_root):
        files = [data_root]
    if not files:
        raise FileNotFoundError(f"h5 파일을 못 찾았다: {data_root}")

    lowdim_specs = {n: s for n, s in obs_spec.items() if s["kind"] != "vision"}
    vision_names = [n for n, s in obs_spec.items() if s["kind"] == "vision"]
    if len(vision_names) > 1:
        raise NotImplementedError(f"vision 모달리티는 1개만 지원한다 — 받은 값 {vision_names}")
    vision_name = vision_names[0] if vision_names else None

    demos: list[Demo] = []
    table: list[dict] = []

    for path in files:
        fname = os.path.basename(path)
        spec_for_file = None if clamp_spec is None else clamp_spec.get(fname)
        if clamp_spec is not None and spec_for_file is None:
            continue
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)
            keys = sorted((k for k in f.keys() if k.startswith("Demo_")),
                          key=lambda s: int(s.split("_")[1]))
            idxs = sorted(spec_for_file) if spec_for_file is not None else \
                [int(k.split("_")[1]) for k in keys]

            for idx in idxs:
                gk = f"Demo_{idx}"
                if gk not in f:
                    raise KeyError(f"{fname}:{gk} 없음")
                g = f[gk]
                n_raw = g[action_key].shape[0]
                clamp = spec_for_file[idx] if spec_for_file is not None else None
                row = {"file": fname, "demo": idx, "n_raw": n_raw,
                       "note": "full" if clamp is None else clamp,
                       "n_used": 0, "seconds": 0.0, "fruit_fixed": 0,
                       "rgb": 0, "status": "", "act_std": 0.0}

                if clamp == "DELETE":
                    row["status"] = "삭제(노트)"
                    table.append(row); continue

                n = n_raw if clamp in (None, "DEAD") else int(clamp)
                if n > n_raw:
                    raise ValueError(f"{fname}:{gk} 클램프 {n} > N {n_raw}")

                action = np.asarray(g[action_key][:n], dtype=np.float32)
                act_std = float(action.std(axis=0).max())
                row["act_std"] = act_std
                if clamp == "DEAD" or act_std < DEAD_ACTION_STD:
                    row["status"] = f"제외(정지, act_std={act_std:.2f})"
                    if not keep_dead:
                        table.append(row); continue
                    row["status"] += " [keep_dead]"

                lowdim, fixed_total = {}, 0
                for name, s in lowdim_specs.items():
                    arr = _concat_keys(g, list(s["keys"]), n)
                    # 과일 신호가 이 모달리티에 포함돼 있으면 위생 처리
                    ks = list(s["keys"])
                    if "30_fruit_pos" in ks and "32_fruit_size" in ks:
                        off, sl = 0, {}
                        for k in ks:
                            sl[k] = slice(off, off + KEY_DIMS[k]); off += KEY_DIMS[k]
                        p, q, fx = sanitize_fruit(arr[:, sl["30_fruit_pos"]],
                                                  arr[:, sl["32_fruit_size"]])
                        arr[:, sl["30_fruit_pos"]] = p
                        arr[:, sl["32_fruit_size"]] = q
                        fixed_total += fx
                    if arr.shape[1] != int(s["shape"]):
                        raise ValueError(
                            f"{fname}:{gk}:{name} 차원 {arr.shape[1]} != config shape {s['shape']}. "
                            f"keys={ks}")
                    lowdim[name] = arr

                if not np.isfinite(action).all() or \
                        not all(np.isfinite(v).all() for v in lowdim.values()):
                    raise ValueError(f"{fname}:{gk} NaN/Inf")

                rgb_index, n_frames = None, 0
                if vision_name is not None:
                    if RGB_JPEG_KEY not in g:
                        raise KeyError(
                            f"{fname}:{gk} 에 {RGB_JPEG_KEY} 가 없다 — 이 데모는 --rgb-on "
                            f"없이 기록됐다. tools/inspect_h5.py 로 전수 확인할 것.")
                    n_frames = len(g[RGB_JPEG_KEY])
                    rgb_index = np.asarray(g[RGB_INDEX_KEY][:n], dtype=np.int64).ravel()
                    row["rgb"] = n_frames

                row.update(n_used=n, seconds=n / 100.0, fruit_fixed=fixed_total,
                           status=row["status"] or "사용")
                table.append(row)
                demos.append(Demo(f"{fname}:{gk}", path, gk, n, lowdim, action,
                                  rgb_index, n_frames, fixed_total, attrs))

    if verbose:
        print_demo_table(table)
    if not demos:
        raise RuntimeError("사용 가능한 데모가 없다")
    return demos, table


def print_demo_table(table: list[dict]) -> None:
    print("=" * 104)
    print("  데모 요약")
    print("=" * 104)
    print(f"  {'파일':<26s} {'demo':>5s} {'N_raw':>7s} {'노트':>7s} {'사용':>7s} "
          f"{'초':>6s} {'act_std':>8s} {'fruit':>6s} {'rgb':>6s}  상태")
    print("  " + "-" * 100)
    tot_used = tot_fix = tot_rgb = n_ok = 0
    for r in table:
        print(f"  {r['file']:<26s} {r['demo']:>5d} {r['n_raw']:>7d} {str(r['note']):>7s} "
              f"{r['n_used']:>7d} {r['seconds']:>6.1f} {r['act_std']:>8.2f} "
              f"{r['fruit_fixed']:>6d} {r['rgb']:>6d}  {r['status']}")
        if r["status"].startswith("사용"):
            n_ok += 1; tot_used += r["n_used"]; tot_fix += r["fruit_fixed"]; tot_rgb += r["rgb"]
    print("  " + "-" * 100)
    print(f"  사용 데모 {n_ok}개 | {tot_used:,} step @100Hz = {tot_used/6000:.1f}분 "
          f"| 과일 보정 {tot_fix:,} ({100*tot_fix/max(1,tot_used):.2f}%) "
          f"| RGB 프레임 {tot_rgb:,}")
    print("=" * 104)


# ══════════════════════════════════════════════════════════════════════════
# 정규화
# ══════════════════════════════════════════════════════════════════════════
class Normalizer:
    """per-feature z-score. 통계는 **train split 프레임만**으로 낸다.

    상수 채널(std < eps)은 **std=1 로 둔다**. refer/ 처럼 1e-6 으로 clip 하면
    배포 때 그 채널이 아주 조금만 움직여도 정규화 값이 10^6 배로 폭발한다.
    (train 에서 상수였다는 건 정보가 없다는 뜻이지, 무한 민감하다는 뜻이 아니다.)
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, const_idx: list[int] | None = None):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.const_idx = list(const_idx or [])

    @classmethod
    def fit(cls, arrays: list[np.ndarray], eps: float = 1e-4) -> "Normalizer":
        cat = np.concatenate(arrays, axis=0)
        mean, std = cat.mean(0), cat.std(0)
        const = np.flatnonzero(std < eps)
        std = std.copy()
        std[const] = 1.0
        return cls(mean, std, const.tolist())

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std, "const_idx": self.const_idx}

    @classmethod
    def from_state(cls, d: dict) -> "Normalizer":
        return cls(d["mean"], d["std"], d.get("const_idx", []))


def fit_normalizers(demos: list[Demo], obs_spec: dict) -> tuple[dict, Normalizer]:
    """모달리티 **그룹별** 통계 + action 통계. vision 은 정규화 대상이 아니다."""
    obs_norm = {}
    for name, s in obs_spec.items():
        if s["kind"] == "vision":
            continue
        obs_norm[name] = Normalizer.fit([d.lowdim[name] for d in demos])
    act_norm = Normalizer.fit([d.action for d in demos])
    return obs_norm, act_norm


# ══════════════════════════════════════════════════════════════════════════
# 윈도우 데이터셋
# ══════════════════════════════════════════════════════════════════════════
class VTWindowDataset(Dataset):
    """모달리티마다 stride/horizon 이 다른 dilated 슬라이딩 윈도우.

    시작점 t(원본 100Hz 인덱스) 하나에 대해
        obs[k]  = x_k[t - (T_k-1-j)*s_k]   j=0..T_k-1     → (T_k, D_k)
        action  = a[t + j*s_a]             j=0..T_p-1     → (T_p, A)
    유효 t: max_k((T_k-1)*s_k) ≤ t ≤ N-1 - (T_p-1)*s_a
    """

    def __init__(self, demos: list[Demo], obs_spec: dict, action_cfg: dict,
                 default_stride: int = 5,
                 obs_norm: dict | None = None, act_norm: Normalizer | None = None,
                 rgb_size: tuple[int, int] = (224, 224), cache_rgb: bool = False):
        self.demos = demos
        self.obs_spec = obs_spec
        self.obs_norm = obs_norm or {}
        self.act_norm = act_norm
        self.rgb_size = rgb_size
        self.cache_rgb = cache_rgb
        self._h5: dict[int, h5py.File] = {}          # pid -> File (워커별 지연 오픈)
        self._cache: dict[tuple[int, int], np.ndarray] = {}

        self.vision_name = next(
            (n for n, s in obs_spec.items() if s["kind"] == "vision"), None)

        # 모달리티별 offset 테이블을 미리 만든다
        self.offsets: dict[str, np.ndarray] = {}
        back = 0
        for name, s in obs_spec.items():
            T, st = int(s["horizon"]), int(s.get("stride") or default_stride)
            self.offsets[name] = np.array([-(T - 1 - j) * st for j in range(T)], dtype=np.int64)
            back = max(back, (T - 1) * st)
        self.a_stride = int(action_cfg.get("stride") or default_stride)
        self.T_pred = int(action_cfg["pred_horizon"])
        self.act_offsets = np.arange(self.T_pred, dtype=np.int64) * self.a_stride
        fwd = (self.T_pred - 1) * self.a_stride
        self.back, self.fwd = back, fwd

        if not demos:
            raise ValueError("데모가 0개다 — data.n_held_out 이 데모 수와 맞는지 확인할 것")

        self.index: list[tuple[int, int]] = []
        self.n_rgb_skipped = 0
        for di, d in enumerate(demos):
            lo, hi = back, d.n - 1 - fwd
            # RGB 를 쓰면 첫 프레임이 도착하기 전 구간(41_rgb_index == -1)은 버린다.
            # 여기서 안 버리면 그 구간 관측이 **미래 프레임**을 끌어와 누출이 된다.
            if self.vision_name is not None and d.rgb_index is not None:
                valid = np.flatnonzero(d.rgb_index >= 0)
                if len(valid) == 0:
                    self.n_rgb_skipped += max(0, hi - lo + 1)
                    continue
                new_lo = max(lo, int(valid[0]) + back)
                self.n_rgb_skipped += max(0, min(new_lo, hi + 1) - lo)
                lo = new_lo
            if hi < lo:
                continue
            self.index.extend((di, t) for t in range(lo, hi + 1))
        if not self.index:
            raise RuntimeError(
                f"윈도우 0개 — 데모가 너무 짧다. 필요 최소 길이 {back + fwd + 1} step "
                f"(lookback {back} + 예측 {fwd}). 가장 긴 데모 {max(d.n for d in demos)} step.")

    # ── RGB ───────────────────────────────────────────────────────────────
    def _file(self, path: str) -> h5py.File:
        """워커 프로세스마다 따로 연다 — h5py File 은 fork 를 넘어 공유하면 안 된다."""
        key = (os.getpid(), path)
        f = self._h5.get(key)
        if f is None:
            f = h5py.File(path, "r")
            self._h5[key] = f
        return f

    def _decode(self, di: int, frame_idx: int) -> np.ndarray:
        """JPEG 한 장 → (3,H,W) float32 [0,1]."""
        ck = (di, frame_idx)
        if self.cache_rgb and ck in self._cache:
            return self._cache[ck].astype(np.float32) / 255.0
        d = self.demos[di]
        buf = np.asarray(self._file(d.path)[d.group][RGB_JPEG_KEY][frame_idx],
                         dtype=np.uint8).tobytes()
        from PIL import Image
        im = Image.open(io.BytesIO(buf)).convert("RGB").resize(
            (self.rgb_size[1], self.rgb_size[0]), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)      # (3,H,W)
        if self.cache_rgb:
            self._cache[ck] = arr
        return arr.astype(np.float32) / 255.0

    def _rgb_window(self, di: int, t: int, offs: np.ndarray) -> np.ndarray:
        d = self.demos[di]
        out = np.zeros((len(offs), 3, *self.rgb_size), dtype=np.float32)
        for j, off in enumerate(offs):
            fi = int(d.rgb_index[t + off])
            if fi < 0:
                # 프레임 없음 → **과거**에서만 찾는다. 미래에서 당겨오면 관측에 미래가
                # 섞인다(원래 구현이 그랬다). 시작 구간은 __init__ 이 이미 걸러 두므로
                # 여기까지 오면 인덱스 생성 쪽 버그다.
                past = d.rgb_index[:t + off + 1]
                past = past[past >= 0]
                if len(past) == 0:
                    raise RuntimeError(
                        f"{d.name} t={t + off}: 이 시점 이전에 RGB 프레임이 없다. "
                        f"윈도우 생성 단계에서 걸러졌어야 한다.")
                fi = int(past[-1])
            out[j] = self._decode(di, fi)
        return out

    # ── Dataset ───────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        di, t = self.index[i]
        d = self.demos[di]
        obs = {}
        for name, s in self.obs_spec.items():
            offs = self.offsets[name]
            if name == self.vision_name:
                obs[name] = self._rgb_window(di, t, offs)
            else:
                x = d.lowdim[name][t + offs]
                nz = self.obs_norm.get(name)
                obs[name] = (nz.normalize(x) if nz else x).astype(np.float32)
        act = d.action[t + self.act_offsets]
        if self.act_norm is not None:
            act = self.act_norm.normalize(act)
        return {"obs": obs, "action": act.astype(np.float32)}

    def raw_action(self, i: int) -> np.ndarray:
        di, t = self.index[i]
        return self.demos[di].action[t + self.act_offsets]

    def close(self):
        for f in self._h5.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5.clear()


def split_demos(demos: list[Demo], n_held_out: int, seed: int) -> tuple[list[Demo], list[Demo]]:
    """데모 단위 분할. 시드 고정이라 재실행 시 같은 홀드아웃."""
    if n_held_out < 1:
        raise ValueError(
            f"data.n_held_out({n_held_out}) 은 1 이상이어야 한다 — "
            f"best 체크포인트를 홀드아웃 액션 MAE 로 고르기 때문에 홀드아웃이 없으면 고를 수 없다")
    if n_held_out >= len(demos):
        raise ValueError(f"n_held_out({n_held_out}) >= 데모 수({len(demos)})")
    order = np.random.RandomState(seed).permutation(len(demos))
    return [demos[i] for i in order[n_held_out:]], [demos[i] for i in order[:n_held_out]]


def build_datasets(cfg: dict, verbose: bool = True):
    """config → (train_ds, val_ds, obs_norm, act_norm, table)."""
    data = cfg["data"]
    demos, table = load_demos(data["root"], cfg["obs_spec"], cfg["action"]["key"],
                              clamp_spec=data.get("clamp_spec"),
                              keep_dead=bool(data.get("keep_dead", False)),
                              verbose=verbose)
    train_d, held_d = split_demos(demos, int(data["n_held_out"]), int(data["seed"]))
    obs_norm, act_norm = fit_normalizers(train_d, cfg["obs_spec"])

    if verbose:
        for name, nz in obs_norm.items():
            msg = f"  [NORM] {name:9s} dim={len(nz.mean):<5d}"
            if nz.const_idx:
                msg += f"  ⚠️ 상수 채널 {nz.const_idx} → std=1 로 고정(정보 없음)"
            print(msg)
        print(f"  [NORM] action    dim={len(act_norm.mean)}")

    rgb_size = (224, 224)
    for s in cfg["obs_spec"].values():
        if s["kind"] == "vision":
            rgb_size = (int(s["shape"][1]), int(s["shape"][2]))

    kw = dict(obs_spec=cfg["obs_spec"], action_cfg=cfg["action"],
              default_stride=int(data.get("ds_stride", 5)),
              obs_norm=obs_norm, act_norm=act_norm, rgb_size=rgb_size,
              cache_rgb=bool(data.get("cache_rgb", False)))
    return (VTWindowDataset(train_d, **kw), VTWindowDataset(held_d, **kw),
            obs_norm, act_norm, table)
