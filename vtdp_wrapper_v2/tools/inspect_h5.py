#!/usr/bin/env python3
"""기록된 HDF5 데모에 무엇이 실제로 들어 있는지 판정한다.

visuo-tactile 정책을 붙이기 전에 답이 필요한 질문들:

  Q1. RGB 가 있나?              레코더의 --rgb-on 은 opt-in 이라 없을 수 있다.
                                없으면 재수집부터 해야 하므로 이게 1순위.
  Q2. 21_paxini_raw 는 살아있나? 1524채널 중 몇 개가 실제로 변하는가.
  Q3. 31_fruit_quat 은 쓸모있나? 항등쿼터니언·상수면 "과일 각도"로 못 쓴다.
  Q4. 카메라 내부파라미터가 있나? fingertip 3D → image plane 투영(공간 grounding)에 필요.
  Q5. RGB 프레임이 얼어붙은 구간이 있나? 41_rgb_index 가 오래 안 변하면 카메라가 죽은 구간.

h5py + numpy 만 있으면 돌아간다(torch·ROS 불필요). JPEG 디코드 검증은 PIL 이나 cv2 가
있으면 하고, 없으면 건너뛴다.

사용:
    python3 tools/inspect_h5.py                      # dp_config.DATA_ROOT 의 *.h5 전부
    python3 tools/inspect_h5.py /path/to/logs        # 폴더 지정
    python3 tools/inspect_h5.py a.h5 b.h5            # 파일 지정
    python3 tools/inspect_h5.py --full               # 데모별 전체 데이터셋 목록까지
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np

# Windows 콘솔 기본이 cp949 라 한글/기호 출력에서 죽는다. Linux 에서는 무해.
for _s in (sys.stdout, sys.stderr):
    if getattr(_s, "encoding", "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 레코더가 쓰는 이름. 없어도 되는 것과 없으면 안 되는 것을 구분한다.
REQUIRED = ["03_hand_j_pos", "04_hand_j_tar", "06_hand_j_kin", "20_paxini_ft",
            "30_fruit_pos", "32_fruit_size"]
OPTIONAL = ["21_paxini_raw", "31_fruit_quat", "40_rgb_jpeg", "41_rgb_index",
            "42_rgb_time", "43_rgb_stamp", "08_hand_tip_pos", "09_hand_tip_quat"]

OK, NO, WARN = "✅", "❌", "⚠️"


def _demo_keys(f: h5py.File) -> list[str]:
    ks = [k for k in f.keys() if k.startswith("Demo_")]
    return sorted(ks, key=lambda s: int(s.split("_")[1]))


def check_layout(f: h5py.File) -> tuple[list[str], list[str]]:
    """대상 포맷인지 먼저 판정한다. `(demo 키, 경고 줄)`.

    ⚠️ 이게 없으면 **구버전 파일을 'RGB 없음 → 재수집 필요' 로 오진한다.**
    실제로 구버전 세션(`demo0` 소문자, 1000Hz, Paxini 없음)에서 그 오진이 났다.
    레이아웃이 다른 것과 데이터가 없는 것은 완전히 다른 문제이고, 대응도 정반대다.
    """
    out: list[str] = []
    demos = _demo_keys(f)
    groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    if demos:
        rate = f.attrs.get("rate_hz")
        if rate is not None and abs(float(rate) - 100.0) > 1e-6:
            out.append(f"{WARN} rate_hz={rate} — 이 프로젝트는 100Hz 를 전제한다 "
                       f"(config 의 stride 가 전부 100Hz 기준이다)")
        return demos, out

    out.append(f"{NO} **대상 포맷이 아니다** — `Demo_0`, `Demo_1` ... 그룹이 없다.")
    if groups:
        low = [k for k in groups if k.lower().startswith("demo")]
        out.append(f"   이 파일에 있는 그룹: {groups[:8]}{' ...' if len(groups) > 8 else ''}")
        if low:
            out.append(f"   → `{low[0]}` 같은 **구버전 이름**이다(소문자/언더바 없음). "
                       f"이 파일은 refer/ros2_hdf5_recorder.py 가 만든 것이 아니다.")
            g = f[low[0]]
            shapes = {k: f"{g[k].shape}" for k in list(g.keys())[:4]}
            out.append(f"   구버전 예시 shape: {shapes}")
            out.append(f"   → 구버전은 (N,1,C) 처럼 중간에 축이 하나 더 있고 키 차원도 다르다"
                       f"(예: 07_hand_j_tac 60ch, 13_franka_Arm_C_pos 16ch). "
                       f"vtdp/data.py 는 (N,C) 를 기대하므로 **그대로는 못 읽는다.**")
    else:
        out.append("   그룹이 아예 없다 — 빈 파일이거나 저장이 중단됐다.")
    out.append(f"   ⇒ 아래 Q1~Q3 판정은 **이 파일에 대해 의미가 없다.** "
               f"먼저 레코더 버전을 확인할 것.")
    return [], out


# ── 촉각 레이아웃: n_part 를 정하기 위한 근거 ────────────────────────────────
def probe_tactile_layout(a: np.ndarray, key: str) -> list[str]:
    """`20_paxini_ft` 같은 F/T 벡터가 몇 부위 × 몇 채널인지 추정을 돕는다.

    `vtdp` 의 촉각 인코더는 `n_part` 로 부위(손가락)를 쪼갠다. **이 숫자가 틀리면
    조용히 엉뚱한 손가락을 학습한다** — 그래서 사람이 눈으로 고를 근거를 뽑아 준다.
    부위-major(`x[p*ch+c]`) 라면 같은 부위의 채널끼리 상관이 더 높게 나온다.
    """
    out = []
    D = a.shape[1]
    cands = [p for p in (4, 5, 6, 8) if D % p == 0]
    if not cands:
        return [f"   {WARN} {key} {D}ch — 4/5/6/8 로 안 나뉜다. n_part 를 직접 정해야 한다"]
    c = np.corrcoef(a.T)
    np.fill_diagonal(c, np.nan)
    out.append(f"   [레이아웃 추정] {key} {D}ch — 부위-major 가정 시 블록 내 상관:")
    for p in cands:
        ch = D // p
        blocks = [np.nanmean(np.abs(c[i * ch:(i + 1) * ch, i * ch:(i + 1) * ch]))
                  for i in range(p)]
        inside = float(np.nanmean(blocks))
        mask = np.ones_like(c, dtype=bool)
        for i in range(p):
            mask[i * ch:(i + 1) * ch, i * ch:(i + 1) * ch] = False
        outside = float(np.nanmean(np.abs(c[mask])))
        mark = " ←" if inside > outside * 1.3 else ""
        out.append(f"      n_part={p:<2d} (부위당 {ch:>4d}ch)  블록내 {inside:.3f} "
                   f"vs 블록외 {outside:.3f}{mark}")
    out.append(f"      → 블록내 >> 블록외 인 n_part 가 부위-major 레이아웃과 맞는다. "
               f"판정이 안 서면 KIST 퍼블리셔 코드를 볼 것.")
    return out


def _alive_channels(a: np.ndarray, eps: float = 1e-6) -> int:
    """시간축으로 실제로 변하는 채널 수 (상수 채널 = 정보 없음)."""
    if a.ndim == 1:
        a = a[:, None]
    return int((a.std(axis=0) > eps).sum())


def _fmt_range(a: np.ndarray) -> str:
    return f"[{a.min():+.4g}, {a.max():+.4g}]"


def check_quat(q: np.ndarray) -> str:
    """31_fruit_quat 이 실제 각도 신호인지 판정.

    검출기가 방향을 추정하지 않으면 항등(0,0,0,1) 이거나 전부 0 으로 채워진다.
    """
    n = len(q)
    norm = np.linalg.norm(q, axis=1)
    n_zero = int((norm < 1e-6).sum())
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    n_ident = int((np.abs(q - identity).max(axis=1) < 1e-6).sum())
    alive = _alive_channels(q)

    if n_zero == n:
        return f"{NO} 전부 0 — 토픽 미수신. 각도 정보 없음"
    if n_ident + n_zero >= 0.99 * n:
        return f"{NO} {100*(n_ident+n_zero)/n:.1f}% 가 항등/0 — 검출기가 방향을 추정하지 않음"
    if alive == 0:
        return f"{NO} 상수 (변하지 않음) — 각도 정보 없음"
    # 실제로 변한다면 얼마나 변하는지
    dq = np.abs(np.diff(q, axis=0)).max()
    return (f"{OK} 살아있음 — 변하는 채널 {alive}/4, 최대 프레임간 변화 {dq:.4g}, "
            f"norm {_fmt_range(norm)}  ★ in-hand 회전 위상 관측으로 검토 가치 있음")


def check_rgb(g: h5py.Group, n_steps: int) -> list[str]:
    out = []
    if "40_rgb_jpeg" not in g:
        out.append(f"{NO} 40_rgb_jpeg 없음 — 이 데모는 --rgb-on 없이 기록됨. "
                   f"시각 입력 불가(재수집 필요)")
        return out

    jpeg = g["40_rgb_jpeg"]
    n_frames = len(jpeg)
    sizes = np.array([len(jpeg[i]) for i in range(n_frames)], dtype=np.int64)
    out.append(f"{OK} 40_rgb_jpeg  고유 프레임 {n_frames}장, "
               f"{sizes.sum()/1e6:.1f}MB, 장당 {sizes.mean()/1e3:.0f}KB")

    if "41_rgb_index" not in g:
        out.append(f"{WARN} 41_rgb_index 없음 — 스텝↔프레임 대응을 복원할 수 없다")
        return out

    idx = np.asarray(g["41_rgb_index"][:]).ravel()
    n_missing = int((idx < 0).sum())
    if n_missing:
        out.append(f"{WARN} 41_rgb_index 에 -1 이 {n_missing}개 "
                   f"({100*n_missing/len(idx):.1f}%) — 프레임 없는 스텝")

    # 유효 구간에서 실효 프레임레이트와 '얼어붙음' 판정
    valid = idx[idx >= 0]
    if len(valid) > 1:
        n_change = int((np.diff(valid) != 0).sum())
        eff_hz = 100.0 * n_change / len(valid)      # 원본 100Hz 기준
        out.append(f"   실효 프레임레이트 ≈ {eff_hz:.1f}Hz "
                   f"(스텝 {len(idx)} → 고유 {n_frames})")
        # 같은 인덱스가 연속으로 이어진 최장 구간 = 카메라가 멈춘 구간
        run, best = 1, 1
        for a, b in zip(valid[:-1], valid[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        if best > 50:                                # 0.5초 이상 동결
            out.append(f"{WARN} 같은 프레임이 최장 {best} 스텝({best/100:.2f}s) 연속 "
                       f"— 카메라 정지 구간 의심")
    if idx.max() >= n_frames:
        out.append(f"{NO} 41_rgb_index 최대 {idx.max()} ≥ 프레임 수 {n_frames} — 인덱스 깨짐")
    return out


def try_decode(g: h5py.Group) -> str:
    """JPEG 한 장을 실제로 디코드해 해상도를 확인. 디코더 없으면 건너뛴다."""
    if "40_rgb_jpeg" not in g or len(g["40_rgb_jpeg"]) == 0:
        return ""
    buf = np.asarray(g["40_rgb_jpeg"][0], dtype=np.uint8).tobytes()
    try:
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(buf))
        return f"   디코드 OK (PIL): {im.size[0]}x{im.size[1]} {im.mode}"
    except ImportError:
        pass
    except Exception as e:
        return f"{NO} JPEG 디코드 실패 (PIL): {e}"
    try:
        import cv2
        im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return f"{NO} JPEG 디코드 실패 (cv2): None 반환"
        return f"   디코드 OK (cv2): {im.shape[1]}x{im.shape[0]}"
    except ImportError:
        return "   (PIL·cv2 둘 다 없음 — 디코드 검증 건너뜀)"
    except Exception as e:
        return f"{NO} JPEG 디코드 실패 (cv2): {e}"


def inspect_file(path: str, full: bool = False) -> dict:
    """파일 하나를 훑고 요약 dict 를 돌려준다."""
    summary = {"path": path, "demos": 0, "steps": 0, "rgb_demos": 0,
               "has_raw_tactile": False, "quat_usable": False, "intrinsics": False,
               "layout_ok": True}

    print("\n" + "=" * 100)
    print(f"  {path}")
    print("=" * 100)

    with h5py.File(path, "r") as f:
        # ── 파일 attrs (카메라 내부파라미터가 여기 있다) ──
        attrs = dict(f.attrs)
        print(f"\n[FILE ATTRS] {len(attrs)}개")
        for k in sorted(attrs):
            print(f"   {k:16s} = {attrs[k]}")

        need = ("rgb_fx", "rgb_fy", "rgb_cx", "rgb_cy")
        if all(k in attrs for k in need):
            summary["intrinsics"] = True
            print(f"\n{OK} 카메라 내부파라미터 있음 "
                  f"fx={attrs['rgb_fx']:.1f} fy={attrs['rgb_fy']:.1f} "
                  f"cx={attrs['rgb_cx']:.1f} cy={attrs['rgb_cy']:.1f} "
                  f"({attrs.get('rgb_width','?')}x{attrs.get('rgb_height','?')})")
            print("   → fingertip 3D 를 image plane 에 투영하는 공간 grounding 이 가능하다")
        else:
            print(f"\n{NO} 카메라 내부파라미터 없음 (rgb_fx/fy/cx/cy) — 3D→2D 투영 불가")

        # ── 레이아웃 판정을 **먼저**. 대상 포맷이 아니면 나머지 판정은 무의미하다 ──
        demos, layout_msgs = check_layout(f)
        if layout_msgs:
            print()
            for m in layout_msgs:
                print(m if m.startswith("   ") else f"  {m}")
        if not demos:
            summary["layout_ok"] = False
            return summary

        summary["demos"] = len(demos)
        print(f"\n[DEMOS] {len(demos)}개: {', '.join(demos)}")

        for dk in demos:
            g = f[dk]
            n = g["04_hand_j_tar"].shape[0] if "04_hand_j_tar" in g else -1
            summary["steps"] += max(0, n)
            print(f"\n{'─' * 100}")
            print(f"  {dk}   {n} step ({n/100:.1f}s @100Hz)")
            print(f"{'─' * 100}")

            missing = [k for k in REQUIRED if k not in g]
            if missing:
                print(f"{NO} 필수 키 누락: {missing}")

            present_opt = [k for k in OPTIONAL if k in g]
            absent_opt = [k for k in OPTIONAL if k not in g]
            print(f"   선택 키 있음: {present_opt}")
            print(f"   선택 키 없음: {absent_opt}")

            if full:
                print("   전체 데이터셋:")
                for k in sorted(g.keys()):
                    d = g[k]
                    print(f"      {k:22s} {str(d.shape):14s} {d.dtype}")

            # ── Q1. RGB ──
            print("\n   [Q1] RGB")
            for line in check_rgb(g, n):
                print("   " + line)
            if "40_rgb_jpeg" in g:
                summary["rgb_demos"] += 1
                msg = try_decode(g)
                if msg:
                    print("   " + msg)

            # ── Q2. raw 촉각 ──
            print("\n   [Q2] 촉각")
            ft = np.asarray(g["20_paxini_ft"][:]) if "20_paxini_ft" in g else None
            if ft is not None:
                if ft.ndim == 1:
                    ft = ft[:, None]
                print(f"   {OK} 20_paxini_ft  {ft.shape}  변하는 채널 "
                      f"{_alive_channels(ft)}/{ft.shape[1]}  범위 {_fmt_range(ft)}")
                # n_part 를 정할 근거 — 첫 데모에서만 뽑는다(전 데모 반복은 노이즈)
                if dk == demos[0] and ft.shape[1] > 1 and _alive_channels(ft) > 1:
                    for ln in probe_tactile_layout(ft, "20_paxini_ft"):
                        print(ln)
            if "21_paxini_raw" in g:
                raw = np.asarray(g["21_paxini_raw"][:])
                alive = _alive_channels(raw)
                summary["has_raw_tactile"] = True
                print(f"   {OK} 21_paxini_raw {raw.shape}  변하는 채널 "
                      f"{alive}/{raw.shape[1]} ({100*alive/raw.shape[1]:.0f}%)  "
                      f"범위 {_fmt_range(raw)}")
                if alive < 0.1 * raw.shape[1]:
                    print(f"   {WARN} 살아있는 채널이 10% 미만 — 센서 일부만 동작했을 수 있다")
            else:
                print(f"   {NO} 21_paxini_raw 없음 — 12차원 F/T 요약만 사용 가능")

            # ── Q3. 과일 각도 ──
            print("\n   [Q3] 과일 방향(회전 위상)")
            if "31_fruit_quat" in g:
                q = np.asarray(g["31_fruit_quat"][:], dtype=np.float64)
                verdict = check_quat(q)
                print("   " + verdict)
                if verdict.startswith(OK):
                    summary["quat_usable"] = True
            else:
                print(f"   {NO} 31_fruit_quat 없음")

            # ── 참고: action 다이내믹스 ──
            if "04_hand_j_tar" in g:
                act = np.asarray(g["04_hand_j_tar"][:], dtype=np.float64)
                print(f"\n   [참고] action std(관절별 최대) = {act.std(axis=0).max():.1f} count"
                      f"   (5 미만이면 '정지 데모')")

    return summary


def main():
    ap = argparse.ArgumentParser(description="HDF5 데모에 무엇이 들어있는지 판정")
    ap.add_argument("paths", nargs="*", help="h5 파일 또는 폴더 (없으면 dp_config.DATA_ROOT)")
    ap.add_argument("--full", action="store_true", help="데모별 전체 데이터셋 목록까지 출력")
    args = ap.parse_args()

    paths = args.paths
    if not paths:
        # refer/ 의 설정을 재사용 (없으면 안내만)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "refer", "diffusion_policy"))
        try:
            from dp_config import DATA_ROOT
            paths = [DATA_ROOT]
            print(f"경로 미지정 → dp_config.DATA_ROOT 사용: {DATA_ROOT}")
        except Exception as e:
            ap.error(f"경로를 지정하세요 (dp_config 로드 실패: {e})")

    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.h5"))))
        elif os.path.exists(p):
            files.append(p)
        else:
            print(f"{NO} 경로 없음: {p}")
    if not files:
        print(f"{NO} h5 파일을 못 찾았다.")
        return 1

    summaries = [inspect_file(p, args.full) for p in files]

    # ── 종합 판정 ──
    print("\n" + "=" * 100)
    print("  종합 판정")
    print("=" * 100)
    bad = [s for s in summaries if not s["layout_ok"]]
    ok = [s for s in summaries if s["layout_ok"]]
    tot_demo = sum(s["demos"] for s in summaries)
    tot_rgb = sum(s["rgb_demos"] for s in summaries)
    tot_step = sum(s["steps"] for s in summaries)
    print(f"  파일 {len(summaries)}개 · 데모 {tot_demo}개 · "
          f"{tot_step:,} step @100Hz = {tot_step/6000:.1f}분")
    if bad:
        print()
        print(f"  {NO} **읽을 수 없는 파일 {len(bad)}/{len(summaries)}개** "
              f"(Demo_* 그룹이 없다 = 대상 레코더 포맷이 아니다):")
        for s in bad:
            print(f"      {os.path.basename(s['path'])}")
        print("      → 이건 'RGB 가 없다'와 **다른 문제**다. 재수집이 아니라 "
              "레코더 버전을 확인해야 한다.")
        if not ok:
            print("      → 읽을 수 있는 파일이 0개다. 아래 판정은 생략한다.")
            return 1
    print()
    if tot_rgb == 0:
        print(f"  {NO} RGB 를 가진 데모가 0개다.")
        print(f"      → 시각 분기를 붙일 수 없다. 레코더를 --rgb-on 으로 재수집해야 한다.")
        print(f"      → 그 전까지는 '촉각만 제대로 인코딩' 경로로 진행 (fruit_pos/size 6차원 유지).")
    elif tot_rgb < tot_demo:
        print(f"  {WARN} RGB 를 가진 데모가 {tot_rgb}/{tot_demo} 개뿐이다.")
        print(f"      → RGB 있는 데모만 쓰면 데이터가 더 줄고, 섞어 쓰려면 modality dropout 이 필요하다.")
    else:
        print(f"  {OK} 모든 데모({tot_demo}개)에 RGB 가 있다 — 시각 분기 진행 가능.")

    if any(s["has_raw_tactile"] for s in summaries):
        print(f"  {OK} 21_paxini_raw(1524ch) 사용 가능 — 촉각 인코더에 저해상 12차원 대신 쓸 수 있다.")
    else:
        print(f"  {WARN} 21_paxini_raw 없음 — 촉각은 12차원 F/T 만.")

    if any(s["quat_usable"] for s in summaries):
        print(f"  {OK} 31_fruit_quat 가 살아있다 — in-hand 회전 위상 관측 후보. "
              f"BASELINES.md 의 '과일 각도가 없다'는 진단을 재검토할 것.")
    else:
        print(f"  {NO} 31_fruit_quat 는 쓸 수 없다 — 회전 위상은 여전히 미관측(POMDP).")

    if not any(s["intrinsics"] for s in summaries):
        print(f"  {WARN} 카메라 내부파라미터 없음 — 3D→2D 투영 기반 공간 grounding 은 배제.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
