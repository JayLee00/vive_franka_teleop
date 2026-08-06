#!/usr/bin/env python3
"""스캔한 메시를 FoundationPose 입력으로 다듬는다 (진단 + 단위보정 + 변환).

아이폰 스캐너 앱 결과물은 그대로 쓰면 대개 세 군데서 걸린다:

  1) 단위     — ARKit 은 m 지만 앱에 따라 cm/mm 로 뽑는다. FoundationPose 는 **m** 다.
                7cm 오렌지를 mm 로 내보내면 extents=[70,70,70] → 70미터짜리 물체로 인식.
  2) 텍스처   — STL 은 형상만 있다. 텍스처가 없으면 FoundationPose 가 메시를
                균일 회색([128,128,128], Utils.py:124)으로 렌더해서 RGB 정합이
                아무 정보도 못 준다. 둥근 과일은 이때 회전이 전혀 안 잡힌다.
  3) 폴리곤 수 — 스캔은 수십만 face 가 흔하다. 렌더가 느려진다.

사용:
    # 진단만
    python3 prepare_mesh.py scan.glb

    # 실측 지름 72mm 에 맞춰 변환
    python3 prepare_mesh.py scan.glb -o assets/orange.obj --target-diameter 0.072
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh

# trimesh 가 읽는 포맷 (usdz 는 없음 → GLB/OBJ 로 내보내야 한다)
GOOD = {".glb", ".gltf", ".obj", ".ply", ".dae", ".off", ".3mf"}
GEOM_ONLY = {".stl"}


def texture_image(m: trimesh.Trimesh):
    """메시에서 텍스처 이미지를 꺼낸다. GLB 는 PBRMaterial.baseColorTexture 에 들어간다.

    ★ FoundationPose 는 `mesh.visual.material.image` 만 본다(Utils.py:109). GLB 를
    그대로 넘기면 PBRMaterial 이라 .image 가 None 이고 거기서 AttributeError 로 죽는다.
    그래서 이 스크립트가 OBJ(SimpleMaterial) 로 바꿔주는 것이다.
    """
    mat = getattr(getattr(m, "visual", None), "material", None)
    if mat is None:
        return None
    return getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)


def describe(m: trimesh.Trimesh) -> dict:
    """텍스처/색 유무와 크기를 판정."""
    has_tex = isinstance(getattr(m, "visual", None), trimesh.visual.texture.TextureVisuals)
    tex_img = texture_image(m) if has_tex else None
    has_vcol = False
    if not has_tex:
        try:
            vc = m.visual.vertex_colors
            # trimesh 는 색이 없어도 기본 회색을 채워 넣는다 → 실제로 변하는지 본다
            has_vcol = vc is not None and len(np.unique(vc[:, :3], axis=0)) > 1
        except Exception:                                        # noqa: BLE001
            has_vcol = False
    return {"has_tex": has_tex and tex_img is not None,
            "has_vcol": has_vcol, "extents": m.extents,
            "nv": len(m.vertices), "nf": len(m.faces)}


def guess_unit(extents: np.ndarray) -> tuple[str, float]:
    """가장 긴 축 길이로 단위를 추정해 (이름, m 로 가는 배율) 반환.

    과일(대략 2~30cm)을 가정한 임계값이다. m/cm/mm 로 각각 0.02~0.3 / 2~30 / 20~300
    이 나오므로 아래처럼 갈라도 겹치지 않는다. 어디까지나 추정이니 확실히 하려면
    --target-diameter 로 실측을 주는 게 맞다.
    """
    d = float(np.max(extents))
    if d < 0.5:
        return "m", 1.0
    if d < 50:
        return "cm", 1e-2
    return "mm", 1e-3


def main():
    ap = argparse.ArgumentParser(description="스캔 메시 → FoundationPose 입력 변환")
    ap.add_argument("src", help="스캔 파일 (.glb/.obj/.ply/.stl …)")
    ap.add_argument("-o", "--out", default=None, help="출력 .obj (없으면 진단만)")
    ap.add_argument("--target-diameter", type=float, default=None,
                    help="실측 지름 [m] 으로 강제 스케일 (예: 0.072). 가장 확실하다")
    ap.add_argument("--scale", type=float, default=None,
                    help="배율 직접 지정 (mm→m 이면 0.001). --target-diameter 와 배타")
    ap.add_argument("--max-faces", type=int, default=50000,
                    help="이보다 많으면 단순화 (0=끔)")
    a = ap.parse_args()

    ext = os.path.splitext(a.src)[1].lower()
    if ext not in GOOD | GEOM_ONLY:
        print(f"✗ trimesh 가 못 읽는 포맷: {ext}")
        print(f"  읽을 수 있는 것: {', '.join(sorted(GOOD | GEOM_ONLY))}")
        if ext == ".usdz":
            print("  → 아이폰 USDZ 는 지원 안 됨. 앱에서 GLB 나 OBJ 로 내보내세요.")
        sys.exit(1)

    m = trimesh.load(a.src, force="mesh")
    info = describe(m)
    unit, to_m = guess_unit(info["extents"])

    print(f"입력      : {a.src}")
    print(f"  정점/면 : {info['nv']:,} / {info['nf']:,}")
    print(f"  크기    : {np.round(info['extents'], 4).tolist()}  → 단위 추정 {unit}")
    print(f"  텍스처  : {'있음 (UV+이미지)' if info['has_tex'] else '없음'}")
    print(f"  정점색  : {'있음' if info['has_vcol'] else '없음'}")

    if not info["has_tex"] and not info["has_vcol"]:
        print("")
        print("  ⚠ 색 정보가 전혀 없습니다. FoundationPose 는 이런 메시를 균일 회색으로")
        print("    렌더하므로 RGB 정합이 회전을 전혀 못 잡습니다. 둥근 과일이면 사실상")
        print("    위치만 쓰게 됩니다. 스캐너 앱에서 **텍스처 포함 GLB/OBJ** 로 다시")
        print("    내보내세요 (STL 은 형상 전용이라 안 됩니다).")

    if a.out is None:
        print("\n(-o 를 주면 변환해서 저장합니다)")
        return

    # ── 스케일 ────────────────────────────────────────────────────────────
    if a.target_diameter and a.scale:
        print("✗ --target-diameter 와 --scale 은 같이 못 씁니다"); sys.exit(1)
    if a.target_diameter:
        cur = float(np.max(m.extents))
        s = a.target_diameter / cur
        print(f"\n스케일    : 최대축 {cur:.4f} → {a.target_diameter:.4f} m  (×{s:.6g})")
    elif a.scale:
        s = a.scale
        print(f"\n스케일    : ×{s:.6g} (직접 지정)")
    else:
        s = to_m
        print(f"\n스케일    : 단위 추정 {unit} → m  (×{s:.6g})")
        print("  ※ 추정입니다. 실측 지름을 아신다면 --target-diameter 를 쓰세요.")
    if s != 1.0:
        m.apply_scale(s)

    # ── 중심 정렬 (FoundationPose 도 내부에서 하지만 파일도 맞춰둔다) ──────
    m.apply_translation(-m.bounding_box.centroid)

    # ── 단순화 ────────────────────────────────────────────────────────────
    if a.max_faces and len(m.faces) > a.max_faces:
        before = len(m.faces)
        try:
            m = m.simplify_quadric_decimation(a.max_faces)
            print(f"단순화    : {before:,} → {len(m.faces):,} face")
        except Exception as e:                                   # noqa: BLE001
            print(f"단순화 실패(무시): {e}")

    # ── PBR(GLB) → SimpleMaterial 로 강등 ────────────────────────────────
    # FoundationPose 가 material.image 만 읽으므로 여기서 맞춰준다.
    img = texture_image(m)
    if img is not None and getattr(m.visual.material, "image", None) is None:
        m.visual.material = trimesh.visual.material.SimpleMaterial(image=img)
        print("재질      : PBRMaterial → SimpleMaterial (FoundationPose 가 .image 만 읽음)")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    m.export(out)          # .obj 로 내보내면 텍스처는 .mtl + 이미지로 같이 나온다
    final = describe(m)
    print(f"\n저장      : {out}")
    print(f"  크기    : {np.round(final['extents'], 4).tolist()} m")
    print(f"  면      : {final['nf']:,}")
    print(f"  텍스처  : {'유지됨' if final['has_tex'] else ('정점색' if final['has_vcol'] else '없음')}")
    print(f"\n실행:  bash foundation_pose/run_foundation_pose.sh")
    print(f"또는 돌아가는 중이면:")
    print(f"  ros2 topic pub --once /fruit/reset std_msgs/String '{{data: \"{out}\"}}'")


if __name__ == "__main__":
    main()
