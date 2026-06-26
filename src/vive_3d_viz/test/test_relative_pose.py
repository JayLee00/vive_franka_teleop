"""relative_pose 코어 단위 테스트 (ROS 의존성 없음).

실행:
  python3 test_relative_pose.py          # 직접 실행 (assert 기반)
  python3 -m pytest test_relative_pose.py # pytest
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from vive_3d_viz.relative_pose import (
        pose_to_matrix, matrix_to_pose, invert_transform, relative_pose,
        ensure_quat_continuity,
    )
except ImportError:  # 설치(소싱) 안 된 상태에서도 직접 실행 가능하게
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from vive_3d_viz.relative_pose import (
        pose_to_matrix, matrix_to_pose, invert_transform, relative_pose,
        ensure_quat_continuity,
    )


def _rand_T(rng):
    p = rng.uniform(-2.0, 2.0, 3)
    rotvec = rng.uniform(-np.pi, np.pi, 3)
    q = Rotation.from_rotvec(rotvec).as_quat()
    return pose_to_matrix(p, q)


def _rot_angle(R):
    """회전행렬의 등가 단일축 회전각 [rad]."""
    return float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))


def test_analytic_inverse():
    """analytic inverse 가 실제 역행렬이고 np.linalg.inv 와 일치."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        T = _rand_T(rng)
        assert np.allclose(invert_transform(T) @ T, np.eye(4), atol=1e-9)
        assert np.allclose(invert_transform(T), np.linalg.inv(T), atol=1e-9)


def test_relative_reconstruction():
    """역검증: T_W_E @ T_E_O == T_W_O."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        T_W_E = _rand_T(rng)
        T_W_O = _rand_T(rng)
        T_E_O = relative_pose(T_W_E, T_W_O)
        assert np.allclose(T_W_E @ T_E_O, T_W_O, atol=1e-9)


def test_identity_at_engage():
    """engage 순간(T_W_O == T_W_E) -> T_E_O == Identity (pos<1e-9, 각<1e-9)."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        T = _rand_T(rng)
        T_E_O = relative_pose(T, T)
        p, _ = matrix_to_pose(T_E_O)
        assert np.linalg.norm(p) < 1e-9
        assert _rot_angle(T_E_O[:3, :3]) < 1e-9


def test_pure_translation_identity_engage():
    """engage 가 단위회전이면 world 평행이동이 그대로 나옴 (손계산)."""
    T_W_E = pose_to_matrix([1.0, 2.0, 3.0], [0, 0, 0, 1])
    T_W_O = pose_to_matrix([1.5, 2.0, 3.0], [0, 0, 0, 1])   # world +X 0.5
    p, _ = matrix_to_pose(relative_pose(T_W_E, T_W_O))
    assert np.allclose(p, [0.5, 0.0, 0.0], atol=1e-12)


def test_pure_translation_rotated_engage():
    """engage 가 world Z +90° 기울어진 상태에서 world +X 이동 -> engage 프레임은 -Y (손계산)."""
    qz = Rotation.from_euler('z', 90, degrees=True).as_quat()
    T_W_E = pose_to_matrix([0, 0, 0], qz)
    T_W_O = pose_to_matrix([0.5, 0, 0], qz)   # 방향 동일, world +X 0.5
    p, _ = matrix_to_pose(relative_pose(T_W_E, T_W_O))
    # Rz(90)^T @ [0.5,0,0] = [0,-0.5,0]
    assert np.allclose(p, [0.0, -0.5, 0.0], atol=1e-9)
    assert _rot_angle(relative_pose(T_W_E, T_W_O)[:3, :3]) < 1e-9


def test_pure_rotation():
    """순수 회전: engage 대비 Z축 +30° (제자리) -> 위치 0, 각 30°, 축 +Z."""
    T_W_E = pose_to_matrix([1, 1, 1], [0, 0, 0, 1])
    qz30 = Rotation.from_euler('z', 30, degrees=True).as_quat()
    T_W_O = pose_to_matrix([1, 1, 1], qz30)
    T_E_O = relative_pose(T_W_E, T_W_O)
    p, _ = matrix_to_pose(T_E_O)
    assert np.allclose(p, [0, 0, 0], atol=1e-12)
    rv = Rotation.from_matrix(T_E_O[:3, :3]).as_rotvec()
    assert abs(np.degrees(np.linalg.norm(rv)) - 30.0) < 1e-6
    assert np.allclose(rv / np.linalg.norm(rv), [0, 0, 1], atol=1e-6)


def test_quat_continuity():
    """double cover: 직전과 dot<0 이면 부호 반전, 아니면 유지."""
    flipped = ensure_quat_continuity(np.array([0, 0, 0, -1.0]), np.array([0, 0, 0, 1.0]))
    assert np.allclose(flipped, [0, 0, 0, 1.0])              # dot<0 -> 반전
    same = ensure_quat_continuity(np.array([0, 0, 0, 1.0]), np.array([0, 0, 0, 1.0]))
    assert np.allclose(same, [0, 0, 0, 1.0])                 # dot>0 -> 유지
    first = ensure_quat_continuity(np.array([0.1, 0.2, 0.3, 0.9]), None)
    assert np.allclose(first, [0.1, 0.2, 0.3, 0.9])          # q_prev None -> 유지


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f'  PASS  {fn.__name__}')
    print(f'\n{len(fns)}/{len(fns)} tests passed ✔')


if __name__ == '__main__':
    _run_all()
