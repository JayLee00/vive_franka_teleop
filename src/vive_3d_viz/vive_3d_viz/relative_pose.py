"""engage 기준 상대 포즈 계산 코어 (ROS 의존성 없는 순수 numpy/scipy).

변환행렬 컨벤션:  ``T_A_B`` = **A 프레임에서 본 B 의 포즈** (= B->A 좌표변환).
출발_도착 표기가 아니라 "A에서 본 B" 의미임에 주의.
  - ``T_W_E`` : World 에서 본 Engage(트래커가 engage된 순간) 포즈   (latch 후 고정)
  - ``T_W_O`` : World 에서 본 Object(현재 트래커) 포즈
  - ``T_E_O`` : Engage 에서 본 Object 포즈  =  ``inv(T_W_E) @ T_W_O``   (출력)

모든 quaternion 은 ``[x, y, z, w]`` (scipy 컨벤션).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(p, q) -> np.ndarray:
    """위치 ``p=[x,y,z]`` + 쿼터니언 ``q=[x,y,z,w]`` -> 4x4 동차변환행렬."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(q, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def matrix_to_pose(T) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 동차변환행렬 -> ``(p[3], q[4]=xyzw)``. quaternion 은 정규화됨."""
    T = np.asarray(T, dtype=float)
    p = T[:3, 3].copy()
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x,y,z,w]
    n = float(np.linalg.norm(q))
    q = q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    return p, q


def invert_transform(T) -> np.ndarray:
    """동차변환행렬의 analytic inverse  ``[[R^T, -R^T p], [0, 1]]``.

    회전부 R 이 직교행렬이라는 성질을 쓰므로 ``np.linalg.inv`` 보다 빠르고
    수치적으로 안정적이다 (스펙 요구사항).
    """
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def relative_pose(T_W_E, T_W_O) -> np.ndarray:
    """Engage 기준 상대 포즈  ``T_E_O = inv(T_W_E) @ T_W_O``.

    engage 순간(``T_W_O == T_W_E``)에는 Identity 가 된다.
    """
    return invert_transform(T_W_E) @ np.asarray(T_W_O, dtype=float)


def ensure_quat_continuity(q, q_prev: Optional[np.ndarray]) -> np.ndarray:
    """double cover(q 와 -q 는 동일 회전) 연속성 처리.

    직전 프레임 quaternion 과 dot product < 0 이면 부호를 반전해 부드럽게 잇는다.
    """
    q = np.asarray(q, dtype=float)
    if q_prev is not None and float(np.dot(q, np.asarray(q_prev, dtype=float))) < 0.0:
        return -q
    return q
