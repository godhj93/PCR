"""KD-Tree 기반 Iterative Closest Point(ICP) 구현.

`flow_matching` 라이브러리의 ``SE3`` 모듈이 제공하는 ``expmap`` / ``logmap`` 을
사용하여 SE(3) 상에서 누적 변환을 업데이트한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor

# flow_matching 및 프로젝트 루트를 임포트 경로에 추가
_REPO_DIR = Path(__file__).resolve().parents[1]
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))
_PKG_DIR = _REPO_DIR / "flow_matching"
if _PKG_DIR.exists():
    pkg_path = str(_PKG_DIR)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

from flow_matching.utils.manifolds.se3 import SE3  # noqa: E402

try:  # SciPy KD-Tree 사용
    from scipy.spatial import cKDTree
except ImportError as exc:  # pragma: no cover - SciPy 미설치 시 안내
    raise ImportError(
        "scipy.spatial.cKDTree 가 필요합니다. SciPy를 설치해 주세요."
    ) from exc

from data import PointRegistrationDataset  # noqa: E402
from metric import compute_metrics  # noqa: E402
from utils import (
    apply_transform,
    compute_metrics_over_transforms,
    create_registration_animation,
    print_metrics,
)


@dataclass
class ICPResult:
    """ICP 최적화 결과."""

    transform: Tensor  # 4×4 동차변환 행렬
    rotation: Tensor   # 3×3 회전 행렬
    translation: Tensor  # 3×1 평행이동 벡터
    transform_history: List[Tensor]
    errors: List[float]  # 반복별 평균 포인트 거리
    iterations: int
    converged: bool


def iterative_closest_point(
    source: Tensor,
    target: Tensor,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    initial_rotation: Optional[Tensor] = None,
    initial_translation: Optional[Tensor] = None,
) -> ICPResult:
    """KD-Tree 기반 point-to-point ICP.

    Args:
        source: (N, 3) 원본 포인트 클라우드.
        target: (M, 3) 타깃 포인트 클라우드.
        max_iterations: 최대 반복 횟수.
        tolerance: 이전 평균 오차 대비 변화량이 이 값 아래로 떨어지면 수렴으로 간주.
        initial_rotation: 초기 회전(3×3).
        initial_translation: 초기 평행이동(3,).
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("source와 target은 (N,3), (M,3)의 2차원 텐서여야 합니다.")
    if source.shape[0] < 3 or target.shape[0] < 3:
        raise ValueError("source와 target 모두 최소 3개의 점을 포함해야 합니다.")

    device = source.device
    dtype = source.dtype

    # 초기 변환
    R0 = (
        initial_rotation.to(device=device, dtype=dtype)
        if initial_rotation is not None
        else torch.eye(3, dtype=dtype, device=device)
    )
    t0 = (
        initial_translation.to(device=device, dtype=dtype)
        if initial_translation is not None
        else torch.zeros(3, dtype=dtype, device=device)
    )
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R0
    T[:3, 3] = t0
    transform_history: List[Tensor] = [T.detach().clone()]

    # KD-Tree는 CPU numpy 배열 사용
    target_cpu = target.detach().cpu().numpy()
    kdtree = cKDTree(target_cpu)

    se3 = SE3().to(device)
    identity = torch.eye(4, dtype=dtype, device=device)

    errors: List[float] = []
    prev_error = torch.tensor(float("inf"), dtype=dtype, device=device)

    source_device = source.to(device=device, dtype=dtype)
    target_device = target.to(device=device, dtype=dtype)
    converged = False

    for iteration in range(1, max_iterations + 1):
        transformed = apply_transform(source_device, T)

        distances, indices = kdtree.query(transformed.detach().cpu().numpy())
        indices_tensor = torch.as_tensor(indices, device=target_device.device, dtype=torch.long)
        matched = target_device.index_select(0, indices_tensor)

        src_centroid = transformed.mean(dim=0, keepdim=True)
        tgt_centroid = matched.mean(dim=0, keepdim=True)

        src_centered = transformed - src_centroid
        tgt_centered = matched - tgt_centroid

        H = src_centered.T @ tgt_centered
        U, _, Vh = torch.linalg.svd(H)
        R_delta = Vh.T @ U.T
        if torch.det(R_delta) < 0:
            Vh[..., -1, :] *= -1
            R_delta = Vh.T @ U.T
        t_delta = (tgt_centroid - src_centroid @ R_delta.T).squeeze(0)

        delta_T = torch.eye(4, dtype=dtype, device=device)
        delta_T[:3, :3] = R_delta
        delta_T[:3, 3] = t_delta
        # 로그맵으로 delta twist 구한 후 expmap으로 누적 업데이트
        xi_hat = se3.logmap(identity, delta_T)
        delta_T_exp = se3.expmap(identity, xi_hat)
        T = se3.projx(delta_T_exp @ T)

        transformed = apply_transform(source_device, T)
        current_error = torch.norm(transformed - matched, dim=1).mean()
        errors.append(float(current_error.item()))

        transform_history.append(T.detach().clone())

        if torch.abs(prev_error - current_error) < tolerance:
            converged = True
            break
        prev_error = current_error

    rotation = T[:3, :3]
    translation = T[:3, 3]
    return ICPResult(
        transform=T,
        rotation=rotation,
        translation=translation,
        transform_history=transform_history,
        errors=errors,
        iterations=len(errors),
        converged=converged,
    )


if __name__ == "__main__":  # PointRegistrationDataset 기반 간단 검증
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    dataset = PointRegistrationDataset(
        num_samples=1,
        num_points=1_000,
        point_scale=1.5,
        translation_scale=0.5,
        target_noise_std=0.01,
        dtype=dtype,
        seed=42,
        shape="sphere",
        overlap_ratio=0.7,
        target_point_ratio=1.2,
    )
    sample = dataset[0]
    source = sample.source.to(device)
    target = sample.target.to(device)
    true_transform = sample.transform.to(device)
    overlap_count = sample.overlap_count
    overlap_source = source[:overlap_count]
    overlap_target = target[:overlap_count]

    axis_noise = torch.tensor([0.2, -0.3, 0.5], device=device, dtype=dtype)
    axis_noise = axis_noise / axis_noise.norm()
    angle_noise = 0.1
    c = torch.cos(torch.tensor(angle_noise, device=device, dtype=dtype))
    s = torch.sin(torch.tensor(angle_noise, device=device, dtype=dtype))
    nx, ny, nz = axis_noise
    R_noise = torch.tensor(
        [
            [c + nx * nx * (1 - c), nx * ny * (1 - c) - nz * s, nx * nz * (1 - c) + ny * s],
            [ny * nx * (1 - c) + nz * s, c + ny * ny * (1 - c), ny * nz * (1 - c) - nx * s],
            [nz * nx * (1 - c) - ny * s, nz * ny * (1 - c) + nx * s, c + nz * nz * (1 - c)],
        ],
        device=device,
        dtype=dtype,
    )
    init_rot = sample.rotation.to(device) @ R_noise
    init_trans = sample.translation.to(device) + torch.tensor([0.05, -0.04, 0.02], device=device, dtype=dtype)

    result = iterative_closest_point(
        source,
        target,
        max_iterations=1000,
        tolerance=1e-9,
        initial_rotation=torch.eye(3, device=device, dtype=dtype),  # 혹은 큰 각도 노이즈
        initial_translation=torch.zeros(3, device=device, dtype=dtype),
    )


    metrics = compute_metrics(
        source=overlap_source,
        target=overlap_target,
        transform_gt=true_transform,
        transform_est=result.transform,
    )

    print("Ground-truth transform T_gt:")
    print(true_transform.cpu().numpy())
    print("\nEstimated transform T_icp:")
    print(result.transform.cpu().numpy())
    print(f"\nOverlap count: {overlap_count} / source={source.shape[0]} / target={target.shape[0]}")
    print(f"\nIterations: {result.iterations} | Converged: {result.converged}")
    if result.errors:
        print(f"Final mean correspondence error: {result.errors[-1]:.6f}")

    print_metrics(metrics, header="Metrics (RMSE / RRE / RTE / RReg)")

    frame_metrics = compute_metrics_over_transforms(
        overlap_source,
        overlap_target,
        result.transform_history,
        transform_gt=true_transform,
    )

    create_registration_animation(
        source,
        target,
        result.transform_history,
        filename="icp_registration.gif",
        frame_times=list(range(len(result.transform_history))),
        frame_metrics=frame_metrics,
        title_prefix="ICP Registration",
    )
    print("\n애니메이션을 'icp_registration.gif' 파일로 저장했습니다.")
