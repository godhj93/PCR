"""3D 포인트 정합용 핵심 평가 지표(RMSE, RRE, RTE, RREg) 정의."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def rmse(predicted: Tensor, target: Tensor) -> Tensor:
    """예측 포인트 집합과 기준 포인트 집합 간 RMSE를 계산한다."""
    if predicted.shape != target.shape:
        raise ValueError("predicted와 target의 텐서 크기가 일치해야 합니다.")
    mse = ((predicted - target) ** 2).mean()
    return torch.sqrt(mse)


def relative_rotation_error(R_gt: Tensor, R_est: Tensor, *, degrees: bool = False) -> Tensor:
    """Relative Rotation Error(RRE): R_gt 와 R_est 간 회전 오차 (라디안 또는 도)."""
    if R_gt.shape != (3, 3) or R_est.shape != (3, 3):
        raise ValueError("R_gt와 R_est 모두 (3, 3) 회전 행렬이어야 합니다.")
    relative = R_gt.T @ R_est
    trace_val = torch.clamp((torch.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.arccos(trace_val)
    return torch.rad2deg(angle) if degrees else angle


def relative_translation_error(t_gt: Tensor, t_est: Tensor) -> Tensor:
    """Relative Translation Error(RTE): 두 평행이동 벡터 간 L2 거리."""
    if t_gt.shape != (3,) or t_est.shape != (3,):
        raise ValueError("t_gt와 t_est 모두 길이 3의 벡터여야 합니다.")
    return torch.linalg.norm(t_gt - t_est, ord=2)


def relative_registration_error(
    source: Tensor,
    target: Tensor,
    transform_gt: Tensor,
    transform_est: Tensor,
) -> Tensor:
    """
    Relative Registration Error(RREg):
    추정 변환과 기준 변환을 적용한 포인트 위치간의 RMSE.
    """
    if transform_gt.shape != (4, 4) or transform_est.shape != (4, 4):
        raise ValueError("transform_gt와 transform_est는 (4, 4) 동차변환 행렬이어야 합니다.")
    R_gt = transform_gt[:3, :3]
    t_gt = transform_gt[:3, 3]
    R_est = transform_est[:3, :3]
    t_est = transform_est[:3, 3]
    target_from_gt = source @ R_gt.T + t_gt
    target_from_est = source @ R_est.T + t_est
    return rmse(target_from_est, target_from_gt)


def compute_metrics(
    *,
    source: Tensor,
    target: Tensor,
    transform_gt: Tensor,
    transform_est: Tensor,
) -> Dict[str, Tensor]:
    """위 네 가지 지표를 모두 계산해 사전(dict)으로 반환한다."""
    R_gt = transform_gt[:3, :3]
    t_gt = transform_gt[:3, 3]
    R_est = transform_est[:3, :3]
    t_est = transform_est[:3, 3]

    predicted_target = source @ R_est.T + t_est

    return {
        "rmse": rmse(predicted_target, target),
        "rre_rad": relative_rotation_error(R_gt, R_est, degrees=False),
        "rre_deg": relative_rotation_error(R_gt, R_est, degrees=True),
        "rte": relative_translation_error(t_gt, t_est),
        "rreg": relative_registration_error(source, target, transform_gt, transform_est),
    }

