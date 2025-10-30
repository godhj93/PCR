"""공통 유틸리티: 변환 적용, 지표 계산, 애니메이션 생성."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
from torch import Tensor

from metric import compute_metrics


def apply_transform(points: Tensor, transform: Tensor) -> Tensor:
    """4×4 동차변환 행렬을 점 집합(N×3)에 적용한다."""
    if points.shape[-1] != 3:
        raise ValueError("points는 (..., 3) 형태여야 합니다.")
    if transform.shape[-2:] != (4, 4):
        raise ValueError("transform은 (..., 4, 4) 형태여야 합니다.")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return points @ rotation.transpose(-1, -2) + translation


def compute_metrics_over_transforms(
    source: Tensor,
    target: Tensor,
    transforms: Sequence[Tensor],
    *,
    transform_gt: Optional[Tensor] = None,
) -> List[dict]:
    """각 변환에 대한 정합 지표를 계산한다."""
    metrics_list: List[dict] = []
    for T in transforms:
        if transform_gt is not None:
            metrics = compute_metrics(
                source=source,
                target=target,
                transform_gt=transform_gt,
                transform_est=T,
            )
        else:
            metrics = {"rmse": torch.sqrt(((apply_transform(source, T) - target) ** 2).mean())}
        metrics_list.append({k: v.item() for k, v in metrics.items()})
    return metrics_list


def print_metrics(metrics: dict, header: str = "Metrics") -> None:
    """정합 지표 사전을 보기 좋게 출력한다."""
    print(f"\n{header}:")
    for key, value in metrics.items():
        scalar = value.item() if isinstance(value, Tensor) else float(value)
        print(f" - {key}: {scalar:.6f}")


def create_registration_animation(
    source: Tensor,
    target: Tensor,
    transforms: Sequence[Tensor],
    *,
    filename: str = "registration_animation.gif",
    frame_times: Optional[Iterable[float]] = None,
    frame_metrics: Optional[Sequence[dict]] = None,
    title_prefix: str = "3D Point Registration",
    fps: int = 15,
) -> None:
    """등록 과정 애니메이션을 생성해 GIF로 저장한다."""
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.animation import FuncAnimation  # type: ignore

    if frame_times is None:
        frame_times = torch.linspace(0.0, 1.0, steps=len(transforms)).tolist()
    time_list = list(frame_times)
    if len(time_list) != len(transforms):
        raise ValueError("frame_times 길이는 transforms 길이와 동일해야 합니다.")

    device = source.device
    dtype = source.dtype

    transformed_points = [
        apply_transform(source, T.to(device=device, dtype=dtype)).cpu().numpy()
        for T in transforms
    ]
    source_np = source.cpu().numpy()
    target_np = target.cpu().numpy()

    # 좌표계 범위 계산
    all_points = torch.cat(
        [source] + [torch.from_numpy(tp).to(device=device, dtype=dtype) for tp in transformed_points] + [target],
        dim=0,
    )
    mins = all_points.min(dim=0).values.cpu().numpy()
    maxs = all_points.max(dim=0).values.cpu().numpy()
    margin = 0.1 * (maxs - mins + 1e-6)
    xlim = (mins[0] - margin[0], maxs[0] + margin[0])
    ylim = (mins[1] - margin[1], maxs[1] + margin[1])
    zlim = (mins[2] - margin[2], maxs[2] + margin[2])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter_source = ax.scatter(source_np[:, 0], source_np[:, 1], source_np[:, 2], s=3, alpha=0.2, label="source")
    scatter_target = ax.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], s=3, alpha=0.4, label="target")
    current_pts = transformed_points[0]
    scatter_current = ax.scatter(current_pts[:, 0], current_pts[:, 1], current_pts[:, 2], s=3, alpha=0.8, label="estimate")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")

    def _format_title(frame_idx: int) -> str:
        rmse = frame_metrics[frame_idx]["rmse"] if frame_metrics is not None else None
        rre = frame_metrics[frame_idx].get("rre_deg") if frame_metrics is not None else None
        rte = frame_metrics[frame_idx].get("rte") if frame_metrics is not None else None
        rreg = frame_metrics[frame_idx].get("rreg") if frame_metrics is not None else None
        parts = [f"{title_prefix}", f"s={time_list[frame_idx]:.2f}"]
        if rmse is not None:
            parts.append(f"RMSE={rmse:.4f}")
        if rre is not None:
            parts.append(f"RRE={rre:.2f}deg")
        if rte is not None:
            parts.append(f"RTE={rte:.4f}")
        if rreg is not None:
            parts.append(f"RReg={rreg:.4f}")
        return " | ".join(parts)

    ax.set_title(_format_title(0))

    def _update(frame: int):
        pts = transformed_points[frame]
        scatter_current._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_title(_format_title(frame))
        return scatter_source, scatter_target, scatter_current

    ani = FuncAnimation(fig, _update, frames=len(transforms), interval=80, blit=False)
    ani.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
