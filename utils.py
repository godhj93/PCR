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
            metrics = {"RMSE": torch.sqrt(((apply_transform(source, T) - target) ** 2).mean())}
        metrics_list.append({k: (v.item() if isinstance(v, Tensor) else float(v)) for k, v in metrics.items()})
    return metrics_list


def train(cfg):
    """Hydra config를 받아 학습을 수행하는 자리 placeholder."""
    raise NotImplementedError("train() 함수는 utils.py에 구현해야 합니다.")


def test(cfg):
    """Hydra config를 받아 평가를 수행하는 자리 placeholder."""
    raise NotImplementedError("test() 함수는 utils.py에 구현해야 합니다.")


def print_metrics(metrics: dict, header: str = "Metrics") -> None:
    """정합 지표 사전을 보기 좋게 출력한다."""
    print(f"\n{header}:")
    for key, value in metrics.items():
        scalar = value.item() if isinstance(value, Tensor) else float(value)
        print(f" - {key}: {scalar:.6f}")


def _save_animation(ani, filename: str, fps: int) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix == ".gif":
        ani.save(filename, writer="pillow", fps=fps)
        return
    if suffix in {".mp4", ".m4v", ".mov", ".avi"}:
        from matplotlib.animation import FFMpegWriter  # type: ignore

        writer = FFMpegWriter(fps=fps)
        ani.save(filename, writer=writer)
        return
    raise ValueError(f"지원하지 않는 애니메이션 포맷입니다: '{suffix}'")


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
        rmse = frame_metrics[frame_idx].get("RMSE") if frame_metrics is not None else None
        rre = frame_metrics[frame_idx].get("RRE(deg)") if frame_metrics is not None else None
        rte = frame_metrics[frame_idx].get("RTE") if frame_metrics is not None else None
        rreg = frame_metrics[frame_idx].get("RREG") if frame_metrics is not None else None
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
    _save_animation(ani, filename, fps)
    plt.close(fig)


def create_registration_comparison_animation(
    source: Tensor,
    target: Tensor,
    transforms_a: Sequence[Tensor],
    transforms_b: Sequence[Tensor],
    *,
    filename: str = "registration_comparison.gif",
    frame_times: Optional[Iterable[float]] = None,
    frame_metrics_a: Optional[Sequence[dict]] = None,
    frame_metrics_b: Optional[Sequence[dict]] = None,
    titles: tuple[str, str] = ("Path A", "Path B"),
    fps: int = 15,
    overlap_count: Optional[int] = None,
    overlap_alpha: float = 0.8,
    unique_alpha: float = 0.25,
    overlap_color: str = "tab:red",
    source_color: str = "tab:blue",
    target_color: str = "tab:orange",
) -> None:
    """두 개의 경로 결과를 나란히 비교하는 애니메이션을 생성한다."""
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.animation import FuncAnimation  # type: ignore

    if len(transforms_a) != len(transforms_b):
        raise ValueError("transforms_a와 transforms_b는 동일한 길이여야 합니다.")

    num_frames = len(transforms_a)
    if frame_times is None:
        frame_times = torch.linspace(0.0, 1.0, steps=num_frames).tolist()
    time_list = list(frame_times)
    if len(time_list) != num_frames:
        raise ValueError("frame_times 길이는 transforms와 동일해야 합니다.")

    device = source.device
    dtype = source.dtype

    def _prepare(transforms: Sequence[Tensor]) -> list:
        return [
            apply_transform(source, T.to(device=device, dtype=dtype)).cpu().numpy()
            for T in transforms
        ]

    transformed_a = _prepare(transforms_a)
    transformed_b = _prepare(transforms_b)
    source_np = source.cpu().numpy()
    target_np = target.cpu().numpy()

    overlap_len = overlap_count or 0
    overlap_len = min(overlap_len, source_np.shape[0], target_np.shape[0])
    source_overlap_np = source_np[:overlap_len] if overlap_len > 0 else None
    source_unique_np = source_np[overlap_len:] if source_np.shape[0] > overlap_len else None
    target_overlap_np = target_np[:overlap_len] if overlap_len > 0 else None
    target_unique_np = target_np[overlap_len:] if target_np.shape[0] > overlap_len else None

    # 모든 경로를 아우르는 범위 계산
    all_points = torch.cat(
        [source]
        + [torch.from_numpy(tp).to(device=device, dtype=dtype) for tp in transformed_a]
        + [torch.from_numpy(tp).to(device=device, dtype=dtype) for tp in transformed_b]
        + [target],
        dim=0,
    )
    mins = all_points.min(dim=0).values.cpu().numpy()
    maxs = all_points.max(dim=0).values.cpu().numpy()
    margin = 0.1 * (maxs - mins + 1e-6)
    xlim = (mins[0] - margin[0], maxs[0] + margin[0])
    ylim = (mins[1] - margin[1], maxs[1] + margin[1])
    zlim = (mins[2] - margin[2], maxs[2] + margin[2])

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(bottom=0.2)  # 하단 여백 확보
    axes = [
        fig.add_subplot(1, 2, idx + 1, projection="3d")
        for idx in range(2)
    ]

    source_overlap_scatter_a: Optional[object] = None
    source_unique_scatter_a: Optional[object] = None
    source_overlap_scatter_b: Optional[object] = None
    source_unique_scatter_b: Optional[object] = None

    for idx, ax in enumerate(axes):
        current_pts = transformed_a[0] if idx == 0 else transformed_b[0]
        title = titles[idx]

        def legend_label(name: str) -> str:
            return name if idx == 0 else "_nolegend_"

        # Static target points
        if target_overlap_np is not None:
            ax.scatter(
                target_overlap_np[:, 0],
                target_overlap_np[:, 1],
                target_overlap_np[:, 2],
                s=4,
                alpha=overlap_alpha,
                color=overlap_color,
                label=legend_label("overlap"),
            )
        if target_unique_np is not None and target_unique_np.shape[0] > 0:
            ax.scatter(
                target_unique_np[:, 0],
                target_unique_np[:, 1],
                target_unique_np[:, 2],
                s=3,
                alpha=unique_alpha,
                color=target_color,
                label=legend_label("target-only"),
            )

        # Dynamic source points
        scatter_overlap = None
        scatter_unique = None
        if overlap_len > 0:
            scatter_overlap = ax.scatter(
                current_pts[:overlap_len, 0],
                current_pts[:overlap_len, 1],
                current_pts[:overlap_len, 2],
                s=4,
                alpha=overlap_alpha,
                color=overlap_color,
                label="_nolegend_",
            )
        if current_pts.shape[0] > overlap_len:
            scatter_unique = ax.scatter(
                current_pts[overlap_len:, 0],
                current_pts[overlap_len:, 1],
                current_pts[overlap_len:, 2],
                s=3,
                alpha=unique_alpha,
                color=source_color,
                label=legend_label("source-only"),
            )

        if idx == 0:
            source_overlap_scatter_a = scatter_overlap
            source_unique_scatter_a = scatter_unique
        else:
            source_overlap_scatter_b = scatter_overlap
            source_unique_scatter_b = scatter_unique

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        axes[0].legend(loc="upper right")

    def _title_text(base_title: str, metrics: Optional[Sequence[dict]], frame_idx: int) -> str:
        lines = [base_title]
        if metrics is not None:
            rmse = metrics[frame_idx].get("RMSE")
            rreg = metrics[frame_idx].get("RREG")
            rre = metrics[frame_idx].get("RRE(deg)")
            rte = metrics[frame_idx].get("RTE")

            top_line = []
            if rmse is not None:
                top_line.append(f"RMSE={rmse:.4f}")
            if rreg is not None:
                top_line.append(f"RReg={rreg:.4f}")
            if top_line:
                lines.append(" | ".join(top_line))

            bottom_line = []
            if rre is not None:
                bottom_line.append(f"RRE={rre:.2f}deg")
            if rte is not None:
                bottom_line.append(f"RTE={rte:.4f}")
            if bottom_line:
                lines.append(" | ".join(bottom_line))
        return "\n".join(lines)

    axes[0].set_title(_title_text(titles[0], frame_metrics_a, 0))
    axes[1].set_title(_title_text(titles[1], frame_metrics_b, 0))

    time_text = fig.text(0.5, 0.05, f"t={time_list[0]:.2f}", ha="center")

    def _update(frame: int):
        pts_a = transformed_a[frame]
        pts_b = transformed_b[frame]
        updated_artists = []

        if overlap_len > 0 and source_overlap_scatter_a is not None:
            source_overlap_scatter_a._offsets3d = (
                pts_a[:overlap_len, 0],
                pts_a[:overlap_len, 1],
                pts_a[:overlap_len, 2],
            )
            updated_artists.append(source_overlap_scatter_a)
        if source_unique_scatter_a is not None:
            source_unique_scatter_a._offsets3d = (
                pts_a[overlap_len:, 0],
                pts_a[overlap_len:, 1],
                pts_a[overlap_len:, 2],
            )
            updated_artists.append(source_unique_scatter_a)

        if overlap_len > 0 and source_overlap_scatter_b is not None:
            source_overlap_scatter_b._offsets3d = (
                pts_b[:overlap_len, 0],
                pts_b[:overlap_len, 1],
                pts_b[:overlap_len, 2],
            )
            updated_artists.append(source_overlap_scatter_b)
        if source_unique_scatter_b is not None:
            source_unique_scatter_b._offsets3d = (
                pts_b[overlap_len:, 0],
                pts_b[overlap_len:, 1],
                pts_b[overlap_len:, 2],
            )
            updated_artists.append(source_unique_scatter_b)

        axes[0].set_title(_title_text(titles[0], frame_metrics_a, frame))
        axes[1].set_title(_title_text(titles[1], frame_metrics_b, frame))
        time_text.set_text(f"t={time_list[frame]:.2f}")
        return updated_artists

    ani = FuncAnimation(fig, _update, frames=num_frames, interval=80, blit=False)
    _save_animation(ani, filename, fps)
    plt.close(fig)
