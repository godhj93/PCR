#!/usr/bin/env python
"""Utility to inspect a single HeLiPR Ouster LiDAR scan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a single Ouster LiDAR frame from the HeLiPR dataset."
    )
    parser.add_argument(
        "sequence",
        help="Sequence folder name under the dataset root (e.g. 'Bridge01').",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--timestamp",
        help="Exact filename (without extension) to load within the sequence's Ouster folder.",
    )
    group.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index into the sorted list of available scans (default: 0).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hj/datasets/scans"),
        help="Root directory that contains the HeLiPR scans (default: %(default)s).",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=1000.0,
        help="Discard points whose |x|, |y|, or |z| exceed this value (meters).",
    )
    parser.add_argument(
        "--min-norm",
        type=float,
        default=1e-3,
        help="Drop points whose Euclidean norm is below this threshold (meters).",
    )
    parser.add_argument(
        "--intensity-range",
        type=float,
        nargs=2,
        default=(0.0, 255.0),
        metavar=("MIN", "MAX"),
        help="Valid intensity range; points outside are filtered out.",
    )
    parser.add_argument(
        "--save-as",
        type=Path,
        help="Optional path to save the filtered point cloud as a NumPy .npz file.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Path to save a rendered 3D scatter plot image (default: scripts/<sequence>_<scan>.png).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip visualization entirely, even if a default save path is available.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20000,
        help="Maximum number of points to include in the visualization (down-sampled if exceeded).",
    )
    return parser.parse_args()


@dataclass
class PointCloudSummary:
    count: int
    centroid: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    intensity_stats: tuple[float, float, float]


def summarize_point_cloud(points: np.ndarray, intensities: np.ndarray) -> PointCloudSummary:
    centroid = points.mean(axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    intensity_mean = float(intensities.mean()) if intensities.size else float("nan")
    intensity_min = float(intensities.min()) if intensities.size else float("nan")
    intensity_max = float(intensities.max()) if intensities.size else float("nan")
    return PointCloudSummary(
        count=points.shape[0],
        centroid=centroid,
        mins=mins,
        maxs=maxs,
        intensity_stats=(intensity_min, intensity_mean, intensity_max),
    )


def load_ouster_scan_bytes(path: Path) -> np.ndarray:
    """Read the raw binary file as a flat float32 array."""
    data = np.fromfile(path, dtype="<f4")
    if data.size == 0:
        raise ValueError(f"{path} is empty.")
    if data.size % 4:
        # Drop partial record at the tail if the total byte count is not a multiple of 16.
        data = data[: data.size - (data.size % 4)]
    return data.reshape(-1, 4)


def filter_points(
    raw_points: np.ndarray,
    *,
    max_abs: float,
    min_norm: float,
    intensity_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = raw_points[:, :3]
    intensities = raw_points[:, 3]

    finite_mask = np.isfinite(raw_points).all(axis=1)
    range_mask = (np.abs(coords) <= max_abs).all(axis=1)
    intensity_mask = (intensities >= intensity_range[0]) & (intensities <= intensity_range[1])

    keep_mask = finite_mask & range_mask & intensity_mask
    if min_norm > 0.0:
        idx = np.nonzero(keep_mask)[0]
        if idx.size:
            selected_norms = np.linalg.norm(coords[idx], axis=1)
            keep_mask[idx] &= selected_norms >= min_norm
    return coords[keep_mask], intensities[keep_mask], keep_mask


def choose_scan_file(sequence_dir: Path, *, timestamp: str | None, index: int) -> Path:
    ouster_dir = sequence_dir / "Ouster"
    if not ouster_dir.exists():
        raise FileNotFoundError(f"{ouster_dir} does not exist.")
    bin_files = sorted(p for p in ouster_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found under {ouster_dir}.")

    if timestamp is not None:
        candidate = ouster_dir / f"{timestamp}.bin"
        if not candidate.exists():
            available = (f.stem for f in bin_files[:10])
            raise FileNotFoundError(
                f"{candidate} not found. Example available timestamps: {', '.join(available)}"
            )
        return candidate

    if index < 0 or index >= len(bin_files):
        raise IndexError(f"Index {index} out of range for {len(bin_files)} scans.")
    return bin_files[index]


def read_pose(sequence_dir: Path, timestamp: str) -> tuple[np.ndarray, np.ndarray] | None:
    gt_path = sequence_dir / "Ouster_gt.txt"
    if not gt_path.exists():
        return None
    target_ts = timestamp.split(".")[0]
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] != target_ts:
                continue
            if len(tokens) != 8:
                continue
            translation = np.array(tokens[1:4], dtype=float)
            quaternion = np.array(tokens[4:8], dtype=float)
            return translation, quaternion
    return None


def format_vector(vec: Iterable[float]) -> str:
    return ", ".join(f"{value: .3f}" for value in vec)


def maybe_visualize(
    points: np.ndarray,
    intensities: np.ndarray,
    *,
    save_path: Path | None,
    max_points: int,
    sequence: str,
    scan_name: str,
) -> None:
    if save_path is None:
        return
    if points.size == 0:
        print("Visualization skipped (no points after filtering).")
        return
    if max_points <= 0:
        print("Visualization skipped (max_plot_points set to zero or negative).")
        return

    count = points.shape[0]
    if count > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(count, size=max_points, replace=False)
        coords = points[indices]
        colors = intensities[indices]
        sampled = True
    else:
        coords = points
        colors = intensities
        sampled = False

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        print(f"Visualization skipped (matplotlib import failed: {exc})")
        return

    color_min = colors.min(initial=0.0)
    color_max = colors.max(initial=1.0)
    denom = color_max - color_min or 1.0
    color_norm = (colors - color_min) / denom

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color_norm,
        cmap="viridis",
        s=1,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    title = f"{sequence} / {scan_name}"
    if sampled:
        title += f" (sampled {coords.shape[0]} of {count} pts)"
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="Normalized intensity")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    print(f"Saved visualization to {save_path.resolve()}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sequence_dir = args.dataset_root / args.sequence
    scan_path = choose_scan_file(sequence_dir, timestamp=args.timestamp, index=args.index)
    raw = load_ouster_scan_bytes(scan_path)
    filtered_points, intensities, mask = filter_points(
        raw,
        max_abs=args.max_abs,
        min_norm=args.min_norm,
        intensity_range=tuple(args.intensity_range),
    )

    summary = summarize_point_cloud(filtered_points, intensities)
    pose = read_pose(sequence_dir, scan_path.stem)

    print(f"Sequence      : {args.sequence}")
    print(f"Scan file     : {scan_path.name}")
    print(f"Raw records   : {raw.shape[0]}")
    print(f"Filtered keep : {summary.count} ({summary.count / raw.shape[0]:.2%})")
    print(f"Centroid [m]  : {format_vector(summary.centroid)}")
    print(f"Min XYZ [m]   : {format_vector(summary.mins)}")
    print(f"Max XYZ [m]   : {format_vector(summary.maxs)}")
    print(
        "Intensity     : "
        f"min={summary.intensity_stats[0]:.2f}, "
        f"mean={summary.intensity_stats[1]:.2f}, "
        f"max={summary.intensity_stats[2]:.2f}"
    )
    if pose is not None:
        translation, quaternion = pose
        print(f"Pose (t) [m]  : {format_vector(translation)}")
        print(f"Pose (quat)   : {format_vector(quaternion)}")

    sample_preview = filtered_points[:5]
    if sample_preview.size:
        print("First 5 points [m]:")
        for idx, (x, y, z) in enumerate(sample_preview):
            print(f"  #{idx}: ({x: .3f}, {y: .3f}, {z: .3f})  intensity={intensities[idx]:.2f}")

    if args.save_as:
        output_path = args.save_as
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            points=filtered_points,
            intensity=intensities,
            keep_mask=mask,
            raw=raw,
        )
        print(f"Saved filtered points to {output_path.resolve()}")

    plot_path: Path | None
    if args.no_plot:
        plot_path = None
    else:
        plot_path = args.save_plot
        if plot_path is None:
            default_dir = Path(__file__).resolve().parent
            scan_stem = scan_path.stem
            plot_path = default_dir / f"{args.sequence}_{scan_stem}.png"

    maybe_visualize(
        filtered_points,
        intensities,
        save_path=plot_path,
        max_points=args.max_plot_points,
        sequence=args.sequence,
        scan_name=scan_path.name,
    )


if __name__ == "__main__":
    main()
