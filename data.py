"""PyTorch를 이용해 3차원 포인트 정합용 합성 데이터를 생성하는 유틸리티."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from pathlib import Path

# 로컬 레포에서 직접 모듈을 가져오기 위해 패키지 루트를 경로에 추가한다.
_REPO_DIR = Path(__file__).resolve().parent
_PKG_DIR = _REPO_DIR / "flow_matching"
if _PKG_DIR.exists():
    pkg_path = str(_PKG_DIR)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

from flow_matching.path.tau_constrained_rigid_path import TauConstrainedRigidPath
from flow_matching.utils.manifolds.se3 import SE3
from torch import Tensor
from torch.utils.data import Dataset

from metric import compute_metrics
from utils import (
    apply_transform,
    compute_metrics_over_transforms,
    create_registration_animation,
    create_registration_comparison_animation,
    print_metrics,
)


def generate_random_point_cloud(
    num_points: int,
    scale: float = 1.0,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
    shape: str = "cube",
) -> Tensor:
    """지정된 기하 형태에 맞추어 포인트 클라우드를 샘플링한다.

    shape:
        - "cube": 한 변의 길이가 `scale`인 정육면체 내부에서 균일 분포로 샘플링.
        - "sphere": 반지름이 `scale`인 구의 표면에서 균일하게 샘플링.
    """
    device = torch.device(device)
    shape = shape.lower()

    if shape == "cube":
        points = torch.rand((num_points, 3), dtype=dtype, device=device, generator=generator)
        points = (points - 0.5) * scale
        return points

    if shape == "sphere":
        points = torch.randn((num_points, 3), dtype=dtype, device=device, generator=generator)
        points = F.normalize(points, dim=-1)
        return points * scale

    raise ValueError(f"지원하지 않는 shape '{shape}' 값입니다. 'cube' 또는 'sphere'를 사용하세요.")


def random_rotation(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """SVD 기반 샘플링으로 SO(3)에서 무작위 회전 행렬을 생성한다."""
    device = torch.device(device)
    normal = torch.randn((3, 3), dtype=dtype, device=device, generator=generator)
    u, _, vh = torch.linalg.svd(normal)
    r = u @ vh
    if torch.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    return r


def random_translation(
    *,
    translation_scale: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """각 좌표축을 [-translation_scale, translation_scale]에서 균일 샘플링해 평행이동 벡터를 만든다."""
    device = torch.device(device)
    t = torch.empty(3, dtype=dtype, device=device)
    t.uniform_(-translation_scale, translation_scale, generator=generator)
    return t


@dataclass
class PointRegistrationSample:
    source: Tensor
    target: Tensor
    transform: Tensor  # 4×4 동차변환행렬 (target = transform @ source)
    rotation: Tensor   # 변환 행렬의 3×3 회전 블록
    translation: Tensor  # 3차원 평행이동 벡터
    overlap_count: int    # source/target 간 실제 겹치는 점 개수


class PointRegistrationDataset(Dataset[PointRegistrationSample]):
    """강체 3D 포인트 정합을 위한 합성 데이터셋."""

    def __init__(
        self,
        num_samples: int,
        num_points: int,
        *,
        point_scale: float = 1.0,
        translation_scale: float = 0.5,
        target_noise_std: float = 0.0,
        overlap_ratio: float = 1.0,
        target_point_ratio: float = 1.0,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        shape: str = "cube",
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples는 1보다 커야 합니다.")
        if num_points <= 0:
            raise ValueError("num_points는 1보다 커야 합니다.")
        if not 0.0 < overlap_ratio <= 1.0:
            raise ValueError("overlap_ratio는 0과 1 사이여야 합니다.")
        if target_point_ratio <= 0.0:
            raise ValueError("target_point_ratio는 0보다 커야 합니다.")
        self.num_samples = num_samples
        self.num_points = num_points
        self.point_scale = point_scale
        self.translation_scale = translation_scale
        self.target_noise_std = target_noise_std
        self.dtype = dtype
        self.shape = shape
        self.overlap_ratio = overlap_ratio
        self.target_point_ratio = target_point_ratio

        self._device = torch.device("cpu")
        self._generator = torch.Generator(device=self._device)
        if seed is not None:
            self._generator.manual_seed(seed)

        (self._sources,
         self._targets,
         self._transforms,
         self._overlap_counts) = self._generate_dataset()

    def _generate_dataset(self) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[int]]:
        sources: list[Tensor] = []
        targets: list[Tensor] = []
        transforms: list[Tensor] = []
        overlap_counts: list[int] = []

        for idx in range(self.num_samples):
            target_total = max(1, int(round(self.num_points * self.target_point_ratio)))
            num_overlap = min(int(round(self.num_points * self.overlap_ratio)), self.num_points, target_total)
            num_unique_source = self.num_points - num_overlap
            num_unique_target = target_total - num_overlap

            shared_points = generate_random_point_cloud(
                num_overlap,
                scale=self.point_scale,
                dtype=self.dtype,
                device=self._device,
                generator=self._generator,
                shape=self.shape,
            )
            source_unique = generate_random_point_cloud(
                num_unique_source,
                scale=self.point_scale,
                dtype=self.dtype,
                device=self._device,
                generator=self._generator,
                shape=self.shape,
            ) if num_unique_source > 0 else torch.empty((0, 3), dtype=self.dtype, device=self._device)

            rotation = random_rotation(
                dtype=self.dtype,
                device=self._device,
                generator=self._generator,
            )
            translation = random_translation(
                translation_scale=self.translation_scale,
                dtype=self.dtype,
                device=self._device,
                generator=self._generator,
            )
            transform = torch.eye(4, dtype=self.dtype, device=self._device)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation

            shared_transformed = apply_transform(shared_points, transform)
            target_unique = generate_random_point_cloud(
                num_unique_target,
                scale=self.point_scale,
                dtype=self.dtype,
                device=self._device,
                generator=self._generator,
                shape=self.shape,
            ) if num_unique_target > 0 else torch.empty((0, 3), dtype=self.dtype, device=self._device)

            source = torch.cat([shared_points, source_unique], dim=0)
            target = torch.cat([shared_transformed, target_unique], dim=0)
            if self.target_noise_std > 0.0:
                noise = torch.randn(
                    (target.shape[0], 3),
                    dtype=self.dtype,
                    device=self._device,
                    generator=self._generator,
                )
                target = target + noise * self.target_noise_std

            sources.append(source)
            targets.append(target)
            transforms.append(transform)
            overlap_counts.append(num_overlap)

        return sources, targets, transforms, overlap_counts

    def __len__(self) -> int:
        return len(self._sources)

    def __getitem__(self, index: int) -> PointRegistrationSample:
        if not 0 <= index < self.num_samples:
            raise IndexError("데이터셋 인덱스가 범위를 벗어났습니다.")

        transform = self._transforms[index]
        return PointRegistrationSample(
            source=self._sources[index],
            target=self._targets[index],
            transform=transform,
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            overlap_count=self._overlap_counts[index],
        )


if __name__ == "__main__":
    # 합성 데이터셋을 초기화하고 샘플을 뽑는다.
    dataset = PointRegistrationDataset(
        num_samples=1,
        num_points=1_000,
        point_scale=2.0,
        translation_scale=10.0,
        target_noise_std=0.02,
        seed=42,
        shape="sphere",
        overlap_ratio=0.7,
    )
    sample = dataset[0]
    source = sample.source
    target = sample.target
    rotation = sample.rotation
    translation = sample.translation
    transform = sample.transform
    overlap_count = sample.overlap_count
    overlap_source = source[:overlap_count]
    overlap_target = target[:overlap_count]

    # SE(3) 로그맵으로 twist를 구하고, 지수맵으로 다시 복원해본다.
    se3 = SE3()
    identity = torch.eye(4, dtype=dataset.dtype)
    twist_matrix = se3.logmap(identity, transform)
    twist6 = se3._vee6(twist_matrix)
    recovered_transform = se3.expmap(identity, twist_matrix)
    metrics = compute_metrics(
        source=overlap_source,
        target=overlap_target,
        transform_gt=transform,
        transform_est=recovered_transform,
    )
    registration_rmse = metrics["RMSE"].item()
    rotation_error_rad = metrics["RRE(rad)"].item()
    rotation_error_deg = metrics["RRE(deg)"].item()
    translation_error = metrics["RTE"].item()
    relative_reg_error = metrics["RREG"].item()

    # 시각화용 NumPy 배열로 변환
    source_np = source.numpy()
    target_np = target.numpy()
    rotation_np = rotation.numpy()
    translation_np = translation.numpy()

    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(source_np[:, 0], source_np[:, 1], source_np[:, 2], s=3, alpha=0.6, label="source")
    ax.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], s=3, alpha=0.6, label="target")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(
        "Synthetic Point Registration Pair\n"
        f"Shape={dataset.shape} | det(R)={torch.linalg.det(rotation):.3f} | "
        f"RMSE={registration_rmse:.4f}, RRE={rotation_error_deg:.2f}deg, "
        f"RTE={translation_error:.4f}, RReg={relative_reg_error:.4f}"
    )

    print("기준 변환 T_true의 회전 행렬 R:")
    print(rotation_np)
    print("\n기준 평행이동 벡터 t:")
    print(translation_np)
    print(f"\n겹치는 점 개수: {overlap_count} / source={source.shape[0]} / target={target.shape[0]}")
    print("\nSE(3) 로그맵에서 얻은 twist [rho(평행이동), phi(회전)]:")
    print(twist6.numpy())
    print_metrics(metrics, header="정합 평가 지표 (RMSE / RRE / RTE / RReg)")

    plt.tight_layout()
    plt.savefig("random_point_cloud.png", dpi=300)

    # 로그맵으로 얻은 변환을 등분해하여 애니메이션으로 저장
    geodesic_steps = 60
    geodesic_ts = torch.linspace(0.0, 1.0, steps=geodesic_steps, dtype=dataset.dtype)
    geodesic_transforms = [se3.expmap(identity, twist_matrix * s) for s in geodesic_ts]
    geodesic_metrics = compute_metrics_over_transforms(
        overlap_source,
        overlap_target,
        geodesic_transforms,
        transform_gt=transform,
    )
    geodesic_final_metrics = geodesic_metrics[-1]
    print_metrics(geodesic_final_metrics, header="SE(3) Geodesic 최종 정합 지표 (RMSE / RRE / RTE / RReg)")
    # create_registration_animation(
    #     source,
    #     target,
    #     geodesic_transforms,
    #     filename="registration_animation.gif",
    #     frame_times=[float(v) for v in geodesic_ts.tolist()],
    #     frame_metrics=geodesic_metrics,
    #     title_prefix="SE(3) Geodesic Interpolation",
    # )
    # print("\n애니메이션을 'registration_animation.gif' 파일로 저장했습니다.")

    # Tau-constrained rigid probability path을 활용한 동일 정합 경로 시각화
    tau_path = TauConstrainedRigidPath(detach_targets=True)
    x0_batch = overlap_source.unsqueeze(0)
    x1_batch = overlap_target.unsqueeze(0)

    tau_steps = 60
    tau_ts = torch.linspace(0.0, 1.0, steps=tau_steps, dtype=dataset.dtype)
    tau_transforms: list[Tensor] = []

    with torch.no_grad():
        mu_P, mu_Q = tau_path._centroids(x0_batch, x1_batch)
        R_star = tau_path._solve_R_star(x0_batch, x1_batch, mu_P=mu_P, mu_Q=mu_Q)
        omega_star = tau_path._log_at_identity(R_star)

        for t_val in tau_ts:
            t_batch = t_val.unsqueeze(0)
            omega_t = omega_star * t_batch.unsqueeze(-1)
            R_t = tau_path._exp_rodrigues(omega_t)
            tau_t = tau_path._tau_star(R_t, mu_P, mu_Q)

            T_t = torch.eye(4, dtype=dataset.dtype)
            T_t[:3, :3] = R_t[0]
            T_t[:3, 3] = tau_t[0]
            tau_transforms.append(T_t)

        midpoint_sample = tau_path.sample(
            x_0=x0_batch,
            x_1=x1_batch,
            t=torch.tensor([0.5], dtype=dataset.dtype),
        )

    tau_metrics = compute_metrics_over_transforms(
        overlap_source,
        overlap_target,
        tau_transforms,
        transform_gt=transform,
    )
    tau_final_metrics = tau_metrics[-1]
    print_metrics(tau_final_metrics, header="Tau-Constrained Rigid Path 최종 정합 지표 (RMSE / RRE / RTE / RReg)")
    # create_registration_animation(
    #     source,
    #     target,
    #     tau_transforms,
    #     filename="tau_constrained_rigid_path.gif",
    #     frame_times=[float(v) for v in tau_ts.tolist()],
    #     frame_metrics=tau_metrics,
    #     title_prefix="Tau-Constrained Rigid ProbPath",
    # )
    # print("tau-constrained rigid 경로 애니메이션을 'tau_constrained_rigid_path.gif'로 저장했습니다.")
    # print(f"midpoint velocity mean norm: {midpoint_velocity_norm:.6f}")

    create_registration_comparison_animation(
        overlap_source,
        overlap_target,
        geodesic_transforms,
        tau_transforms,
        filename="registration_comparison.gif",
        frame_times=[float(v) for v in geodesic_ts.tolist()],
        frame_metrics_a=geodesic_metrics,
        frame_metrics_b=tau_metrics,
        titles=("SE(3) Geodesic", "Tau-Constrained Rigid Path"),
    )
    print("비교 애니메이션을 'registration_comparison.gif'로 저장했습니다.")
