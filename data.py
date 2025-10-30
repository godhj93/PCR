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

from flow_matching.utils.manifolds.se3 import SE3
from torch import Tensor
from torch.utils.data import Dataset

from metric import compute_metrics
from utils import (
    apply_transform,
    compute_metrics_over_transforms,
    create_registration_animation,
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
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        shape: str = "cube",
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples는 1보다 커야 합니다.")
        if num_points <= 0:
            raise ValueError("num_points는 1보다 커야 합니다.")
        self.num_samples = num_samples
        self.num_points = num_points
        self.point_scale = point_scale
        self.translation_scale = translation_scale
        self.target_noise_std = target_noise_std
        self.dtype = dtype
        self.shape = shape

        self._device = torch.device("cpu")
        self._generator = torch.Generator(device=self._device)
        if seed is not None:
            self._generator.manual_seed(seed)

        self._sources, self._targets, self._transforms = self._generate_dataset()

    def _generate_dataset(self) -> tuple[Tensor, Tensor, Tensor]:
        sources = torch.zeros(
            (self.num_samples, self.num_points, 3), dtype=self.dtype, device=self._device
        )
        targets = torch.zeros_like(sources)
        transforms = torch.zeros((self.num_samples, 4, 4), dtype=self.dtype, device=self._device)

        for idx in range(self.num_samples):
            source = generate_random_point_cloud(
                self.num_points,
                scale=self.point_scale,
                dtype=self.dtype,
                device=self._device,
                generator=self._generator,
                shape=self.shape,
            )
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
            target = (source @ rotation.T) + translation
            if self.target_noise_std > 0.0:
                noise = torch.randn(
                    (self.num_points, 3),
                    dtype=self.dtype,
                    device=self._device,
                    generator=self._generator,
                )
                target = target + noise * self.target_noise_std

            transform = torch.eye(4, dtype=self.dtype, device=self._device)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation

            sources[idx] = source
            targets[idx] = target
            transforms[idx] = transform

        return sources, targets, transforms

    def __len__(self) -> int:
        return self.num_samples

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
    )
    sample = dataset[0]
    source = sample.source
    target = sample.target
    rotation = sample.rotation
    translation = sample.translation
    transform = sample.transform

    # SE(3) 로그맵으로 twist를 구하고, 지수맵으로 다시 복원해본다.
    se3 = SE3()
    identity = torch.eye(4, dtype=dataset.dtype)
    twist_matrix = se3.logmap(identity, transform)
    twist6 = se3._vee6(twist_matrix)
    recovered_transform = se3.expmap(identity, twist_matrix)
    metrics = compute_metrics(
        source=source,
        target=target,
        transform_gt=transform,
        transform_est=recovered_transform,
    )
    registration_rmse = metrics["rmse"].item()
    rotation_error_rad = metrics["rre_rad"].item()
    rotation_error_deg = metrics["rre_deg"].item()
    translation_error = metrics["rte"].item()
    relative_reg_error = metrics["rreg"].item()

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
    print("\nSE(3) 로그맵에서 얻은 twist [rho(평행이동), phi(회전)]:")
    print(twist6.numpy())
    print_metrics(metrics, header="정합 평가 지표 (RMSE / RRE / RTE / RReg)")

    plt.tight_layout()
    plt.savefig("random_point_cloud.png", dpi=300)

    # 로그맵으로 얻은 변환을 등분해하여 애니메이션으로 저장
    steps = 60
    ts = torch.linspace(0.0, 1.0, steps=steps, dtype=dataset.dtype)
    frame_transforms = [se3.expmap(identity, twist_matrix * s) for s in ts]
    frame_metrics = compute_metrics_over_transforms(
        source,
        target,
        frame_transforms,
        transform_gt=transform,
    )
    create_registration_animation(
        source,
        target,
        frame_transforms,
        filename="registration_animation.gif",
        frame_times=[float(v) for v in ts.tolist()],
        frame_metrics=frame_metrics,
        title_prefix="SE(3) Geodesic Interpolation",
    )
    print("\n애니메이션을 'registration_animation.gif' 파일로 저장했습니다.")
