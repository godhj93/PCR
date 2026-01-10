# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import PathSample
from flow_matching.utils.manifolds.so3 import SO3


def _kabsch_rotation(Pc: Tensor, Qc: Tensor) -> Tensor:
    """
    Batched Kabsch alignment that returns the optimal rotation with det(R)=+1.
    Pc, Qc: (..., N, 3) centered point sets.
    """
    H = torch.einsum("...ni,...nj->...ij", Pc, Qc)  # (..., 3, 3)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    Ut = U.transpose(-1, -2)
    det = torch.det(V @ Ut)
    batch_shape = det.shape
    D = torch.eye(3, dtype=H.dtype, device=H.device).expand(batch_shape + (3, 3)).clone()
    det_sign = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    D[..., 2, 2] = det_sign
    return V @ D @ Ut


class TauConstrainedRigidPath(ProbPath):
    r"""
    τ*-제약 강체 확률 경로.

    주어진 배치 점군 P, Q에 대해
      R* = argmin_R Σ_i ||R x_i' - y_i'||^2  (Kabsch)
      τ*(R) = μ_Q - R μ_P

    경로는 Φ_t(x) = R_t x + τ*(R_t) 로 정의하며
      R_t = exp(t · ω*),  ω* = vee(log R*),  ω_t^world = R_t ω*

    결과적으로 FM 타깃 속도장은
      u_t(y) = [ω_t^world]_× (y - μ_Q)
    """

    def __init__(self, so3: Optional[SO3] = None, detach_targets: bool = True):
        super().__init__()
        self.so3 = so3 if so3 is not None else SO3()
        self.detach_targets = detach_targets

    @torch.no_grad()
    def _centroids(self, P: Tensor, Q: Tensor) -> tuple[Tensor, Tensor]:
        # P, Q: (B, N, 3)
        return P.mean(dim=-2), Q.mean(dim=-2)

    @torch.no_grad()
    def _solve_R_star(
        self,
        P: Tensor,
        Q: Tensor,
        mu_P: Optional[Tensor] = None,
        mu_Q: Optional[Tensor] = None,
    ) -> Tensor:
        if mu_P is None or mu_Q is None:
            mu_P, mu_Q = self._centroids(P, Q)
        Pc = P - mu_P.unsqueeze(-2)
        Qc = Q - mu_Q.unsqueeze(-2)
        return _kabsch_rotation(Pc, Qc)

    @torch.no_grad()
    def _log_at_identity(self, R: Tensor) -> Tensor:
        B = R.shape[0]
        I = torch.eye(3, dtype=R.dtype, device=R.device).expand(B, 3, 3).contiguous()
        Omega = self.so3.logmap(I, R)  # (B, 3, 3)
        return self.so3._vee(Omega)

    @torch.no_grad()
    def _tau_star(self, R: Tensor, mu_P: Tensor, mu_Q: Tensor) -> Tensor:
        return mu_Q - torch.einsum("bij,bj->bi", R, mu_P)

    @torch.no_grad()
    def _exp_rodrigues(self, omega: Tensor) -> Tensor:
        """
        Small-angle safe Rodrigues exponential using SO3 hat/vee utilities.
        omega: (B, 3) -> (B, 3, 3)
        """
        theta = torch.linalg.norm(omega, dim=-1)  # (B,)
        A = self.so3._hat(omega)  # (B, 3, 3)
        A2 = torch.matmul(A, A)
        eps = self.so3.EPS.get(omega.dtype, 1e-7)

        sin_by_theta = torch.where(
            theta > eps,
            torch.sin(theta) / theta,
            torch.ones_like(theta) - (theta**2) / 6.0,
        )
        one_minus_cos_by_theta2 = torch.where(
            theta > eps,
            (1.0 - torch.cos(theta)) / (theta**2),
            0.5 * torch.ones_like(theta) - (theta**2) / 24.0,
        )

        sin_by_theta = sin_by_theta.unsqueeze(-1).unsqueeze(-1)
        one_minus_cos_by_theta2 = one_minus_cos_by_theta2.unsqueeze(-1).unsqueeze(-1)
        I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(A)

        R = I + sin_by_theta * A + one_minus_cos_by_theta2 * A2
        return self.so3.projx(R)

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        """
        Args:
            x_0: P (B, N, 3)
            x_1: Q (B, N, 3)
            t:   (B,)
        Returns:
            PathSample(x_0, x_1, x_t, t, dx_t)
        """
        self.assert_sample_shape(x_0, x_1, t)
        assert x_0.ndim == 3 and x_0.shape[-1] == 3, "x_0 must be (B, N, 3)"
        assert x_1.ndim == 3 and x_1.shape[-1] == 3, "x_1 must be (B, N, 3)"

        if t.dtype != x_0.dtype:
            t = t.to(x_0.dtype)

        # 1) Solve for rigid alignment and logarithmic velocity
        mu_P, mu_Q = self._centroids(x_0, x_1)
        R_star = self._solve_R_star(x_0, x_1, mu_P=mu_P, mu_Q=mu_Q)
        omega_star = self._log_at_identity(R_star)

        # 2) Geodesic rotation path R_t = exp(t * ω*)
        omega_t = omega_star * t.unsqueeze(-1)
        R_t = self._exp_rodrigues(omega_t)

        # 3) τ*(R_t) with centroid constraint
        tau_t = self._tau_star(R_t, mu_P, mu_Q)

        # 4) Transform source points
        X_t = torch.einsum("bij,bnj->bni", R_t, x_0) + tau_t.unsqueeze(-2)

        # 5) World angular velocity and induced velocity field
        omega_world = torch.einsum("bij,bj->bi", R_t, omega_star)
        hat_world = self.so3._hat(omega_world)
        Y_c = X_t - mu_Q.unsqueeze(-2)
        U_t = torch.einsum("bij,bnj->bni", hat_world, Y_c)
        if self.detach_targets:
            U_t = U_t.detach()

        return PathSample(
            x_0=x_0,
            x_1=x_1,
            x_t=X_t,
            t=t,
            dx_t=U_t,
        )
