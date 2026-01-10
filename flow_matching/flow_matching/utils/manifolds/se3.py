# se3.py
import torch
from torch import Tensor

from flow_matching.utils.manifolds.so3 import SO3


class SE3(SO3):
    """
    Special Euclidean group SE(3): (R, t)
    - Right-invariant convention.
    - Uses parent's robust SO(3) log (near-pi SVD axis) and projection.
    """

    # SO3의 EPS를 그대로 사용

    # --------------------------
    # 6D twist hat/vee utilities
    # --------------------------
    @staticmethod
    def _hat6(xi: Tensor) -> Tensor:
        """
        xi: (..., 6) with (rho, phi) where rho: translation, phi: rotation
        return: (..., 4, 4) se(3) matrix
        """
        rho, phi = xi[..., :3], xi[..., 3:]
        O = SE3._hat(phi)  # from SO3
        Xi = torch.zeros(xi.shape[:-1] + (4, 4), dtype=xi.dtype, device=xi.device)
        Xi[..., :3, :3] = O
        Xi[..., :3, 3]  = rho
        return Xi

    @staticmethod
    def _vee6(Xi: Tensor) -> Tensor:
        """
        Xi: (..., 4, 4) se(3) matrix
        return: (..., 6) with (rho, phi)
        """
        rho = Xi[..., :3, 3]
        phi = SE3._vee(Xi[..., :3, :3])  # from SO3
        return torch.cat([rho, phi], dim=-1)

    # --------------------------
    # projection to SE(3)
    # --------------------------
    def projx(self, T: Tensor) -> Tensor:
        """
        Project arbitrary 4x4 matrix to SE(3) by orthonormalizing the rotation block.
        T: (...,4,4)
        """
        R = T[..., :3, :3]
        t = T[..., :3,  3]
        R_proj = super().projx(R)
        Tout = torch.zeros_like(T)
        Tout[..., :3, :3] = R_proj
        Tout[..., :3,  3] = t
        Tout[...,  3,  3] = 1.0
        return Tout

    # --------------------------
    # projection of ambient to tangent at T
    # --------------------------
    def proju(self, T: Tensor, U: Tensor) -> Tensor:
        """
        Project arbitrary 4x4 U to the tangent space at T (right-invariant):
        proju(T, U) = T * [ skew(R^T U_R), R^T U_t ; 0, 0 ]
        """
        R = T[..., :3, :3]
        Rt = R.transpose(-1, -2)
        UR = U[..., :3, :3]
        Ut = U[..., :3,  3]
        Omega = self._skew(Rt @ UR)
        v = torch.einsum('...ij,...j->...i', Rt, Ut)

        U_tan = torch.zeros_like(U)
        U_tan[..., :3, :3] = R @ Omega
        U_tan[..., :3, 3] = torch.einsum('...ij,...j->...i', R, v)

        # bottom row already zeros
        return U_tan

    # --------------------------
    # helpers: V(θ) for SE(3) exp
    # --------------------------
    def _V_from_omega(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute V = I + B*Omega + C*Omega^2  (elementwise 보장)
        omega: (...,3)
        returns: (...,3,3)
        """
        dtype = omega.dtype
        device = omega.device
        eps = self.EPS[dtype]

        # ---- 안전한 hat (elementwise) ----
        ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
        Omega = torch.zeros(omega.shape[:-1] + (3, 3), dtype=dtype, device=device)
        Omega[..., 0, 1] = -oz; Omega[..., 0, 2] =  oy
        Omega[..., 1, 0] =  oz; Omega[..., 1, 2] = -ox
        Omega[..., 2, 0] = -oy; Omega[..., 2, 1] =  ox

        # ---- 나머지 계산 ----
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (...,1)
        Omega2 = torch.einsum('...ij,...jk->...ik', Omega, Omega)  # Elementwise batch matmul

        # A = sin θ / θ, B = (1 - cos θ)/θ^2, C = (θ - sin θ)/θ^3  (작각 가드)
        A = torch.where(theta > eps, torch.sin(theta) / theta,
                        1.0 - (theta**2)/6.0)
        B = torch.where(theta > eps, (1.0 - torch.cos(theta)) / (theta**2),
                        0.5 - (theta**2)/24.0)
        C = torch.where(theta > eps, (theta - torch.sin(theta)) / (theta**3),
                        (1.0/6.0) - (theta**2)/120.0)

        # theta는 (..., 1) 형태이므로, A/B/C도 (..., 1)
        # 이를 (..., 1, 1)로 변환하여 (... 3, 3) 행렬과 곱할 수 있게 함
        A = A.unsqueeze(-1)  # (...,1,1)
        B = B.unsqueeze(-1)
        C = C.unsqueeze(-1)

        # I를 Omega와 같은 shape으로 만들기
        batch_shape = Omega.shape[:-2]
        I = torch.eye(3, dtype=dtype, device=device)
        if len(batch_shape) > 0:
            I = I.unsqueeze(0).expand(*batch_shape, -1, -1).clone()
        
        V = I + B * Omega + C * Omega2

        return V.contiguous()


    # --------------------------
    # exponential map on SE(3)
    # --------------------------
    def expmap(self, T: Tensor, U: Tensor) -> Tensor:
        """
        T: (...,4,4) current pose
        U: (...,4,4) tangent at T  (right-invariant: U = T * [Omega, v; 0,0])

        Returns: T_next = T * Exp([Omega, v])
        """
        R = T[..., :3, :3]
        t = T[..., :3,  3]
        Rt = R.transpose(-1, -2)

        # Body twist extraction: T^{-1} U = [ Rt*U_R, Rt*U_t ; 0 0 ]
        UR = U[..., :3, :3]
        Ut = U[..., :3,  3]
        Omega_body = self._skew(Rt @ UR)             # (...,3,3)
        omega = self._vee(Omega_body)                # (...,3)
        v      = torch.einsum('...ij,...j->...i', Rt, Ut)

        # Rotation increment via parent's SO3.expmap (reuse robust small-angle)
        # Build a tangent matrix at R: U_rot = R * Omega_body
        U_rot = R @ Omega_body
        # SO3.expmap calls self.projx, which is SE3.projx, causing issues
        # Instead, compute rotation directly using SO3 formula
        theta_rot = torch.linalg.norm(omega, dim=-1, keepdim=True)
        eps = self.EPS[R.dtype]
        
        sin_by_theta = torch.where(
            theta_rot > eps,
            torch.sin(theta_rot) / theta_rot,
            1.0 - (theta_rot ** 2) / 6.0
        )
        one_minus_cos_by_theta2 = torch.where(
            theta_rot > eps,
            (1.0 - torch.cos(theta_rot)) / (theta_rot ** 2),
            0.5 - (theta_rot ** 2) / 24.0
        )
        
        batch_shape = Omega_body.shape[:-2]
        I = torch.eye(3, dtype=R.dtype, device=R.device)
        if len(batch_shape) > 0:
            I = I.unsqueeze(0).expand(*batch_shape, -1, -1).clone()
        
        sin_by_theta = sin_by_theta.unsqueeze(-1)  # (..., 1, 1)
        one_minus_cos_by_theta2 = one_minus_cos_by_theta2.unsqueeze(-1)
        
        Omega_body2 = torch.einsum('...ij,...jk->...ik', Omega_body, Omega_body)
        Exp = I + sin_by_theta * Omega_body + one_minus_cos_by_theta2 * Omega_body2
        R_next = R @ Exp
        R_next = SO3.projx(self, R_next)  # Call SO3's projx directly

        # Translation increment t_inc = V(omega) * v  (body frame)
        V = self._V_from_omega(omega)                # (...,3,3)
        t_inc  = torch.einsum('...ij,...j->...i', V, v)

        # Compose
        t_next = t + torch.einsum('...ij,...j->...i', R, t_inc)
        T_next = torch.zeros_like(T)
        T_next[..., :3, :3] = R_next
        T_next[..., :3,  3] = t_next
        T_next[...,  3,  3] = 1.0

        # Re-project rotation just in case
        return self.projx(T_next)

    # --------------------------
    # logarithmic map on SE(3)
    # --------------------------
    def logmap(self, T: Tensor, Y: Tensor) -> Tensor:
        R = T[..., :3, :3]
        t = T[..., :3,  3]
        Rt = R.transpose(-1, -2)

        # 1) Delta = T^{-1} Y
        RY = Y[..., :3, :3]
        tY = Y[..., :3,  3]
        R_d = Rt @ RY
        t_d = torch.einsum('...ij,...j->...i', Rt, tY - t)        # (...,3)

        # 2) robust SO3 log: logmap at identity of R_d
        # SO3.logmap(I, R_d) computes tangent at identity
        batch_shape = R.shape[:-2]
        I3 = torch.eye(3, dtype=R.dtype, device=R.device)
        
        # For each rotation in the batch, compute log independently
        if len(batch_shape) == 0:
            # Single matrix case
            Omega_I = super().logmap(I3, R_d)
        else:
            # Batch case: need to handle each element independently
            # Expand I3 to match batch dimensions
            I3_batch = I3.unsqueeze(0).expand(*batch_shape, -1, -1).contiguous()
            Omega_I = super().logmap(I3_batch, R_d)  # Returns (..., 3, 3)
            
            # If SO3.logmap returns pairwise results (B, B, 3, 3), extract diagonal
            if Omega_I.dim() > R_d.dim():
                # Extract diagonal: (B, B, 3, 3) -> (B, 3, 3)
                Omega_I = torch.diagonal(Omega_I, dim1=0, dim2=1).permute(-1, 0, 1).contiguous()
        
        omega = self._vee(Omega_I)  # (...,3)

        # 3) solve V(omega) * rho = t_d
        V   = self._V_from_omega(omega)                           # (...,3,3)
        rho = torch.linalg.solve(V, t_d.unsqueeze(-1)).squeeze(-1)# (...,3)

        # 4) U = T * [Omega_I, rho]
        U = torch.zeros_like(T)
        U[..., :3, :3] = torch.einsum('...ij,...jk->...ik', R, Omega_I)
        U[..., :3,  3] = torch.einsum('...ij,...j->...i',  R, rho)
        return U

    # --------------------------
    # distance on SE(3) (simple weighted log-norm)
    # --------------------------
    def dist(self, T: Tensor, Y: Tensor, *, keepdim: bool = False,
             trans_weight: float = 1.0, rot_weight: float = 1.0) -> Tensor:
        """
        d(T,Y) = || [ trans_weight * rho,  rot_weight * omega ] ||_2
        where [rho, omega] = vee6( log( T^{-1} Y ) )

        Note: scale weights according to your unit conventions (m vs rad).
        """
        # Compute body log at T
        U = self.logmap(T, Y)                         # (...,4,4)
        xi = self._vee6(torch.linalg.solve(T, U))     # not needed actually; alternative below

        # Simpler & correct: extract directly from log at Identity of Delta
        R = T[..., :3, :3]; t = T[..., :3,  3]; Rt = R.transpose(-1, -2)
        RY = Y[..., :3, :3]; tY = Y[..., :3,  3]
        R_d = Rt @ RY
        t_d = (Rt @ (tY - t)[..., None])[..., 0]

        I3 = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
        Omega_at_I = super().logmap(I3, R_d)
        omega = self._vee(Omega_at_I)                 # (...,3)
        V = self._V_from_omega(omega)
        rho = torch.linalg.solve(V, t_d.unsqueeze(-1)).squeeze(-1)  # (N,3)

        v = trans_weight * rho
        w = rot_weight   * omega
        d = torch.sqrt((v**2).sum(dim=-1) + (w**2).sum(dim=-1))  # (...)
        if keepdim:
            return d[..., None, None]
        return d


if __name__ == "__main__":

    import math
    
    torch.set_printoptions(precision=4, sci_mode=False)

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    se3 = SE3().to(device)
    so3 = SE3().to(device)  # 부모 SO3 기능 재사용 목적

    # --------------------------
    # Utilities
    # --------------------------
    def hat3(omega: torch.Tensor) -> torch.Tensor:
        ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
        O = torch.zeros(omega.shape[:-1] + (3, 3), dtype=omega.dtype, device=omega.device)
        O[..., 0, 1] = -oz; O[..., 0, 2] =  oy
        O[..., 1, 0] =  oz; O[..., 1, 2] = -ox
        O[..., 2, 0] = -oy; O[..., 2, 1] =  ox
        return O

    def axis_angle_to_R(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
        th = angle.squeeze(-1)
        omega = axis * th[..., None]
        Omega = hat3(omega)
        I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(Omega.shape)
        sin_by_th = (torch.sin(th) / th).where(th.abs() > 1e-12, torch.ones_like(th))
        one_minus_cos_by_th2 = ((1 - torch.cos(th)) / (th * th)).where(
            th.abs() > 1e-12, 0.5 * torch.ones_like(th)
        )
        sin_by_th = sin_by_th[..., None, None]
        one_minus_cos_by_th2 = one_minus_cos_by_th2[..., None, None]
        R = I + sin_by_th * Omega + one_minus_cos_by_th2 * (Omega @ Omega)
        return R

    def Rt(T: torch.Tensor) -> torch.Tensor:
        return T[..., :3, :3].transpose(-1, -2)

    def make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = R.shape[:-2]
        T = torch.zeros(B + (4, 4), dtype=R.dtype, device=R.device)
        T[..., :3, :3] = R
        T[..., :3,  3] = t
        T[...,  3,  3] = 1.0
        return T

    # --------------------------
    # 1) projx test
    # --------------------------
    B = 32
    R_noise = torch.randn(B, 3, 3, dtype=dtype, device=device)
    t_noise = torch.randn(B, 3,     dtype=dtype, device=device)
    T_noise = make_T(R_noise, t_noise)
    T_proj = se3.projx(T_noise)

    should_be_I = T_proj[..., :3, :3].transpose(-1, -2) @ T_proj[..., :3, :3]
    det_R = torch.det(T_proj[..., :3, :3])

    print("\n[SE3.projx] ||R^T R - I|| (mean):",
          (should_be_I - torch.eye(3, dtype=dtype, device=device)).norm(dim=(-2, -1)).mean().item())
    print("[SE3.projx] det(R) (mean, min, max):",
          det_R.mean().item(), det_R.min().item(), det_R.max().item())

    # --------------------------
    # 2) proju test: tangent check
    # --------------------------
    R0_axis = torch.randn(B, 3, dtype=dtype, device=device)
    R0_axis = R0_axis / (R0_axis.norm(dim=-1, keepdim=True) + 1e-9)
    R0_angle = torch.rand(B, 1, dtype=dtype, device=device) * math.pi
    R0 = axis_angle_to_R(R0_axis, R0_angle)
    t0 = torch.randn(B, 3, dtype=dtype, device=device)
    T0 = make_T(R0, t0)

    U_amb = torch.randn(B, 4, 4, dtype=dtype, device=device)
    U_tan = se3.proju(T0, U_amb)

    # Check skew on body rotation block: R^T (U_tan_R) should be skew
    RtUtanR = R0.transpose(-1, -2) @ U_tan[..., :3, :3]
    skew_sym_err = (RtUtanR + RtUtanR.transpose(-1, -2)).norm(dim=(-2, -1)).mean().item()
    print("\n[SE3.proju] rotation skew-symmetry error (mean):", skew_sym_err)

    # --------------------------
    # 3) exp ∘ log ≈ id (general angles)
    # --------------------------
    # Random target Y near-ish (not too close to pi)
    R1_axis = torch.randn(B, 3, dtype=dtype, device=device); R1_axis = R1_axis / (R1_axis.norm(dim=-1, keepdim=True)+1e-9)
    R2_axis = torch.randn(B, 3, dtype=dtype, device=device); R2_axis = R2_axis / (R2_axis.norm(dim=-1, keepdim=True)+1e-9)
    R1_angle = torch.rand(B, 1, dtype=dtype, device=device) * (math.pi/4)
    R2_angle = torch.rand(B, 1, dtype=dtype, device=device) * (math.pi/4)
    R1 = axis_angle_to_R(R1_axis, R1_angle)
    R2 = axis_angle_to_R(R2_axis, R2_angle)
    t1 = torch.randn(B, 3, dtype=dtype, device=device)
    t2 = torch.randn(B, 3, dtype=dtype, device=device)

    T1 = make_T(R1, t1)
    T2 = make_T(R2, t2)

    U12 = se3.logmap(T1, T2)
    T2_hat = se3.expmap(T1, U12)

    # Rotation error (deg) + translation error (L2)
    def so3_angle(Ra, Rb):
        tr = torch.diagonal(Ra.transpose(-1, -2) @ Rb, dim1=-2, dim2=-1).sum(-1)
        x = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.arccos(x)

    rot_err_deg = (so3_angle(T2_hat[..., :3, :3], T2[..., :3, :3]) * 180.0 / math.pi).mean().item()
    trans_err = (T2_hat[..., :3, 3] - T2[..., :3, 3]).norm(dim=-1).mean().item()

    print("\n[exp∘log≈id] rotation error (deg, mean):", rot_err_deg)
    print("[exp∘log≈id] translation error (L2, mean):", trans_err)

    # --------------------------
    # 4) ORACLE test (generate via exp then recover via log)
    # --------------------------
    # Sample body twist xi=(rho, phi), then Y = T1 * Exp(xi)
    rho_true = torch.randn(B, 3, dtype=dtype, device=device)
    phi_axis = torch.randn(B, 3, dtype=dtype, device=device)
    phi_axis = phi_axis / (phi_axis.norm(dim=-1, keepdim=True) + 1e-9)
    phi_angle = torch.rand(B, 1, dtype=dtype, device=device) * (math.pi - 1e-2)  # avoid exact pi
    phi_true = phi_axis * phi_angle

    Xi_true = torch.zeros(B, 4, 4, dtype=dtype, device=device)
    Xi_true[..., :3, :3] = hat3(phi_true)
    Xi_true[..., :3,  3] = rho_true

    Y = se3.expmap(T1, T1 @ Xi_true)  # note: U = T * [Omega, rho]

    U_rec = se3.logmap(T1, Y)
    Y_rec = se3.expmap(T1, U_rec)

    rot_err_deg_oracle = (so3_angle(Y_rec[..., :3, :3], Y[..., :3, :3]) * 180.0 / math.pi).mean().item()
    trans_err_oracle = (Y_rec[..., :3, 3] - Y[..., :3, 3]).norm(dim=-1).mean().item()

    print("\n[oracle exp∘log≈id] rotation error (deg, mean):", rot_err_deg_oracle)
    print("[oracle exp∘log≈id] translation error (L2, mean):", trans_err_oracle)

    # --------------------------
    # 5) small-angle test (log ∘ exp ≈ id on tangent)
    # --------------------------
    small = 1e-4
    rho_s = torch.randn(B, 3, dtype=dtype, device=device)
    phi_s = torch.randn(B, 3, dtype=dtype, device=device)
    phi_s = phi_s / (phi_s.norm(dim=-1, keepdim=True) + 1e-9) * small

    Xi_s = torch.zeros(B, 4, 4, dtype=dtype, device=device)
    Xi_s[..., :3, :3] = hat3(phi_s)
    Xi_s[..., :3,  3] = rho_s * small

    T_small = se3.expmap(T1, T1 @ Xi_s)
    U_back = se3.logmap(T1, T_small)

    back_err = (U_back - (T1 @ Xi_s)).norm(dim=(-2, -1)).mean().item()
    print("\n[log∘exp≈id] tangent recon error (small, mean):", back_err)

    # --------------------------
    # 6) dist sanity (weighted log-norm)
    # --------------------------
    d = se3.dist(T1, T2, trans_weight=1.0, rot_weight=1.0)
    print("\n[dist] example value (mean):", d.mean().item())

   