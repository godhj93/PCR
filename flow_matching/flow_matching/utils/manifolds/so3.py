import torch
from torch import Tensor

from flow_matching.utils.manifolds import Manifold

class SO3(Manifold):
    """Represents the special orthogonal group SO(3) of 3D rotations."""

    EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

    @staticmethod
    def _skew(A: Tensor) -> Tensor:
        # (..,3,3) -> skew-symmetric matrix
        return 0.5 * (A - A.transpose(-1, -2))
    
    @staticmethod
    def _vee(Omega: Tensor) -> Tensor:
        # (..,3,3) skew -> (..,3) vector
        
        return torch.stack([
            Omega[..., 2, 1],
            Omega[..., 0, 2],
            Omega[..., 1, 0]
        ], dim=-1)
        
    @staticmethod
    def _hat(omega: Tensor) -> Tensor:
        # (..,3) vector -> (..,3,3) skew
        ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
        O = torch.zeros(omega.shape[:-1] + (3, 3), dtype=omega.dtype, device=omega.device)
        O[..., 0, 1] = -oz
        O[..., 0, 2] = oy
        O[..., 1, 0] = oz
        O[..., 1, 2] = -ox
        O[..., 2, 0] = -oy
        O[..., 2, 1] = ox
        return O
    
    @staticmethod
    def _trace(M: Tensor) -> Tensor:
        return torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1)
        
    def projx(self, R: Tensor) -> Tensor:
        # Project to SO(3) via SVD
        U, S, Vh = torch.linalg.svd(R)
        # det(U @ Vh): 배치 det
        det = torch.det(U @ Vh)
        # D를 배치-아이덴으로 만들고 마지막 축만 부호 교정
        D = torch.eye(3, dtype=R.dtype, device=R.device).expand(R.shape[:-2] + (3, 3)).clone()
        # 마지막 축을 det의 부호로 설정 (부호 음수면 마지막 축 flip)
        D[..., 2, 2] = torch.sign(det)
        return U @ D @ Vh
    
    def proju(self, R: Tensor, U: Tensor) -> Tensor:
        """
        Project arbitrary matrix U to the tangent space at R
        proju(R, U) = R * skew(R^T * U)"""
        
        RtU = R.transpose(-1, -2) @ U
        Omega = self._skew(RtU)
        return R @ Omega
    
    def expmap(self, R: Tensor, U: Tensor) -> Tensor:
        
        # Rodrigues + Small Angle Approximation (Taylor 1st order)
        eps = self.EPS[U.dtype]
        
        # body algebra
        Omega = self._skew(R.transpose(-1, -2) @ U)
        omega = self._vee(Omega)
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
        
        # Rodrigues formula
        A = self._hat(omega)
        A2 = A @ A
        
        # sin(theta) / theta
        sin_by_theta = torch.where(
            theta > eps,
            torch.sin(theta) / theta, 
            1.0 - (theta ** 2) / 6.0 # When theta -> 0, sin(theta)/theta becomes 1 - theta^2/6
        )
        
        # (1 - cos(theta)) / theta^2
        one_minus_cos_by_theta2 = torch.where(
            theta > eps,
            (1.0 - torch.cos(theta)) / (theta ** 2),
            0.5 - (theta ** 2) / 24.0 # When theta -> 0, (1 - cos(theta))/theta^2 becomes 0.5 - theta^2/24
        )
        
        I = torch.eye(3, dtype=R.dtype, device=R.device).expand(R.shape)
        
        Exp = I + sin_by_theta[..., None, None] * A + one_minus_cos_by_theta2[..., None, None] * A2 # Rodrigues formula
        
        R_next = R @ Exp
        
        return self.projx(R_next) # re-project to SO(3) in case of numerical errors
        
    def logmap(self, R: Tensor, Y: Tensor) -> Tensor:
        """
        Logarithmic map on SO(3) with robust π-branch handling (SVD-based axis).
        """
        delta = R.transpose(-1, -2) @ Y

        tr = self._trace(delta)  # no clamp here
        x = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # clamp on x
        theta = torch.arccos(x)[..., None, None]


        # common terms
        skew = 0.5 * (delta - delta.transpose(-1, -2))
        sin_th = torch.sin(theta)

        # masks (shapes: (...,1,1)) — adjust near_pi threshold if needed
        zero_eps = self.EPS[delta.dtype]
        pi = torch.tensor(torch.pi, dtype=delta.dtype, device=delta.device)
        near_zero = (theta <= zero_eps)
        near_pi   = (pi - theta <= 5e-3)   # ← 1e-3 → 1e-2로 살짝 넓힘 (필요시 튜닝)

        # (1) θ≈0: exact limit
        Omega_small = skew

        # (2) general θ: exact formula
        scale = theta / (2.0 * sin_th)
        Omega_general = scale * (delta - delta.transpose(-1, -2))

        # (3) θ≈π: robust axis via SVD of (delta - I)
        # Find a ≈ null(delta - I): last right-singular vector of (delta - I)
        I = torch.eye(3, dtype=delta.dtype, device=delta.device).expand_as(delta)
        M = delta - I  # (...,3,3)
        # batched SVD
        U_, S_, Vh_ = torch.linalg.svd(M)
        v_min = Vh_[..., -1, :]                   # (...,3) right-singular vector for smallest σ
        a = v_min / (v_min.norm(dim=-1, keepdim=True) + 1e-12)

        # Make axis sign consistent with skew(delta) (optional, improves determinism)
        # Choose sign so that a^T * vee(skew) >= 0
        vee_skew = self._vee(skew)
        sign_align = torch.sign((a * vee_skew).sum(dim=-1, keepdim=True))
        sign_align = torch.where(sign_align == 0, torch.ones_like(sign_align), sign_align)
        a = a * sign_align


        theta_scalar = theta.squeeze(-1).squeeze(-1)   # (...,)
        omega_pi = a * theta_scalar[..., None]         # (...,3)
        Omega_pi = self._hat(omega_pi)                 # (...,3,3)

        # merge branches (broadcast masks)
        Omega = torch.where(near_zero, Omega_small, Omega_general)
        Omega = torch.where(near_pi,   Omega_pi,     Omega)

        U = R @ Omega
        return U


    
    def dist(self, R: Tensor, Y: Tensor, *, keepdim = False) -> Tensor:
        """
        Geodesic distance on SO(3): rotation angle between R and Y
        """
        delta = R.transpose(-1, -2) @ Y
        tr = self._trace(delta)  # no clamp here
        x = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # same tiny epsilon
        theta = torch.arccos(x)

        return theta[..., None, None] if keepdim else theta
    
if __name__ == "__main__":
    import math
    import torch

    torch.set_printoptions(precision=4, sci_mode=False)

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    so3 = SO3().to(device)

    def hat(omega: torch.Tensor) -> torch.Tensor:
        ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
        O = torch.zeros(omega.shape[:-1] + (3, 3), dtype=omega.dtype, device=omega.device)
        O[..., 0, 1] = -oz; O[..., 0, 2] =  oy
        O[..., 1, 0] =  oz; O[..., 1, 2] = -ox
        O[..., 2, 0] = -oy; O[..., 2, 1] =  ox
        return O

    def axis_angle_to_R(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        axis: (B, 3)  unit vector
        angle: (B, 1) or (B,)  rotation angle
        return: (B, 3, 3)
        """
        # 1) 형태 정리
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)  # ensure unit
        th = angle.squeeze(-1)  # (B,) 로 맞춤

        # 2) 회전벡터/hat
        omega = axis * th[..., None]        # (B,3)
        ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
        Omega = torch.zeros(omega.shape[:-1] + (3, 3), dtype=omega.dtype, device=omega.device)
        Omega[..., 0, 1] = -oz; Omega[..., 0, 2] =  oy
        Omega[..., 1, 0] =  oz; Omega[..., 1, 2] = -ox
        Omega[..., 2, 0] = -oy; Omega[..., 2, 1] =  ox

        # 3) Rodrigues 계수 (B,) → (B,1,1)
        sin_by_th = (torch.sin(th) / th).where(th.abs() > 1e-12, torch.ones_like(th))
        one_minus_cos_by_th2 = ((1 - torch.cos(th)) / (th * th)).where(
            th.abs() > 1e-12, 0.5 * torch.ones_like(th)
        )
        sin_by_th = sin_by_th[..., None, None]               # (B,1,1)
        one_minus_cos_by_th2 = one_minus_cos_by_th2[..., None, None]

        # 4) 최종 R
        I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(Omega.shape)
        A2 = Omega @ Omega
        R = I + sin_by_th * Omega + one_minus_cos_by_th2 * A2   # (B,3,3)
        return R

    # ---------- 1) projx 테스트 ----------
    B = 8
    R_noise = torch.randn(B, 3, 3, dtype=dtype, device=device)
    R_proj = so3.projx(R_noise)
    should_be_I = R_proj.transpose(-1, -2) @ R_proj
    det_R = torch.det(R_proj)

    print("\n[projx] ||R^T R - I|| (mean):",
          (should_be_I - torch.eye(3, dtype=dtype, device=device)).norm(dim=(-2, -1)).mean().item())
    print("[projx] det(R) (mean, min, max):",
          det_R.mean().item(), det_R.min().item(), det_R.max().item())

    # ---------- 2) proju 테스트: R^T (proju(R,U))가 skew? ----------
    R0_axis = torch.randn(B, 3, dtype=dtype, device=device)
    R0_axis = R0_axis / (R0_axis.norm(dim=-1, keepdim=True) + 1e-9)
    R0_angle = torch.rand(B, 1, dtype=dtype, device=device) * math.pi
    R0 = axis_angle_to_R(R0_axis, R0_angle)

    U = torch.randn(B, 3, 3, dtype=dtype, device=device)
    U_tan = so3.proju(R0, U)
    RtUtan = R0.transpose(-1, -2) @ U_tan
    skew_sym_err = (RtUtan + RtUtan.transpose(-1, -2)).norm(dim=(-2, -1)).mean().item()
    print("\n[proju] skew-symmetry error (mean):", skew_sym_err)

    # ---------- 3) exp ◦ log ~ id, log ◦ exp ~ id ---------- 
    # # (a) 일반 각 테스트 
    MARGIN = 1e-2  # 필요시 5e-3~2e-2 사이 튜닝
    R1_angle = torch.rand(B, 1, dtype=dtype, device=device) * (math.pi/4) 
    R2_angle = torch.rand(B, 1, dtype=dtype, device=device) * (math.pi/4) 
    
    # print angles(degree)
    print("\nTest angles (degree):")
    print("R1 angles:", (R1_angle.squeeze(-1).cpu().numpy() * 180.0 / math.pi))
    print("R2 angles:", (R2_angle.squeeze(-1).cpu().numpy() * 180.0 / math.pi))   

    R1_axis = torch.randn(B, 3, dtype=dtype, device=device) 
    R1_axis = R1_axis / (R1_axis.norm(dim=-1, keepdim=True) + 1e-9)
    #R1_angle = torch.rand(B, 1, dtype=dtype, device=device) * math.pi 
    R1 = axis_angle_to_R(R1_axis, R1_angle) 
    R2_axis = torch.randn(B, 3, dtype=dtype, device=device) 
    R2_axis = R2_axis / (R2_axis.norm(dim=-1, keepdim=True) + 1e-9) 
    #R2_angle = torch.rand(B, 1, dtype=dtype, device=device) * math.pi 
    R2 = axis_angle_to_R(R2_axis, R2_angle) 
    # log_R1(R2) -> U, then exp_R1(U) ≈ R2 
    U12 = so3.logmap(R1, R2) 
    R2_hat = so3.expmap(R1, U12) 
    recon_err_general = so3.dist(R2_hat, R2).mean().item()
    # radian to degree
    recon_err_general = recon_err_general * 180.0 / math.pi
    print("\n[exp∘log≈id] angle(degree) error (general angles, mean):", recon_err_general)
    
    # (a-2) ORACLE TEST: 생성-복원 (θ < π)
    MARGIN = 1e-2
    axis = torch.randn(B, 3, dtype=dtype, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
    theta = torch.rand(B, 1, dtype=dtype, device=device) * (math.pi - MARGIN)
    omega_true = axis * theta                        # (B,3)
    U_true = R1 @ so3._hat(omega_true)               # tangent at R1
    Y = so3.expmap(R1, U_true)                       # 생성한 타겟

    U_rec = so3.logmap(R1, Y)                        # 복원
    Y_rec = so3.expmap(R1, U_rec)

    oracle_err = so3.dist(Y_rec, Y).mean().item()
    # radian to degree
    oracle_err = oracle_err * 180.0 / math.pi
    print("[exp∘log≈id] oracle angle(degree) error (θ<π, mean):", oracle_err)
    


    # (b) 소각 테스트
    small_angle = 1e-4
    v_small = torch.randn(B, 3, dtype=dtype, device=device)
    v_small = v_small / (v_small.norm(dim=-1, keepdim=True) + 1e-9) * small_angle
    U_small = R1 @ hat(v_small)   # tangent at R1: R1 * Omega
    R_small = so3.expmap(R1, U_small)
    U_back = so3.logmap(R1, R_small)
    back_err_small = (U_back - U_small).norm(dim=(-2, -1)).mean().item()
    # radian to degree
    back_err_small = back_err_small * 180.0 / math.pi
    
    print("[log∘exp≈id] tangent recon error (small angles(degree), mean):", back_err_small)

    # ---------- 4) dist 테스트 ----------
    # 이론상 dist(R1,R2)=theta( R1^T R2 )
    Delta = R1.transpose(-1, -2) @ R2
    tr = torch.diagonal(Delta, dim1=-2, dim2=-1).sum(-1)               # no clamp here
    x = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-8, 1.0 - 1e-8)              # clamp on x
    theta_gt = torch.arccos(x)
    theta_est = so3.dist(R1, R2)
    dist_err = (theta_est - theta_gt).abs().mean().item()
    print("\n[dist] angle error (mean):", dist_err)

    # ---------- 5) autograd 간단 체크 ----------
    # 목표: R_target에 가깝게 만드는 U를 학습하는 toy loss
    R_target = R2.detach()
    # --- 수정(A안: 3-vector ω 학습) ---
    omega_var = (torch.randn(B, 3, dtype=dtype, device=device) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([omega_var], lr=3e-3)  # Adam 권장

    for it in range(100):  # step 수 늘리면 수렴 관찰 용이
        U_tan = R1 @ so3._hat(omega_var)    # 접공간: U = R1 [ω]_x
        R_step = so3.expmap(R1, U_tan)      # R_{k+1} = R1 exp([ω]_x)

        # 지오데식 잔차 기반 loss: ξ = vee( log( R_step^T R_target ) )
        Delta = R_step.transpose(-1, -2) @ R_target
        U_log = so3.logmap(torch.eye(3, dtype=dtype, device=device).expand_as(R_step), Delta)
        xi = so3._vee(U_log)                # (..., 3)
        loss = 0.5 * (xi**2).sum(dim=-1).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it in (0, 10, 50, 99):
            print(f"[autograd-ω] iter {it:02d} loss: {loss.item():.6f}")