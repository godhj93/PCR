import torch
from torch import Tensor
import torch.nn as nn
from flow_matching.utils.manifolds import Manifold

class SE3(Manifold):
    """
    The Special Euclidean Group SE(3) manifold.
    Elements are represented as 4x4 matrices:
        [[R, t],
         [0, 1]]
    where R in SO(3) and t in R^3.
    """
    
    def __init__(self):
        super().__init__()
    
    def projx(self, x: Tensor) -> Tensor:
        """
        Project 4x4 matrix onto SE(3).
        R is projected to SO(3) using SVD, and the last row is forced to [0, 0, 0, 1].
        """
        # Handle shape issues from flow_matching library's geodesic function
        # original_shape = x.shape
        if x.dim() > 2 and x.shape[-2] != 4:
            # Reshape from (..., 4, K, 4) to (..., 4, 4) by squeezing extra dims
            while x.dim() > 2 and x.shape[-2] != x.shape[-1]:
                x = x.squeeze(-2)
        
        # x: (..., 4, 4)
        R = x[..., :3, :3]
        t = x[..., :3, 3]
        
        # Project R onto SO(3) using SVD
        # torch.linalg.svd returns (U, S, Vh) where A = U @ diag(S) @ Vh
        U, _, Vh = torch.linalg.svd(R)
        
        # Initial projection: R_proj = U @ Vh
        R_proj = torch.matmul(U, Vh)
        
        # Ensure determinant is +1 (not -1, which would be a reflection)
        det = torch.linalg.det(R_proj)
        
        # Create correction matrix: if det < 0, flip the last column of U
        # This is equivalent to multiplying by diag(1, 1, sign(det))
        correction = torch.ones_like(det)
        correction[det < 0] = -1
        
        # Apply correction to last column of U
        U_corrected = U.clone()
        U_corrected[..., :, 2] = U[..., :, 2] * correction.view(-1, 1)
        
        # Final projection
        R_proj = torch.matmul(U_corrected, Vh)
        
        # Reconstruct SE(3) matrix
        x_proj = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0).expand_as(x).clone()
        x_proj[..., :3, :3] = R_proj
        x_proj[..., :3, 3] = t
        
        return x_proj

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Project ambient vector u onto the tangent space T_x SE(3).
        For Left-Invariant metric: u_proj = x * skew(x^-1 * u)
        Actually, simpler: T_x SE(3) = { x * xi | xi in se(3) }
        So we project xi = x^-1 * u onto se(3).
        se(3) structure:
            [[ skew, vec ],
             [ 0   , 0   ]]
        """
        # Handle shape issues
        if x.dim() > 2 and x.shape[-2] != 4:
            while x.dim() > 2 and x.shape[-2] != x.shape[-1]:
                x = x.squeeze(-2)
        if u.dim() > 2 and u.shape[-2] != 4:
            while u.dim() > 2 and u.shape[-2] != u.shape[-1]:
                u = u.squeeze(-2)
        
        # Calculate x_inv
        # Since x is SE(3), inv(x) = [[R^T, -R^T t], [0, 1]]
        # But we can just use generic inverse or solve if batch size allows.
        # For stability/speed with 4x4, torch.linalg.inv is fine.
        x_inv = torch.linalg.inv(x)
        
        # xi_ambient = x^-1 * u
        xi_amb = torch.matmul(x_inv, u)
        
        # Project xi_amb onto se(3) algebra
        # 1. Top-left 3x3 must be skew-symmetric
        # skew(A) = 0.5 * (A - A^T)
        A = xi_amb[..., :3, :3]
        A_skew = 0.5 * (A - A.transpose(-1, -2))
        
        # 2. Bottom row must be 0
        
        # Construct projected xi
        xi_proj = torch.zeros_like(xi_amb)
        xi_proj[..., :3, :3] = A_skew
        xi_proj[..., :3, 3] = xi_amb[..., :3, 3] # Translation part is free
        # xi_proj[..., 3, :] is already 0
        
        # Map back to tangent space at x: u_proj = x * xi_proj
        u_proj = torch.matmul(x, xi_proj)
        
        return u_proj

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """
        Exponential map at x.
        exp_x(u) = x * exp(x^-1 * u) = x * exp(xi)
        """
        # Handle shape issues from geodesic interpolation
        if x.dim() > 2 and x.shape[-2] != 4:
            while x.dim() > 2 and x.shape[-2] != x.shape[-1]:
                x = x.squeeze(-2)
        if u.dim() > 2 and u.shape[-2] != 4:
            while u.dim() > 2 and u.shape[-2] != u.shape[-1]:
                u = u.squeeze(-2)
        
        # u is typically in T_x SE(3), so x^-1 u is in se(3).
        # We can compute matrix exponential directly.
        
        # xi = x^-1 * u
        x_inv = torch.linalg.inv(x)
        xi = torch.matmul(x_inv, u)
        
        # Matrix exponential
        # For se(3), closed form (Rodrigues-like) exists, 
        # but torch.linalg.matrix_exp is robust enough for batch 4x4.
        exp_xi = torch.linalg.matrix_exp(xi)
        
        return torch.matmul(x, exp_xi)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Logarithmic map at x.
        log_x(y) = x * log(x^-1 * y)
        """
        # Handle shape issues
        if x.dim() > 2 and x.shape[-2] != 4:
            while x.dim() > 2 and x.shape[-2] != x.shape[-1]:
                x = x.squeeze(-2)
        if y.dim() > 2 and y.shape[-2] != 4:
            while y.dim() > 2 and y.shape[-2] != y.shape[-1]:
                y = y.squeeze(-2)
        
        x_inv = torch.linalg.inv(x)
        # diff = x^-1 * y (This is a relative transform in SE(3))
        diff = torch.matmul(x_inv, y)
        
        # Check relative transform is SE(3) (it should be if x, y are)
        # Compute matrix logarithm
        # Currently PyTorch doesn't have a stable batch matrix_log for general matrices,
        # but for SE(3) we can use a custom 'log_se3' or assume inputs are close enough to I.
        # However, writing a robust log_se3 (inverse Rodrigues) is safer.
        # Here, I'll implement a robust log map for SE(3) or use a simplified approximation 
        # if the library assumes existence of a generic log.
        
        # Since we need a robust implementation:
        xi = self._se3_log(diff)
        
        # Return tangent vector u = x * xi
        return torch.matmul(x, xi)

    def _se3_log(self, T: Tensor) -> Tensor:
        """
        Logarithm map for SE(3) matrices T -> xi in se(3) (4x4 matrix).
        Uses closed form for rotation log and translation mapping.
        """
        R = T[..., :3, :3]
        t = T[..., :3, 3]
        
        # 1. Rotation Log (SO(3) -> so(3))
        # trace(R) = 1 + 2 cos(theta)
        # theta = arccos((tr(R) - 1) / 2)
        trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_theta)
        
        # case 1: theta != 0 (general case)
        # ln(R) = (theta / (2 sin theta)) * (R - R^T)
        sin_theta = torch.sin(theta)
        coef = theta / (2 * sin_theta)
        # Avoid division by zero for small theta
        coef = torch.where(torch.abs(theta) < 1e-4, 0.5 + theta**2/12, coef)
        
        R_minus_Rt = R - R.transpose(-1, -2)
        omega_skew = coef[..., None, None] * R_minus_Rt  # (..., 3, 3)
        
        # 2. Translation V_inv (J^-1)
        # V^-1 = I - 0.5 * omega_skew + (1 - theta * cot(theta/2) / 2) * (omega_skew / theta)^2
        # Note: omega_skew is theta * K. So K = omega_skew / theta.
        # A simpler formula exists using omega vector.
        # Let's use the power series expansion for V_inv for small theta, standard for large.
        
        I = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(R)
        omega_sq = torch.matmul(omega_skew, omega_skew)
        
        # coef2 = 1/theta^2 * (1 - (theta/2) * cot(theta/2))
        # Derived: (1 - 0.5 * theta * (cos/sin)) / theta^2
        half_theta = theta / 2
        cot_half_theta = torch.cos(half_theta) / torch.sin(half_theta)
        coef2 = (1 - half_theta * cot_half_theta) / (theta**2)
        
        # Small angle approx for coef2
        # Series: 1/12 + theta^2/720 ...
        coef2 = torch.where(torch.abs(theta) < 1e-4, 1/12.0, coef2)
        
        V_inv = I - 0.5 * omega_skew + coef2[..., None, None] * omega_sq
        
        # u = V^-1 * t
        u = torch.matmul(V_inv, t.unsqueeze(-1)).squeeze(-1)
        
        # Construct xi 4x4
        xi = torch.zeros_like(T)
        xi[..., :3, :3] = omega_skew
        xi[..., :3, 3] = u
        
        return xi
    
if __name__ == '__main__':
    # Test SE3 Manifold
    print("Testing SE3 Manifold...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    se3 = SE3().to(device)
    
    # 1. Random Generation (via Exp map from random tangent)
    B = 4
    # Random algebra element
    xi_vec = torch.randn(B, 6, device=device)
    # Convert to 4x4 se(3) matrix manually for test input
    omega = xi_vec[:, :3]
    v = xi_vec[:, 3:]
    
    # Skew-symmetric map
    # [[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]]
    O = torch.zeros(B, 3, 3, device=device)
    O[:, 0, 1] = -omega[:, 2]; O[:, 0, 2] = omega[:, 1]
    O[:, 1, 0] = omega[:, 2]; O[:, 1, 2] = -omega[:, 0]
    O[:, 2, 0] = -omega[:, 1]; O[:, 2, 1] = omega[:, 0]
    
    xi = torch.zeros(B, 4, 4, device=device)
    xi[:, :3, :3] = O
    xi[:, :3, 3] = v
    
    # Identity
    I = torch.eye(4, device=device).expand(B, 4, 4)
    
    # Generate random SE(3) points
    x = se3.expmap(I, xi)  # exp_I(xi) = exp(xi)
    
    # 2. Test projx
    # Add noise
    noise = torch.randn_like(x) * 0.1
    x_noisy = x + noise
    x_proj = se3.projx(x_noisy)
    
    # Check determinant of R is 1
    det_R = torch.linalg.det(x_proj[:, :3, :3])
    print(f"Projx Det(R) error: {(det_R - 1.0).abs().max().item():.6f}")
    # Check bottom row
    bottom_err = (x_proj[:, 3, :] - torch.tensor([0., 0., 0., 1.], device=device)).abs().max()
    print(f"Projx Bottom row error: {bottom_err.item():.6f}")

    # 3. Test expmap & logmap consistency
    # y = exp_x(u) -> u_rec = log_x(y)
    # Generate random tangent u at x
    # u = x * xi_rand
    xi_rand = torch.randn_like(xi)
    xi_rand[:, :3, :3] = 0.5 * (xi_rand[:, :3, :3] - xi_rand[:, :3, :3].transpose(-1, -2)) # make skew
    xi_rand[:, 3, :] = 0
    u = torch.matmul(x, xi_rand)
    
    # Project u just in case
    u = se3.proju(x, u)
    
    y = se3.expmap(x, u)
    u_rec = se3.logmap(x, y)
    
    recon_err = (u - u_rec).norm() / u.norm()
    print(f"Exp/Log Consistency Error: {recon_err.item():.6f}")
    
    # 4. Geodesic Path check (t=0 -> x, t=1 -> y)
    # path(t) = exp_x(t * log_x(y))
    t = 0.5
    u_xy = se3.logmap(x, y)
    mid_pt = se3.expmap(x, t * u_xy)
    
    # Check if mid_pt is on manifold
    mid_proj = se3.projx(mid_pt)
    dist = (mid_pt - mid_proj).norm()
    print(f"Midpoint Manifold Constraint Error: {dist.item():.6f}")

    print("Test Finished.")