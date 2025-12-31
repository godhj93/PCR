import torch
import torch.nn as nn
from typing import Optional, Tuple

# flow_matching 라이브러리 구조 가정 (import 경로가 맞는지 확인 필요)
from flow_matching.utils.manifolds import Manifold
from flow_matching.solver import Solver # 만약 base class가 없다면 생략 가능

class SE3VectorField(nn.Module):
    """
    [Bridge Class]
    Deep Learning Model (takes Point Cloud) <-> ODE Solver (takes SE3 Pose)
    
    This class wraps the trained agent and behaves like a mathematical Vector Field 
    defined on the SE(3) manifold.
    
    u = V(t, x) where u in T_x SE(3)
    """
    def __init__(self, model: nn.Module, p_src: torch.Tensor, q_tgt: torch.Tensor):
        super().__init__()
        self.model = model
        self.p_src = p_src  # (B, 3, N) Initial Source
        self.q_tgt = q_tgt  # (B, 3, N) Fixed Target
        
        # Ensure dimensions
        if self.p_src.shape[1] != 3: self.p_src = self.p_src.transpose(1, 2)
        if self.q_tgt.shape[1] != 3: self.q_tgt = self.q_tgt.transpose(1, 2)

    def _vec2mat_se3(self, v_vec: torch.Tensor) -> torch.Tensor:
        """
        Helper: 6D Vector -> 4x4 Lie Algebra Matrix (xi)
        Input: (B, 6) [vx, vy, vz, wx, wy, wz]
        Output: (B, 4, 4) se(3) matrix
        """
        v = v_vec[..., :3]
        w = v_vec[..., 3:]
        
        zero = torch.zeros_like(w[..., 0])
        w_hat = torch.stack([
            torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1),
            torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1),
            torch.stack([-w[..., 1], w[..., 0], zero], dim=-1)
        ], dim=-2)
        
        B = v_vec.shape[0]
        xi = torch.zeros((B, 4, 4), device=v_vec.device, dtype=v_vec.dtype)
        xi[..., :3, :3] = w_hat
        xi[..., :3, 3] = v
        return xi

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the tangent vector at current pose x.
        
        Args:
            t: Scalar or (1,) tensor time
            x: Current Pose T (B, 4, 4)
            
        Returns:
            u: Tangent vector in T_x SE(3) (B, 4, 4)
               For left-invariant metric: u = x @ xi
        """
        # 1. Transform Point Cloud to current pose x
        # P_t = R * P_0 + t
        R_curr = x[..., :3, :3]
        t_curr = x[..., :3, 3].unsqueeze(-1)
        P_curr = torch.matmul(R_curr, self.p_src) + t_curr
        
        # 2. Handle Time Input
        if isinstance(t, float) or t.ndim == 0:
            t_tensor = torch.full((x.shape[0],), float(t), device=x.device)
        else:
            t_tensor = t.expand(x.shape[0])
            
        # 3. Model Inference (Feedback Loop)
        # Model returns Body Velocity (Lie Algebra vector)
        v_body_vec, _, _ = self.model(P_curr, t_tensor, self.q_tgt)
        
        # 4. Convert to Matrix (Lie Algebra xi)
        xi_mat = self._vec2mat_se3(v_body_vec)
        
        # 5. Push-forward to Tangent Space
        # u = x * xi
        u_tangent = torch.matmul(x, xi_mat)
        
        return u_tangent


class RiemannianEulerSolver:
    """
    Riemannian Euler Solver compatible with flow_matching library style.
    """
    def __init__(self, vector_field: nn.Module, manifold: Manifold, step_size: float = 0.05):
        self.vector_field = vector_field
        self.manifold = manifold
        self.step_size = step_size
        
    def step(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Perform one Euler step on the manifold.
        x_{t+1} = Exp_x( u * dt )
        """
        # 1. Get Tangent Vector u at x
        u = self.vector_field(t, x)
        
        # 2. Scale by step size
        dt = self.step_size
        v = u * dt
        
        # 3. Exponential Map Update
        # x_next = x * exp(x^-1 * v) = x * exp(xi * dt)
        x_next = self.manifold.expmap(x, v)
        
        return x_next

    def sample(self, x_init: torch.Tensor, t0: float = 0.0, t1: float = 1.0) -> torch.Tensor:
        """
        Integrate from t0 to t1.
        """
        x_curr = x_init.clone()
        
        # Create time steps
        steps = int((t1 - t0) / self.step_size)
        times = torch.linspace(t0, t1, steps)
        
        for t in times:
            x_curr = self.step(t, x_curr)
            
        return x_curr