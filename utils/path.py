import torch
from torch import Tensor
from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import PathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like
from flow_matching.utils.manifolds import Manifold

class SE3GeodesicProbPath(ProbPath):
    r"""
    SE(3) Geodesic Path that returns 6D velocity vectors.
    """

    def __init__(self, manifold: Manifold, scheduler: ConvexScheduler = None):
        self.manifold = manifold
        self.scheduler = scheduler

    def _mat2vec_se3(self, mat: Tensor) -> Tensor:
        """
        Helper: Convert 4x4 Lie Algebra matrix (se(3)) to 6D vector.
        Args:
            mat: (B, 4, 4)
                [[ [w]x,  v ],
                 [  0  ,  0 ]]
        Returns:
            vec: (B, 6) -> [v_x, v_y, v_z, w_x, w_y, w_z]
        """
        # 1. Linear Velocity (v)
        v = mat[..., :3, 3] # (B, 3)
        
        # 2. Angular Velocity (w) from skew-symmetric part
        # [ 0  -wz  wy ]
        # [ wz  0  -wx ]
        # [-wy  wx  0  ]
        wx = mat[..., 2, 1]
        wy = mat[..., 0, 2]
        wz = mat[..., 1, 0]
        w = torch.stack([wx, wy, wz], dim=-1) # (B, 3)
        
        # Concatenate [v, w]
        return torch.cat([v, w], dim=-1)

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        
        # 1. Time Scheduling
        if self.scheduler is not None:
            t_expanded = expand_tensor_like(t, x_1[..., 0:1])
            # ! Edit: Retrieve d_alpha_t for correct velocity scaling
            sched_out = self.scheduler(t_expanded)
            time_val = sched_out.alpha_t.squeeze(-1)
            d_alpha_t = sched_out.d_alpha_t.squeeze(-1)
        else:
            time_val = t
            d_alpha_t = torch.ones_like(t) # ! Edit: Linear case derivative is 1
        
        t_view = time_val.view(-1, 1, 1)
        d_alpha_view = d_alpha_t.view(-1, 1) # ! Edit: View for broadcasting

        # 2. Compute Relative Transform
        # T_rel = x_0^{-1} @ x_1
        x_0_inv = torch.linalg.inv(x_0)
        t_rel = torch.matmul(x_0_inv, x_1)
        
        # 3. Compute Log Map (Lie Algebra Matrix)
        identity = torch.eye(4, device=x_0.device, dtype=x_0.dtype).expand_as(x_0)
        
        # u_static_mat: (B, 4, 4) Lie Algebra Matrix at Identity
        # SE3.logmap은 (x, y) -> x @ log(x^-1 y)를 반환하므로,
        # x=Identity를 넣으면 순수한 Lie Algebra Matrix xi가 나옵니다.
        u_static_mat = self.manifold.logmap(identity, t_rel)
        
        # 4. Extract 6D Target Velocity (Vector)
        # [Key Update] 4x4 행렬을 6차원 벡터로 변환하여 반환
        # ! Edit: Apply time derivative scale (d_alpha_t) to target velocity
        dx_t_vec = self._mat2vec_se3(u_static_mat) * d_alpha_view # (B, 6)

        # 5. Compute Intermediate Pose x_t
        # Integrate: x_t = x_0 * exp(t * u)
        u_t = u_static_mat * t_view
        exp_u_t = self.manifold.expmap(identity, u_t)
        x_t = torch.matmul(x_0, exp_u_t)

        # Return PathSample
        # dx_t에는 이제 6차원 벡터가 들어갑니다.
        return PathSample(x_t=x_t, dx_t=dx_t_vec, x_1=x_1, x_0=x_0, t=t)