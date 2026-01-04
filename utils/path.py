import torch
from torch import Tensor
from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import PathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like
from flow_matching.utils.manifolds import Manifold

class SE3GeodesicProbPath(ProbPath):
    r"""
    SE(3) Geodesic Path following the standard Riemannian flow matching framework.
    Returns both 4x4 tangent matrices (for integration) and 6D vectors (for training).
    """

    def __init__(self, manifold: Manifold, scheduler: ConvexScheduler = None):
        self.manifold = manifold
        self.scheduler = scheduler

    def _mat2vec_se3(self, mat: Tensor) -> Tensor:
        """
        Convert 4x4 se(3) matrix to 6D vector [v, w].
        Args:
            mat: (B, 4, 4) - Lie algebra matrix
        Returns:
            vec: (B, 6) - [v_x, v_y, v_z, w_x, w_y, w_z]
        """
        # Linear velocity
        v = mat[..., :3, 3]  # (B, 3)
        
        # Angular velocity from skew-symmetric part
        # [[ 0  -wz  wy ],
        #  [ wz  0  -wx ],
        #  [-wy  wx  0  ]]
        wx = mat[..., 2, 1]
        wy = mat[..., 0, 2]
        wz = mat[..., 1, 0]
        w = torch.stack([wx, wy, wz], dim=-1)  # (B, 3)
        
        return torch.cat([v, w], dim=-1)

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        """
        Sample from the geodesic path on SE(3).
        
        Args:
            x_0: (B, 4, 4) - Initial poses (usually identity)
            x_1: (B, 4, 4) - Target poses
            t: (B,) - Time values in [0, 1]
            
        Returns:
            PathSample with:
                - x_t: (B, 4, 4) - Interpolated pose
                - dx_t: (B, 6) - Target velocity as 6D vector (for training)
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)
        
        B = x_0.shape[0]
        
        # 1. Time Scheduling
        # t is (B,), we need to reshape for broadcasting
        t_reshaped = t.view(B, 1, 1)  # (B, 1, 1) for matrix operations
        
        if self.scheduler is not None:
            # Scheduler expects (B, 1) shape
            t_for_scheduler = t.view(B, 1)
            sched_out = self.scheduler(t_for_scheduler)
            alpha_t = sched_out.alpha_t.view(B, 1, 1)  # (B, 1, 1)
            d_alpha_t = sched_out.d_alpha_t.view(B, 1, 1)  # (B, 1, 1)
        else:
            alpha_t = t_reshaped
            d_alpha_t = torch.ones_like(t_reshaped)
        
        # 2. Compute Relative Transform in Lie algebra
        # xi = log(x_0^{-1} @ x_1) - This is the "direction" from x_0 to x_1
        identity = torch.eye(4, device=x_0.device, dtype=x_0.dtype).unsqueeze(0).expand(B, 4, 4)
        x_0_inv = torch.linalg.inv(x_0)
        x_rel = torch.matmul(x_0_inv, x_1)  # (B, 4, 4)
        
        # Get the Lie algebra element (4x4 matrix in se(3))
        xi = self.manifold.logmap(identity, x_rel)  # (B, 4, 4)
        
        # 3. Compute Intermediate Pose x_t
        # x_t = x_0 @ exp(alpha_t * xi)
        xi_t = xi * alpha_t  # (B, 4, 4) * (B, 1, 1) -> (B, 4, 4)
        exp_xi_t = self.manifold.expmap(identity, xi_t)
        x_t = torch.matmul(x_0, exp_xi_t)  # (B, 4, 4)
        
        # 4. Compute Target Velocity
        # In the tangent space at identity: velocity = d_alpha_t * xi
        # Convert to 6D vector for model training
        velocity_at_identity = xi * d_alpha_t  # (B, 4, 4)
        dx_t_vec = self._mat2vec_se3(velocity_at_identity)  # (B, 6)

        return PathSample(x_t=x_t, dx_t=dx_t_vec, x_1=x_1, x_0=x_0, t=t)