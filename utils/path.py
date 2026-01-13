import torch
from torch import Tensor
from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import PathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like
from flow_matching.utils.manifolds import Manifold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    def _vec2mat_se3(self, vec: Tensor) -> Tensor:
        """
        Convert 6D vector [v, w] to 4x4 se(3) matrix.
        Args:
            vec: (B, 6) - [v_x, v_y, v_z, w_x, w_y, w_z]
        Returns:
            mat: (B, 4, 4) - Lie algebra matrix
        """
        B = vec.shape[0]
        v = vec[..., :3]  # (B, 3)
        w = vec[..., 3:]  # (B, 3)
        
        wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
        
        zero = torch.zeros(B, device=vec.device, dtype=vec.dtype)
        
        mat = torch.zeros(B, 4, 4, device=vec.device, dtype=vec.dtype)
        mat[..., 0, 1] = -wz
        mat[..., 0, 2] = wy
        mat[..., 1, 0] = wz
        mat[..., 1, 2] = -wx
        mat[..., 2, 0] = -wy
        mat[..., 2, 1] = wx
        mat[..., :3, 3] = v
        
        return mat
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


def visualize_se3_path():
    """
    Visualize SE(3) geodesic path interpolation.
    Shows how poses evolve from x_0 to x_1 over time.
    """
    from utils.se3 import SE3    
    # Create SE(3) manifold
    manifold = SE3()
    path = SE3GeodesicProbPath(manifold=manifold)
    
    # Create initial and target poses
    B = 1
    device = 'cpu'
    
    # x_0: Identity pose
    x_0 = torch.eye(4, device=device).unsqueeze(0)  # (1, 4, 4)
    
    # x_1: Target pose with rotation and translation
    # Rotation: 90 degrees around z-axis
    angle = torch.tensor([torch.pi / 2], device=device)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    R = torch.zeros(1, 3, 3, device=device)
    R[0, 0, 0] = cos_a
    R[0, 0, 1] = -sin_a
    R[0, 1, 0] = sin_a
    R[0, 1, 1] = cos_a
    R[0, 2, 2] = 1.0
    
    # Translation: [2, 1, 0.5]
    t_vec = torch.tensor([[2.0, 1.0, 0.5]], device=device)
    
    # Build x_1
    x_1 = torch.eye(4, device=device).unsqueeze(0)
    x_1[0, :3, :3] = R[0]
    x_1[0, :3, 3] = t_vec[0]
    
    # Sample along the path
    num_steps = 20
    t_values = torch.linspace(0, 1, num_steps, device=device)
    
    positions = []
    orientations = []
    velocities = []
    
    for t_val in t_values:
        t = t_val.unsqueeze(0).expand(B)
        sample = path.sample(x_0, x_1, t)
        
        # Extract position (translation)
        pos = sample.x_t[0, :3, 3].cpu().numpy()
        positions.append(pos)
        
        # Extract orientation (rotation matrix x-axis direction)
        x_axis = sample.x_t[0, :3, 0].cpu().numpy()
        orientations.append(x_axis)
        
        # Extract velocity
        vel = sample.dx_t[0].cpu().numpy()
        velocities.append(vel)
    
    positions = torch.tensor(positions)
    orientations = torch.tensor(orientations)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Path')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', s=100, marker='s', label='End')
    
    # Draw orientation vectors at several points
    for i in range(0, num_steps, 4):
        ax1.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                   orientations[i, 0], orientations[i, 1], orientations[i, 2],
                   length=0.3, color='orange', arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('SE(3) Geodesic Path')
    ax1.legend()
    ax1.grid(True)
    
    # Position over time
    ax2 = fig.add_subplot(132)
    ax2.plot(t_values.cpu().numpy(), positions[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(t_values.cpu().numpy(), positions[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(t_values.cpu().numpy(), positions[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Position')
    ax2.set_title('Translation Components')
    ax2.legend()
    ax2.grid(True)
    
    # Velocity over time
    ax3 = fig.add_subplot(133)
    velocities_array = torch.tensor(velocities)
    ax3.plot(t_values.cpu().numpy(), velocities_array[:, 0], label='v_x', linewidth=2)
    ax3.plot(t_values.cpu().numpy(), velocities_array[:, 1], label='v_y', linewidth=2)
    ax3.plot(t_values.cpu().numpy(), velocities_array[:, 2], label='v_z', linewidth=2)
    ax3.plot(t_values.cpu().numpy(), velocities_array[:, 3], '--', label='ω_x', linewidth=2)
    ax3.plot(t_values.cpu().numpy(), velocities_array[:, 4], '--', label='ω_y', linewidth=2)
    ax3.plot(t_values.cpu().numpy(), velocities_array[:, 5], '--', label='ω_z', linewidth=2)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Velocity (6D: linear + angular)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/hj/Projects/scan_matching/flow_icp/se3_path_visualization.png', dpi=150)
    plt.show()
    
    print("Path visualization saved to: se3_path_visualization.png")
    print(f"Start pose:\n{x_0[0]}")
    print(f"\nEnd pose:\n{x_1[0]}")


if __name__ == "__main__":
    visualize_se3_path()
    
