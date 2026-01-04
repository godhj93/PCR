import os
import torch
import torch.nn as nn
import argparse
import hydra
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from omegaconf import OmegaConf
from termcolor import colored
from typing import List, Tuple

# flow_matching library imports
from flow_matching.utils.manifolds import Manifold 
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper

# User modules
from utils.se3 import SE3  
from utils.data import data_loader  

# -----------------------------------------------------------------------------
# 0. Helper Functions (Metrics)
# -----------------------------------------------------------------------------

def compute_metrics(T_pred: torch.Tensor, T_gt: torch.Tensor) -> Tuple[float, float]:
    """
    Computes RRE (degrees) and RTE.
    Auto-casts tensors to the same device (CPU) to avoid device mismatch.
    """
    # Safe device handling: Move everything to CPU for metric calculation
    T_pred = T_pred.cpu()
    T_gt = T_gt.cpu()

    R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

    # RTE (Relative Translation Error)
    rte = torch.norm(t_pred - t_gt).item()

    # RRE (Relative Rotation Error)
    R_diff = torch.matmul(R_gt.T, R_pred)
    trace = torch.trace(R_diff)
    
    # Numerical stability clamping
    val = (trace - 1) / 2
    val = torch.clamp(val, -1 + 1e-6, 1 - 1e-6)
    
    rre_rad = torch.acos(val)
    rre_deg = torch.rad2deg(rre_rad).item()
    
    return rre_deg, rte

# -----------------------------------------------------------------------------
# 1. Physics & Solver Classes
# -----------------------------------------------------------------------------

class SE3VectorField(ModelWrapper):
    """
    Wraps the velocity prediction model for use with RiemannianODESolver.
    Follows the standard flow_matching ModelWrapper interface.
    """
    def __init__(self, model: nn.Module, p_src: torch.Tensor, q_tgt: torch.Tensor):
        super().__init__(model)
        self.p_src = p_src  # (B, 3, N)
        self.q_tgt = q_tgt  # (B, 3, N)
        
        # Ensure dimensions
        if self.p_src.shape[1] != 3: self.p_src = self.p_src.transpose(1, 2)
        if self.q_tgt.shape[1] != 3: self.q_tgt = self.q_tgt.transpose(1, 2)

    def _vec2mat_se3(self, v_vec: torch.Tensor) -> torch.Tensor:
        """Convert 6D velocity vector to 4x4 se(3) matrix."""
        v = v_vec[..., :3]  # Linear velocity
        w = v_vec[..., 3:]  # Angular velocity
        
        # Construct skew-symmetric matrix from angular velocity
        zero = torch.zeros_like(w[..., 0])
        w_hat = torch.stack([
            torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1),
            torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1),
            torch.stack([-w[..., 1], w[..., 0], zero], dim=-1)
        ], dim=-2)
        
        # Construct 4x4 se(3) matrix
        B = v_vec.shape[0]
        xi = torch.zeros((B, 4, 4), device=v_vec.device, dtype=v_vec.dtype)
        xi[..., :3, :3] = w_hat
        xi[..., :3, 3] = v
        return xi

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        """
        Standard ModelWrapper interface: forward(x, t, **extras)
        
        Args:
            x: (B, 4, 4) - Current SE(3) poses
            t: (B,) or scalar - Time
            
        Returns:
            u: (B, 4, 4) - Tangent vector at x in T_x SE(3)
        """
        # Extract current transformation
        R_curr = x[..., :3, :3]
        t_curr = x[..., :3, 3].unsqueeze(-1)
        
        # Transform source points to current pose
        P_curr = torch.matmul(R_curr, self.p_src) + t_curr
        
        # Handle time tensor formatting
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t_tensor = torch.full((x.shape[0],), float(t), device=x.device).unsqueeze(1)
        else:
            t_tensor = t.view(-1, 1) if t.ndim == 1 else t
            
        # Model prediction (6D velocity vector)
        outputs = self.model(P_curr, t_tensor, self.q_tgt)
        v_pred_vec = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Convert to se(3) matrix (global frame)
        xi_global = self._vec2mat_se3(v_pred_vec)
        
        # Convert to tangent vector at x: u = x @ xi
        # This ensures u is in the tangent space T_x SE(3)
        u_tangent = torch.matmul(x, xi_global)
        
        return u_tangent


# Note: RiemannianEulerSolver is now replaced by flow_matching.solver.RiemannianODESolver
# The library version is more robust and supports multiple integration methods

# -----------------------------------------------------------------------------
# 2. Visualization Class
# -----------------------------------------------------------------------------

class FlowAnimator:
    def __init__(self, p_src: torch.Tensor, q_tgt: torch.Tensor, 
                 trajectory: torch.Tensor, T_gt: torch.Tensor = None):
        """
        Args:
            p_src: Source points (3, N) or (N, 3)
            q_tgt: Target points (3, N) or (N, 3)
            trajectory: (T, B, 4, 4) or (T, 4, 4) - Trajectory of SE(3) poses
            T_gt: Ground truth transformation (4, 4)
        """
        self.p_src = p_src.detach().cpu().numpy()
        self.q_tgt = q_tgt.detach().cpu().numpy()
        # Handle trajectory from RiemannianODESolver
        self.trajectory = trajectory.detach().cpu() if isinstance(trajectory, torch.Tensor) else trajectory
        self.T_gt = T_gt.detach().cpu() if T_gt is not None else None
        
        if self.p_src.shape[0] == 3: self.p_src = self.p_src.T
        if self.q_tgt.shape[0] == 3: self.q_tgt = self.q_tgt.T
        
    def _apply_transform(self, T: torch.Tensor, points: np.ndarray) -> np.ndarray:
        # T is CPU tensor
        R = T[:3, :3].numpy()
        t = T[:3, 3].numpy()
        return (points @ R.T) + t

    def save_animation(self, save_path: str, fps: int = 10):
        # Bounds calculation
        p_gt_final = self._apply_transform(self.T_gt, self.p_src) if self.T_gt is not None else self.p_src
        all_points = np.concatenate([self.p_src, self.q_tgt, p_gt_final], axis=0)
        min_lim = all_points.min() - 0.5
        max_lim = all_points.max() + 0.5
        
        # Handle trajectory shape: (T, 4, 4) or (T, B, 4, 4)
        if self.trajectory.ndim == 4:
            trajectory = self.trajectory[:, 0]  # Take first batch element
        else:
            trajectory = self.trajectory
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame_idx):
            ax.clear()
            ax.set_xlim(min_lim, max_lim); ax.set_ylim(min_lim, max_lim); ax.set_zlim(min_lim, max_lim)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            
            T_curr = trajectory[frame_idx] # (4, 4) CPU
            
            # Title with Metrics
            title_txt = f"Step {frame_idx}"
            if self.T_gt is not None:
                # T_curr is CPU, self.T_gt is CPU -> Safe
                rre, rte = compute_metrics(T_curr, self.T_gt)
                title_txt += f"\nRRE: {rre:.2f}°, RTE: {rte:.4f}"
            ax.set_title(title_txt)

            # Scatter Plots
            ax.scatter(self.q_tgt[:, 0], self.q_tgt[:, 1], self.q_tgt[:, 2], 
                       c='blue', s=2, alpha=0.3, label='Target')
            
            if self.T_gt is not None:
                p_gt = self._apply_transform(self.T_gt, self.p_src)
                ax.scatter(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2],
                           c='green', s=2, alpha=0.2, label='GT')

            p_curr = self._apply_transform(T_curr, self.p_src)
            ax.scatter(p_curr[:, 0], p_curr[:, 1], p_curr[:, 2], 
                       c='red', s=5, alpha=0.9, label='Flow')
            
            ax.legend(loc='upper right')
        
        # Get number of frames
        n_frames = trajectory.shape[0]
        ani = FuncAnimation(fig, update, frames=n_frames, interval=100)
        
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        plt.close()

# -----------------------------------------------------------------------------
# 3. Main Inference Controller
# -----------------------------------------------------------------------------

class FlowMatchingInference:
    def __init__(self, ckpt_path: str, overrides: List[str] = None):
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = self._load_config(overrides)
        self.model = self._load_model()
        self.manifold = SE3() 

    def _load_config(self, overrides):
        yaml_path = os.path.join(os.path.dirname(self.ckpt_path), '.hydra', 'config.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config not found at {yaml_path}")
        cfg = OmegaConf.load(yaml_path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cfg

    def _load_model(self):
        model = hydra.utils.instantiate(self.cfg.model).to(self.device)
        weight_path = os.path.join(self.ckpt_path, "weights/ckpt.pt")
        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location=self.device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(colored(f"[Model] Weights loaded from {weight_path}", "green"))
        else:
            print(colored("[Warning] Random Weights (Checkpoint not found)", "yellow"))
        return model

    def run_visualization(self, save_dir: str = "vis_outputs", max_batches: int = None):
        os.makedirs(save_dir, exist_ok=True)
        _, test_loader = data_loader(self.cfg) 
        
        self.model.eval()
        global_idx = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                # 1. Prepare Batch Data (CUDA)
                p_init = batch['p'].to(self.device)
                q_tgt = batch['q'].to(self.device)
                
                R_gt = batch['R_pq'].to(self.device)  
                t_gt_vec = batch['t_pq'].to(self.device)  
                
                # GT Matrices (B, 4, 4) - CUDA
                B = p_init.shape[0]
                T_gt = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
                T_gt[:, :3, :3] = R_gt
                T_gt[:, :3, 3] = t_gt_vec

                # 2. Batch Integration using flow_matching RiemannianODESolver
                vf = SE3VectorField(self.model, p_init, q_tgt)
                solver = RiemannianODESolver(manifold=self.manifold, velocity_model=vf)
                
                print(colored(f"[Inference] Integrating Batch {batch_idx} (Size: {B})...", "cyan"))
                
                T_identity = torch.eye(4, device=self.device).unsqueeze(0).repeat(B, 1, 1)
                
                # Use RiemannianODESolver with return_intermediates=True for trajectory
                time_grid = torch.linspace(0.0, 1.0, 20, device=self.device)  # 20 frames
                trajectory_batch = solver.sample(
                    x_init=T_identity,
                    step_size=0.05,
                    method="midpoint",  # More accurate than euler
                    time_grid=time_grid,
                    return_intermediates=True,
                    projx=True,  # Project onto manifold at each step
                    proju=True   # Project velocity onto tangent space
                )
                # trajectory_batch: (T, B, 4, 4)
                
                # 3. Process Each Sample
                for b in range(B):
                    # Extract Trajectory for sample b -> (T, 4, 4) Tensor
                    traj_sample = trajectory_batch[:, b]  # (T, 4, 4)
                    
                    p_sample = p_init[b] # CUDA
                    q_sample = q_tgt[b]  # CUDA
                    
                    # Move GT to CPU for metrics and visualization
                    T_gt_sample = T_gt[b].cpu() 
                    
                    # Final Error Check
                    T_final = traj_sample[-1].cpu() # (4, 4) CPU
                    rre, rte = compute_metrics(T_final, T_gt_sample)
                    
                    print(colored(f"  > Sample {global_idx}: Final RRE={rre:.2f}°, RTE={rte:.4f}", "yellow"))
                    
                    # Animate (p_sample, q_sample will be moved to cpu inside class)
                    animator = FlowAnimator(
                        p_src=p_sample, 
                        q_tgt=q_sample, 
                        trajectory=traj_sample,
                        T_gt=T_gt_sample 
                    )
                    
                    save_name = os.path.join(save_dir, f"vis_batch{batch_idx}_sample{b}_ID{global_idx}.gif")
                    animator.save_animation(save_name)
                    global_idx += 1

# -----------------------------------------------------------------------------
# 4. Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory path")
    parser.add_argument("--max_batches", type=int, default=1, help="Max number of batches")
    parser.add_argument("overrides", nargs="*", help="Hydra config overrides")
    args = parser.parse_args()

    try:
        runner = FlowMatchingInference(args.ckpt, args.overrides)
        runner.run_visualization(save_dir=os.path.join(args.ckpt, "animations"), max_batches=args.max_batches)
        
    except Exception as e:
        print(colored(f"[Fatal] Process terminated: {e}", "red"))
        import traceback
        traceback.print_exc() # 상세 에러 위치 확인용

if __name__ == "__main__":
    main()