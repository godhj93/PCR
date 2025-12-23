import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial import cKDTree as KDTree
from utils.data import data_loader
from iterative_closet_point.bunny import run_icp, calculatenormal

# ==================================================================================
# 1. Math & Metrics Helper
# ==================================================================================
class GeometryUtils:
    @staticmethod
    def get_gravity_rotation(g_src, g_tgt):
        """ Calculate rotation matrix to align g_src to g_tgt """
        u = g_src / (np.linalg.norm(g_src) + 1e-8)
        v = g_tgt / (np.linalg.norm(g_tgt) + 1e-8)
        axis = np.cross(u, v)
        norm_axis = np.linalg.norm(axis)
        dot = np.dot(u, v)
        
        if norm_axis < 1e-6:
            if dot > 0: return np.eye(3) 
            else: 
                if np.abs(u[0]) > 0.9: return np.diag([-1, 1, -1]) 
                else: return np.diag([1, -1, -1]) 
            
        K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
        R = np.eye(3) + K + (K @ K) * ((1 - dot) / (norm_axis ** 2))
        return R

    @staticmethod
    def compute_full_metrics(P_pred, P_gt, Q_target, R_pred, R_gt, t_pred, t_gt, ir_thresh=0.1):
        # 1. RRE
        R_diff = R_pred.T @ R_gt
        trace = np.trace(R_diff)
        trace = np.clip((trace - 1) / 2, -1.0, 1.0)
        rre = np.degrees(np.arccos(trace))
        
        # 2. RTE
        centroid_pred = np.mean(P_pred, axis=0)
        centroid_gt = np.mean(P_gt, axis=0)
        rte = np.linalg.norm(centroid_pred - centroid_gt)
        
        # 3. RMSE
        # Note: RMSE only makes sense if P_pred and P_gt are point-to-point correspondents
        # For partial overlap, this might be high even if aligned well visually
        mse = np.mean(np.sum((P_pred - P_gt)**2, axis=1))
        rmse = np.sqrt(mse)
        
        # 4. Inlier Ratio
        tree = KDTree(Q_target)
        dists, _ = tree.query(P_pred, k=1)
        ir = np.mean(dists < ir_thresh) * 100.0
        
        return rre, rte, rmse, ir

# ==================================================================================
# 2. Standard ICP Solver Class
# ==================================================================================
class StandardICPSolver:
    def __init__(self, P, Q, max_iter=60, tol=1e-6):
        self.P_raw = P
        self.Q_raw = Q
        self.max_iter = max_iter
        self.tol = tol
        
        self.norm_P = calculatenormal(P, k=20)
        self.norm_Q = calculatenormal(Q, k=20)
        
        # Initial Centering
        self.centroid_P = np.mean(P, axis=0)
        self.centroid_Q = np.mean(Q, axis=0)
        self.P_centered = P + (self.centroid_Q - self.centroid_P)

    def register_standard(self):
        # Adaptive Threshold
        tree = KDTree(self.P_centered)
        d, _ = tree.query(self.P_centered[:100], k=2)
        resolution = np.mean(d[:, 1])
        
        # Standard ICP
        R, t, _ = run_icp(
            self.P_centered, self.Q_raw, method="p2p", 
            normals_P=self.norm_P, normals_Q=self.norm_Q,
            max_iter=self.max_iter, tol=self.tol, dist_thresh=10.0*resolution,
            R_init=np.eye(3), t_init=np.zeros((3,1)), verbose=False
        )
        
        # Recover Final Pose
        t_center = self.centroid_Q - self.centroid_P
        P_final = (R @ self.P_centered.T).T + t.flatten()
        t_total = (R @ t_center.reshape(3, 1)).flatten() + t.flatten()
        
        return P_final, R, t_total

# ==================================================================================
# 3. Visualization
# ==================================================================================
def set_axes_equal_union(ax, points):
    center = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0 * 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()

def draw_gravity_arrow(ax, origin, vector, color, label, scale):
    v_norm = vector / (np.linalg.norm(vector) + 1e-6)
    u, v, w = v_norm * scale
    x, y, z = origin
    ax.quiver(x, y, z, u, v, w, color=color, length=1.0, normalize=False, linewidth=4.0, arrow_length_ratio=0.25)
    ax.text(x + u, y + v, z + w, label, color=color, fontsize=12, fontweight='bold')

def draw_result(ax, P, Q, g_p, g_q, title, metrics_str=None, color_title='black', ir_thresh=0.1):
    all_pts = np.concatenate([P, Q], axis=0)
    scale = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.6
    
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=3, c='blue', alpha=0.3, label='Source')
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=3, c='orange', alpha=0.3, label='Target')
    
    if g_p is not None:
        draw_gravity_arrow(ax, np.mean(P, axis=0), g_p, 'blue', "g_P", scale)
    if g_q is not None:
        draw_gravity_arrow(ax, np.mean(Q, axis=0), g_q, 'red', "g_Q", scale)

    ax.set_title(f"{title}\n{metrics_str}" if metrics_str else title, fontsize=12, color=color_title, fontweight='bold')
    set_axes_equal_union(ax, all_pts)

def visualize_samples(ckpt_path, loader, num_samples=5):
    print(f"Processing samples: Standard ICP (Real Data Loader Input)...")
    save_dir = os.path.join(ckpt_path, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    try: batch = next(iter(loader)) 
    except StopIteration: return
    
    src_batch = batch['p'].numpy() 
    if src_batch.shape[1] == 3: src_batch = src_batch.transpose(0, 2, 1)
    
    grav_p_batch = batch['gravity_p'].numpy()
    grav_q_batch = batch['gravity_q'].numpy()
    
    algo_names = ["Standard ICP"]
    stats = {name: [] for name in algo_names}

    actual_samples = min(len(src_batch), num_samples)
    
    for i in range(actual_samples):
        # [Correct Data Loading] No hardcoded transformation here!
        P_raw = src_batch[i]
        
        # Load Q from dataset directly (Respects partial overlap settings)
        Q = batch['q'][i].numpy()
        if Q.shape[0] == 3: Q = Q.T
            
        g_p = grav_p_batch[i]
        g_q = grav_q_batch[i]
        
        # Load GT for metrics
        R_gt = batch['R_pq'][i].numpy()
        t_gt = batch['t_pq'][i].numpy()
        
        # P_gt is where P should be after perfect transformation
        # Note: If P and Q are different crops, P_gt will overlap with Q but not match point-to-point.
        P_gt = (R_gt @ P_raw.T).T + t_gt
        
        # --- [Prep] Gravity Alignment Visualization ---
        R_grav = GeometryUtils.get_gravity_rotation(g_p, g_q)
        
        center_P = np.mean(P_raw, axis=0)
        center_Q = np.mean(Q, axis=0)
        P_grav_aligned = ((R_grav @ (P_raw - center_P).T).T) + center_Q
        g_p_aligned = R_grav @ g_p

        # --- Solver ---
        solver = StandardICPSolver(P_raw, Q)
        tree = KDTree(solver.P_centered); d, _ = tree.query(solver.P_centered[:100], k=2)
        ir_thresh = 3.0 * np.mean(d[:, 1])
        
        gt_str = f"Rot: {np.degrees(np.arccos(np.clip((np.trace(R_gt)-1)/2, -1, 1))):.1f}°"
        
        # --- Run Standard ICP ---
        P_std, R_std, t_std = solver.register_standard()
        
        met_std = GeometryUtils.compute_full_metrics(P_std, P_gt, Q, R_std, R_gt, t_std, t_gt, ir_thresh)
        stats["Standard ICP"].append(met_std)

        # --- Visualization (2x2 Layout) ---
        fig = plt.figure(figsize=(15, 15))
        
        # 1. Initial State
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        draw_result(ax1, solver.P_centered, Q, g_p, g_q, "1. Initial (Centered)", ir_thresh=ir_thresh)
        
        # 2. Gravity Aligned
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        draw_result(ax2, P_grav_aligned, Q, g_p_aligned, g_q, "2. Gravity Aligned", color_title='green', ir_thresh=ir_thresh)
        
        # 3. Ground Truth
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        draw_result(ax3, P_gt, Q, g_q, g_q, "3. Ground Truth (Target Pose)", gt_str, ir_thresh=ir_thresh)
        
        # 4. Standard ICP Result
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        met_str = f"RRE:{met_std[0]:.1f}, RTE:{met_std[1]:.3f}"
        color = 'red' if met_std[3] < 50 else 'blue'
        draw_result(ax4, P_std, Q, R_std@g_p, g_q, "4. Standard ICP", met_str, color, ir_thresh=ir_thresh)
        
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"result_{i}.png"), dpi=100); plt.close()
        print(f"Sample {i} | {gt_str} | Std ICP RRE: {met_std[0]:.2f}")

    # --- Summary ---
    print("\n" + "="*90)
    print(f" FINAL SUMMARY (Over {actual_samples} samples)")
    print("="*90)
    print(f"{'Algorithm':<20} | {'Succ Rate (%)':<15} | {'Mean RRE':<10} | {'Mean RTE':<10} | {'Mean IR':<10}")
    print("-" * 90)
    for name in algo_names:
        arr = np.array(stats[name])
        success_mask = (arr[:, 0] < 5.0) & (arr[:, 1] < 0.1)
        success_rate = np.mean(success_mask) * 100.0
        mean_rre = np.mean(arr[:, 0])
        mean_rte = np.mean(arr[:, 1])
        mean_ir  = np.mean(arr[:, 3])
        print(f"{name:<20} | {success_rate:<15.1f} | {mean_rre:<10.2f} | {mean_rte:<10.3f} | {mean_ir:<10.1f}")
    print("="*90)

def main(ckpt):
    yaml_path = os.path.join(os.path.dirname(ckpt), '.hydra', 'config.yaml')
    cfg = OmegaConf.load(yaml_path)
    
    # Load config as-is. Data generation is now fully delegated to RegistrationDataset via loader
    if not hasattr(cfg.data, 'partial_overlap'): cfg.data.partial_overlap = True
    
    _, test_loader = data_loader(cfg)
    visualize_samples(os.path.dirname(ckpt), test_loader, num_samples=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args.ckpt)