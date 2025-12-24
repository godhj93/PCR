import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from omegaconf import OmegaConf
from scipy.spatial import cKDTree as KDTree
from iterative_closet_point.bunny import run_icp, calculatenormal
from utils.data import data_loader

# ==================================================================================
# 1. Math & Metrics Helper
# ==================================================================================
class GeometryUtils:
    @staticmethod
    def get_gravity_rotation(g_src, g_tgt):
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
    def get_rotation_around_axis(axis, angle_deg):
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        theta = np.radians(angle_deg)
        K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    @staticmethod
    def compute_chamfer_distance(P, Q):
        tree_q = KDTree(Q); dist_p2q, _ = tree_q.query(P, k=1)
        tree_p = KDTree(P); dist_q2p, _ = tree_p.query(Q, k=1)
        return np.mean(dist_p2q) + np.mean(dist_q2p)

    @staticmethod
    def compute_full_metrics(P_pred, P_gt, Q_target, R_pred, R_gt, t_pred, t_gt, ir_thresh=0.1):
        # 1. RRE (Relative Rotation Error) in degrees
        R_diff = R_pred.T @ R_gt
        trace = np.trace(R_diff)
        trace = np.clip((trace - 1) / 2, -1.0, 1.0)
        rre = np.degrees(np.arccos(trace))
        # 2. RTE (Relative Translation Error)
        centroid_pred = np.mean(P_pred, axis=0)
        centroid_gt = np.mean(P_gt, axis=0)
        rte = np.linalg.norm(centroid_pred - centroid_gt)
        # 3. Chamfer Distance
        cd = GeometryUtils.compute_chamfer_distance(P_pred, P_gt)
        # 4. Inlier Ratio
        tree = KDTree(Q_target)
        dists, _ = tree.query(P_pred, k=1)
        ir = np.mean(dists < ir_thresh) * 100.0
        return rre, rte, cd, ir

# ==================================================================================
# 2. Multi-Strategy ICP Solvers (동일함)
# ==================================================================================
class ICPSolvers:
    def __init__(self, P, Q, g_p, g_q, max_iter=60, tol=1e-6):
        self.P_raw = P
        self.Q_raw = Q
        self.g_p = g_p / (np.linalg.norm(g_p) + 1e-8)
        self.g_q = g_q / (np.linalg.norm(g_q) + 1e-8)
        self.max_iter = max_iter
        self.tol = tol
        self.norm_P = calculatenormal(P, k=20)
        self.norm_Q = calculatenormal(Q, k=20)
        self.centroid_P = np.mean(P, axis=0)
        self.centroid_Q = np.mean(Q, axis=0)
        self.tree_Q = KDTree(self.Q_raw)

    def _get_resolution(self, pcd):
        tree = KDTree(pcd); d, _ = tree.query(pcd[:100], k=2)
        return np.mean(d[:, 1])

    # 1. Standard ICP
    def run_standard_icp(self):
        P_init = self.P_raw + (self.centroid_Q - self.centroid_P)
        resolution = self._get_resolution(P_init)
        R, t, _ = run_icp(P_init, self.Q_raw, "p2p", self.norm_P, self.norm_Q, self.max_iter, self.tol, 10.0*resolution, np.eye(3), np.zeros((3,1)), False)
        t_center = self.centroid_Q - self.centroid_P
        P_final = (R @ P_init.T).T + t.flatten()
        t_total = (R @ t_center.reshape(3, 1)).flatten() + t.flatten()
        return P_final, R, t_total

    # 2. Gravity + ICP
    def run_gravity_icp(self):
        R_grav = GeometryUtils.get_gravity_rotation(self.g_p, self.g_q)
        P_aligned = (R_grav @ self.P_raw.T).T
        t_center = self.centroid_Q - np.mean(P_aligned, axis=0)
        P_init = P_aligned + t_center
        norm_P_aligned = (R_grav @ self.norm_P.T).T
        resolution = self._get_resolution(P_init)
        R_ref, t_ref, _ = run_icp(P_init, self.Q_raw, "p2p", norm_P_aligned, self.norm_Q, self.max_iter, self.tol, 10.0*resolution, np.eye(3), np.zeros((3,1)), False)
        R_total = R_ref @ R_grav
        t_total = (R_ref @ t_center.reshape(3,1)).flatten() + t_ref.flatten()
        P_final = (R_total @ self.P_raw.T).T + t_total
        return P_final, R_total, t_total

    # 3. Gravity + Yaw Search (1 degree) + ICP
    def run_gravity_yaw_search_icp(self):
        R_grav = GeometryUtils.get_gravity_rotation(self.g_p, self.g_q)
        P_grav = (R_grav @ self.P_raw.T).T
        norm_P_grav = (R_grav @ self.norm_P.T).T
        t_center_base = self.centroid_Q - np.mean(P_grav, axis=0)
        P_grav_centered = P_grav + t_center_base
        
        best_angle = 0
        min_cost = float('inf')
        search_indices = np.random.choice(len(P_grav_centered), min(500, len(P_grav_centered)), replace=False)
        P_subset = P_grav_centered[search_indices]
        angles = np.arange(0, 360, 1)
        
        for angle in angles:
            R_yaw = GeometryUtils.get_rotation_around_axis(self.g_q, angle)
            P_rotated = (R_yaw @ P_subset.T).T
            dists, _ = self.tree_Q.query(P_rotated, k=1)
            cost = np.mean(dists)
            if cost < min_cost:
                min_cost = cost
                best_angle = angle
                
        R_best_yaw = GeometryUtils.get_rotation_around_axis(self.g_q, best_angle)
        P_best_init = (R_best_yaw @ P_grav_centered.T).T
        norm_P_best = (R_best_yaw @ norm_P_grav.T).T
        resolution = self._get_resolution(P_best_init)
        R_ref, t_ref, _ = run_icp(P_best_init, self.Q_raw, "p2p", norm_P_best, self.norm_Q, self.max_iter, self.tol, 10.0*resolution, np.eye(3), np.zeros((3,1)), False)
        
        R_total = R_ref @ R_best_yaw @ R_grav
        P_final_exact = (R_ref @ P_best_init.T).T + t_ref.flatten()
        t_total = np.mean(P_final_exact, axis=0) - (R_total @ np.mean(self.P_raw, axis=0))
        return P_final_exact, R_total, t_total

    # 4-5. Constrained Solvers
    def run_gravity_inclination_icp(self): return self._run_constrained_solver(use_inclination=True, use_height=False)
    def run_fused_icp(self): return self._run_constrained_solver(use_inclination=True, use_height=True)

    def _run_constrained_solver(self, use_inclination=False, use_height=False):
        R_grav = GeometryUtils.get_gravity_rotation(self.g_p, self.g_q)
        P_curr = (R_grav @ self.P_raw.T).T
        t_center = self.centroid_Q - np.mean(P_curr, axis=0)
        P_curr += t_center
        norm_P_curr = (R_grav @ self.norm_P.T).T
        
        inc_Q = np.dot(self.norm_Q, self.g_q)
        h_Q = np.dot(self.Q_raw, self.g_q)
        R_accum = np.eye(3); t_accum = np.zeros(3)
        resolution = self._get_resolution(P_curr)
        dist_thresh = 10.0 * resolution

        for i in range(self.max_iter):
            dists, indices = self.tree_Q.query(P_curr, k=1)
            valid = dists < dist_thresh
            if use_inclination:
                inc_P = np.dot(norm_P_curr, self.g_q)
                valid &= (np.abs(inc_P - inc_Q[indices]) < 0.2)
            if use_height:
                h_P = np.dot(P_curr, self.g_q)
                h_diff = h_Q[indices] - h_P
                median_h_diff = np.median(h_diff[valid]) if np.sum(valid) > 0 else 0
                valid &= (np.abs(h_diff - median_h_diff) < 0.3)

            if np.sum(valid) < 10: break
            src_match = P_curr[valid]; tgt_match = self.Q_raw[indices][valid]
            mu_s = np.mean(src_match, 0); mu_t = np.mean(tgt_match, 0)
            W = (src_match - mu_s).T @ (tgt_match - mu_t)
            U, _, Vt = np.linalg.svd(W)
            R_step = Vt.T @ U.T
            if np.linalg.det(R_step) < 0: Vt[2,:] *= -1; R_step = Vt.T @ U.T
            t_step = mu_t - R_step @ mu_s
            P_curr = (R_step @ P_curr.T).T + t_step
            norm_P_curr = (R_step @ norm_P_curr.T).T
            R_accum = R_step @ R_accum
            t_accum = (R_step @ t_accum.reshape(3,1)).flatten() + t_step

        R_total = R_accum @ R_grav
        t_total = (R_accum @ t_center.reshape(3,1)).flatten() + t_accum
        P_final = (R_total @ self.P_raw.T).T + t_total
        return P_final, R_total, t_total

# ==================================================================================
# 3. Visualization & Logging (수정됨)
# ==================================================================================
# ... (set_axes_equal_union, draw_gravity_arrow, draw_result 함수는 동일하여 생략) ...
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
    ax.quiver(x, y, z, u, v, w, color=color, length=1.0, normalize=False, linewidth=3.0, arrow_length_ratio=0.25)
    ax.text(x + u, y + v, z + w, label, color=color, fontsize=10, fontweight='bold')

def draw_result(ax, P, Q, g_p, g_q, title, metrics_str=None, color_title='black', ir_thresh=0.1):
    all_pts = np.concatenate([P, Q], axis=0)
    scale = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.6
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c='blue', alpha=0.3, label='Source')
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=2, c='orange', alpha=0.3, label='Target')
    if g_p is not None: draw_gravity_arrow(ax, np.mean(P, axis=0), g_p, 'blue', "g_P", scale)
    if g_q is not None: draw_gravity_arrow(ax, np.mean(Q, axis=0), g_q, 'red', "g_Q", scale)
    ax.set_title(f"{title}\n{metrics_str}" if metrics_str else title, fontsize=10, color=color_title, fontweight='bold')
    set_axes_equal_union(ax, all_pts)


def visualize_samples(ckpt_path, loader, num_samples=50):
    print(f"Processing samples: 5-Way Robustness Comparison (Target: {num_samples})...")
    save_dir = os.path.join(ckpt_path, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    algo_names = ["Standard", "Gravity", "YawSearch", "Inc", "Fused"]
    stats = {name: [] for name in algo_names}
    total_processed = 0
    
    # --- [LOGGING FIX] Define Table Column Widths & Header ---
    CW_ID = 6   # Sample ID width
    CW_AL = 15  # Algorithm column width (enough for "XXX.X/X.XX")
    
    # Print Header
    print("\n" + "="*(CW_ID + 5*CW_AL + 16))
    header_top = f"{'Sample':<{CW_ID}} | " + " | ".join([f"{name:<{CW_AL}}" for name in algo_names])
    header_sub = f"{'ID':<{CW_ID}} | " + " | ".join([f"{'RRE(°)/RTE(m)':<{CW_AL}}"] * 5)
    print(header_top)
    print(header_sub)
    print("="*(CW_ID + 5*CW_AL + 16))
    # ---------------------------------------------------------

    for batch_idx, batch in enumerate(loader):
        if total_processed >= num_samples: break
        # (데이터 로딩 부분 생략 - 동일함)
        src_batch = batch['p'].numpy(); src_batch = src_batch.transpose(0, 2, 1) if src_batch.shape[1] == 3 else src_batch
        grav_p_batch = batch['gravity_p'].numpy()
        grav_q_batch = batch['gravity_q'].numpy()
        R_gt_batch = batch['R_pq'].numpy()
        t_gt_batch = batch['t_pq'].numpy()
        tgt_batch = batch['q'].numpy(); tgt_batch = tgt_batch.transpose(0, 2, 1) if tgt_batch.shape[1] == 3 else tgt_batch
        
        batch_size = len(src_batch)
        for i in range(batch_size):
            if total_processed >= num_samples: break
            
            P_raw = src_batch[i]; Q = tgt_batch[i]; g_p = grav_p_batch[i]; g_q = grav_q_batch[i]
            R_gt = R_gt_batch[i]; t_gt = t_gt_batch[i]
            P_gt = (R_gt @ P_raw.T).T + t_gt
            
            solver = ICPSolvers(P_raw, Q, g_p, g_q)
            ir_thresh = 3.0 * solver._get_resolution(P_raw)
            gt_str = f"Rot: {np.degrees(np.arccos(np.clip((np.trace(R_gt)-1)/2, -1, 1))):.1f}°"
            results = {}; current_metrics = []

            # --- Run Algorithms ---
            # 1. Standard
            P_std, R_std, t_std = solver.run_standard_icp()
            m = GeometryUtils.compute_full_metrics(P_std, P_gt, Q, R_std, R_gt, t_std, t_gt, ir_thresh)
            stats["Standard"].append(m); results["Standard"] = (P_std, R_std)
            current_metrics.append(m)

            # 2. Gravity
            P_grav, R_grav, t_grav = solver.run_gravity_icp()
            m = GeometryUtils.compute_full_metrics(P_grav, P_gt, Q, R_grav, R_gt, t_grav, t_gt, ir_thresh)
            stats["Gravity"].append(m); results["Gravity"] = (P_grav, R_grav)
            current_metrics.append(m)

            # 3. YawSearch
            P_yaw, R_yaw, t_yaw = solver.run_gravity_yaw_search_icp()
            m = GeometryUtils.compute_full_metrics(P_yaw, P_gt, Q, R_yaw, R_gt, t_yaw, t_gt, ir_thresh)
            stats["YawSearch"].append(m); results["YawSearch"] = (P_yaw, R_yaw)
            current_metrics.append(m)

            # 4. Inc
            P_inc, R_inc, t_inc = solver.run_gravity_inclination_icp()
            m = GeometryUtils.compute_full_metrics(P_inc, P_gt, Q, R_inc, R_gt, t_inc, t_gt, ir_thresh)
            stats["Inc"].append(m); results["Inc"] = (P_inc, R_inc)
            current_metrics.append(m)

            # 5. Fused
            P_fused, R_fused, t_fused = solver.run_fused_icp()
            m = GeometryUtils.compute_full_metrics(P_fused, P_gt, Q, R_fused, R_gt, t_fused, t_gt, ir_thresh)
            stats["Fused"].append(m); results["Fused"] = (P_fused, R_fused)
            current_metrics.append(m)

            # --- [LOGGING FIX] Print Aligned Row ---
            # Format: "RRE(6.1f)/RTE(<4.2f)" padded to CW_AL width
            log_strs = [f"{m[0]:6.1f}/{m[1]:<4.2f}" for m in current_metrics]
            row_str = f"{total_processed:<{CW_ID}} | " + " | ".join([f"{s:<{CW_AL}}" for s in log_strs])
            print(row_str)
            # ---------------------------------------

            # --- Visualization (생략 - 동일함) ---
            
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(2, 4, 1, projection='3d')
            draw_result(ax1, solver.P_raw+(solver.centroid_Q-solver.centroid_P), Q, g_p, g_q, "1. Input", ir_thresh=ir_thresh)
            ax2 = fig.add_subplot(2, 4, 2, projection='3d')
            draw_result(ax2, P_gt, Q, g_q, g_q, "2. GT", gt_str, ir_thresh=ir_thresh)
            ax3 = fig.add_subplot(2, 4, 3, projection='3d'); m = stats["Standard"][-1]
            draw_result(ax3, results["Standard"][0], Q, results["Standard"][1]@g_p, g_q, "3. Standard", f"RRE:{m[0]:.1f}", 'black', ir_thresh)
            ax4 = fig.add_subplot(2, 4, 4, projection='3d'); m = stats["Gravity"][-1]
            draw_result(ax4, results["Gravity"][0], Q, results["Gravity"][1]@g_p, g_q, "4. Gravity", f"RRE:{m[0]:.1f}", 'blue', ir_thresh)
            ax5 = fig.add_subplot(2, 4, 5, projection='3d'); m = stats["YawSearch"][-1]
            draw_result(ax5, results["YawSearch"][0], Q, results["YawSearch"][1]@g_p, g_q, "5. YawSearch", f"RRE:{m[0]:.1f}", 'purple', ir_thresh)
            ax6 = fig.add_subplot(2, 4, 6, projection='3d'); m = stats["Inc"][-1]
            draw_result(ax6, results["Inc"][0], Q, results["Inc"][1]@g_p, g_q, "6. Inc", f"RRE:{m[0]:.1f}", 'magenta', ir_thresh)
            ax7 = fig.add_subplot(2, 4, 7, projection='3d'); m = stats["Fused"][-1]
            col = 'green' if m[0] < 5.0 and m[1] < 0.1 else 'red'
            draw_result(ax7, results["Fused"][0], Q, results["Fused"][1]@g_p, g_q, "7. Fused", f"RRE:{m[0]:.1f}", col, ir_thresh)
            ax8 = fig.add_subplot(2, 4, 8, projection='3d'); R_g = GeometryUtils.get_gravity_rotation(g_p, g_q)
            P_g_view = (R_g @ P_raw.T).T + (solver.centroid_Q - np.mean(P_raw,0))
            draw_result(ax8, P_g_view, Q, R_g@g_p, g_q, "8. Gravity View", color_title='gray', ir_thresh=ir_thresh)
            plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"result_{total_processed}.png"), dpi=100); plt.close()
        
            total_processed += 1

    # --- Summary (동일함) ---
    THRESH_RRE = 5.0; THRESH_RTE = 0.1
    print("\n" + "="*120)
    print(f" FINAL SUMMARY (Over {total_processed} samples)")
    print(f" * Criteria: Success if RRE < {THRESH_RRE}° AND RTE < {THRESH_RTE}")
    print("="*120)
    print(f"{'Algorithm':<15} | {'Succ Rate':<10} | {'Mean RRE':<10} | {'Mean RTE':<10} | {'Mean CD':<10}")
    print("-" * 120)
    for name in algo_names:
        arr = np.array(stats[name])
        if len(arr) == 0: continue
        success_mask = (arr[:, 0] < THRESH_RRE) & (arr[:, 1] < THRESH_RTE)
        success_rate = np.mean(success_mask) * 100.0
        print(f"{name:<15} | {success_rate:<10.1f} | {arr[:,0].mean():<10.4f} | {arr[:,1].mean():<10.4f} | {arr[:,2].mean():<10.4f}")
    print("="*120)

def main(ckpt):
    yaml_path = os.path.join(os.path.dirname(ckpt), '.hydra', 'config.yaml')
    cfg = OmegaConf.load(yaml_path)
    if not hasattr(cfg.data, 'partial_overlap'): cfg.data.partial_overlap = True
    _, test_loader = data_loader(cfg)
    visualize_samples(os.path.dirname(ckpt), test_loader, num_samples=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args.ckpt)