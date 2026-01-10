import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import math
from omegaconf import OmegaConf
from scipy.spatial import cKDTree as KDTree
from iterative_closet_point.bunny import run_icp, calculatenormal
from utils.data import data_loader
import hydra
from hydra.core.hydra_config import HydraConfig
from termcolor import colored

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
        R_diff = R_pred.T @ R_gt
        trace = np.trace(R_diff)
        trace = np.clip((trace - 1) / 2, -1.0, 1.0)
        rre = np.degrees(np.arccos(trace))
        centroid_pred = np.mean(P_pred, axis=0)
        centroid_gt = np.mean(P_gt, axis=0)
        rte = np.linalg.norm(centroid_pred - centroid_gt)
        cd = GeometryUtils.compute_chamfer_distance(P_pred, P_gt)
        tree = KDTree(Q_target)
        dists, _ = tree.query(P_pred, k=1)
        ir = np.mean(dists < ir_thresh) * 100.0
        return rre, rte, cd, ir

    # ! New Implementation: Lie Algebra Utilities from provided code
    @staticmethod
    def skew(v):
        x, y, z = v.ravel()
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)

    @staticmethod
    def exp_so3(omega):
        theta = np.linalg.norm(omega)
        if theta < 1e-12: return np.eye(3)
        K = GeometryUtils.skew(omega / theta)
        return np.eye(3) + math.sin(theta)*K + (1 - math.cos(theta))*(K @ K)

# ==================================================================================
# 2. Multi-Strategy ICP Solvers
# ==================================================================================
class ICPSolvers:
    def __init__(self, P, Q, g_p_gt, g_q_gt, 
                 neural_model=None, neural_preds=None, batch_idx=None,
                 max_iter=60, tol=1e-6, chi2_thresh=9.0, method='p2p'):
        self.P_raw = P
        self.Q_raw = Q
        self.g_p_gt = g_p_gt / (np.linalg.norm(g_p_gt) + 1e-8)
        self.g_q_gt = g_q_gt / (np.linalg.norm(g_q_gt) + 1e-8)
        
        self.model = neural_model
        self.neural_preds = neural_preds
        self.batch_idx = batch_idx
        
        self.max_iter = max_iter
        self.tol = tol
        self.method = method # 'p2p', 'p2l', 'l2l'
        
        # ! New Implementation: Normals are needed for both P and Q depending on method
        self.norm_P = calculatenormal(P, k=20)
        self.norm_Q = calculatenormal(Q, k=20)
        
        self.centroid_P = np.mean(P, axis=0)
        self.centroid_Q = np.mean(Q, axis=0)
        self.tree_Q = KDTree(self.Q_raw)

        self.chi2 = chi2_thresh
        
    def _get_resolution(self, pcd):
        tree = KDTree(pcd); d, _ = tree.query(pcd[:100], k=2)
        return np.mean(d[:, 1])

    # ! New Implementation: Integrated Gauss-Newton Solver Step (Replacing SVD)
    # This logic matches the provided 'run_icp' structure but allows Neural Gating injection.
    def _step_gauss_newton(self, Pt, Qt, nP, nQ, R_curr):
        A = np.zeros((6,6)); b = np.zeros((6,1))
        N = Pt.shape[0]
        I3 = np.eye(3)

        # 1. Assemble Linear System (A * delta = b)
        if self.method == 'p2p':
            for i in range(N):
                x = Pt[i] - Qt[i]
                J = np.zeros((3,6))
                J[:, :3] = I3
                J[:, 3:] = -GeometryUtils.skew(Pt[i])
                A += J.T @ J
                b += J.T @ x.reshape(3,1)
                
        elif self.method == 'p2l':
            # Needs Target Normals (nQ)
            for i in range(N):
                n = nQ[i].reshape(1,3)
                e = float(n @ (Pt[i] - Qt[i]).reshape(3,1))
                J = np.zeros((1,6))
                J[0,:3] = n
                J[0,3:] = -n @ GeometryUtils.skew(Pt[i])
                A += J.T @ J
                b += J.T * e
                
        elif self.method == 'l2l':
            # Needs Source (nP) and Target (nQ) Normals
            # nP must be rotated by current R to match frame
            # But here we assume nP is already rotated outside or we handle it here.
            # The provided code assumes R handles rotation. 
            # In our loop, Pt is already rotated. So nP should be rotated too.
            for i in range(N):
                nq = nQ[i].reshape(3,1)
                np_curr = nP[i].reshape(3,1) 
                
                Tq = I3 - nq @ nq.T
                Tp = I3 - np_curr @ np_curr.T
                # Mi approx = Tq + Tp (since R is handled by input points)
                Mi = Tq + Tp 
                
                ri = (Pt[i] - Qt[i]).reshape(3,1)
                J = np.zeros((3,6))
                J[:, :3] = I3
                J[:, 3:] = -GeometryUtils.skew(Pt[i])
                A += J.T @ Mi @ J
                b += J.T @ Mi @ ri

        # 2. Solve for Delta
        try:
            delta = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(A, b, rcond=None)[0]
            
        dt = delta[:3].reshape(3,1)
        dw = delta[3:].ravel()
        dR = GeometryUtils.exp_so3(dw)
        
        return dR, dt.flatten()

    # -------------------------------------------------------------------------
    # A. Baseline Methods (Standard & GT Gravity)
    # -------------------------------------------------------------------------
    def run_standard_icp(self):
        P_init = self.P_raw + (self.centroid_Q - self.centroid_P)
        resolution = self._get_resolution(P_init)
        # ! New Implementation: Pass self.method to external run_icp
        R, t, _ = run_icp(P_init, self.Q_raw, self.method, self.norm_P, self.norm_Q, self.max_iter, self.tol, 10.0*resolution, np.eye(3), np.zeros((3,1)), False)
        t_center = self.centroid_Q - self.centroid_P
        P_final = (R @ P_init.T).T + t.flatten()
        t_total = (R @ t_center.reshape(3, 1)).flatten() + t.flatten()
        return P_final, R, t_total

    def run_gravity_icp(self):
        R_grav = GeometryUtils.get_gravity_rotation(self.g_p_gt, self.g_q_gt)
        P_aligned = (R_grav @ self.P_raw.T).T
        t_center = self.centroid_Q - np.mean(P_aligned, axis=0)
        P_init = P_aligned + t_center
        norm_P_aligned = (R_grav @ self.norm_P.T).T
        resolution = self._get_resolution(P_init)
        # ! New Implementation: Pass self.method
        R_ref, t_ref, _ = run_icp(P_init, self.Q_raw, self.method, norm_P_aligned, self.norm_Q, self.max_iter, self.tol, 10.0*resolution, np.eye(3), np.zeros((3,1)), False)
        R_total = R_ref @ R_grav
        t_total = (R_ref @ t_center.reshape(3,1)).flatten() + t_ref.flatten()
        P_final = (R_total @ self.P_raw.T).T + t_total
        return P_final, R_total, t_total

    def run_gravity_yaw_search_icp(self):
        R_grav = GeometryUtils.get_gravity_rotation(self.g_p_gt, self.g_q_gt)
        P_grav = (R_grav @ self.P_raw.T).T
        t_center_base = self.centroid_Q - np.mean(P_grav, axis=0)
        P_grav_centered = P_grav + t_center_base
        
        best_angle = 0; min_cost = float('inf')
        search_indices = np.random.choice(len(P_grav_centered), min(500, len(P_grav_centered)), replace=False)
        P_subset = P_grav_centered[search_indices]
        
        for angle in np.arange(0, 360, 5): 
            R_yaw = GeometryUtils.get_rotation_around_axis(self.g_q_gt, angle)
            P_rotated = (R_yaw @ P_subset.T).T
            dists, _ = self.tree_Q.query(P_rotated, k=1)
            cost = np.mean(dists)
            if cost < min_cost: min_cost = cost; best_angle = angle
                
        R_best_yaw = GeometryUtils.get_rotation_around_axis(self.g_q_gt, best_angle)
        P_best_init = (R_best_yaw @ P_grav_centered.T).T
        resolution = self._get_resolution(P_best_init)
        norm_P_best = (R_best_yaw @ (R_grav @ self.norm_P.T).T.T).T

        # ! New Implementation: Pass self.method
        R_ref, t_ref, _ = run_icp(P_best_init, self.Q_raw, self.method, norm_P_best, self.norm_Q, self.max_iter, self.tol, 10.0*resolution, np.eye(3), np.zeros((3,1)), False)
        R_total = R_ref @ R_best_yaw @ R_grav
        P_final = (R_ref @ P_best_init.T).T + t_ref.flatten()
        t_total = np.mean(P_final, 0) - (R_total @ np.mean(self.P_raw, 0))
        return P_final, R_total, t_total

    def _run_constrained_solver(self, use_inclination=False, use_height=False):
        # ! New Implementation: Updated to use Gauss-Newton Step
        R_grav = GeometryUtils.get_gravity_rotation(self.g_p_gt, self.g_q_gt)
        P_curr = (R_grav @ self.P_raw.T).T
        t_center = self.centroid_Q - np.mean(P_curr, axis=0)
        P_curr += t_center
        norm_P_curr = (R_grav @ self.norm_P.T).T
        
        inc_Q = np.dot(self.norm_Q, self.g_q_gt)
        h_Q = np.dot(self.Q_raw, self.g_q_gt)
        R_accum = np.eye(3); t_accum = np.zeros(3)
        resolution = self._get_resolution(P_curr)
        dist_thresh = 10.0 * resolution

        for i in range(self.max_iter):
            dists, indices = self.tree_Q.query(P_curr, k=1)
            valid = dists < dist_thresh
            if use_inclination:
                inc_P = np.dot(norm_P_curr, self.g_q_gt)
                valid &= (np.abs(inc_P - inc_Q[indices]) < 0.2)
            if use_height:
                h_P = np.dot(P_curr, self.g_q_gt)
                h_diff = h_Q[indices] - h_P
                median_h_diff = np.median(h_diff[valid]) if np.sum(valid) > 0 else 0
                valid &= (np.abs(h_diff - median_h_diff) < 0.3)

            if np.sum(valid) < 6: break
            
            src_match = P_curr[valid]; tgt_match = self.Q_raw[indices][valid]
            nP_match = norm_P_curr[valid]; nQ_match = self.norm_Q[indices][valid]
            
            # Optimization Step
            dR, dt = self._step_gauss_newton(src_match, tgt_match, nP_match, nQ_match, R_accum)
            
            # Update
            P_curr = (P_curr @ dR.T) + dt
            norm_P_curr = (norm_P_curr @ dR.T) # Rotate Normals
            
            R_accum = dR @ R_accum
            t_accum = (dR @ t_accum.reshape(3,1)).flatten() + dt

        R_total = R_accum @ R_grav
        t_total = (R_accum @ t_center.reshape(3,1)).flatten() + t_accum
        P_final = (R_total @ self.P_raw.T).T + t_total
        return P_final, R_total, t_total

    def run_gravity_inclination_icp(self): return self._run_constrained_solver(use_inclination=True, use_height=False)
    def run_fused_icp(self): return self._run_constrained_solver(use_inclination=True, use_height=True)

    # -------------------------------------------------------------------------
    # B. Neural Methods (BNN vs DNN)
    # -------------------------------------------------------------------------
    def run_neural_constrained_icp(self):
        """ BNN: Uses Predicted Gravity + Kappa-based Hypothesis Testing """
        if self.model is None: return self.P_raw, np.eye(3), np.zeros(3)
        g_p_pred = self.neural_preds['mu_p'].detach().cpu().numpy()
        g_q_pred = self.neural_preds['mu_q'].detach().cpu().numpy()
        
        R_grav = GeometryUtils.get_gravity_rotation(g_p_pred, g_q_pred)
        P_curr = (R_grav @ self.P_raw.T).T
        t_center = self.centroid_Q - np.mean(P_curr, axis=0)
        P_curr += t_center
        norm_P_curr = (R_grav @ self.norm_P.T).T # ! New Implementation: Track normals
        
        R_accum = np.eye(3); t_accum = np.zeros(3)
        resolution = self._get_resolution(P_curr)
        dist_thresh = 10.0 * resolution

        for i in range(self.max_iter):
            dists, indices = self.tree_Q.query(P_curr, k=1)
            valid_geom = dists < dist_thresh
            
            # [KEY] Kappa-based Hypothesis Testing
            P_indices = np.arange(len(P_curr))
            is_inlier_neural, _ = self.model.encoder.check_correspondence_validity(
                batch_idx=self.batch_idx, P_indices=P_indices, Q_indices=indices,
                g_p=self.neural_preds['mu_p'], kappa_p=self.neural_preds['kappa_p'],
                g_q=self.neural_preds['mu_q'], kappa_q=self.neural_preds['kappa_q'],
                chi2_thresh= self.chi2
            )
            valid = valid_geom & is_inlier_neural
            
            if np.sum(valid) < 6: break
            
            src_match = P_curr[valid]; tgt_match = self.Q_raw[indices][valid]
            nP_match = norm_P_curr[valid]; nQ_match = self.norm_Q[indices][valid]
            
            # ! New Implementation: Optimization Step using Gauss-Newton
            dR, dt = self._step_gauss_newton(src_match, tgt_match, nP_match, nQ_match, R_accum)
            
            # Update
            P_curr = (P_curr @ dR.T) + dt
            norm_P_curr = (norm_P_curr @ dR.T) # Rotate Normals
            
            R_accum = dR @ R_accum
            t_accum = (dR @ t_accum.reshape(3,1)).flatten() + dt

        R_total = R_accum @ R_grav
        t_total = (R_accum @ t_center.reshape(3,1)).flatten() + t_accum
        P_final = (R_total @ self.P_raw.T).T + t_total
        return P_final, R_total, t_total

    def run_neural_no_kappa_icp(self):
        """ DNN: Uses Predicted Gravity + Fixed Threshold (No Kappa) """
        if self.model is None: return self.P_raw, np.eye(3), np.zeros(3)
        g_p_pred = self.neural_preds['mu_p'].detach().cpu().numpy()
        g_q_pred = self.neural_preds['mu_q'].detach().cpu().numpy()
        
        R_grav = GeometryUtils.get_gravity_rotation(g_p_pred, g_q_pred)
        P_curr = (R_grav @ self.P_raw.T).T
        t_center = self.centroid_Q - np.mean(P_curr, axis=0)
        P_curr += t_center
        norm_P_curr = (R_grav @ self.norm_P.T).T 
        
        R_accum = np.eye(3); t_accum = np.zeros(3)
        resolution = self._get_resolution(P_curr)
        dist_thresh = 10.0 * resolution
        
        inc_Q = np.dot(self.norm_Q, g_q_pred)

        for i in range(self.max_iter):
            dists, indices = self.tree_Q.query(P_curr, k=1)
            valid = dists < dist_thresh
            
            # [KEY] Fixed Threshold (0.2)
            inc_P = np.dot(norm_P_curr, g_q_pred)
            valid &= (np.abs(inc_P - inc_Q[indices]) < 0.2) 
            
            if np.sum(valid) < 6: break
            
            src_match = P_curr[valid]; tgt_match = self.Q_raw[indices][valid]
            nP_match = norm_P_curr[valid]; nQ_match = self.norm_Q[indices][valid]
            
            # ! New Implementation: Optimization Step using Gauss-Newton
            dR, dt = self._step_gauss_newton(src_match, tgt_match, nP_match, nQ_match, R_accum)
            
            # Update
            P_curr = (P_curr @ dR.T) + dt
            norm_P_curr = (norm_P_curr @ dR.T)
            
            R_accum = dR @ R_accum
            t_accum = (dR @ t_accum.reshape(3,1)).flatten() + dt

        R_total = R_accum @ R_grav
        t_total = (R_accum @ t_center.reshape(3,1)).flatten() + t_accum
        P_final = (R_total @ self.P_raw.T).T + t_total
        return P_final, R_total, t_total

    def run_neural_prediction(self):
        """ 
        Flow Model: ODE integration using predicted velocity field.
        (Centering Logic Removed as requested)
        """
        if self.model is None: 
            return self.P_raw, np.eye(3), np.zeros(3)
        
        device = next(self.model.parameters()).device
        
        # [Step 1] Raw Input Preparation (No Centering)
        # P_raw를 그대로 Tensor로 변환하여 사용합니다.
        P_raw_tensor = torch.from_numpy(self.P_raw).float().to(device).unsqueeze(0).transpose(1, 2)
        Q_target = torch.from_numpy(self.Q_raw).float().to(device).unsqueeze(0).transpose(1, 2)
        
        # [Step 2] 초기 포즈 설정 (T_init)
        # Centering을 하지 않으므로, 원점(Identity)에서 시작하거나
        # 필요하다면 여기서 초기 위치를 잡아줄 수 있습니다.
        # 일단 "Centering 제거" 요청에 맞춰 완전한 Identity에서 시작합니다.
        T_init = torch.eye(4, device=device).unsqueeze(0)
        
        # [Step 3] Solver 설정
        from utils.se3 import SE3
        from utils.inference import SE3VectorField, RiemannianEulerSolver
        
        se3_manifold = SE3().to(device)
        
        # Vector Field에 Centered가 아닌 P_raw_tensor를 넘깁니다.
        vf = SE3VectorField(self.model, P_raw_tensor, Q_target) 
        
        solver = RiemannianEulerSolver(
            vector_field=vf,
            manifold=se3_manifold,
            step_size=0.1 # t=0 -> t=1 (10 steps)
        )
        
        # [Step 4] 적분 수행 (AttributeError 방지 수정)
        # sample() 대신 sample_trajectory()를 호출하고 마지막 스텝([-1])을 가져옵니다.
        trajectory = solver.sample_trajectory(T_init, t0=0.0, t1=1.0)
        T_pred = trajectory[-1] # (1, 4, 4)
        
        # [Step 5] 결과 적용
        T_pred_np = T_pred[0].detach().cpu().numpy()
        R_pred = T_pred_np[:3, :3]
        t_pred = T_pred_np[:3, 3]
        
        # P_raw에 대해 예측된 R, t를 바로 적용
        P_final = (R_pred @ self.P_raw.T).T + t_pred
        
        # t_total은 별도의 보정 없이 예측값 그대로입니다.
        t_total = t_pred
        
        return P_final, R_pred, t_total

# ==================================================================================
# 3. Visualization & Main Logic
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

def visualize_samples(ckpt_path, model, chi2, method, loader, num_samples=50):
    print(f"Processing samples: Robustness Comparison (Target: {num_samples})...")
    save_dir = os.path.join(ckpt_path, "results_comprehensive")
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    algo_names = [
        "Standard", "Gravity(GT)", "YawSearch", "Inc(GT)", "Fused(GT)", 
        "Model(Flow)", "DNN", "BNN", "BNN(Iter)"
    ]
    stats = {name: [] for name in algo_names}
    total_processed = 0
    
    THRESH_RRE = 5.0
    THRESH_RTE = 0.2
    
    CW_ID, CW_AL = 6, 13
    print("\n" + "="*(CW_ID + len(algo_names)*CW_AL + 20))
    header = f"{'Sample':<{CW_ID}} | " + " | ".join([f"{name[:12]:<{CW_AL}}" for name in algo_names])
    print(header)
    print("="*(CW_ID + len(algo_names)*CW_AL + 20))

    with torch.no_grad():
        for batch in loader:
            if total_processed >= num_samples: break
            
            p_tensor = batch['p'].to(device)
            q_tensor = batch['q'].to(device)
            
            # GravityFlowAgent requires (x, t, context_q)
            t_dummy = torch.zeros(p_tensor.shape[0], 1, device=device)
            _, (mu_p_b, k_p_b), (mu_q_b, k_q_b) = model(p_tensor, t_dummy, q_tensor)
            
            batch_p_normals_backup = model.encoder.p_normals
            batch_q_normals_backup = model.encoder.q_normals
            
            src_batch = batch['p'].numpy().transpose(0, 2, 1) 
            tgt_batch = batch['q'].numpy().transpose(0, 2, 1)
            grav_p_gt_batch = batch['g_p_init'].numpy()
            grav_q_gt_batch = batch['g_q'].numpy()
            R_gt_batch = batch['R_pq'].numpy()
            t_gt_batch = batch['t_pq'].numpy()
            
            batch_size = len(src_batch)
            for i in range(batch_size):
                if total_processed >= num_samples: break
                
                # Set normals for current sample only (batch_idx will be 0)
                model.encoder.p_normals = batch_p_normals_backup[i:i+1]
                model.encoder.q_normals = batch_q_normals_backup[i:i+1]
                
                P_raw = src_batch[i]; Q = tgt_batch[i]
                g_p_gt = grav_p_gt_batch[i]; g_q_gt = grav_q_gt_batch[i]
                
                neural_preds_init = {
                    'mu_p': mu_p_b[i], 'kappa_p': k_p_b[i], 
                    'mu_q': mu_q_b[i], 'kappa_q': k_q_b[i]
                }
                
                # ! New Implementation: Pass method
                # batch_idx=0 because we sliced normals to single sample
                solver = ICPSolvers(P_raw, Q, g_p_gt, g_q_gt, 
                                    neural_model=model, neural_preds=neural_preds_init, 
                                    batch_idx=0, chi2_thresh=chi2, method=method)
                
                ir_thresh = 3.0 * solver._get_resolution(P_raw)
                P_gt = (R_gt_batch[i] @ P_raw.T).T + t_gt_batch[i]
                
                current_metrics = []
                results = {}

                for algo in algo_names:
                    P_res, R_res, t_res = None, None, None

                    if algo == "Standard": P_res, R_res, t_res = solver.run_standard_icp()
                    elif algo == "Gravity(GT)": P_res, R_res, t_res = solver.run_gravity_icp()
                    elif algo == "YawSearch": P_res, R_res, t_res = solver.run_gravity_yaw_search_icp()
                    elif algo == "Inc(GT)": P_res, R_res, t_res = solver.run_gravity_inclination_icp()
                    elif algo == "Fused(GT)": P_res, R_res, t_res = solver.run_fused_icp()
                    elif algo == "Model(Flow)": P_res, R_res, t_res = solver.run_neural_prediction()
                    elif algo == "DNN": P_res, R_res, t_res = solver.run_neural_no_kappa_icp() 
                    elif algo == "BNN": P_res, R_res, t_res = solver.run_neural_constrained_icp() 
                    
                    elif algo == "BNN(Iter)":
                        P_curr = P_raw.copy()
                        R_accum = np.eye(3)
                        t_accum = np.zeros(3)
                        max_outer_iter = 5 
                        for iter_idx in range(max_outer_iter):
                            p_tensor_curr = torch.from_numpy(P_curr).float().to(device).transpose(0, 1).unsqueeze(0)
                            t_curr = torch.zeros(1, 1, device=device)
                            _, (mu_p_curr, k_p_curr), _ = model(p_tensor_curr, t_curr, q_tensor[i].unsqueeze(0))
                            curr_preds = {
                                'mu_p': mu_p_curr[0], 'kappa_p': k_p_curr[0],
                                'mu_q': mu_q_b[i],    'kappa_q': k_q_b[i] 
                            }
                            solver_step = ICPSolvers(P_curr, Q, g_p_gt, g_q_gt, 
                                                     neural_model=model, neural_preds=curr_preds, 
                                                     batch_idx=0, chi2_thresh=chi2, method=method)
                            P_next, dR, dt = solver_step.run_neural_constrained_icp()
                            trace = np.trace(dR)
                            trace = np.clip((trace - 1) / 2, -1.0, 1.0)
                            angle_diff = np.degrees(np.arccos(trace))
                            dist_diff = np.linalg.norm(dt)
                            R_accum = dR @ R_accum
                            t_accum = (dR @ t_accum.reshape(3,1)).flatten() + dt
                            P_curr = P_next
                            if angle_diff < 0.1 and dist_diff < 1e-4: break
                        P_res, R_res, t_res = P_curr, R_accum, t_accum

                    m = GeometryUtils.compute_full_metrics(P_res, P_gt, Q, R_res, R_gt_batch[i], t_res, t_gt_batch[i], ir_thresh)
                    stats[algo].append(m)
                    results[algo] = P_res
                    current_metrics.append(m)
                
                log_strs = []
                for m in current_metrics:
                    rre, rte = m[0], m[1]
                    val_str = f"{rre:5.1f}/{rte:<4.2f}"
                    padded_str = f"{val_str:<{CW_AL}}"
                    if rre < THRESH_RRE and rte < THRESH_RTE:
                        log_strs.append(colored(padded_str, 'green'))
                    else:
                        log_strs.append(colored(padded_str, 'red'))
                
                row_str = f"{total_processed:<{CW_ID}} | " + " | ".join(log_strs)
                print(row_str)
                
                fig = plt.figure(figsize=(24, 18))
                ax1 = fig.add_subplot(3, 4, 1, projection='3d')
                draw_result(ax1, P_raw + (solver.centroid_Q - solver.centroid_P), Q, g_p_gt, g_q_gt, "Input (Std Init)", ir_thresh=ir_thresh)
                for idx, algo in enumerate(algo_names):
                    ax = fig.add_subplot(3, 4, idx+2, projection='3d')
                    g_vis = neural_preds_init['mu_p'].detach().cpu().numpy() if "Neural" in algo or "BNN" in algo or "DNN" in algo or "Model" in algo else g_p_gt
                    draw_result(ax, results[algo], Q, g_vis, g_q_gt, algo, f"RRE:{stats[algo][-1][0]:.1f}", ir_thresh=ir_thresh)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{total_processed}.png"))
                plt.close()
                total_processed += 1
                
    print("\n" + "="*140)
    print(f" FINAL SUMMARY (Over {total_processed} samples)")
    print("="*140)
    print(f"{'Algorithm':<15} | {'Succ Rate':<10} | {'Mean RRE':<10} | {'Mean RTE':<10} | {'Mean CD':<10}")
    print("-" * 140)
    for name in algo_names:
        arr = np.array(stats[name])
        if len(arr) == 0: continue
        success_mask = (arr[:, 0] < THRESH_RRE) & (arr[:, 1] < THRESH_RTE)
        success_rate = np.mean(success_mask) * 100.0
        print(f"{name:<15} | {success_rate:<10.1f} | {arr[:,0].mean():<10.4f} | {arr[:,1].mean():<10.4f} | {arr[:,2].mean():<10.4f}")
    print("="*140)

def main(ckpt, chi2, method, overrides=None):
    yaml_path = os.path.join(os.path.dirname(ckpt), '.hydra', 'config.yaml')
    if os.path.exists(yaml_path):
        cfg = OmegaConf.load(yaml_path)
        if overrides:
            print(colored(f"[Info] Applying overrides: {overrides}", "yellow"))
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
        try:
            model = hydra.utils.instantiate(cfg.model).to(cfg.device)
        except Exception as e:
            print(colored(f"[Error] Model instantiation failed: {e}", "red"))
            return
        weight_path = os.path.join(ckpt, "weights/ckpt.pt")
        if os.path.exists(weight_path):            
            checkpoint = torch.load(weight_path, map_location=cfg.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(colored(f"Loaded Neural Model Weights from {weight_path}", "green"))
    else:
        print("[Error] Config not found.")
        return
    print(cfg)
    _, test_loader = data_loader(cfg)
    visualize_samples(os.path.dirname(ckpt), model, chi2, method, test_loader, num_samples=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory or file")
    parser.add_argument("--chi2", type=float, default=9.0, help="Chi-squared threshold for BNN correspondence validity")
    parser.add_argument("--method", type=str, default='p2p', help="ICP method to use: ['p2p', 'p2l', 'l2l']")
    args, unknown_args = parser.parse_known_args()
    main(args.ckpt, args.chi2, args.method, overrides=unknown_args)