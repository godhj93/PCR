import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial import cKDTree as KDTree
from utils.data import load_modelnet40_data, load_bunny_data, RegistrationDataset

# 사용자가 제공한 라이브러리
from iterative_closet_point.bunny import run_icp, calculatenormal

def data_loader(cfg):
    """ Config 설정에 따라 RegistrationDataset 초기화 """
    dataset_name = cfg.data.name.lower()
    partial_overlap = getattr(cfg.data, 'partial_overlap', False)
    print(f"[DataLoader] Partial Overlap Mode: {partial_overlap}")

    if dataset_name == 'modelnet40':
        print(f"Loading ModelNet40 data (num_points: {cfg.data.num_points})...")
        test_data = load_modelnet40_data('test')
        test_dataset = RegistrationDataset(
            dataset_name=dataset_name,
            data_source=test_data,
            num_points=cfg.data.num_points,
            partition='test',
            gaussian_noise=False,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor,
            partial_overlap=partial_overlap
        )
    elif dataset_name == 'bunny':
        if not hasattr(cfg.data, 'bunny_path'):
            raise KeyError("Config Error: Bunny 데이터셋을 위해 'cfg.data.bunny_path'가 필요합니다.")
        print(f"Loading Bunny from {cfg.data.bunny_path}...")
        shared_data = load_bunny_data(cfg.data.bunny_path)
        test_dataset = RegistrationDataset(
            dataset_name=dataset_name,
            data_source=shared_data,
            num_points=cfg.data.num_points,
            partition='test',
            gaussian_noise=False,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor,
            partial_overlap=partial_overlap
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False
    )
    return test_loader

def compute_resolution(pcd_numpy):
    if pcd_numpy.shape[0] == 0: return 1.0
    tree = KDTree(pcd_numpy)
    n_samples = min(len(pcd_numpy), 1000)
    indices = np.random.choice(len(pcd_numpy), n_samples, replace=False)
    sample_pts = pcd_numpy[indices]
    dists, _ = tree.query(sample_pts, k=2)
    valid_dists = dists[:, 1][dists[:, 1] > 1e-6] 
    if len(valid_dists) == 0: return 0.01
    return np.mean(valid_dists)

def calculate_errors(R_pred, t_pred, R_gt, t_gt):
    R_diff = R_pred.T @ R_gt
    trace = np.trace(R_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rot_error = np.degrees(np.arccos(trace))
    trans_error = np.linalg.norm(t_pred - t_gt)
    return rot_error, trans_error

def set_axes_equal(ax, points):
    """ 3D 플롯의 축 비율을 1:1:1로 맞춤 """
    center = points.mean(axis=0)
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect((1, 1, 1))

def draw_gravity_arrow(ax, origin, vector, color, label, scale_factor=0.8):
    """
    중력 벡터를 화살표로 시각화
    scale_factor: 점군 크기에 비례하여 화살표 길이 조정
    """
    # 벡터 정규화
    v_norm = vector / (np.linalg.norm(vector) + 1e-6)
    
    u, v, w = v_norm * scale_factor
    x, y, z = origin
    
    # 화살표 그리기
    ax.quiver(x, y, z, u, v, w, color=color, length=1.0, normalize=False, 
              linewidth=2.5, arrow_length_ratio=0.3, zorder=100)
    
    # 라벨 표시 (화살표 끝부분)
    ax.text(x + u, y + v, z + w, label, color=color, fontsize=12, fontweight='bold')

def draw_alignment_result(ax, P_aligned, Q, threshold=0.1, num_lines=100):
    """ ICP 결과 및 Correspondence 시각화 """
    ax.scatter(P_aligned[:, 0], P_aligned[:, 1], P_aligned[:, 2], s=1, c='blue', alpha=0.3, label='P (Aligned)')
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=1, c='orange', alpha=0.3, label='Q (Target)')
    
    if len(P_aligned) > 0 and len(Q) > 0:
        indices = np.random.choice(len(P_aligned), min(len(P_aligned), num_lines), replace=False)
        P_samp = P_aligned[indices]
        
        tree = KDTree(Q)
        dists, nn_indices = tree.query(P_samp, k=1)
        Q_nn = Q[nn_indices]
        
        for i in range(len(P_samp)):
            p = P_samp[i]
            q = Q_nn[i]
            dist = dists[i]
            
            color = 'green' if dist < threshold else 'red'
            alpha = 0.6 if dist < threshold else 0.2
            
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c=color, linewidth=0.5, alpha=alpha)

def visualize_samples(ckpt_path, loader, num_samples=5):
    print("Processing samples from DataLoader...")
    save_dir = os.path.join(ckpt_path, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        batch = next(iter(loader)) 
    except StopIteration:
        return

    src_batch = batch['p'].numpy() 
    tgt_batch = batch['q'].numpy()
    R_gt_batch = batch['R_pq'].numpy()
    t_gt_batch = batch['t_pq'].numpy()
    
    # [New] Gravity Vectors from Batch
    g_p_batch = batch['gravity_p'].numpy()
    g_q_batch = batch['gravity_q'].numpy()
    
    if src_batch.shape[1] == 3:
        src_batch = src_batch.transpose(0, 2, 1)
        tgt_batch = tgt_batch.transpose(0, 2, 1)

    for i in range(min(len(src_batch), num_samples)):
        P_raw = src_batch[i]
        Q = tgt_batch[i]
        R_gt = R_gt_batch[i]
        t_gt = t_gt_batch[i]
        g_p = g_p_batch[i] # Source Gravity
        g_q = g_q_batch[i] # Target Gravity
        
        # Hard Mode: No Initialization
        t_init_vec = np.zeros(3)

        # Parameter 계산
        resolution = compute_resolution(P_raw)
        ir_thresh = 3.0 * resolution 
        icp_dist_thresh = 5.0 * resolution
        
        # --- [ICP 수행] ---
        norm_P = calculatenormal(P_raw, k=20)
        norm_Q = calculatenormal(Q, k=20)
        
        R_icp, t_icp, _ = run_icp(
            P_raw, Q, method="p2p", normals_P=norm_P, normals_Q=norm_Q,
            max_iter=60, tol=1e-6, dist_thresh=icp_dist_thresh,
            R_init=np.eye(3), t_init=t_init_vec.reshape(3,1), verbose=False
        )
        
        P_aligned = (R_icp @ P_raw.T).T + t_icp.flatten()
        
        # --- [평가 지표 계산] ---
        rot_err, trans_err = calculate_errors(R_icp, t_icp.flatten(), R_gt, t_gt)
        tree = KDTree(Q)
        dists, _ = tree.query(P_aligned, k=1)
        inlier_ratio = np.mean(dists < ir_thresh) * 100.0
        
        # --- [시각화 준비] ---
        fig = plt.figure(figsize=(24, 10))
        
        # 물체 크기 계산 (화살표 길이 결정을 위해)
        scale_p = (P_raw.max(axis=0) - P_raw.min(axis=0)).max() * 0.6
        scale_q = (Q.max(axis=0) - Q.min(axis=0)).max() * 0.6
        
        # Fig 1: P (Centered) + Gravity P
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        P_center = P_raw - np.mean(P_raw, axis=0) 
        ax1.scatter(P_center[:, 0], P_center[:, 1], P_center[:, 2], s=2, c='blue', alpha=0.5)
        # [Visual] g_p 그리기 (중심에서 시작)
        draw_gravity_arrow(ax1, [0,0,0], g_p, 'black', '$g_p$', scale_factor=scale_p)
        
        ax1.set_title(f"1. Source P (Input)\nPoints: {len(P_raw)}", fontsize=14)
        ax1.set_axis_off()
        set_axes_equal(ax1, P_center)
        
        # Fig 2: Q (Centered) + Gravity Q
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        Q_center = Q - np.mean(Q, axis=0)
        ax2.scatter(Q_center[:, 0], Q_center[:, 1], Q_center[:, 2], s=2, c='orange', alpha=0.5)
        # [Visual] g_q 그리기
        draw_gravity_arrow(ax2, [0,0,0], g_q, 'red', '$g_q$', scale_factor=scale_q)
        
        ax2.set_title(f"2. Target Q (Input)\nPoints: {len(Q)}", fontsize=14)
        ax2.set_axis_off()
        set_axes_equal(ax2, Q_center)
        
        # Fig 3: Alignment Result + Gravity Alignment Check
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        draw_alignment_result(ax3, P_aligned, Q, threshold=ir_thresh)
        
        # [Visual] 결과 확인용 중력 그리기
        # 1. Target Q의 중력 (빨강) - Q의 무게중심에서
        centroid_Q = np.mean(Q, axis=0)
        draw_gravity_arrow(ax3, centroid_Q, g_q, 'red', '$g_q$', scale_factor=scale_q)
        
        # 2. ICP로 회전된 P의 중력 (파랑) - P_aligned의 무게중심에서
        #    g_p_aligned = R_icp * g_p
        g_p_aligned = R_icp @ g_p
        centroid_P_aligned = np.mean(P_aligned, axis=0)
        # 살짝 옆으로 이동해서 겹치지 않게 보이도록 함
        offset = np.array([0.1, 0.1, 0.1]) * scale_q 
        draw_gravity_arrow(ax3, centroid_P_aligned + offset, g_p_aligned, 'blue', "$g_p'$", scale_factor=scale_q)
        
        ax3.set_title(f"3. ICP Result (Gravity Check)\nR_err: {rot_err:.2f}°, Inlier: {inlier_ratio:.1f}%", fontsize=14)
        ax3.legend(loc='upper right', fontsize='small')
        ax3.set_axis_off()
        
        all_pts = np.concatenate([P_aligned, Q], axis=0)
        set_axes_equal(ax3, all_pts)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"icp_result_{i}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path} | R_err: {rot_err:.2f}°, IR: {inlier_ratio:.1f}%")

def main(ckpt):
    yaml_path = os.path.join(os.path.dirname(ckpt), '.hydra', 'config.yaml')
    cfg = OmegaConf.load(yaml_path)
    
    if not hasattr(cfg.data, 'partial_overlap'):
        cfg.data.partial_overlap = True
    
    print("Loading Dataset...")
    test_loader = data_loader(cfg)
    
    # ckpt 경로의 상위 폴더를 기준으로 results 저장
    visualize_samples(os.path.dirname(ckpt), test_loader, num_samples=5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args.ckpt)