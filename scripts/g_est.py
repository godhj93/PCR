import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from tqdm import tqdm
# VN layers
from utils.model import GravityEstimationModel, ProbabilisticGravityModel
from utils.data import RegistrationDataset
from utils.loss import GravityLoss
from utils.common import AverageMeter, count_parameters
from utils.pointnet import PointNetGravityBaseline
from termcolor import colored
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation
from iterative_closet_point.bunny import run_icp, apply_RT, build_kdtree, calculatenormal

NUM_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 10
DATA = 'modelnet40'
# DATA = 'bunny'

def load_dataset():
    
    train_ds = RegistrationDataset(
        dataset_name=DATA,
        file_path='data/bunny/reconstruction/bun_zipper.ply',
        num_points=NUM_POINTS,
        partition='train',
        gaussian_noise=True,
        unseen=False,
        factor=1,
    )
    
    test_ds = RegistrationDataset(
        dataset_name=DATA,
        file_path='data/bunny/reconstruction/bun_zipper.ply',
        num_points=NUM_POINTS,
        partition='test',
        gaussian_noise=True,
        unseen=True,
        factor=1.0,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    
    return train_loader, test_loader
    
def train(VN_model, classical_model, train_loader, test_loader, criterion, optimizer, optimizer_classical, device):

    AvgMeter = AverageMeter()
    AvgMeter_classical = AverageMeter()
    
    for epoch in range(EPOCHS):
        
        pbar = tqdm(enumerate(train_loader), ncols=0, total=len(train_loader), desc=f"Epoch [{epoch+1} / {EPOCHS}]")
        
        VN_model.train()
        classical_model.train()
        
        AvgMeter.reset()
        AvgMeter_classical.reset()
        
        for idx, batch in pbar:
            
            q = batch['q'].to(device)
            gravity_q = batch['gravity_q'].to(device)
            
            g_pred = VN_model(q)  # (B, 3)
            g_pred_classical = classical_model(q)  # (B, 3)
            
            loss = criterion(g_pred, gravity_q)
            loss_classical = criterion(g_pred_classical, gravity_q)
            optimizer.zero_grad()
            optimizer_classical.zero_grad()
            loss.backward()
            loss_classical.backward()
            optimizer.step()
            optimizer_classical.step()
            
            AvgMeter.update(loss.item(), n=q.size(0))
            AvgMeter_classical.update(loss_classical.item(), n=q.size(0))
                        
            pbar.set_description(f"Epoch [{epoch+1} / {EPOCHS}] Loss: {AvgMeter.avg:.4f} | Classical Loss: {AvgMeter_classical.avg:.4f}")
            
        # Evaluation
        VN_model.eval()
        classical_model.eval()
        
        AvgMeter.reset()
        AvgMeter_classical.reset()
        
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), ncols=0, total=len(test_loader), desc=f"Eval [{epoch+1} / {EPOCHS}]")
            for idx, batch in pbar:
                
                q = batch['q'].to(device)
                gravity_q = batch['gravity_q'].to(device)
                g_pred = VN_model(q)  # (B, 3)
                g_pred_classical = classical_model(q)  # (B, 3)
                
                loss = criterion(g_pred, gravity_q)
                loss_classical = criterion(g_pred_classical, gravity_q)
                
                AvgMeter.update(loss.item(), n=q.size(0))
                AvgMeter_classical.update(loss_classical.item(), n=q.size(0))
                            
                pbar.set_description(colored(f"Eval [{epoch+1} / {EPOCHS}] Loss: {AvgMeter.avg:.4f} | Classical Loss: {AvgMeter_classical.avg:.4f}", "cyan"))
                
        # Save the best model based on test loss
        os.makedirs(f"checkpoints/{DATA}", exist_ok=True)
        if epoch == 0:
            best_loss = AvgMeter.avg
            torch.save(VN_model.state_dict(), f"checkpoints/{DATA}/best_vn_model.pth")
            torch.save(classical_model.state_dict(), f"checkpoints/{DATA}/best_classical_model.pth")
        else:
            if AvgMeter.avg < best_loss:
                best_loss = AvgMeter.avg
                torch.save(VN_model.state_dict(), f"checkpoints/{DATA}/best_vn_model.pth")
            if AvgMeter_classical.avg < best_loss:
                best_loss = AvgMeter_classical.avg
                torch.save(classical_model.state_dict(), f"checkpoints/{DATA}/best_classical_model.pth")
      
def align_points_by_gravity(points, predicted_gravity, target_gravity=np.array([0.0, -1.0, 0.0])):
    """
    점군을 회전시켜 예측된 중력 방향이 타겟 중력 방향(예: [0, -1, 0])과 일치하도록 만듭니다.
    """
    # 1. 벡터 정규화
    pred_g = predicted_gravity / (np.linalg.norm(predicted_gravity) + 1e-8)
    target_g = target_gravity / (np.linalg.norm(target_gravity) + 1e-8)
    
    # 2. 회전 축(Cross Product)과 각도(Dot Product) 계산
    axis = np.cross(pred_g, target_g)
    axis_norm = np.linalg.norm(axis)
    
    # 3. 예외 처리: 이미 정렬되어 있거나(0도), 정반대인 경우(180도)
    if axis_norm < 1e-6:
        dot_prod = np.dot(pred_g, target_g)
        if dot_prod > 0:
            return points, np.eye(3)
        else:
            axis = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(pred_g, axis)) > 0.99: 
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(pred_g, target_g), -1.0, 1.0))

    # 4. 로드리게스 공식
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # 5. 점군 회전 적용
    aligned_points = (R_align @ points.T).T
    
    return aligned_points, R_align
    
def compute_resolution(pcd_numpy):
    """
    점군의 평균 해상도(점들 사이의 평균 거리)를 계산합니다.
    """
    if pcd_numpy.shape[0] == 0:
        return 1.0
    n_samples = min(len(pcd_numpy), 1000)
    indices = np.random.choice(len(pcd_numpy), n_samples, replace=False)
    sample_pts = pcd_numpy[indices]
    tree = KDTree(sample_pts)
    dists, _ = tree.query(sample_pts, k=2) 
    mean_dist = np.mean(dists[:, 1])
    return mean_dist

def evaluate_correspondence(source_pts, target_pts, R_gt, t_gt, dist_thresh=0.1):
    """
    Ground Truth 변환을 알고 있을 때, k-d tree가 찾은 매칭의 정확도(Accuracy)를 평가
    """
    # 1. Transform Source to GT Target Frame
    source_gt = (source_pts @ R_gt.T) + t_gt.reshape(1, 3)
    
    # 2. Build Tree for Target
    target_tree = build_kdtree(target_pts)
    
    # 3. Query (1-NN)
    # 현재 위치에서의 매칭 찾기
    dists, indices = target_tree.query(source_pts, k=1)
    
    # 4. Check Correctness
    # "올바른 매칭" 정의: GT 위치 근처에 있는 점을 찾았는가?
    # 하지만 여기서는 "k-d tree가 찾은 점이 GT 변환 후의 점과 얼마나 가까운가"를 봅니다.
    # 즉, Alignment가 안 되어 있으면 dists가 큼 -> 매칭 실패로 간주
    
    # GT Correspondence Distance (Ideal)
    # 실제 GT 짝꿍과의 거리 (여기서는 Point-to-Point 대응이 1:1이라고 가정하지 않고,
    # 가장 가까운 점을 정답으로 봅니다.)
    gt_dists, _ = target_tree.query(source_gt, k=1)
    
    # Metric: Inlier Ratio (Threshold 내에 들어온 점의 비율)
    inlier_mask = dists < dist_thresh
    inlier_ratio = np.mean(inlier_mask)
    mean_dist = np.mean(dists)
    
    return inlier_ratio, mean_dist

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------------------------------
    # 1. 모델 & 데이터 로드 (기존 동일)
    # -------------------------------------------------------------------------
    train_loader, test_loader = load_dataset()
    
    VN_model = GravityEstimationModel(pooling='max', normal_channel=False).to(device)
    classical_model = PointNetGravityBaseline(channel=3).to(device)
    BVN_model = ProbabilisticGravityModel(pooling='max', normal_channel=False).to(device)
    print(colored(f"[Info] VN Model Parameters: {count_parameters(VN_model)}", "green"))
    print(colored(f"[Info] Classical Model Parameters: {count_parameters(classical_model)}", "green"))
    print(colored(f"[Info] BVN Model Parameters: {count_parameters(BVN_model)}", "green"))
        
    if os.path.exists(f"checkpoints/{DATA}/best_vn_model.pth") and os.path.exists(f"checkpoints/{DATA}/best_classical_model.pth") and os.path.exists(f"checkpoints/{DATA}/best_bvn_model.pth"):
        VN_model.load_state_dict(torch.load(f"checkpoints/{DATA}/best_vn_model.pth"))
        classical_model.load_state_dict(torch.load(f"checkpoints/{DATA}/best_classical_model.pth"))
        BVN_model.load_state_dict(torch.load(f"checkpoints/{DATA}/best_bvn_model.pth"))
        print(colored("[Init] Pretrained models loaded successfully.", "green"))
    else:
        print(colored("[Warning] No checkpoints found! Running with random weights.", "red"))
        train(VN_model, classical_model, train_loader, test_loader, GravityLoss(), torch.optim.Adam(VN_model.parameters(), lr=0.001), torch.optim.Adam(classical_model.parameters(), lr=0.001), device)
        
    print(colored("\n" + "="*120, "yellow"))
    print(colored(" Experiment: Classical vs Gravity-Aligned ICP (Full Error Analysis with Inlier Ratio)", "yellow"))
    print(colored("="*120, "yellow"))

    # -------------------------------------------------------------------------
    # 2. 데이터 준비 & GT 추출 (기존 동일)
    # -------------------------------------------------------------------------
    ICP_Baseline_SR = 0.0
    ICP_PointNet_SR = 0.0
    ICP_VN_SR = 0.0
    
    batch = next(iter(test_loader))

    for idx in range(batch['p'].shape[0]):
        points_P = batch['p'][idx].numpy().T  
        points_Q = batch['q'][idx].numpy().T  
        gt_gravity_q = batch['gravity_q'][idx].numpy() 
        
        R_gt = batch['R_pq'][idx].numpy()  
        t_gt = batch['t_pq'][idx].numpy()  
        
        r = Rotation.from_matrix(R_gt)
        rpy_gt = r.as_euler('xyz', degrees=True)

        print(colored("\n[Ground Truth Transformation (Target)]", "cyan"))
        print(f" - RPY (deg)   : R={rpy_gt[0]:.2f}, P={rpy_gt[1]:.2f}, Y={rpy_gt[2]:.2f}")
        print(f" - Translation : x={t_gt[0]:.2f}, y={t_gt[1]:.2f}, z={t_gt[2]:.2f}")
        print("-" * 120)

        print(colored(f"[Preprocessing] Computing Normals (k=20)...", "cyan"))
        normals_P = calculatenormal(points_P, k=20)
        normals_Q = calculatenormal(points_Q, k=20)
        
        # -------------------------------------------------------------------------
        # 3. 중력 예측 및 오차 계산 (기존 동일)
        # -------------------------------------------------------------------------
        VN_model.eval()
        classical_model.eval()
        
        with torch.no_grad():
            input_tensor = batch['q'][idx:idx+1].to(device)
            pred_g_vn = VN_model(input_tensor)[0].cpu().numpy().flatten()
            pred_g_cls = classical_model(input_tensor)[0].cpu().numpy().flatten()
        
        def calc_gravity_error(v_pred, v_true):
            v_p = v_pred / np.linalg.norm(v_pred)
            v_t = v_true / np.linalg.norm(v_true)
            dot = np.clip(np.dot(v_p, v_t), -1.0, 1.0)
            return np.degrees(np.arccos(dot))

        g_err_vn = calc_gravity_error(pred_g_vn, gt_gravity_q)
        g_err_cls = calc_gravity_error(pred_g_cls, gt_gravity_q)

        # --- Alignment Setup ---
        pts_Q_raw, R_align_raw = points_Q, np.eye(3)
        norm_Q_raw = normals_Q
        
        pts_Q_pnet, R_align_pnet = align_points_by_gravity(points_Q, pred_g_cls)
        norm_Q_pnet = (R_align_pnet @ normals_Q.T).T 
        
        pts_Q_vn, R_align_vn = align_points_by_gravity(points_Q, pred_g_vn)
        norm_Q_vn = (R_align_vn @ normals_Q.T).T 
        
        pts_P_base = points_P
        norm_P_base = normals_P

        # -------------------------------------------------------------------------
        # 4. ICP Loop & Error Analysis (수정됨)
        # -------------------------------------------------------------------------
        resolution = compute_resolution(points_P)
        dist_thresh = 5.0 * resolution
        ir_thresh = 2.0 * resolution  # ★ Inlier 판단 기준 (해상도의 2배, 약 2~4cm)
        
        def calc_rot_error(R_pred, R_true):
            R_diff = R_pred @ R_true.T
            trace = np.trace(R_diff)
            cos_theta = (trace - 1) / 2.0
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        def calc_trans_error(t_pred, t_true):
            return np.linalg.norm(t_pred - t_true)

        # ★ 테이블 헤더에 IR(%) 추가
        print(f"\n{'Method':<12} | {'g_err (deg)':<12} | {'R_err (deg)':<12} | {'t_err (m)':<12} | {'RMSE':<12} | {'IR (%)':<10} | {'Status':<10}")
        print("=" * 120)

        icp_method = "p2p"
        
        modes = [
            ("Raw (Base)", pts_Q_raw, norm_Q_raw, R_align_raw, None),
            ("PointNet",   pts_Q_pnet, norm_Q_pnet, R_align_pnet, g_err_cls),
            ("VN (Ours)",  pts_Q_vn,   norm_Q_vn,   R_align_vn,   g_err_vn)
        ]
        
        for mode_name, q_pts, q_norm, R_align_q, g_err_val in modes:
            
            # 1. 초기 위치(t_init) 계산
            centroid_P = np.mean(pts_P_base, axis=0)
            centroid_Q = np.mean(q_pts, axis=0)
            t_init_vec = centroid_Q - centroid_P
            
            # 2. Yaw 초기화
            # (필요시 estimate_initial_yaw 함수 호출 코드 추가)
            
            # 3. ICP 수행
            R_icp, t_icp, _ = run_icp(
                pts_P_base, q_pts, 
                method=icp_method, 
                normals_P=norm_P_base, 
                normals_Q=q_norm,
                max_iter=60, 
                tol=1e-6, 
                dist_thresh=dist_thresh,
                R_init=np.eye(3),
                t_init=t_init_vec.reshape(3,1), # 중심 정렬만 수행
                verbose=False
            )
            
            # 4. 최종 변환 행렬 복원
            R_final = R_align_q.T @ R_icp
            t_final = (R_align_q.T @ t_icp).flatten()
            
            # 5. 오차 계산
            r_err = calc_rot_error(R_final, R_gt)
            t_err = calc_trans_error(t_final, t_gt)
            
            # 6. RMSE & Inlier Ratio 계산 (Nearest Neighbor 방식)
            P_final = (R_final @ pts_P_base.T).T + t_final
            tree_Q = KDTree(points_Q)
            dists, _ = tree_Q.query(P_final, k=1) # 각 점에서 가장 가까운 Q점까지의 거리
            
            rmse = np.sqrt(np.mean(dists**2))
            
            # ★ Inlier Ratio 계산 (Threshold 이내인 점의 비율)
            inlier_ratio = np.mean(dists < ir_thresh) * 100.0
            
            # 상태 표시
            status = "FAIL"
            color = "red"
            if r_err < 5.0 and t_err < 0.1:
                status = "SUCCESS"
                color = "green"
            
            if g_err_val is None:
                g_str = "N/A"
            else:
                g_str = f"{g_err_val:.4f}"
                
            print(f"{mode_name:<12} | {g_str:<12} | {r_err:8.4f}°   | {t_err:8.4f} m   | {rmse:8.4f}   | {inlier_ratio:8.2f} % | {colored(status, color)}")

            if mode_name == "Raw (Base)":
                ICP_Baseline_SR += 1.0 if status == "SUCCESS" else 0.0
            elif mode_name == "PointNet":
                ICP_PointNet_SR += 1.0 if status == "SUCCESS" else 0.0
            elif mode_name == "VN (Ours)":
                ICP_VN_SR += 1.0 if status == "SUCCESS" else 0.0
                
        print("=" * 120)
        
    total_samples = batch['p'].shape[0]
    print(colored("\n" + "="*120, "yellow"))
    print(colored(" Summary of Success Rates:", "yellow"))
    print(colored(f" - ICP Baseline Success Rate   : {(ICP_Baseline_SR / total_samples) * 100.0:.2f} %", "yellow"))
    print(colored(f" - ICP + PointNet Success Rate  : {(ICP_PointNet_SR / total_samples) * 100.0:.2f} %", "yellow"))
    print(colored(f" - ICP + VN (Ours) Success Rate : {(ICP_VN_SR / total_samples) * 100.0:.2f} %", "yellow"))
    print(colored("="*120 + "\n", "yellow"))