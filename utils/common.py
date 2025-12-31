import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from utils.se3 import SE3
from utils.inference import RiemannianEulerSolver, SE3VectorField

def AverageMeter():
    """Computes and stores the average and current value"""
    class Meter:
        def __init__(self):
            self.reset()
        
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
            
    return Meter()

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_angular_error(v1, v2):
    """
    두 벡터(배치) 사이의 각도 오차(Degree) 계산
    Input: (B, 3) or (3,)
    Output: (B,) or scalar
    """
    # 1. Convert to numpy if tensor
    if isinstance(v1, torch.Tensor):
        v1 = v1.detach().cpu().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.detach().cpu().numpy()
    
    # 2. Normalize (Batch 처리를 위해 axis, keepdims 필수)
    # 1차원(단일 벡터)일 경우와 2차원(배치)일 경우를 구분하거나 axis를 안전하게 처리
    if v1.ndim == 1:
        norm1 = np.linalg.norm(v1) + 1e-8
        norm2 = np.linalg.norm(v2) + 1e-8
    else:
        norm1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8
        
    v1 = v1 / norm1
    v2 = v2 / norm2
    
    # 3. Dot product (Row-wise)
    # np.dot 대신 element-wise 곱 후 합을 구해야 함
    if v1.ndim == 1:
        dot_product = np.dot(v1, v2)
    else:
        dot_product = np.sum(v1 * v2, axis=1) # (B, 3) * (B, 3) -> (B, 3) sum -> (B,)

    # 4. Clip & Arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return np.degrees(angle)

def get_rotation_from_vectors(u, v):
    """
    벡터 u를 벡터 v로 정렬시키는 회전 행렬 R을 계산 (Rodrigues' rotation formula)
    """
    u = u / (np.linalg.norm(u) + 1e-8)
    v = v / (np.linalg.norm(v) + 1e-8)
    
    cross = np.cross(u, v)
    dot = np.dot(u, v)
    
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0: return np.eye(3)
        else: return -np.eye(3)

    S = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    
    R = np.eye(3) + S + S @ S * (1 / (1 + dot))
    return R
 
def train_one_epoch(model, data_loader, optimizer, loss_fn, epoch, metric, cfg):
    
    model.train() 
    pbar = tqdm(data_loader['train'], ncols=0)
    AvgMeter_train = metric['train']
    
    for batch in pbar:
        
        # Data Load
        p_t = batch['P_t'].to(cfg.device)         # (B, 3, N)
        q = batch['q'].to(cfg.device)           # (B, 3, N)
        t = batch['t'].to(cfg.device)           # (B, 1)
        v_target = batch['v_target'].to(cfg.device) # (B, 6)
        g_p = batch['g_p_t'].to(cfg.device) # (B, 3)
        g_q = batch['g_q'].to(cfg.device) # (B, 3
        
        # Forward Pass
        v_pred, (mu_p, kappa_p), (mu_q, kappa_q) = model(p_t, t, q)
        
        loss, log_dict = loss_fn(
            v_pred = v_pred,
            v_target = v_target,
            variational_params = {'mu_p': mu_p, 'kappa_p': kappa_p, 'mu_q': mu_q, 'kappa_q': kappa_q},
            gt_physics = {'g_p': g_p, 'g_q': g_q}
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        AvgMeter_train.update(loss.item(), q.size(0))
        
        pbar.set_description(f"Epoch [{epoch} / {cfg.training.epochs}] Train Loss: {AvgMeter_train.avg:.4f}, kappa: {(kappa_p.mean() + kappa_q.mean()).item() / 2:.4f}")
        
    # Validation
    if epoch % 10 == 0 or epoch == cfg.training.epochs - 1:
        integrate = True
    else:
        integrate = False
        
    val_loss, val_log_dict = test_one_epoch(model, data_loader['test'], loss_fn, metric, cfg, epoch, visualize=False, integrate=integrate)
    
    return {'train_loss': AvgMeter_train.avg, 'train_loss_dict': log_dict, 'val_loss': val_loss, 'val_loss_dict': val_log_dict}
    
def test_one_epoch(model, test_loader, loss_fn, metric, cfg, epoch=0, visualize=False, integrate=True):
    
    model.eval()
    AvgMeter_val = metric['val']
    
    # Integration을 위한 매니폴드 객체 (한 번만 생성)
    se3_manifold = SE3().to(cfg.device)
    
    # 에러 통계를 내기 위한 리스트
    rot_errors_list = []
    trans_errors_list = []
    
    pbar = tqdm(test_loader, ncols=0)
    
    # Visualization 결과 저장소
    if visualize:
        results = {
            'p': [], 'q': [],
            'gravity_p': [], 'gravity_q': [],
            'mu_p': [], 'kappa_p': [],
            'mu_q': [], 'kappa_q': [],
            'T_pred': [] # 적분된 포즈도 저장
        }
    
    # 마지막 배치의 log_dict를 저장하기 위한 변수 (초기화)
    last_log_dict = {}

    with torch.no_grad():
        for batch in pbar:
            
            # -----------------------------------------------------------
            # 1. Data Load & Loss Calculation
            # -----------------------------------------------------------
            p_t = batch['P_t'].to(cfg.device)     # (B, 3, N) - Loss 계산용 (중간 경로)
            q = batch['q'].to(cfg.device)         # (B, 3, N)
            t = batch['t'].to(cfg.device)         # (B, 1)
            v_target = batch['v_target'].to(cfg.device) 
            g_p = batch['g_p_t'].to(cfg.device) 
            g_q = batch['g_q'].to(cfg.device) 
            
            # Forward Pass
            v_pred, (mu_p, kappa_p), (mu_q, kappa_q) = model(p_t, t, q)
            
            loss, log_dict = loss_fn(
                v_pred = v_pred,
                v_target = v_target,
                variational_params = {'mu_p': mu_p, 'kappa_p': kappa_p, 'mu_q': mu_q, 'kappa_q': kappa_q},
                gt_physics = {'g_p': g_p, 'g_q': g_q}
            )
            
            # Update Average Loss
            AvgMeter_val.update(loss.item(), q.size(0))
            last_log_dict = log_dict # 나중에 반환할 기본 log 정보 업데이트
            
            # -----------------------------------------------------------
            # 2. Integration Logic (Optional)
            # -----------------------------------------------------------
            current_r_err = 0.0
            current_t_err = 0.0
            
            if integrate:
                # A. 초기 데이터 (t=0) 가져오기
                P_init = batch['p'].to(cfg.device) # (B, 3, N) - Augmentation된 초기 소스
                if P_init.shape[1] != 3: P_init = P_init.transpose(1, 2)
                
                # B. GT Pose 구성
                R_gt = batch['R_pq'].to(cfg.device)
                t_gt = batch['t_pq'].to(cfg.device)
                T_gt = torch.eye(4, device=cfg.device).repeat(P_init.shape[0], 1, 1)
                T_gt[:, :3, :3] = R_gt
                T_gt[:, :3, 3] = t_gt
                
                # C. Solver 설정 (Vector Field)
                vf = SE3VectorField(model, P_init, q)
                
                # Euler Solver (Step size 0.1 -> 10 Steps)
                solver = RiemannianEulerSolver(
                    vector_field=vf,
                    manifold=se3_manifold,
                    step_size=0.1 
                )
                
                # D. 적분 수행 (Identity -> T_pred)
                T_init = torch.eye(4, device=cfg.device).repeat(P_init.shape[0], 1, 1)
                T_pred = solver.sample(T_init, t0=0.0, t1=1.0)
                
                # E. 에러 계산 및 저장
                r_err, t_err = compute_errors(T_pred, T_gt)
                rot_errors_list.append(r_err)
                trans_errors_list.append(t_err)
                
                current_r_err = r_err.mean().item()
                current_t_err = t_err.mean().item()

                if visualize:
                    results['T_pred'].append(T_pred.cpu())
            
            # -----------------------------------------------------------
            # Logging (Pbar Description)
            # -----------------------------------------------------------
            desc = f"Ep [{epoch}] Val Loss: {AvgMeter_val.avg:.4f}"
            if integrate:
                desc += f" | R: {current_r_err:.2f}°, T: {current_t_err:.4f}"
            pbar.set_description(desc)
            
            # Visualization Data Collection
            if visualize:
                results['p'].append(p_t.cpu()); results['q'].append(q.cpu())
                results['gravity_p'].append(g_p.cpu()); results['gravity_q'].append(g_q.cpu())
                results['mu_p'].append(mu_p.cpu()); results['kappa_p'].append(kappa_p.cpu())
                results['mu_q'].append(mu_q.cpu()); results['kappa_q'].append(kappa_q.cpu())
    
    # -----------------------------------------------------------
    # 3. Finalize & Merge Metrics into log_dict
    # -----------------------------------------------------------
    # 기존 log_dict 복사 (마지막 배치의 Loss 정보 등 포함)
    final_log_dict = last_log_dict.copy()
    
    # 통합된 Avg Loss 업데이트
    final_log_dict['val_loss'] = AvgMeter_val.avg
    
    # Integration 결과가 있으면 평균을 내서 log_dict에 추가
    if integrate and len(rot_errors_list) > 0:
        all_r = torch.cat(rot_errors_list)
        all_t = torch.cat(trans_errors_list)
        
        final_log_dict['R_error_mean'] = all_r.mean().item()
        final_log_dict['R_error_med'] = all_r.median().item()
        final_log_dict['T_error_mean'] = all_t.mean().item()
        final_log_dict['T_error_med'] = all_t.median().item()
        
        print(f"\n=== Validation Result (Epoch {epoch}) ===")
        print(f"  R_Error: {final_log_dict['R_error_mean']:.4f}° (Mean), {final_log_dict['R_error_med']:.4f}° (Med)")
        print(f"  T_Error: {final_log_dict['T_error_mean']:.4f} (Mean), {final_log_dict['T_error_med']:.4f} (Med)")

    # -----------------------------------------------------------
    # 4. Return
    # -----------------------------------------------------------
    if visualize:
        for key in results:
            if len(results[key]) > 0:
                results[key] = torch.cat(results[key], dim=0)
        # visualize=True일 때는 (Loss, Results) 반환 (기존 로직 유지)
        return AvgMeter_val.avg, results
    
    # 평소에는 (Loss, Log_dict) 반환
    return AvgMeter_val.avg, final_log_dict

def compute_errors(T_pred, T_gt):
    """
    Compute rotation (degree) and translation errors
    """
    # 1. Rotation Error
    R_gt = T_gt[:, :3, :3]
    R_pred = T_pred[:, :3, :3]
    
    R_diff = torch.matmul(R_gt.transpose(1, 2), R_pred)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    
    rot_err_rad = torch.acos(cos_theta)
    rot_err_deg = rot_err_rad * (180 / np.pi)
    
    # 2. Translation Error
    t_gt = T_gt[:, :3, 3]
    t_pred = T_pred[:, :3, 3]
    trans_err = torch.norm(t_gt - t_pred, dim=1)
    
    return rot_err_deg, trans_err
def logging_tensorboard(writer, result, epoch, optimizer):
    
    train_loss = result['train_loss']
    train_loss_v_pred = result['train_loss_dict']['loss_action']
    train_loss_g_pred = result['train_loss_dict']['loss_perception']
    train_loss_g_p_pred = result['train_loss_dict']['loss_g_p']
    train_loss_g_q_pred = result['train_loss_dict']['loss_g_q']
    
    val_loss = result['val_loss']
    val_loss_v_pred = result['val_loss_dict']['loss_action']
    val_loss_g_pred = result['val_loss_dict']['loss_perception']
    val_loss_g_p_pred = result['val_loss_dict']['loss_g_p']
    val_loss_g_q_pred = result['val_loss_dict']['loss_g_q']
    
    writer.add_scalar("Loss/train/total", train_loss, epoch)
    writer.add_scalar("Loss/train/v_pred", train_loss_v_pred, epoch)
    writer.add_scalar("Loss/train/g_pred", train_loss_g_pred, epoch)
    writer.add_scalar("Loss/train/g_p_pred", train_loss_g_p_pred, epoch)
    writer.add_scalar("Loss/train/g_q_pred", train_loss_g_q_pred, epoch)
    
    writer.add_scalar("Loss/val/total", val_loss, epoch)
    writer.add_scalar("Loss/val/v_pred", val_loss_v_pred, epoch)
    writer.add_scalar("Loss/val/g_pred", val_loss_g_pred, epoch)
    writer.add_scalar("Loss/val/g_p_pred", val_loss_g_p_pred, epoch)
    writer.add_scalar("Loss/val/g_q_pred", val_loss_g_q_pred, epoch)
    
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    
    return train_loss, val_loss
    
def visualize_registration(P, Q, R, t, vis, title="Registration Result"):
    """
    P: (3, N) source point cloud
    Q: (3, N) target point cloud
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    """
    # P를 변환하여 Q와 정합 (P_transformed = R * P + t)
    P_transformed = R @ P + t[:, None]

    fig = plt.figure(figsize=(12, 6))

    # 1. 정합 전 (Before Registration)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(P[0], P[1], P[2], c='blue', s=2, label='Source (P)')
    ax1.scatter(Q[0], Q[1], Q[2], c='red', s=2, label='Target (Q)')
    ax1.set_title("Before Registration")
    ax1.legend()

    # 2. 정합 후 (After Registration)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(P_transformed[0], P_transformed[1], P_transformed[2], c='blue', s=2, label='Transformed P')
    ax2.scatter(Q[0], Q[1], Q[2], c='red', s=2, label='Target (Q)')
    ax2.set_title("After Registration (GT)")
    ax2.legend()

    if vis:
        plt.show()
    
    