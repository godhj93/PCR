import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from utils.se3 import SE3
from utils.inference import SE3VectorField
from flow_matching.solver import RiemannianODESolver

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
 
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Linear Warmup 후 Cosine Annealing을 수행하는 스케줄러 생성
    """
    def lr_lambda(current_step):
        # 1. Warmup 구간
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. Cosine Decay 구간
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_one_epoch(model, data_loader, optimizer, scheduler, loss_fn, epoch, metric, cfg):
    
    model.train() 
    pbar = tqdm(data_loader['train'], ncols=0, leave = False)
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
        v_pred = model(p_t, q, t)
        
        loss, log_dict = loss_fn(
            v_pred = v_pred,
            v_target = v_target
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        AvgMeter_train.update(loss.item(), q.size(0))
        
        # Progress bar with detailed loss breakdown
        desc = (f"Epoch [{epoch}/{cfg.training.epochs}] "
                f"Loss: {AvgMeter_train.avg:.4f} "
                f"Angular: {log_dict['angular']:.4f}, "
                f"Linear: {log_dict['linear']:.4f}) ")
        pbar.set_description(desc)
        
        scheduler.step()
        
    # Validation
    if epoch % 1 == 0 or epoch == cfg.training.epochs - 1:
        integrate = True
    else:
        integrate = False
        
    val_loss, val_log_dict = test_one_epoch(model, data_loader['test'], loss_fn, metric, cfg, epoch, visualize=False, integrate=integrate)
    
    return {'train_loss': AvgMeter_train.avg, 'train_loss_dict': log_dict, 'val_loss': val_loss, 'val_loss_dict': val_log_dict}
    
def test_one_epoch(model, test_loader, loss_fn, metric, cfg, epoch=0, visualize=False, integrate=True):
    
    model.eval()
    AvgMeter_val = metric['val']
    
    # Import for integration and animation
    from utils.inference import SE3VectorField, FlowAnimator
    
    # Integration을 위한 매니폴드 객체 (한 번만 생성)
    se3_manifold = SE3().to(cfg.device)
    
    # Store first batch for later integration
    first_batch = None
    
    pbar = tqdm(test_loader, ncols=0, leave = False)
    
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
        for batch_idx, batch in enumerate(pbar):
            
            # Store first batch for later integration
            if batch_idx == 0:
                first_batch = batch
            
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
            v_pred = model(p_t, q, t)
            
            loss, log_dict = loss_fn(
                v_pred, v_target
            )
            
            # Update Average Loss
            AvgMeter_val.update(loss.item(), q.size(0))
            last_log_dict = log_dict # 나중에 반환할 기본 log 정보 업데이트
            
            # -----------------------------------------------------------
            # Logging (Pbar Description)
            # -----------------------------------------------------------
            desc = (f"Val Epoch [{epoch}/{cfg.training.epochs}] "
                    f"Loss: {AvgMeter_val.avg:.4f} "
                    f"(Angular: {log_dict['angular']:.4f}, "
                    f"Linear: {log_dict['linear']:.4f})")
            pbar.set_description(desc)
            
    # -----------------------------------------------------------
    # 3. Integration on First Sample Only (After all batches)
    # -----------------------------------------------------------
    final_log_dict = last_log_dict.copy()
    final_log_dict['val_loss'] = AvgMeter_val.avg
    
    if integrate and first_batch is not None:
        # Take only first sample from first batch
        P_init = first_batch['p'].to(cfg.device)[:1]  # (1, 3, N)
        q = first_batch['q'].to(cfg.device)[:1]       # (1, 3, N)
        R_gt = first_batch['R_pq'].to(cfg.device)[:1]
        t_gt = first_batch['t_pq'].to(cfg.device)[:1]
        
        if P_init.shape[1] != 3: P_init = P_init.transpose(1, 2)
        if q.shape[1] != 3: q = q.transpose(1, 2)
        
        # GT Pose
        T_gt = torch.eye(4, device=cfg.device).unsqueeze(0)
        T_gt[:, :3, :3] = R_gt
        T_gt[:, :3, 3] = t_gt
        
        # Setup solver
        vf = SE3VectorField(model, P_init, q)
        solver = RiemannianODESolver(
            manifold=se3_manifold,
            velocity_model=vf
        )
        
        # Integration
        T_init = torch.eye(4, device=cfg.device).unsqueeze(0)
        time_grid = torch.tensor([0.0, 1.0], device=cfg.device)
        
        T_pred = solver.sample(
            x_init=T_init,
            step_size=0.05,
            method="midpoint",
            time_grid=time_grid,
            return_intermediates=False,
            projx=True,
            proju=True
        )
        
        # Compute errors
        r_err, t_err = compute_errors(T_pred, T_gt)
        
        final_log_dict['R_error'] = r_err.item()
        final_log_dict['T_error'] = t_err.item()
        
        print(f"\n=== Validation Result (Epoch {epoch}) ===")
        print(f"  R_Error: {final_log_dict['R_error']:.4f}°")
        print(f"  T_Error: {final_log_dict['T_error']:.4f}")

    # -----------------------------------------------------------
    # 4. Generate Animation for First Sample (if integrate)
    # -----------------------------------------------------------
    if integrate and epoch % cfg.test.vis_every_epoch == 0 and first_batch is not None:
        try:
            from pathlib import Path
            import os
            
            # Use already stored first batch
            p_init = first_batch['p'].to(cfg.device)[:1]
            q_tgt = first_batch['q'].to(cfg.device)[:1]
            R_gt = first_batch['R_pq'].to(cfg.device)[:1]
            t_gt = first_batch['t_pq'].to(cfg.device)[:1]
            
            # GT Matrix
            T_gt = torch.eye(4, device=cfg.device).unsqueeze(0)
            T_gt[:, :3, :3] = R_gt
            T_gt[:, :3, 3] = t_gt
            
            # Setup solver with trajectory
            vf = SE3VectorField(model, p_init, q_tgt)
            solver = RiemannianODESolver(manifold=se3_manifold, velocity_model=vf)
            
            # Generate trajectory
            T_init = torch.eye(4, device=cfg.device).unsqueeze(0)
            time_grid = torch.linspace(0.0, 1.0, 20, device=cfg.device)
            
            trajectory = solver.sample(
                x_init=T_init,
                step_size=0.1,
                method="rk4",
                time_grid=time_grid,
                return_intermediates=True,
                projx=True,
                proju=True
            )  # (T, 1, 4, 4)
            
            # Create animator and save
            animator = FlowAnimator(
                p_src=p_init[0],
                q_tgt=q_tgt[0],
                trajectory=trajectory[:, 0],  # (T, 4, 4)
                T_gt=T_gt[0].cpu()
            )
            
            # Save path
            from hydra.core.hydra_config import HydraConfig
            output_dir = Path(HydraConfig.get().runtime.output_dir)
            anim_dir = output_dir / "animations"
            anim_dir.mkdir(exist_ok=True)
            
            save_path = anim_dir / f"epoch_{epoch:03d}.gif"
            animator.save_animation(str(save_path), fps=10)
            print(f"\n[Animation] Saved to {save_path}")
            
        except Exception as e:
            print(f"\n[Warning] Animation generation failed: {e}")
    
    # -----------------------------------------------------------
    # 5. Return
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
    train_loss_angular = result['train_loss_dict']['angular']
    train_loss_linear = result['train_loss_dict']['linear']
    # train_loss_g_pred = result['train_loss_dict']['loss_perception']
    # train_loss_g_p_pred = result['train_loss_dict']['loss_g_p']
    # train_loss_g_q_pred = result['train_loss_dict']['loss_g_q']
    
    val_loss = result['val_loss']
    val_loss_angular = result['val_loss_dict']['angular']
    val_loss_linear = result['val_loss_dict']['linear']
    # val_loss_g_pred = result['val_loss_dict']['loss_perception']
    # val_loss_g_p_pred = result['val_loss_dict']['loss_g_p']
    # val_loss_g_q_pred = result['val_loss_dict']['loss_g_q']
    
    writer.add_scalar("Loss/train/total", train_loss, epoch)
    writer.add_scalar("Loss/train/angular", train_loss_angular, epoch)
    writer.add_scalar("Loss/train/linear", train_loss_linear, epoch)
    # writer.add_scalar("Loss/train/g_pred", train_loss_g_pred, epoch)
    # writer.add_scalar("Loss/train/g_p_pred", train_loss_g_p_pred, epoch)
    # writer.add_scalar("Loss/train/g_q_pred", train_loss_g_q_pred, epoch)
    
    writer.add_scalar("Loss/val/total", val_loss, epoch)
    writer.add_scalar("Loss/val/angular", val_loss_angular, epoch)
    writer.add_scalar("Loss/val/linear", val_loss_linear, epoch)
    # writer.add_scalar("Loss/val/g_pred", val_loss_g_pred, epoch)
    # writer.add_scalar("Loss/val/g_p_pred", val_loss_g_p_pred, epoch)
    # writer.add_scalar("Loss/val/g_q_pred", val_loss_g_q_pred, epoch)
    
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
    
    