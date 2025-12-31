import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

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
    val_loss, val_log_dict = test_one_epoch(model, data_loader['test'], loss_fn, metric, cfg, epoch)
    
    return {'train_loss': AvgMeter_train.avg, 'train_loss_dict': log_dict, 'val_loss': val_loss, 'val_loss_dict': val_log_dict}
    
def test_one_epoch(model, test_loader, loss_fn, metric, cfg, epoch=0, visualize=False):
    
    model.eval()
    AvgMeter_val = metric['val']
    
    pbar = tqdm(test_loader, ncols=0)
    
    if visualize:
        results = {
            'p': [], 'q': [],
            'gravity_p': [], 'gravity_q': [],
            'mu_p': [], 'kappa_p': [],
            'mu_q': [], 'kappa_q': []
        }
    
    with torch.no_grad():
        for batch in pbar:
            
            # Data Load
            p_t = batch['P_t'].to(cfg.device)         # (B, 3, N)
            q = batch['q'].to(cfg.device)         # (B, 3, N)
            t = batch['t'].to(cfg.device)         # (B, 1)
            v_target = batch['v_target'].to(cfg.device) # (B, 6)
            g_p = batch['g_p_t'].to(cfg.device) # (B, 3)
            g_q = batch['g_q'].to(cfg.device) # (B, 3)
            
            # Forward Pass
            v_pred, (mu_p, kappa_p), (mu_q, kappa_q) = model(p_t, t,q)
            
            loss, log_dict = loss_fn(
                v_pred = v_pred,
                v_target = v_target,
                variational_params = {'mu_p': mu_p, 'kappa_p': kappa_p, 'mu_q': mu_q, 'kappa_q': kappa_q},
                gt_physics = {'g_p': g_p, 'g_q': g_q}
            )
            
            AvgMeter_val.update(loss.item(), q.size(0))
            
            pbar.set_description(f"Epoch [{epoch} / {cfg.training.epochs}] Val Loss: {AvgMeter_val.avg:.4f}, kappa: {(kappa_p.mean() + kappa_q.mean()).item() / 2:.4f}")
            
            if visualize:
                results['p'].append(p_t.cpu())
                results['q'].append(q.cpu())
                results['gravity_p'].append(g_p.cpu())
                results['gravity_q'].append(g_q.cpu())
                results['mu_p'].append(mu_p.cpu())
                results['kappa_p'].append(kappa_p.cpu())
                results['mu_q'].append(mu_q.cpu())
                results['kappa_q'].append(kappa_q.cpu())
    
    if visualize:
        for key in results:
            results[key] = torch.cat(results[key], dim=0)
        return AvgMeter_val.avg, results
    
    return AvgMeter_val.avg, log_dict

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
    
    