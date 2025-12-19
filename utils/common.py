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
    
def train_one_epoch(model, data_loader, optimizer, loss_fn, epoch, metric, cfg):
    
    model.train() 
    pbar = tqdm(data_loader['train'], desc=f"Epoch [{epoch}] Training", leave=True)
    
    AvgMeter_train = metric['train']
    
    for batch in pbar:
        
        # Data Load
        p = batch['p'].to(cfg.device)         # (B, 3, N)
        q = batch['q'].to(cfg.device)         # (B, 3, N)
        
        gravity_p = batch['gravity_p'].to(cfg.device) # (B, 3)
        gravity_q = batch['gravity_q'].to(cfg.device) # (B, 3)
        
        # Forward Pass
        mu_p, mu_q, kappa = model(p, q)
        
        loss_p = loss_fn(mu_p, kappa, gravity_p)
        loss_q = loss_fn(mu_q, kappa, gravity_q)
        loss = (loss_p + loss_q) / 2.0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        AvgMeter_train.update(loss.item(), q.size(0))
        
        pbar.set_description(f"Epoch [{epoch}] Train Loss: {AvgMeter_train.avg:.4f}, kappa: {kappa.mean().item():.4f}")
        
    # Validation
    val_loss = test_one_epoch(model, data_loader['test'], loss_fn, metric, cfg)
    
    return {'train_loss': AvgMeter_train.avg, 'val_loss': val_loss}
    
def test_one_epoch(model, test_loader, loss_fn, metric, cfg, epoch=0, visualize=False):
    
    model.eval()
    AvgMeter_val = metric['val']
    
    pbar = tqdm(test_loader, desc=f"Epoch [{epoch}] Validation", leave=True)
    
    with torch.no_grad():
        for batch in pbar:
            
            # Data Load
            p = batch['p'].to(cfg.device)         # (B, 3, N)
            q = batch['q'].to(cfg.device)         # (B, 3, N)
            
            gravity_p = batch['gravity_p'].to(cfg.device) # (B, 3)
            gravity_q = batch['gravity_q'].to(cfg.device) # (B, 3)
            
            # Forward Pass
            mu_p, mu_q, kappa = model(p, q)
            
            loss_p = loss_fn(mu_p, kappa, gravity_p)
            loss_q = loss_fn(mu_q, kappa, gravity_q)
            loss = (loss_p + loss_q) / 2.0
            
            AvgMeter_val.update(loss.item(), q.size(0))
            
            pbar.set_description(f"Epoch [{epoch}] Val Loss: {AvgMeter_val.avg:.4f}, kappa: {kappa.mean().item():.4f}")
            
    return AvgMeter_val.avg 

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
    
    