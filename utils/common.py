import torch
from tqdm import tqdm
import numpy as np

def compute_rotation_error(R_pred, R_gt):
    """
    Compute Relative Rotation Error (RRE) in degrees.
    R_pred, R_gt: (B, 3, 3) rotation matrices
    Returns: (B,) array of rotation errors in degrees
    """
    # R_pred^T @ R_gt should be identity if perfect match
    R_diff = torch.bmm(R_pred.transpose(2, 1), R_gt)
    
    # trace(R) = 1 + 2*cos(theta) for rotation by angle theta
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    
    # Clamp to avoid numerical errors with arccos
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # Rotation angle in radians, then convert to degrees
    angle_rad = torch.acos(cos_theta)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg

def compute_translation_error(t_pred, t_gt):
    """
    Compute Relative Translation Error (RTE) as Euclidean distance.
    t_pred, t_gt: (B, 3) translation vectors
    Returns: (B,) array of translation errors
    """
    return torch.norm(t_pred - t_gt, dim=1)

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

def train_one_epoch(model, data_loader, optimizer, scheduler, loss_fn, epoch, metric, cfg):
    
    model.train() 
    pbar = tqdm(data_loader['train'], ncols=0, leave = False)
    AvgMeter_train = metric['train']
    
    for batch in pbar:

        # Data Load
        P = batch['P'].to(cfg.device)         
        Q = batch['Q'].to(cfg.device)         
        g_p = batch['g_p'].to(cfg.device) 
        g_q = batch['g_q'].to(cfg.device) 
        R_gt = batch['R_gt'].to(cfg.device)
        t_gt = batch['t_gt'].to(cfg.device)
        
        # Forward Pass
        R_pq, t_pq, R_qp, t_qp, gravity = model(P, Q)
        loss, loss_dict = loss_fn((R_pq, t_pq, R_gt, t_gt, R_qp, t_qp), (g_p, g_q), gravity)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        AvgMeter_train.update(loss.item(), P.size(0))
        
        # Progress bar with detailed loss breakdown
        desc = (f"Epoch [{epoch}/{cfg.training.epochs}] "
                f"Loss Avg: {AvgMeter_train.avg:.4f} (Loss: {loss_dict['dcp']['loss']:.4f} | Cycle: {loss_dict['dcp']['cycle_loss']:.4f} | VMF: {loss_dict['vmf_loss']:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        pbar.set_description(desc)
    
    scheduler.step()
    
    train_loss = AvgMeter_train.avg    
    metrics = test_one_epoch(model, data_loader['test'], loss_fn, metric, cfg, epoch)
    
    # add train loss to metrics
    metrics['train_loss'] = train_loss
    
    return metrics

def test_one_epoch(model, test_loader, loss_fn, metric, cfg, epoch=0):
    
    model.eval()
    AvgMeter_val = metric['val']
    
    # Additional metrics for RRE and RTE
    rre_list = []
    rte_list = []
    
    pbar = tqdm(test_loader, ncols=0, leave = False)
    
    with torch.no_grad():
        for batch in pbar:
            
            P = batch['P'].to(cfg.device)         
            Q = batch['Q'].to(cfg.device)           
            g_p = batch['g_p'].to(cfg.device) 
            g_q = batch['g_q'].to(cfg.device) 
            R_gt = batch['R_gt'].to(cfg.device)   
            t_gt = batch['t_gt'].to(cfg.device)   
            
            # Forward Pass
            R_pq, t_pq, R_qp, t_qp, gravity = model(P, Q)
            loss, loss_dict = loss_fn((R_pq, t_pq, R_gt, t_gt, R_qp, t_qp), (g_p, g_q), gravity)
            
            # Compute RRE and RTE
            rre = compute_rotation_error(R_pq, R_gt)  # (B,) degrees
            rte = compute_translation_error(t_pq, t_gt)  # (B,)
            
            rre_list.append(rre.cpu())
            rte_list.append(rte.cpu())
            
            # Update Average Loss
            AvgMeter_val.update(loss.item(), P.size(0))
            
            # Compute average metrics so far
            rre_avg = torch.cat(rre_list).mean().item()
            rte_avg = torch.cat(rte_list).mean().item()
            
            desc = (f"Val Epoch [{epoch}/{cfg.training.epochs}] "
                    f"Loss: {AvgMeter_val.avg:.4f} | RRE: {rre_avg:.2f}° | RTE: {rte_avg:.4f}")
            pbar.set_description(desc)
    
    # Calculate final metrics
    rre_all = torch.cat(rre_list)
    rte_all = torch.cat(rte_list)
    
    metrics = {
        'val_loss': AvgMeter_val.avg,
        'val_rre_mean': rre_all.mean().item(),
        'val_rte_mean': rte_all.mean().item(),
    }
            
    return metrics

def logging_tensorboard(writer, result, epoch, optimizer):
    
    
    writer.add_scalar("Loss/train/total", result['train_loss'], epoch)
    writer.add_scalar("Loss/val/total", result['val_loss'], epoch)
    
    writer.add_scalar("Metrics/val/RRE_mean", result['val_rre_mean'], epoch)
    writer.add_scalar("Metrics/val/RTE_mean", result['val_rte_mean'], epoch)
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    