import torch
from tqdm import tqdm
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
        R_pq, t_pq, R_qp, t_qp = model(P, Q)
        loss, loss_dict = loss_fn(R_pq, t_pq, R_gt, t_gt, R_qp, t_qp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        AvgMeter_train.update(loss.item(), P.size(0))
        
        # Progress bar with detailed loss breakdown
        desc = (f"Epoch [{epoch}/{cfg.training.epochs}] "
                f"Loss Avg: {AvgMeter_train.avg:.4f} (Loss: {loss_dict['loss']:.4f} | Cycle: {loss_dict['cycle_loss']:.4f}) "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        pbar.set_description(desc)
    
    scheduler.step()
    
    train_loss = AvgMeter_train.avg    
    val_loss = test_one_epoch(model, data_loader['test'], loss_fn, metric, cfg, epoch)
    
    return {'train_loss': train_loss,
            'val_loss': val_loss}
    
def test_one_epoch(model, test_loader, loss_fn, metric, cfg, epoch=0):
    
    model.eval()
    AvgMeter_val = metric['val']
    
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
            R_pq, t_pq, R_qp, t_qp = model(P, Q)
            loss, loss_dict = loss_fn(R_pq, t_pq, R_gt, t_gt, R_qp, t_qp)
            
            # Update Average Loss
            AvgMeter_val.update(loss.item(), P.size(0))
            
            desc = (f"Val Epoch [{epoch}/{cfg.training.epochs}] "
                    f"Loss AVg: {AvgMeter_val.avg:.4f} (Loss: {loss_dict['loss']:.4f} | Cycle: {loss_dict['cycle_loss']:.4f})")
            pbar.set_description(desc)
            
    return AvgMeter_val.avg

def logging_tensorboard(writer, result, epoch, optimizer):
    
    train_loss = result['train_loss']
    val_loss = result['val_loss']
    
    writer.add_scalar("Loss/train/total", train_loss, epoch)
    writer.add_scalar("Loss/val/total", val_loss, epoch)
    
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    
    return train_loss, val_loss