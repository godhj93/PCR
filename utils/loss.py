import torch
import torch.nn as nn
import torch.nn.functional as F

class VMFLoss(nn.Module):
    """
    Von Mises-Fisher Loss for directional prediction
    Minimizing this is equivalent to Minimizing KL Divergence between q(z) and p(z)
    """
    def __init__(self):
        super(VMFLoss, self).__init__()
    
    def forward(self, pred_mu, pred_kappa, gt_gravity):
        """
        pred_mu: (B, 3) - Predicted mean direction
        pred_kappa: (B, 1) - Predicted concentration
        gt_gravity: (B, 3) - Ground Truth gravity vector
        """
        # 1. Cosine Similarity (Alignment)
        dot_prod = (pred_mu * gt_gravity).sum(dim=1, keepdim=True) # (B, 1)
        
        # 2. Log Partition Function Approximation
        k = pred_kappa
        
        # log(sinh(k)) approximation for numerical stability
        log_sinh_k = torch.zeros_like(k)
        
        mask = k < 10.0
        if mask.any():
            log_sinh_k[mask] = torch.log(torch.sinh(k[mask]) + 1e-6)
            
        mask_large = ~mask
        if mask_large.any():
            log_sinh_k[mask_large] = k[mask_large] - 0.693147 # k - log(2)
            
        # log(C_3(k)) = log(k) - log(4pi) - log(sinh(k))
        # We ignore constants (-log(4pi)) as they don't affect gradients
        log_c3 = torch.log(k + 1e-6) - log_sinh_k
        
        # Negative Log Likelihood
        loss = - (k * dot_prod) - log_c3
        
        return loss.mean()

class DCPLoss(nn.Module):
    def __init__(self, beta = 0.1, cycle = True):
        super(DCPLoss, self).__init__()

        self.beta = beta
        self.cycle = cycle
        
    def forward(self, *inputs):
        """
        rotation_ab_pred: (B, 3, 3) - Predicted rotation from A to B
        translation_ab_pred: (B, 3, 1) - Predicted translation from A to B
        rotation_ab: (B, 3, 3) - Ground Truth rotation from A to B
        translation_ab: (B, 3, 1) - Ground Truth translation from A to B
        """
        
        rotation_ab_pred = inputs[0]
        translation_ab_pred = inputs[1]
        rotation_ab = inputs[2] # GT
        translation_ab = inputs[3] # GT
        rotation_ba_pred = inputs[4] 
        translation_ba_pred = inputs[5]
        
        batch_size = rotation_ab_pred.shape[0]
        identity = torch.eye(3, device=rotation_ab_pred.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        
        if self.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity)
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss
            loss = loss + self.beta * cycle_loss
            
        return loss, {'loss': loss.item(), 'cycle_loss': cycle_loss.item() if self.cycle else 0.0}