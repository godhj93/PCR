import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------
# Loss Function: Cosine Similarity Loss
# -----------------------------------------------------------
class GravityLoss(nn.Module):
    def __init__(self):
        super(GravityLoss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        # pred: (B, 3) 예측된 중력
        # target: (B, 3) 실제 중력 (R * [0, -1, 0])
        
        # Cosine Similarity: 1이면 일치, -1이면 반대
        # Loss = 1 - CosineSimilarity (0에 가까울수록 좋음)
        cos_sim = F.cosine_similarity(pred, target, dim=1)
        loss = 1.0 - cos_sim.mean()
        
        return loss


class VMFLoss(nn.Module):
    """
    Von Mises-Fisher Loss for directional prediction
    Uses negative log-likelihood of vMF distribution
    """
    def __init__(self):
        super(VMFLoss, self).__init__()
    
    def forward(self, pred_mu, pred_kappa, gt_gravity):
        """
        pred_mu: (B, 3) - Predicted mean direction
        pred_kappa: (B, 1) - Predicted concentration
        gt_gravity: (B, 3) - Ground Truth gravity vector
        """
        # 1. 내적 (Cosine Similarity)
        # mu와 GT가 가까울수록 1, 멀수록 -1
        dot_prod = (pred_mu * gt_gravity).sum(dim=1, keepdim=True) # (B, 1)
        
        # 2. Loss 계산 (NLL)
        # L = - log( C_3(k) * exp(k * mu^T * g) )
        #   = - log(C_3(k)) - k * (mu^T * g)
        
        # log(C_3(k)) = log(k) - log(4pi) - log(sinh(k))
        # log(sinh(k)) 처리가 핵심 (Overflow 방지)
        
        k = pred_kappa
        
        # log(sinh(k)) 근사 계산
        # k < 10: 정확한 식 log(sinh(k))
        # k >= 10: 근사 식 k - log(2)
        
        log_sinh_k = torch.zeros_like(k)
        
        mask = k < 10.0
        if mask.any():
            # sinh(x)가 0 근처에서 매우 작으므로 수치 안정성을 위해 clamp
            log_sinh_k[mask] = torch.log(torch.sinh(k[mask]) + 1e-6)
            
        mask_large = ~mask
        if mask_large.any():
            log_sinh_k[mask_large] = k[mask_large] - 0.693147 # log(2)
            
        # log(C_3(k)) 부분 (상수항 4pi는 제외 가능하지만 명시함)
        log_c3 = torch.log(k) - 1.0 * log_sinh_k # -log(4pi)는 생략 (상수)
        
        # Final NLL Loss
        # 식: - (k * dot_prod) - log_c3
        # 부호 정리하면: - k * dot - log(k) + log(sinh(k))
        loss = - (k * dot_prod) - log_c3
        
        return loss.mean()