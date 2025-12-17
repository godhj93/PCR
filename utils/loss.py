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