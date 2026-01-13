import torch
import torch.nn as nn

# ==============================================================================
# 1. Perception Loss: VMFLoss
# ==============================================================================
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

# ==============================================================================
# 2. Action Loss: FlowStepLoss
# ==============================================================================
class FlowStepLoss(nn.Module):
    """
    Measures the discrepancy between predicted velocity and target geodesic velocity.
    Equivalent to maximizing E[log p(v | z)] in ELBO.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, v_pred, v_target):
        """
        Args:
            v_pred: (B, 6) - Predicted velocity [v, w]
            v_target: (B, 6) - Target velocity from Geodesic Path
        """
        return self.mse(v_pred, v_target), {'angular': v_pred[:, 3:].norm(dim=1).mean().item(), 'linear': v_pred[:, :3].norm(dim=1).mean().item()}

# ==============================================================================
# 3. Integrated Objective: ELBOObjective
# ==============================================================================
class ELBOObjective(nn.Module):
    """
    Optimization Objective for Riemannian Flow Matching with Latent Gravity.
    
    Maximize ELBO <=> Minimize Loss
    Loss = Action_Loss + beta * Perception_Loss
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.perception_loss_fn = VMFLoss()
        self.action_loss_fn = FlowStepLoss()

    def forward(self, 
                v_pred: torch.Tensor, v_target: torch.Tensor, 
                variational_params: dict, 
                gt_physics: dict):
        """
        Args:
            v_pred: (B, 6)
            v_target: (B, 6)
            variational_params: {'mu_p', 'kappa_p', 'mu_q', 'kappa_q'}
            gt_physics: {'g_p', 'g_q'}
            
        Returns:
            total_loss, log_dict
        """
        # 1. Action Loss (Reconstruction)
        action_loss = self.action_loss_fn(v_pred, v_target)

        # 2. Perception Loss (Regularization / KL Divergence)
        loss_g_p = self.perception_loss_fn(
            variational_params['mu_p'], variational_params['kappa_p'], gt_physics['g_p']
        )
        loss_g_q = self.perception_loss_fn(
            variational_params['mu_q'], variational_params['kappa_q'], gt_physics['g_q']
        )
        
        perception_loss = (loss_g_p + loss_g_q) * 0.5

        # 3. Total Loss (Negative ELBO)
        total_loss = self.alpha * action_loss + self.beta * perception_loss
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "loss_action": action_loss.item(),
            "loss_perception": perception_loss.item(),
            "loss_g_p": loss_g_p.item(),
            "loss_g_q": loss_g_q.item()
        }