import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_vn import PointNetEncoder
from utils.dgcnn_vn import VN_DGCNN_Encoder
from utils.layers import VNLinearLeakyReLU, VNLinear, VNInvariant, VN_Attention, VN_Cross_Gating, VNMaxPool, NormalEstimator
import numpy as np

class PointNet_VN_Gravity(nn.Module):
    def __init__(self, pooling, normal_channel=True):
        super(PointNet_VN_Gravity, self).__init__()
        self.pooling = pooling
        channel = 6 if normal_channel else 3
        
        # PointNetEncoder는 보통 1024//3 = 341 채널을 반환한다고 가정
        self.feat = PointNetEncoder(self.pooling, global_feat=True, feature_transform=True, channel=channel)
        
        # (B, 341, 3) -> (B, 512, 3)
        self.vn_fc1 = VNLinearLeakyReLU(1024//3, 512, dim=3)
        self.vn_fc2 = VNLinearLeakyReLU(512, 128, dim=3)
        self.vn_fc3 = VNLinear(128, 1) 

    def forward(self, x):
        x = self.feat(x)
        x = self.vn_fc1(x)
        x = self.vn_fc2(x)
        x = self.vn_fc3(x)
        g_pred = x.view(-1, 3)
        g_pred = F.normalize(g_pred, p=2, dim=1)
        return g_pred

class PointNet_VN_Gravity_Bayes(PointNet_VN_Gravity):
    def __init__(self, pooling, normal_channel=True):
        super(PointNet_VN_Gravity_Bayes, self).__init__(pooling, normal_channel)
        
        self.vn_invariant = VNInvariant(128)
        self.kappa_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.vn_fc1(x)
        x = self.vn_fc2(x)
        
        # Mu Branch
        mu = self.vn_fc3(x)
        mu = mu.view(-1, 3)
        mu = F.normalize(mu, p=2, dim=1)
        
        # Kappa Branch
        x_inv = self.vn_invariant(x)
        kappa = self.kappa_mlp(x_inv)
        kappa = kappa + 1.0
        
        return mu, kappa
    
class PointNet_VN_Gravity_Bayes_v2(nn.Module):
    def __init__(self, pooling='attentive', normal_channel=False, mode='equi', stride=16):
        """
        Args:
            pooling: 'mean', 'max', or 'attentive' 
            mode: 'equi' (Vector Neuron) or 'normal' (Standard PointNet)
        """
        super(PointNet_VN_Gravity_Bayes_v2, self).__init__()
        
        self.pooling = pooling
        self.mode = mode
        self.feat_dim = 1024
        self.stride = stride
        
        # Normal Estimator
        self.normal_estimator = NormalEstimator(k=30)
        self.p_normals = None
        self.q_normals = None
        
        # ==========================================
        # Mode 1: Equivariant (Vector Neuron)
        # ==========================================
        if self.mode == 'equi':
            vn_in_channels = 2 # (XYZ, Normal)
            
            # Backbone
            self.vn_fc1 = VNLinearLeakyReLU(vn_in_channels, 64, dim=4, negative_slope=0.2)
            self.vn_fc2 = VNLinearLeakyReLU(64, 64, dim=4, negative_slope=0.2)
            self.vn_fc3 = VNLinearLeakyReLU(64, self.feat_dim, dim=4, negative_slope=0.2)
            
            # Interaction
            self.vn_self_attn = VN_Attention(self.feat_dim)
            self.vn_cross_gating = VN_Cross_Gating(self.feat_dim)
            
            if self.pooling == 'attentive':
                raise ValueError("Attentive pooling for 'equi' mode is not implemented in this version.")
                self.vn_invariant_layer = VNInvariant(self.feat_dim)
                self.attention_mlp = nn.Sequential(
                    nn.Linear(self.feat_dim * 3, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 1) 
                )
            elif self.pooling == 'max':
                self.vn_pool = VNMaxPool(self.feat_dim)
                
            self.vn_invariant = VNInvariant(self.feat_dim)
            
            # Heads
            self.vn_mu_head = nn.Sequential(
                VNLinearLeakyReLU(self.feat_dim, self.feat_dim, dim=3, negative_slope=0.0),
                VNLinear(self.feat_dim, 1)
            )
            self.kappa_mlp = nn.Sequential(
                nn.Linear(self.feat_dim * 3, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 1), nn.Softplus()
            )

        # ==========================================
        # Mode 2: Normal (Standard NN)
        # ==========================================
        elif self.mode == 'normal':
            std_in_channels = 9 # (XYZ + Sign-Invariant Normal Features)
            
            # Backbone
            self.std_fc1 = nn.Sequential(nn.Conv1d(std_in_channels, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc3 = nn.Sequential(nn.Conv1d(64, self.feat_dim, 1), nn.BatchNorm1d(self.feat_dim), nn.LeakyReLU(0.2))
            
            # Interaction
            self.std_self_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)
            self.std_cross_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)
            
            # [NEW] Attentive Pooling Components
            if self.pooling == 'attentive':
                self.attention_mlp = nn.Sequential(
                    nn.Linear(self.feat_dim, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, 1)
                )
            
            # Heads
            self.std_mu_head = nn.Sequential(
                nn.Linear(self.feat_dim, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 3) 
            )
            self.kappa_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 1), nn.Softplus()
            )

    def extract_feat(self, x):
        """
        Extract features. 
        For 'normal' mode, converts normals to sign-invariant outer product features (9 channels).
        """
        B, D, N = x.shape
        
        # 1. Normal Estimation
        if D == 3:
            x = self.normal_estimator(x) 
            
        # Capture raw normals for Hypothesis Testing
        normals = x[:, 3:, :].contiguous() 
        
        # 2. Mode-specific Feature Processing
        if self.mode == 'normal':
            # Sign-Invariant Transformation
            pos = x[:, :3, :] 
            n = x[:, 3:, :]   
            
            # Outer Product (n * n^T) terms
            n_xx = n[:, 0:1, :] * n[:, 0:1, :]
            n_yy = n[:, 1:2, :] * n[:, 1:2, :]
            n_zz = n[:, 2:3, :] * n[:, 2:3, :]
            n_xy = n[:, 0:1, :] * n[:, 1:2, :]
            n_xz = n[:, 0:1, :] * n[:, 2:3, :]
            n_yz = n[:, 1:2, :] * n[:, 2:3, :]
            
            # Concatenate to form 9-channel input
            x = torch.cat([pos, n_xx, n_yy, n_zz, n_xy, n_xz, n_yz], dim=1) 
            
            x = self.std_fc1(x)
            x = self.std_fc2(x)
            x = self.std_fc3(x) 

        elif self.mode == 'equi':
            x = x.view(B, 2, 3, N)
            x = self.vn_fc1(x)
            x = self.vn_fc2(x)
            x = self.vn_fc3(x) 
            
        return x, normals

    def process_interaction(self, f_p, f_q):
        if self.mode == 'equi':
            f_p = self.vn_self_attn(f_p)
            f_q = self.vn_self_attn(f_q)
            f_p_out = self.vn_cross_gating(x=f_p, y=f_q)
            f_q_out = self.vn_cross_gating(x=f_q, y=f_p)
        elif self.mode == 'normal':
            p_in = f_p.permute(0, 2, 1); q_in = f_q.permute(0, 2, 1)
            p_self, _ = self.std_self_attn(p_in, p_in, p_in)
            q_self, _ = self.std_self_attn(q_in, q_in, q_in)
            p_in = p_in + p_self; q_in = q_in + q_self
            p_cross, _ = self.std_cross_attn(query=p_in, key=q_in, value=q_in)
            q_cross, _ = self.std_cross_attn(query=q_in, key=p_in, value=p_in)
            p_out = p_in + p_cross; q_out = q_in + q_cross
            f_p_out = p_out.permute(0, 2, 1); f_q_out = q_out.permute(0, 2, 1)
        return f_p_out, f_q_out

    def apply_pooling(self, feat):
        
        if self.pooling == 'attentive':
            if self.mode == 'equi':
                # feat: [B, C, 3, N]
                inv_feat = self.vn_invariant_layer(feat) # [B, C*3, N]
                scores = self.attention_mlp(inv_feat.permute(0, 2, 1)) # [B, N, 1]
                weights = F.softmax(scores, dim=1) 
                
                # Weighted Sum (Broadcasting)
                weights = weights.permute(0, 2, 1).unsqueeze(1) # (B, 1, 1, N)
                global_feat = torch.sum(feat * weights, dim=-1) # (B, C, 3)
                
            elif self.mode == 'normal':
                # feat: [B, C, N]
                scores = self.attention_mlp(feat.permute(0, 2, 1)) # [B, N, 1]
                weights = F.softmax(scores, dim=1) # (B, N, 1)
                
                weights = weights.permute(0, 2, 1) # (B, 1, N)
                global_feat = torch.sum(feat * weights, dim=-1) # (B, C)
            return global_feat
            
        elif self.pooling == 'mean':
            return torch.mean(feat, dim=-1)
        elif self.pooling == 'max':
             if self.mode == 'equi': return self.vn_pool(feat)
             else: return torch.max(feat, dim=-1)[0]

    def forward_head(self, g_feat):
        if self.mode == 'equi':
            mu = self.vn_mu_head(g_feat).mean(dim=1)
            mu = F.normalize(mu, p=2, dim=1)
            g_inv = self.vn_invariant(g_feat)
            kappa = self.kappa_mlp(g_inv) + 1.0
        elif self.mode == 'normal':
            mu = self.std_mu_head(g_feat)
            mu = F.normalize(mu, p=2, dim=1)
            kappa = self.kappa_mlp(g_feat) + 1.0
        return mu, kappa

    def forward(self, p, q):
        # 1. Backbone (includes Sign-Invariant input for normal mode)
        f_p, n_p = self.extract_feat(p) 
        f_q, n_q = self.extract_feat(q)
        
        self.p_normals = n_p
        self.q_normals = n_q
        
        # 2. Downsampling
        stride = self.stride 
        if self.mode == 'equi':
            f_p_small = f_p[:, :, :, ::stride]
            f_q_small = f_q[:, :, :, ::stride]
        else:
            f_p_small = f_p[:, :, ::stride]
            f_q_small = f_q[:, :, ::stride]
        
        # 3. Interaction
        f_p_gated, f_q_gated = self.process_interaction(f_p_small, f_q_small)
        
        # 4. Pooling
        g_p = self.apply_pooling(f_p_gated)
        g_q = self.apply_pooling(f_q_gated)
            
        # 5. Head
        mu_p, kappa_p = self.forward_head(g_p)
        mu_q, kappa_q = self.forward_head(g_q)
        
        return (mu_p, kappa_p), (mu_q, kappa_q)
        
    def check_correspondence_validity(self, batch_idx, P_indices, Q_indices, g_p, kappa_p, g_q, kappa_q, chi2_thresh=9.0):
        """
        Analytic Covariance Hypothesis Testing with Robustness Floor
        """
        if self.p_normals is None or self.q_normals is None:
            return np.ones(len(P_indices), dtype=bool), np.zeros(len(P_indices))

        # 1. Retrieve Stored Normals
        curr_p_normals = self.p_normals[batch_idx].transpose(0, 1) 
        curr_q_normals = self.q_normals[batch_idx].transpose(0, 1) 
        
        dev = curr_p_normals.device
        if not isinstance(P_indices, torch.Tensor): P_indices = torch.tensor(P_indices, device=dev)
        if not isinstance(Q_indices, torch.Tensor): Q_indices = torch.tensor(Q_indices, device=dev)
        
        n_p = curr_p_normals[P_indices] 
        n_q = curr_q_normals[Q_indices] 
        
        # 2. Inclination & Geometric Sensitivity
        I_p = torch.matmul(n_p, g_p)
        I_q = torch.matmul(n_q, g_q)
        
        # sin^2 term (Geometric Sensitivity) using Absolute Inclination for Sign Robustness
        # Note: sin^2(x) = 1 - cos^2(x). Effect is same for +I or -I.
        sin2_p = torch.clamp(1.0 - I_p**2, min=1e-6)
        sin2_q = torch.clamp(1.0 - I_q**2, min=1e-6)
        
        # 3. Geometric Covariance & Variance
        # Additive Noise Floor (Critical for BNN robustness)
        base_variance = 0.01 
        
        term_p = sin2_p / (kappa_p + 1e-6)
        term_q = sin2_q / (kappa_q + 1e-6)
        
        # Simplified joint variance (assuming independence for robustness against sign flips)
        sigma_sq_total = term_p + term_q + base_variance
        
        # 4. Test Statistic (Absolute Difference)
        # [KEY FIX] Compare Absolute Values to ignore Sign Ambiguity of Raw Normals
        diff = torch.abs(I_p) - torch.abs(I_q)
        residual_sq = diff**2
        
        M2_score = residual_sq / sigma_sq_total
        
        valid_mask = M2_score < chi2_thresh
        
        return valid_mask.cpu().numpy(), M2_score.cpu().numpy()
    
class DGCNN_VN_Gravity(nn.Module):
    def __init__(self, k=20, normal_channel=False):
        super(DGCNN_VN_Gravity, self).__init__()
        
        # 1. VN DGCNN Encoder
        self.feat = VN_DGCNN_Encoder(k=k, embed_dim=1024)
        
        # [수정] 에러 로그에 따르면 Encoder 출력은 1024 채널입니다.
        # (30x1024 input error implies last dim is 1024)
        encoder_dim = 1024 
        
        # 2. Regression Head
        self.vn_fc1 = VNLinearLeakyReLU(encoder_dim, 512, dim=3)
        self.vn_fc2 = VNLinearLeakyReLU(512, 128, dim=3)
        self.vn_fc3 = VNLinear(128, 1) 

    def forward(self, x):
        # Input Handling (XYZ only)
        if x.size(1) > 3:
            x = x[:, :3, :]
            
        x = self.feat(x)       # (B, 1024, 3)
        x = self.vn_fc1(x)     # (B, 512, 3)
        x = self.vn_fc2(x)     # (B, 128, 3)
        x = self.vn_fc3(x)     # (B, 1, 3)
        
        g_pred = x.view(-1, 3)
        g_pred = F.normalize(g_pred, p=2, dim=1)
        
        return g_pred

class DGCNN_VN_Gravity_Bayes(DGCNN_VN_Gravity):
    def __init__(self, k=20, normal_channel=False):
        super(DGCNN_VN_Gravity_Bayes, self).__init__(k, normal_channel)
        
        self.vn_invariant = VNInvariant(128) 
        self.kappa_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x):
        if x.size(1) > 3:
            x = x[:, :3, :]

        # Backbone
        x = self.feat(x)
        x = self.vn_fc1(x)
        x = self.vn_fc2(x)
        
        # Mu Branch
        mu = self.vn_fc3(x)
        mu = mu.view(-1, 3)
        mu = F.normalize(mu, p=2, dim=1)
        
        # Kappa Branch
        x_inv = self.vn_invariant(x)
        kappa = self.kappa_mlp(x_inv)
        kappa = kappa + 1.0
        
        return mu, kappa
    
if __name__ == '__main__':
    
    # 1. Setup Data
    B, C, N = 10, 3, 30 
    points = torch.randn(B, C, N)
    print("Input Points Shape:", points.shape)
    
    # 2. Rotate
    theta = np.pi / 4
    rotation_matrix = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=torch.float32)
    
    rotated_points = torch.matmul(rotation_matrix, points) 
    print("Rotated Points Shape:", rotated_points.shape)
    
    # 3. Model Init
    print("Initializing Model...")
    model = DGCNN_VN_Gravity_Bayes(k=20, normal_channel=False)
    model.eval()
    
    # 4. Inference & Validation
    with torch.no_grad():
        # [수정] 모델이 Tuple (mu, kappa)를 반환하므로 unpacking 필요
        mu1, kappa1 = model(points)
        mu2, kappa2 = model(rotated_points)
    
    # [수정] Equivariance 체크는 방향 벡터(mu)에 대해서만 수행
    # mu1을 회전시킨 것과 mu2가 같아야 함
    mu1_rotated = mu1 @ rotation_matrix.T 
    
    loss = F.mse_loss(mu2, mu1_rotated)
    print(f"Equivariance Loss (Mu): {loss.item():.6f}")
    
    # Kappa는 Invariant 해야 함 (회전해도 불확실성은 같아야 함)
    kappa_loss = F.mse_loss(kappa1, kappa2)
    print(f"Invariance Loss (Kappa): {kappa_loss.item():.6f}")
    print("Output Shape (Mu):", mu1.shape)