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
    
class PointNet_VN_Gravity_Bayes_Joint(PointNet_VN_Gravity_Bayes):
    def __init__(self, pooling, normal_channel=True):
        super(PointNet_VN_Gravity_Bayes_Joint, self).__init__(pooling, normal_channel)
    
    def forward(self, p, q):
        # 1. Feature Extraction (Siamese)
        feat_p = self.feat(p)  # (B, 1024//3, 3)
        feat_q = self.feat(q)  # (B, 1024//3, 3)
        
        # 2. Shared MLP (Backbone)
        feat_p = self.vn_fc1(feat_p)
        feat_p = self.vn_fc2(feat_p) # (B, 128, 3)
        
        feat_q = self.vn_fc1(feat_q)
        feat_q = self.vn_fc2(feat_q) # (B, 128, 3)
        
        # -----------------------------------------------------------
        # 3. Mu Branch (Independent)
        # -----------------------------------------------------------
        mu_p = self.vn_fc3(feat_p)
        mu_p = F.normalize(mu_p.view(-1, 3), p=2, dim=1)
        
        mu_q = self.vn_fc3(feat_q)
        mu_q = F.normalize(mu_q.view(-1, 3), p=2, dim=1)
        
        # -----------------------------------------------------------
        # 4. Kappa Branch (Joint)
        # -----------------------------------------------------------
        inv_feat_p = self.vn_invariant(feat_p) # (B, 128)
        inv_feat_q = self.vn_invariant(feat_q) # (B, 128)
        
        combined_inv_feat = inv_feat_p + inv_feat_q # (B, 128)
        
        kappa = self.kappa_mlp(combined_inv_feat)
        kappa = kappa + 1.0
        
        # 반환: P의 중력, Q의 중력, 그리고 공통된 확신도
        return mu_p, mu_q, kappa
    
class PointNet_VN_Gravity_Bayes_v2(nn.Module):
    def __init__(self, pooling='mean', normal_channel=False, mode='equi'):
        """
        Args:
            mode: 'equi' (Vector Neuron + Gating) 
                  or 'normal' (Standard PointNet + Self/Cross Attention)
        """
        super(PointNet_VN_Gravity_Bayes_v2, self).__init__()
        
        self.pooling = pooling
        self.mode = mode
        self.feat_dim = 1024
        
        # Normal Estimator
        self.normal_estimator = NormalEstimator(k=30)
        
        # ==========================================
        # Mode 1: Equivariant (Vector Neuron)
        # ==========================================
        if self.mode == 'equi':
            vn_in_channels = 2 # (XYZ, Normal)
            
            # Backbone
            self.vn_fc1 = VNLinearLeakyReLU(vn_in_channels, 64, dim=4, negative_slope=0.2)
            self.vn_fc2 = VNLinearLeakyReLU(64, 64, dim=4, negative_slope=0.2)
            self.vn_fc3 = VNLinearLeakyReLU(64, self.feat_dim, dim=4, negative_slope=0.2)
            
            # Interaction (VN은 구조상 Gating이 효율적)
            self.vn_self_attn = VN_Attention(self.feat_dim)
            self.vn_cross_gating = VN_Cross_Gating(self.feat_dim)
            
            # Pooling & Invariant
            if pooling == 'max':
                self.vn_pool = VNMaxPool(self.feat_dim)
            self.vn_invariant = VNInvariant(self.feat_dim)
            
            # Heads
            self.vn_mu_head = nn.Sequential(
                VNLinearLeakyReLU(self.feat_dim, self.feat_dim, dim=3, negative_slope=0.0),
                VNLinear(self.feat_dim, 1)
            )
            # VNInvariant 출력(feat_dim * 3)에 맞춘 MLP
            self.kappa_mlp = nn.Sequential(
                nn.Linear(self.feat_dim * 3, 128), 
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64), 
                nn.LeakyReLU(0.2), 
                nn.Linear(64, 1), 
                nn.Softplus()
            )

        # ==========================================
        # Mode 2: Normal (Standard NN + Transformer Attention)
        # ==========================================
        elif self.mode == 'normal':
            std_in_channels = 6 # XYZ(3) + Normal(3)
            
            # 1. Backbone (PointNet)
            self.std_fc1 = nn.Sequential(nn.Conv1d(std_in_channels, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc3 = nn.Sequential(nn.Conv1d(64, self.feat_dim, 1), nn.BatchNorm1d(self.feat_dim), nn.LeakyReLU(0.2))
            
            # 2. Interaction (Standard Transformer Attention)
            # Self Attention: 내 점들끼리의 관계 파악
            self.std_self_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)
            
            # Cross Attention: 상대방 점들을 참조 (P queries Q)
            self.std_cross_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)
            
            # Heads
            self.std_mu_head = nn.Sequential(
                nn.Linear(self.feat_dim, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 3) # 3D vector output
            )
            # Global Pooling 출력(feat_dim)에 맞춘 MLP
            self.kappa_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 128), 
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64), 
                nn.LeakyReLU(0.2), 
                nn.Linear(64, 1), 
                nn.Softplus()
            )

    def extract_feat(self, x):
        B, D, N = x.shape
        if D == 3:
            x = self.normal_estimator(x) # [B, 6, N]
        
        if self.mode == 'equi':
            x = x.view(B, 2, 3, N)
            x = self.vn_fc1(x)
            x = self.vn_fc2(x)
            x = self.vn_fc3(x) # [B, 1024, 3, N]
        elif self.mode == 'normal':
            x = self.std_fc1(x)
            x = self.std_fc2(x)
            x = self.std_fc3(x) # [B, 1024, N]
        return x

    def process_interaction(self, f_p, f_q):
        """
        Mode에 따른 Interaction 처리
        Equi: VN Self Attn -> VN Cross Gating
        Normal: Self Attn -> Cross Attn
        """
        if self.mode == 'equi':
            # 1. Self Interaction
            f_p = self.vn_self_attn(f_p)
            f_q = self.vn_self_attn(f_q)
            
            # 2. Cross Interaction (Gating)
            f_p_out = self.vn_cross_gating(x=f_p, y=f_q)
            f_q_out = self.vn_cross_gating(x=f_q, y=f_p)
            
        elif self.mode == 'normal':
            # Input: [B, C, N] -> [B, N, C] for MultiheadAttention
            p_in = f_p.permute(0, 2, 1) 
            q_in = f_q.permute(0, 2, 1)
            
            # 1. Self Attention (Residual)
            # Query=P, Key=P, Value=P
            p_self, _ = self.std_self_attn(p_in, p_in, p_in)
            q_self, _ = self.std_self_attn(q_in, q_in, q_in)
            
            p_in = p_in + p_self
            q_in = q_in + q_self
            
            # 2. Cross Attention (Residual)
            # P gets info from Q: Query=P, Key=Q, Value=Q
            p_cross, _ = self.std_cross_attn(query=p_in, key=q_in, value=q_in)
            # Q gets info from P: Query=Q, Key=P, Value=P
            q_cross, _ = self.std_cross_attn(query=q_in, key=p_in, value=p_in)
            
            p_out = p_in + p_cross
            q_out = q_in + q_cross
            
            # Back to [B, C, N]
            f_p_out = p_out.permute(0, 2, 1)
            f_q_out = q_out.permute(0, 2, 1)
            
        return f_p_out, f_q_out

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
        # 1. Backbone
        f_p = self.extract_feat(p) 
        f_q = self.extract_feat(q)
        
        # 2. Downsampling
        stride = 16 
        if self.mode == 'equi':
            f_p_small = f_p[:, :, :, ::stride]
            f_q_small = f_q[:, :, :, ::stride]
        else:
            f_p_small = f_p[:, :, ::stride]
            f_q_small = f_q[:, :, ::stride]
        
        # 3. Interaction (Self -> Cross)
        f_p_gated, f_q_gated = self.process_interaction(f_p_small, f_q_small)
        
        # 4. Pooling
        if self.pooling == 'mean':
            g_p = torch.mean(f_p_gated, dim=-1)
            g_q = torch.mean(f_q_gated, dim=-1)
        else:
            if self.mode == 'equi':
                g_p = self.vn_pool(f_p_gated)
                g_q = self.vn_pool(f_q_gated)
            else:
                g_p = torch.max(f_p_gated, dim=-1)[0]
                g_q = torch.max(f_q_gated, dim=-1)[0]
            
        # 5. Head
        mu_p, kappa_p = self.forward_head(g_p)
        mu_q, kappa_q = self.forward_head(g_q)
        
        return (mu_p, kappa_p), (mu_q, kappa_q)
        
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