import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_vn import PointNetEncoder
from utils.dgcnn_vn import VN_DGCNN_Encoder
from utils.layers import VNLinearLeakyReLU, VNLinear, VNInvariant
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