import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pointnet_vn import PointNetEncoder
from utils.layers import VNLinearLeakyReLU, VNLinear 
import numpy as np

class GravityEstimationModel(nn.Module):
    def __init__(self, pooling, normal_channel=True):
        super(GravityEstimationModel, self).__init__()
        self.pooling = pooling
        channel = 6 if normal_channel else 3
        
        # 1. VN Encoder (Backbone)
        # global_feat=True -> (B, 1024, 3) 형태의 Equivariant Feature가 나온다고 가정
        self.feat = PointNetEncoder(self.pooling, global_feat=True, feature_transform=True, channel=channel)
        
        # 2. Regression Head
        # (B, 1024, 3) -> (B, 512, 3)
        self.vn_fc1 = VNLinearLeakyReLU(1024//3, 512, dim=3) # dim=3은 3D vector라는 뜻
        
        # (B, 512, 3) -> (B, 128, 3)
        self.vn_fc2 = VNLinearLeakyReLU(512, 128, dim=3)
        
        # (B, 128, 3) -> (B, 1, 3) -> 최종 벡터 1개 추출
        # 마지막은 Activation 없이 선형 변환만 (Rotation만 허용)
        self.vn_fc3 = VNLinear(128, 1) 

    def forward(self, x):
        # x: (B, C, N)
        # feat: (B, 1024, 3) <- VN Encoder의 출력 (Global Feature)
        x = self.feat(x)
        
        # Head 통과
        x = self.vn_fc1(x)
        x = self.vn_fc2(x)
        x = self.vn_fc3(x) # (B, 1, 3)
        
        # 차원 축소: (B, 1, 3) -> (B, 3)
        g_pred = x.view(-1, 3)
        
        # Unit Vector로 정규화 (중력'방향'만 중요하므로)
        g_pred = F.normalize(g_pred, p=2, dim=1)
        
        return g_pred



if __name__ == '__main__':
    
    # Check Equivariance of PointNetEncoder
    # [수정] N(점 개수)과 C(채널/좌표)를 올바르게 설정
    # (Batch, Channel=3, Num_Points=30)
    B, C, N = 10, 3, 30 
    
    # Generate random point cloud: (1, 3, 30)
    points = torch.randn(B, C, N)
    print("Input Points Shape:", points.shape) # (1, 3, 30) 확인
    
    # Rotate point cloud
    # 회전은 (3, 3) 행렬이므로 채널 차원(dim=1)에 대해 곱해야 함
    theta = np.pi / 4  # 45 degrees
    rotation_matrix = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=torch.float32)
    
    # (3, 3) x (1, 3, 30) -> (1, 3, 30)
    # einsum을 쓰거나 transpose 후 matmul 사용
    # 간단하게 구현:
    rotated_points = torch.matmul(rotation_matrix, points) 
    
    print("Rotated Points Shape:", rotated_points.shape)
    
    # 모델 생성 (기존 코드와 동일)
    model = GravityEstimationModel(pooling='max', normal_channel=False)
    
    # 추론
    pred1 = model(points)          # f(x)
    pred2 = model(rotated_points)  # f(R * x)
    
    # 검증: f(R * x) == R * f(x) 이어야 함 (Equivariance)
    # pred1(결과 벡터)도 회전시켜서 pred2와 비교해야 함
    
    # [주의] pred1의 shape은 (B, C_out, 3) 일 것임 (GravityEncoder 수정본 기준)
    # 따라서 pred1의 마지막 차원(3)에 대해 회전을 적용해야 함
    
    pred1_rotated = pred1 @ rotation_matrix.T  # (B, C, 3) x (3, 3)
    
    loss = F.mse_loss(pred2, pred1_rotated)
    print("Equivariance Loss:", loss.item())
    
    print("Output Shape:", pred1.shape)
    