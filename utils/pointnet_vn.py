import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from utils.layers import *


class STNkd(nn.Module):
    def __init__(self, pooling, d=64):
        super(STNkd, self).__init__()
        self.pooling = pooling
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, pooling, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.pooling = pooling
        self.n_knn = 20
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        # ! Disabled STD Feature for equivariant feature extraction not invariant feature
        # self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(self.pooling, d=64//3)

    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        x = self.conv1(x)
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        x = x.mean(dim=-1)
        
        return x
        
    def old_forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        x = self.conv1(x)
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

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
    model = PointNetEncoder(pooling='max', global_feat=True, feature_transform=True, channel=3)
    
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
    
    # 일단 에러 없이 돌아가는지 확인:
    print("Output Shape:", pred1.shape)