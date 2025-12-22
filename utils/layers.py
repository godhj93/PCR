import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class NormalEstimator(nn.Module):
    def __init__(self, k=30):
        super(NormalEstimator, self).__init__()
        self.k = k

    def forward(self, x):
        """
        Input: x (B, 3, N) - Pure XYZ Point Cloud
        Output: x_out (B, 6, N) - XYZ + Estimated Normal
        """
        B, C, N = x.shape
        
        # 1. KNN Search
        x_t = x.transpose(1, 2).contiguous() # (B, N, 3)
        
        # 거리 행렬 계산 (x^2 + y^2 - 2xy)
        xx = torch.sum(x_t ** 2, dim=2, keepdim=True)
        # matmul의 두 번째 인자는 x (B, 3, N)를 그대로 사용
        pairwise_dist = xx + xx.transpose(2, 1) - 2 * torch.matmul(x_t, x)
        
        # 가장 가까운 k개 이웃 인덱스 (B, N, k)
        idx = pairwise_dist.topk(k=self.k, dim=-1, largest=False)[1]

        # 2. Gather Neighbors
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, self.k)
        # Gather를 위한 인덱스 확장
        # idx는 (B, N, k) 형태
        
        # (B, N, k, 3) 형태로 이웃 점 가져오기
        neighbors = x_t[batch_indices, idx, :] 

        # 3. PCA (Local Plane Estimation)
        centroid = neighbors.mean(dim=2, keepdim=True)
        centered = neighbors - centroid # (B, N, k, 3)
        
        # 공분산 행렬: (B, N, 3, k) @ (B, N, k, 3) -> (B, N, 3, 3)
        cov = torch.matmul(centered.transpose(2, 3), centered)

        # 4. Eigen Decomposition
        # e: 고유값, v: 고유벡터 (오름차순 정렬됨)
        e, v = torch.linalg.eigh(cov) 
        
        # 가장 작은 고유값에 해당하는 벡터 = Normal
        normals = v[:, :, :, 0] # (B, N, 3)

        # 5. Orientation Correction
        # 시선 방향(-x_t)과 내적하여 양수가 되도록 뒤집기
        view_dir = -x_t 
        dot_prod = (view_dir * normals).sum(dim=-1, keepdim=True)
        mask = (dot_prod < 0).float()
        normals = normals * (1 - 2 * mask)

        # (B, 3, N) 형태로 복구
        normals = normals.transpose(1, 2).contiguous()
        
        # 6. Concatenate
        x_out = torch.cat([x, normals], dim=1) # (B, 6, N)
        return x_out
    
class VN_Attention(nn.Module):
    def __init__(self, in_channels):
        super(VN_Attention, self).__init__()
        self.q_proj = VNLinear(in_channels, in_channels)
        self.k_proj = VNLinear(in_channels, in_channels)
        self.v_proj = VNLinear(in_channels, in_channels)
        self.scale = in_channels ** -0.5

    def forward(self, x):
        # x: [B, C, 3, N]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # [수정] 3차원 벡터(u=3)에 대해 내적을 수행해야 함
        # q: b, c, u, i (u=3, i=N)
        # k: b, c, u, j (u=3, j=N)
        # 결과: b, c, i, j (N x N Attention Map)
        attn_logits = torch.einsum('bcui, bcuj -> bcij', q, k) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # [수정] Attention Weight(N x N)를 Value(3 x N)에 반영
        # weights: b, c, i, j
        # v: b, c, u, j
        # 결과: b, c, u, i (u=3, i=N)
        out = torch.einsum('bcij, bcuj -> bcui', attn_weights, v)
        
        return x + out

class VN_Cross_Gating(nn.Module):
    def __init__(self, in_channels):
        super(VN_Cross_Gating, self).__init__()
        self.q_proj = VNLinear(in_channels, in_channels)
        self.k_proj = VNLinear(in_channels, in_channels)
        self.scale = in_channels ** -0.5
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # x: [B, C, 3, N]
        q = self.q_proj(x)
        k = self.k_proj(y)
        
        # [수정] u=3 (벡터 차원)에 대해 내적
        # q: bcui, k: bcuj -> map: bcij
        attn_logits = torch.einsum('bcui, bcuj -> bcij', q, k) * self.scale
        
        # Max Pooling over reference points (dim=-1)
        # [B, C, N, M] -> [B, C, N]
        relevance_score, _ = torch.max(attn_logits, dim=-1)
        
        # Gate 생성
        # [B, C, N] -> [B, N, C] (Linear용)
        gate = self.gate_mlp(relevance_score.transpose(1, 2))
        # [B, N, C] -> [B, C, 1, N] (Broadcasting용: 1은 벡터차원, N은 점 차원)
        gate = gate.transpose(1, 2).unsqueeze(2)
        
        out = x * gate
        return x + out

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is not None: # dynamic knn graph
            idx = knn(x, k=k)
        else:             # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device  # 입력 텐서의 device 사용

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0
    
class VNInvariant(nn.Module):
    def __init__(self, in_channels):
        super(VNInvariant, self).__init__()
    
    def forward(self, x):
        # x: (B, C, 3) -> Norm -> (B, C)
        return torch.norm(x, dim=-1)