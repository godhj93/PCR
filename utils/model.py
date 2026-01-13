"""
Flow Matching Registration Model
Architecture: DGCNN + Transformer -> 6D velocity vector

Input:
    - P: Source point cloud (B, 3, N)
    - Q: Target point cloud (B, 3, N)
    
Output:
    - v: 6D velocity vector (B, 6) [vx, vy, vz, wx, wy, wz]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def knn(x, k):
    """
    K-nearest neighbors for point cloud
    Args:
        x: (B, C, N) point features
        k: number of neighbors
    Returns:
        idx: (B, N, k) indices of k-nearest neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Extract edge features for DGCNN
    Args:
        x: (B, C, N)
        k: number of neighbors
        idx: (B, N, k) precomputed indices (optional)
    Returns:
        feature: (B, 2C, N, k) concatenated [x_i, x_j - x_i]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, k, C)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    
    return feature


class DGCNN_Encoder(nn.Module):
    """
    DGCNN encoder for point cloud feature extraction
    """
    def __init__(self, k=20, emb_dims=512):
        super(DGCNN_Encoder, self).__init__()
        self.k = k
        self.emb_dims = emb_dims
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, N) point cloud
        Returns:
            x: (B, emb_dims, N) per-point features
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # EdgeConv 1
        x1 = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        x1 = self.conv1(x1)  # (B, 64, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # (B, 64, N)
        
        # EdgeConv 2
        x2 = get_graph_feature(x1, k=self.k)  # (B, 128, N, k)
        x2 = self.conv2(x2)  # (B, 64, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # (B, 64, N)
        
        # EdgeConv 3
        x3 = get_graph_feature(x2, k=self.k)  # (B, 128, N, k)
        x3 = self.conv3(x3)  # (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # (B, 128, N)
        
        # EdgeConv 4
        x4 = get_graph_feature(x3, k=self.k)  # (B, 256, N, k)
        x4 = self.conv4(x4)  # (B, 256, N, k)
        x4 = x4.max(dim=-1, keepdim=False)[0]  # (B, 256, N)
        
        # Concatenate all features
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        
        # Final feature embedding
        x = self.conv5(x)  # (B, emb_dims, N)
        
        return x

class TransformerFusion(nn.Module):
    """
    Transformer for fusing P and Q features
    Standard Transformer architecture
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerFusion, self).__init__()
        
        self.d_model = d_model
        self.type_embedding = nn.Embedding(2, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, p_feat, q_feat):
        """
        Args:
            p_feat: (B, D, N)
            q_feat: (B, D, N)
        """
        device = p_feat.device
        B, D, N = p_feat.size()
        
        # (B, N, D)로 변환
        p_feat = p_feat.transpose(1, 2)
        q_feat = q_feat.transpose(1, 2)
        
        # [NEW] Type Embedding 추가 (P와 Q 구분)
        # P에는 0번 임베딩, Q에는 1번 임베딩을 더해줌
        type_p = self.type_embedding(torch.zeros(B, N, dtype=torch.long, device=device)) # (B, N, D)
        type_q = self.type_embedding(torch.ones(B, N, dtype=torch.long, device=device))  # (B, N, D)
        
        p_feat = p_feat + type_p
        q_feat = q_feat + type_q
        
        # Concat
        combined = torch.cat([p_feat, q_feat], dim=1)  # (B, 2N, D)
        
        # Transformer
        transformed = self.transformer_encoder(combined)  # (B, 2N, D)
        
        # Global Pooling (Mean)
        # 전체를 평균내는 것보다, P(Source)가 어떻게 변해야 하는지가 중요하므로
        # P 부분에 해당하는 토큰들만 Pooling하는 전략도 유효합니다.
        # 여기서는 기존대로 전체 Mean을 유지하되, Type Embedding 덕분에 구분이 가능합니다.
        fused = transformed.mean(dim=1)
        
        return fused

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        # log-space에서 frequency 생성 (1 ~ 10000)
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.register_buffer('freqs', torch.exp(torch.arange(self.half_dim) * -self.emb))

    def forward(self, t):
        """
        t: (B,) tensor
        Returns: (B, dim)
        """
        # t를 (B, 1)로 변환
        t = t.view(-1, 1)
        args = t * self.freqs.view(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # dim이 홀수일 경우 zero padding (보통 짝수로 씀)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

class VelocityHead(nn.Module):
    """
    [Modified] Input dim = feature_dim + time_dim
    """
    def __init__(self, feature_dim=512, time_dim=64, hidden_dims=[256, 128], output_dim=6):
        super(VelocityHead, self).__init__()
        
        # 입력 차원: Global Feature + Time Embedding
        input_dim = feature_dim + time_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x, t_emb):
        """
        Args:
            x: (B, D) fused global feature
            t_emb: (B, time_dim) time embedding
        """
        # Feature와 Time Embedding을 이어 붙임
        x_input = torch.cat([x, t_emb], dim=1)  # (B, D + time_dim)
        return self.mlp(x_input)

class FlowModel(nn.Module):
    """
    Complete Flow Matching Registration Model
    DGCNN -> Transformer -> Velocity Head
    """
    def __init__(self, 
                 k=20, 
                 emb_dims=512, 
                 nhead=8, 
                 num_layers=6, 
                 dim_feedforward=2048,
                 dropout=0.1,
                 time_dim=64):
        super(FlowModel, self).__init__()
        
        # DGCNN encoders for P and Q (separate weights)
        self.dgcnn_p = DGCNN_Encoder(k=k, emb_dims=emb_dims)
        self.dgcnn_q = DGCNN_Encoder(k=k, emb_dims=emb_dims)
        
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Transformer fusion
        self.transformer = TransformerFusion(
            d_model=emb_dims,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Velocity prediction head
        self.velocity_head = VelocityHead(
            feature_dim=emb_dims,
            time_dim=time_dim,
            hidden_dims=[256, 128],
            output_dim=6
        )
    
    def forward(self, p, q, t):
        """
        Args:
            p: (B, 3, N) Source (x_t)
            q: (B, 3, N) Target (x_1)
            t: (B,) Time t in [0, 1]  <-- [NEW] 입력 추가
        """
        # 1. Time Embedding
        t_emb = self.time_mlp(t) # (B, time_dim)
        
        # 2. Extract Features
        p_feat = self.dgcnn_p(p)
        q_feat = self.dgcnn_q(q)
        
        # 3. Fuse Features (Global Feature)
        fused_feat = self.transformer(p_feat, q_feat) # (B, emb_dims)
        
        # 4. Predict Velocity with Time Info
        v_pred = self.velocity_head(fused_feat, t_emb) # (B, 6)
        
        return v_pred


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """
    Test forward and backward pass
    """
    print("="*80)
    print("Flow Matching Registration Model Test")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Model hyperparameters
    batch_size = 4
    num_points = 512
    k = 20
    emb_dims = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    print(f"\nModel Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of points: {num_points}")
    print(f"  - K-neighbors: {k}")
    print(f"  - Embedding dims: {emb_dims}")
    print(f"  - Transformer heads: {nhead}")
    print(f"  - Transformer layers: {num_layers}")
    print(f"  - Feedforward dims: {dim_feedforward}")
    print(f"  - Dropout: {dropout}")
    
    # Create model
    model = FlowMatchingRegistration(
        k=k,
        emb_dims=emb_dims,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create dummy data (matching data.py format)
    print(f"\n{'='*80}")
    print("Testing Forward Pass")
    print("="*80)
    
    # Random point clouds
    p = torch.randn(batch_size, 3, num_points).to(device)
    q = torch.randn(batch_size, 3, num_points).to(device)
    
    print(f"\nInput shapes:")
    print(f"  P (source): {p.shape}")
    print(f"  Q (target): {q.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        t = torch.rand(batch_size).to(device)  # Random time t in [0, 1]
        v_pred = model(p, q, t)  # Corrected order: (p, q, t)
    
    print(f"\nOutput shape:")
    print(f"  v_pred (velocity): {v_pred.shape}")
    print(f"\nSample prediction (first batch):")
    print(f"  Linear velocity (vx, vy, vz): {v_pred[0, :3].cpu().numpy()}")
    print(f"  Angular velocity (wx, wy, wz): {v_pred[0, 3:].cpu().numpy()}")
    
    # Test backward pass
    print(f"\n{'='*80}")
    print("Testing Backward Pass")
    print("="*80)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy target
    v_target = torch.randn(batch_size, 6).to(device)
    t_train = torch.rand(batch_size).to(device)
    
    print(f"\nTarget velocity shape: {v_target.shape}")
    
    # Forward
    v_pred = model(p, q, t_train)
    
    # Compute loss (MSE)
    loss = F.mse_loss(v_pred, v_target)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters() if p.requires_grad)
    
    if has_gradients:
        print("✓ Gradients computed successfully!")
        
        # Optimizer step
        optimizer.step()
        print("✓ Optimizer step completed!")
        
        # Verify parameters updated
        print("\nGradient statistics:")
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        print(f"  Total gradient norm: {total_norm:.6f}")
    else:
        print("✗ No gradients computed!")
    
    # Test with different batch sizes
    print(f"\n{'='*80}")
    print("Testing Different Batch Sizes")
    print("="*80)
    
    model.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
    for bs in [1, 2, 8, 16]:
        p_test = torch.randn(bs, 3, num_points).to(device)
        q_test = torch.randn(bs, 3, num_points).to(device)
        t_test = torch.rand(bs).to(device)
        
        with torch.no_grad():
            v_test = model(p_test, q_test, t_test)
        
        print(f"  Batch size {bs:2d}: Input {p_test.shape} -> Output {v_test.shape} ✓")
    
    print(f"\n{'='*80}")
    print("All tests passed successfully! ✓")
    print("="*80)
    
