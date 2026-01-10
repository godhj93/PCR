import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.layers import *

def knn(x, k):
    """
    x: (B, C*3, N) 형태의 Flattened Feature
    Distance 계산은 Rotation Invariant 해야 하므로, 
    Vector 차원(3)과 Feature 차원(C)을 합쳐서 Norm을 구합니다.
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def get_graph_feature_vn(x, k=20, idx=None):
    """
    Vector Neuron을 위한 Graph Feature 생성 함수
    x: (B, C, 3, N) - 4D Vector Tensor
    """
    B, C, D, N = x.size()
    
    # 1. KNN 검색 (Rotation Invariant Distance)
    # (B, C, 3, N) -> (B, C*3, N)
    x_flat = x.view(B, -1, N)
    
    if idx is None:
        idx = knn(x_flat, k=k)  # (B, N, k)
    
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    
    # 2. Gathering Neighbors
    # x: (B, C, 3, N) -> (B, N, C, 3) -> (B*N, C, 3)
    x_view = x.permute(0, 3, 1, 2).contiguous().view(B*N, C, D)
    
    # feature: (B*N*k, C, 3)
    feature = x_view[idx, :, :]
    
    # Reshape back: (B, N, k, C, 3) -> (B, C, 3, N, k)
    feature = feature.view(B, N, k, C, D).permute(0, 3, 4, 1, 2).contiguous()
    
    # 3. Edge Feature Calculation
    # x_central: (B, C, 3, N, k)
    x_central = x.unsqueeze(-1).repeat(1, 1, 1, 1, k)
    
    # Output: Concat([x_j - x_i, x_i]) -> Channel dim(1)이 2배가 됨
    # (B, 2*C, 3, N, k)
    feature = torch.cat((feature - x_central, x_central), dim=1)
    
    return feature

class VN_DGCNN_Encoder(nn.Module):
    def __init__(self, k=20, embed_dim=1024):
        super(VN_DGCNN_Encoder, self).__init__()
        self.k = k
        
        # -------------------------------------------------------
        # Channel Definition (Hardcoded to avoid rounding errors)
        # Original DGCNN: 64 -> 64 -> 128 -> 256
        # VN DGCNN:       21 -> 21 -> 42  -> 85
        # -------------------------------------------------------
        self.c1 = 64   # int(64/3)
        self.c2 = 64   # int(64/3)
        self.c3 = 128   # int(128/3)
        self.c4 = 256   # int(256/3)
        self.c_out = 1024 # int(1024/3)

        # Layer 1: Input (1 ch) -> Edge (2 ch) -> Output (c1)
        self.conv1 = VNLinearLeakyReLU(2, self.c1, dim=5, negative_slope=0.0)
        self.pool1 = VNMaxPool(self.c1)
        
        # Layer 2: Input (c1) -> Edge (2*c1) -> Output (c2)
        self.conv2 = VNLinearLeakyReLU(self.c1*2, self.c2, dim=5, negative_slope=0.0)
        self.pool2 = VNMaxPool(self.c2)
        
        # Layer 3: Input (c2) -> Edge (2*c2) -> Output (c3)
        self.conv3 = VNLinearLeakyReLU(self.c2*2, self.c3, dim=5, negative_slope=0.0)
        self.pool3 = VNMaxPool(self.c3)
        
        # Layer 4: Input (c3) -> Edge (2*c3) -> Output (c4)
        self.conv4 = VNLinearLeakyReLU(self.c3*2, self.c4, dim=5, negative_slope=0.0)
        self.pool4 = VNMaxPool(self.c4)
        
        # Aggregation Layer (Conv5)
        # Concatenation of all layers: c1 + c2 + c3 + c4
        self.total_dim = self.c1 + self.c2 + self.c3 + self.c4  # 21+21+42+85 = 169
        
        # dim=4 because input is (B, C, 3, N)
        self.conv5 = VNLinearLeakyReLU(self.total_dim, self.c_out, dim=4, negative_slope=0.0)
        
    def forward(self, x):
        # x: (B, 3, N)
        B, D, N = x.size()
        
        # (B, 3, N) -> (B, 1, 3, N) : VN 입력을 위해 채널 차원 추가
        x = x.unsqueeze(1)
        
        # --- Layer 1 ---
        # (B, 1, 3, N) -> Graph construction -> (B, 2, 3, N, k)
        x = get_graph_feature_vn(x, k=self.k)
        x = self.conv1(x)                   # (B, c1, 3, N, k)
        x1 = x.mean(dim=-1)                 # Pool over k -> (B, c1, 3, N)
        
        # --- Layer 2 ---
        x = get_graph_feature_vn(x1, k=self.k)
        x = self.conv2(x)                   # (B, c2, 3, N, k)
        x2 = x.mean(dim=-1)                 # (B, c2, 3, N)
        
        # --- Layer 3 ---
        x = get_graph_feature_vn(x2, k=self.k)
        x = self.conv3(x)                   # (B, c3, 3, N, k)
        x3 = x.mean(dim=-1)                 # (B, c3, 3, N)
        
        # --- Layer 4 ---
        x = get_graph_feature_vn(x3, k=self.k)
        x = self.conv4(x)                   # (B, c4, 3, N, k)
        x4 = x.mean(dim=-1)                 # (B, c4, 3, N)
        
        # --- Concatenation ---
        # x1(21) + x2(21) + x3(42) + x4(85) = 169 channels
        x_concat = torch.cat((x1, x2, x3, x4), dim=1) # (B, 169, 3, N)
        
        # --- Aggregation ---
        x_out = self.conv5(x_concat)        # (B, 341, 3, N)
        
        # Global Pooling (Mean over N points)
        # Result: (B, 341, 3) -> Global Equivariant Feature
        x_global = x_out.mean(dim=-1)       
        
        return x_global

if __name__ == '__main__':
    # Test Code for Shape & Equivariance
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}...")

    # 1. Setup Input
    B, N = 4, 1024
    points = torch.randn(B, 3, N).to(device)
    
    # 2. Setup Model
    model = VN_DGCNN_Encoder(k=20).to(device)
    model.eval()
    
    # 3. Setup Rotation Matrix (Z-axis 45 degree)
    theta = np.pi / 4
    rot_z = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=torch.float32).to(device)
    
    # 4. Rotate Input
    rotated_points = torch.matmul(rot_z, points)
    
    # 5. Inference
    with torch.no_grad():
        start = time.time()
        out_original = model(points)          # f(x)
        out_rotated = model(rotated_points)   # f(Rx)
        end = time.time()
        
    print(f"Inference Time (Batch {B}): {end - start:.4f} sec")
    print(f"Output Shape: {out_original.shape}") # Should be (B, 341, 3)

    # 6. Check Equivariance: f(Rx) == R * f(x)
    # Rotate the output of original input
    # out_original: (B, C, 3) -> Need to rotate the last dim
    out_original_rotated = torch.matmul(out_original, rot_z.T)
    
    # Calculate Error
    diff = out_rotated - out_original_rotated
    mse = torch.mean(diff ** 2).item()
    
    print(f"Equivariance MSE: {mse:.2e}")
    
    if mse < 1e-5:
        print("✅ Success: The model is rotation equivariant!")
    else:
        print("❌ Failure: The model is NOT equivariant.")