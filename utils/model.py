import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import VNLinearLeakyReLU, VNLinear, VNInvariant, VN_Attention, VN_Cross_Gating, VNMaxPool, NormalEstimator
import numpy as np

class TimeEmbedding(nn.Module):
    """
    Scalar 시간 t를 고차원 벡터로 변환하여 모델이 시간 흐름에 민감하게 반응하도록 함.
    Gaussian Fourier Projection 방식 사용.
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # 학습되지 않는 고정된 랜덤 주파수 (Fixed Random Frequencies)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, t):
        # t: [B] -> [B, 1]
        t = t.view(-1, 1) if t.dim() == 1 else t
        # t_proj: [B, embed_dim/2]
        t_proj = t * self.W[None, :] * 2 * np.pi
        # output: [B, embed_dim]
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
    
class PointNet_VN_Gravity_Bayes_v2(nn.Module):
    def __init__(self, pooling='attentive', mode='equi', stride=16):
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
            
            # Attentive Pooling Components
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
            # x = torch.cat([pos, n, n_xx, n_yy, n_zz, n_xy, n_xz, n_yz], dim=1)
            
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

    def forward(self, p, q, return_feat=False):
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
        
        if return_feat:
            return g_p, g_q, mu_p, mu_q, kappa_p, kappa_q
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
    
class PhysicsAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, query, key_value):
        # query: [B, 1, D] (Physics)
        # key_value: [B, 2, D] (Geometry [P, Q])
        
        # Q가 K, V를 참조하여 필요한 정보를 추출
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out) # Residual Connection

class GravityVelocityDecoder(nn.Module):
    def __init__(self, feat_dim, time_embed_dim=64, hidden_dim=512, encoder_mode='equi'):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.hidden_dim = hidden_dim
        
        # 1. Time Embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # 2. Feature Adapter
        if self.encoder_mode == 'equi':
            input_feat_dim = feat_dim * 3
        else:
            input_feat_dim = feat_dim
            
        self.geom_proj = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 3. Physics Adapter (수정됨!)
        # 기존: Mu(3*2) + Kappa(1*2) + Time
        # 변경: Sampled_G(3*2) + Time
        # Kappa는 더 이상 입력으로 받지 않음 (노이즈로 녹아들어감)
        physics_dim = (3 * 2) + time_embed_dim 
        
        self.physics_proj = nn.Sequential(
            nn.Linear(physics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 4. Cross Attention
        self.cross_attn = PhysicsAttention(hidden_dim, num_heads=8)
        
        # 5. Velocity Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)
        )
        
        # Zero Init (안정적인 학습 시작을 위해 추천)
        nn.init.uniform_(self.head[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, g_p, g_q, g_sample_p, g_sample_q, t):
        # 인자 변경: mu_p, kappa_p -> g_sample_p (샘플링된 중력)
        
        B = g_p.shape[0]
        
        # --- [Step 1] Prepare Geometry ---
        flat_p = g_p.reshape(B, -1)
        flat_q = g_q.reshape(B, -1)
        h_p = self.geom_proj(flat_p)
        h_q = self.geom_proj(flat_q)
        geom_kv = torch.stack([h_p, h_q], dim=1)
        
        # --- [Step 2] Prepare Physics (수정됨!) ---
        t_emb = self.time_embed(t)
        
        # Mu, Kappa 대신 샘플링된 벡터를 사용
        # g_sample: [B, 3] (노이즈가 섞인 중력)
        phys_raw = torch.cat([g_sample_p, g_sample_q, t_emb], dim=1) 
        
        phys_q = self.physics_proj(phys_raw).unsqueeze(1)
        
        # --- [Step 3] Cross Attention ---
        fused_feat = self.cross_attn(query=phys_q, key_value=geom_kv)
        fused_feat = fused_feat.squeeze(1)
        
        # --- [Step 4] Predict ---
        v_pred = self.head(fused_feat)
        
        return v_pred

class GravityFlowAgent(nn.Module):
    def __init__(self, pooling='attentive', mode='normal', stride=4):
        super().__init__()
        self.encoder = PointNet_VN_Gravity_Bayes_v2(pooling=pooling, mode=mode, stride=stride)
        self.decoder = GravityVelocityDecoder(
            feat_dim=self.encoder.feat_dim,
            encoder_mode=self.encoder.mode
        )

    def reparameterize(self, mu, kappa):
        """
        vMF Reparameterization Trick (Gaussian Approximation)
        input: mu (B, 3), kappa (B, 1)
        output: sampled_vector (B, 3)
        """
        # Training 모드일 때만 노이즈 주입
        if self.training:
            # 1. Generate Noise epsilon ~ N(0, I)
            eps = torch.randn_like(mu)
            
            # 2. Scale Noise by Uncertainty (1/sqrt(kappa))
            # kappa가 작으면(불확실하면) 노이즈가 커짐 -> Decoder가 고통받음 -> kappa를 키우게 됨
            # kappa에 1e-6을 더해 0으로 나누는 것 방지
            scaled_noise = eps / torch.sqrt(kappa + 1e-6)
            
            # 3. Add to Mean
            # 접평면 투영 등 복잡한 것보다, 단순히 더하고 정규화하는 게 학습엔 더 효과적임
            z = mu + scaled_noise
            
            # 4. Normalize to unit sphere
            return F.normalize(z, p=2, dim=1)
        else:
            # Test/Validation 시에는 노이즈 없이 평균값(mu) 사용
            return mu

    def forward(self, x, t, context_q):
        # 1. 상태 인지 (Perception)
        # Encoder는 기존과 동일하게 mu, kappa를 뱉음
        feat_p, feat_q, mu_p, mu_q, kappa_p, kappa_q = self.encoder(x, context_q, return_feat=True)
        
        # 2. 샘플링 (핵심!)
        # 여기서 미분 가능한 샘플링 수행
        g_sample_p = self.reparameterize(mu_p, kappa_p)
        g_sample_q = self.reparameterize(mu_q, kappa_q)
        
        # 3. 행동 결정 (Action)
        # Decoder에게는 파라미터 대신 '샘플'을 줌
        v_pred = self.decoder(
            g_p=feat_p, g_q=feat_q,
            g_sample_p=g_sample_p, g_sample_q=g_sample_q, # 변경된 입력
            t=t
        )
        
        # 반환값은 그대로 유지 (Loss 계산에는 원본 mu, kappa가 필요하므로)
        return v_pred, (mu_p, kappa_p), (mu_q, kappa_q)
    
if __name__ == '__main__':
    print("=== Model & Loss Test Start ===\n")

    # 1. Encoder 및 Agent 인스턴스 생성
    # (PointNet_VN_Gravity_Bayes_v2, GravityVelocityDecoder는 이미 정의되어 있다고 가정)
    encoder = PointNet_VN_Gravity_Bayes_v2(mode='normal', pooling='attentive', stride=4)
    agent = GravityFlowAgent(encoder)
    
    from utils.loss import VMFLoss
    # Loss 함수 인스턴스
    vmf_loss_fn = VMFLoss()
    
    # 파라미터 개수 확인
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # 2. Dummy Inputs 생성
    B, N = 4, 1024
    P_t = torch.randn(B, 3, N) # 변형된 Source Point Cloud
    Q = torch.randn(B, 3, N)   # Target Point Cloud
    t = torch.rand(B)          # Random Time [0, 1]
    
    # 3. Dummy Targets (Ground Truth) 생성
    # Flow Matching 정답 속도 (Linear 3 + Angular 3)
    u_gt = torch.randn(B, 6)
    
    # Gravity 정답 벡터 (단위 벡터여야 함)
    g_gt_p = F.normalize(torch.randn(B, 3), p=2, dim=1) # P_t 시점의 정답 중력
    g_gt_q = F.normalize(torch.randn(B, 3), p=2, dim=1) # Q 시점의 정답 중력

    # 4. Forward Pass
    print("\n[Step 1] Forward Pass...")
    # 수정된 Agent는 3가지 튜플을 반환함
    v_pred, (mu_p, kappa_p), (mu_q, kappa_q) = agent(P_t, t, Q)
    
    print(f"  - Velocity Shape : {v_pred.shape}")   # [4, 6]
    print(f"  - Mu P Shape     : {mu_p.shape}")     # [4, 3]
    print(f"  - Kappa P Shape  : {kappa_p.shape}")  # [4, 1]
    
    # 5. Loss Calculation
    print("\n[Step 2] Computing Losses...")
    
    # (A) Flow Loss (MSE)
    loss_flow = F.mse_loss(v_pred, u_gt)
    print(f"  - Flow Loss      : {loss_flow.item():.4f}")
    
    # (B) Gravity Loss (VMFLoss for P and Q)
    loss_g_p = vmf_loss_fn(mu_p, kappa_p, g_gt_p)
    loss_g_q = vmf_loss_fn(mu_q, kappa_q, g_gt_q)
    loss_gravity = (loss_g_p + loss_g_q) * 0.5
    print(f"  - Gravity Loss P : {loss_g_p.item():.4f}")
    print(f"  - Gravity Loss Q : {loss_g_q.item():.4f}")
    
    # (C) Total Loss
    beta = 1.0
    total_loss = loss_flow + beta * loss_gravity
    print(f"  - Total Loss     : {total_loss.item():.4f}")

    # 6. Backward Pass (Gradient Check)
    print("\n[Step 3] Backward Pass...")
    
    # Backward 실행
    try:
        total_loss.backward()
        print("  - Backward execution successful!")
        
        # [수정] net[0] 대신 head[0] 또는 geom_proj[0]의 gradient를 확인
        # agent.decoder.head[0]은 Velocity Head의 첫 번째 Linear Layer입니다.
        grad_norm = agent.decoder.head[0].weight.grad.norm()
        
        print(f"  - Gradient Norm at Decoder Head Layer 0: {grad_norm.item():.4f}")
        
        if grad_norm > 0:
            print("  => SUCCESS: Gradients are flowing correctly.")
        else:
            print("  => WARNING: Gradient is zero.")
            
    except RuntimeError as e:
        print(f"  => FAILURE: Backward pass failed with error: {e}")