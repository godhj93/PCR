import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import VNLinearLeakyReLU, VNLinear, VNInvariant, VN_Attention, VN_Cross_Gating, VNMaxPool, NormalEstimator
import numpy as np

# ==============================================================================
# [Common] Utility Modules
# ==============================================================================
class TimeEmbedding(nn.Module):
    """
    Scalar 시간 t를 고차원 벡터로 변환 (Gaussian Fourier Projection)
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, t):
        t = t.view(-1, 1) if t.dim() == 1 else t
        t_proj = t * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

# ==============================================================================
# [Legacy] PointNet Encoder & MLP Decoder
# ==============================================================================
class PointNet_VN_Gravity_Bayes_v2(nn.Module):
    def __init__(self, pooling='attentive', mode='equi', stride=16):
        super(PointNet_VN_Gravity_Bayes_v2, self).__init__()
        self.pooling = pooling
        self.mode = mode
        self.feat_dim = 1024
        self.stride = stride
        self.normal_estimator = NormalEstimator(k=30)
        self.p_normals = None; self.q_normals = None
        
        if self.mode == 'equi':
            vn_in_channels = 2
            self.vn_fc1 = VNLinearLeakyReLU(vn_in_channels, 64, dim=4, negative_slope=0.2)
            self.vn_fc2 = VNLinearLeakyReLU(64, 64, dim=4, negative_slope=0.2)
            self.vn_fc3 = VNLinearLeakyReLU(64, self.feat_dim, dim=4, negative_slope=0.2)
            self.vn_self_attn = VN_Attention(self.feat_dim)
            self.vn_cross_gating = VN_Cross_Gating(self.feat_dim)
            if self.pooling == 'attentive':
                raise ValueError("Attentive pooling for 'equi' mode is not implemented.")
            elif self.pooling == 'max':
                self.vn_pool = VNMaxPool(self.feat_dim)
            self.vn_invariant = VNInvariant(self.feat_dim)
            self.vn_mu_head = nn.Sequential(VNLinearLeakyReLU(self.feat_dim, self.feat_dim, dim=3, negative_slope=0.0), VNLinear(self.feat_dim, 1))
            self.kappa_mlp = nn.Sequential(nn.Linear(self.feat_dim * 3, 128), nn.LeakyReLU(0.2), nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Softplus())

        elif self.mode == 'normal':
            std_in_channels = 9
            self.std_fc1 = nn.Sequential(nn.Conv1d(std_in_channels, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc3 = nn.Sequential(nn.Conv1d(64, self.feat_dim, 1), nn.BatchNorm1d(self.feat_dim), nn.LeakyReLU(0.2))
            self.std_self_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)
            self.std_cross_attn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=4, batch_first=True)
            if self.pooling == 'attentive':
                self.attention_mlp = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
            self.std_mu_head = nn.Sequential(nn.Linear(self.feat_dim, 512), nn.LeakyReLU(0.2), nn.Linear(512, 3))
            self.kappa_mlp = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Softplus())

    def extract_feat(self, x):
        B, D, N = x.shape
        if D == 3: x = self.normal_estimator(x) 
        normals = x[:, 3:, :].contiguous() 
        if self.mode == 'normal':
            pos = x[:, :3, :]; n = x[:, 3:, :]   
            n_xx = n[:, 0:1, :] * n[:, 0:1, :]; n_yy = n[:, 1:2, :] * n[:, 1:2, :]; n_zz = n[:, 2:3, :] * n[:, 2:3, :]
            n_xy = n[:, 0:1, :] * n[:, 1:2, :]; n_xz = n[:, 0:1, :] * n[:, 2:3, :]; n_yz = n[:, 1:2, :] * n[:, 2:3, :]
            x = torch.cat([pos, n_xx, n_yy, n_zz, n_xy, n_xz, n_yz], dim=1) 
            x = self.std_fc1(x); x = self.std_fc2(x); x = self.std_fc3(x) 
        elif self.mode == 'equi':
            x = x.view(B, 2, 3, N); x = self.vn_fc1(x); x = self.vn_fc2(x); x = self.vn_fc3(x) 
        return x, normals

    def process_interaction(self, f_p, f_q):
        if self.mode == 'equi':
            f_p = self.vn_self_attn(f_p); f_q = self.vn_self_attn(f_q)
            f_p_out = self.vn_cross_gating(x=f_p, y=f_q); f_q_out = self.vn_cross_gating(x=f_q, y=f_p)
        elif self.mode == 'normal':
            p_in = f_p.permute(0, 2, 1); q_in = f_q.permute(0, 2, 1)
            p_self, _ = self.std_self_attn(p_in, p_in, p_in); q_self, _ = self.std_self_attn(q_in, q_in, q_in)
            p_in = p_in + p_self; q_in = q_in + q_self
            p_cross, _ = self.std_cross_attn(query=p_in, key=q_in, value=q_in); q_cross, _ = self.std_cross_attn(query=q_in, key=p_in, value=p_in)
            p_out = p_in + p_cross; q_out = q_in + q_cross
            f_p_out = p_out.permute(0, 2, 1); f_q_out = q_out.permute(0, 2, 1)
        return f_p_out, f_q_out

    def apply_pooling(self, feat):
        if self.pooling == 'attentive':
            if self.mode == 'equi':
                inv_feat = self.vn_invariant_layer(feat) 
                scores = self.attention_mlp(inv_feat.permute(0, 2, 1))
                weights = F.softmax(scores, dim=1).permute(0, 2, 1).unsqueeze(1)
                global_feat = torch.sum(feat * weights, dim=-1)
            elif self.mode == 'normal':
                scores = self.attention_mlp(feat.permute(0, 2, 1))
                weights = F.softmax(scores, dim=1).permute(0, 2, 1)
                global_feat = torch.sum(feat * weights, dim=-1)
            return global_feat
        elif self.pooling == 'mean': return torch.mean(feat, dim=-1)
        elif self.pooling == 'max': return self.vn_pool(feat) if self.mode == 'equi' else torch.max(feat, dim=-1)[0]

    def forward_head(self, g_feat):
        if self.mode == 'equi':
            mu = self.vn_mu_head(g_feat).mean(dim=1); mu = F.normalize(mu, p=2, dim=1)
            g_inv = self.vn_invariant(g_feat); kappa = self.kappa_mlp(g_inv) + 1.0
        elif self.mode == 'normal':
            mu = self.std_mu_head(g_feat); mu = F.normalize(mu, p=2, dim=1)
            kappa = self.kappa_mlp(g_feat) + 1.0
        return mu, kappa

    def forward(self, p, q, return_feat=False):
        f_p, n_p = self.extract_feat(p); f_q, n_q = self.extract_feat(q)
        self.p_normals = n_p; self.q_normals = n_q
        stride = self.stride 
        if self.mode == 'equi': f_p_small = f_p[:, :, :, ::stride]; f_q_small = f_q[:, :, :, ::stride]
        else: f_p_small = f_p[:, :, ::stride]; f_q_small = f_q[:, :, ::stride]
        
        f_p_gated, f_q_gated = self.process_interaction(f_p_small, f_q_small)
        g_p = self.apply_pooling(f_p_gated); g_q = self.apply_pooling(f_q_gated)
        mu_p, kappa_p = self.forward_head(g_p); mu_q, kappa_q = self.forward_head(g_q)
        
        # [Interface Compatibility]
        # Legacy model returns Global Features (B, C)
        if return_feat: return g_p, g_q, mu_p, mu_q, kappa_p, kappa_q
        return (mu_p, kappa_p), (mu_q, kappa_q)

    def check_correspondence_validity(self, batch_idx, P_indices, Q_indices, g_p, kappa_p, g_q, kappa_q, chi2_thresh=9.0):
        # Implementation identical to previous version
        if self.p_normals is None or self.q_normals is None: return np.ones(len(P_indices), dtype=bool), np.zeros(len(P_indices))
        curr_p_normals = self.p_normals[batch_idx].transpose(0, 1); curr_q_normals = self.q_normals[batch_idx].transpose(0, 1) 
        dev = curr_p_normals.device
        if not isinstance(P_indices, torch.Tensor): P_indices = torch.tensor(P_indices, device=dev)
        if not isinstance(Q_indices, torch.Tensor): Q_indices = torch.tensor(Q_indices, device=dev)
        n_p = curr_p_normals[P_indices]; n_q = curr_q_normals[Q_indices] 
        I_p = torch.matmul(n_p, g_p); I_q = torch.matmul(n_q, g_q)
        sin2_p = torch.clamp(1.0 - I_p**2, min=1e-6); sin2_q = torch.clamp(1.0 - I_q**2, min=1e-6)
        base_variance = 0.01 
        term_p = sin2_p / (kappa_p + 1e-6); term_q = sin2_q / (kappa_q + 1e-6)
        sigma_sq_total = term_p + term_q + base_variance
        diff = torch.abs(I_p) - torch.abs(I_q)
        M2_score = (diff**2) / sigma_sq_total
        return (M2_score < chi2_thresh).cpu().numpy(), M2_score.cpu().numpy()

class PhysicsAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, query, key_value):
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_out)

class GravityVelocityDecoder(nn.Module):
    def __init__(self, feat_dim, time_embed_dim=64, hidden_dim=512, encoder_mode='equi'):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.hidden_dim = hidden_dim
        self.time_embed = TimeEmbedding(time_embed_dim)
        input_feat_dim = feat_dim * 3 if self.encoder_mode == 'equi' else feat_dim
        self.geom_proj = nn.Sequential(nn.Linear(input_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        physics_dim = (3 * 2) + time_embed_dim 
        self.physics_proj = nn.Sequential(nn.Linear(physics_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.cross_attn = PhysicsAttention(hidden_dim, num_heads=8)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6))
        nn.init.uniform_(self.head[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.head[-1].bias, 0)

    # [FIXED] Arguments renamed to feat_p, feat_q to match Agent interface
    def forward(self, feat_p, feat_q, g_sample_p, g_sample_q, t):
        # Takes Global Features (B, C) from Legacy Encoder
        # Alias feat_p -> g_p for clarity within the method
        g_p = feat_p
        g_q = feat_q

        B = g_p.shape[0]
        flat_p = g_p.reshape(B, -1); flat_q = g_q.reshape(B, -1)
        h_p = self.geom_proj(flat_p); h_q = self.geom_proj(flat_q)
        geom_kv = torch.stack([h_p, h_q], dim=1) # (B, 2, H)
        
        t_emb = self.time_embed(t)
        phys_raw = torch.cat([g_sample_p, g_sample_q, t_emb], dim=1) 
        phys_q = self.physics_proj(phys_raw).unsqueeze(1) # (B, 1, H)
        
        fused_feat = self.cross_attn(query=phys_q, key_value=geom_kv).squeeze(1)
        v_pred = self.head(fused_feat)
        return v_pred


# ==============================================================================
# [New] Transformer Encoder & Decoder
# ==============================================================================

class LocalPointNetBackbone(nn.Module):
    """
    Backbone only: Extracts features without pooling/heads.
    """
    def __init__(self, feat_dim=512, mode='normal'):
        super().__init__()
        self.mode = mode
        self.feat_dim = feat_dim
        self.normal_estimator = NormalEstimator(k=30)
        
        if self.mode == 'equi':
            self.vn_fc1 = VNLinearLeakyReLU(2, 64, dim=4, negative_slope=0.2)
            self.vn_fc2 = VNLinearLeakyReLU(64, 64, dim=4, negative_slope=0.2)
            self.vn_fc3 = VNLinearLeakyReLU(64, self.feat_dim, dim=4, negative_slope=0.2)
        elif self.mode == 'normal':
            std_in_channels = 9
            self.std_fc1 = nn.Sequential(nn.Conv1d(std_in_channels, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            self.std_fc3 = nn.Sequential(nn.Conv1d(64, self.feat_dim, 1), nn.BatchNorm1d(self.feat_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        B, D, N = x.shape
        if D == 3: x = self.normal_estimator(x) 
        normals = x[:, 3:, :].contiguous()
        
        if self.mode == 'normal':
            pos = x[:, :3, :]; n = x[:, 3:, :]
            n_xx = n[:, 0:1, :] * n[:, 0:1, :]; n_yy = n[:, 1:2, :] * n[:, 1:2, :]; n_zz = n[:, 2:3, :] * n[:, 2:3, :]
            n_xy = n[:, 0:1, :] * n[:, 1:2, :]; n_xz = n[:, 0:1, :] * n[:, 2:3, :]; n_yz = n[:, 1:2, :] * n[:, 2:3, :]
            x = torch.cat([pos, n_xx, n_yy, n_zz, n_xy, n_xz, n_yz], dim=1)
            x = self.std_fc1(x); x = self.std_fc2(x); out = self.std_fc3(x)
        elif self.mode == 'equi':
            x = x.view(B, 2, 3, N)
            x = self.vn_fc1(x); x = self.vn_fc2(x); out = self.vn_fc3(x)
        return out, normals

class GravityTransformerEncoder(nn.Module):
    """
    [PointNet] -> [Concat P, Q] -> [Transformer Encoder] -> [Split] -> [Gravity Head]
    Returns Sequence Features (B, N, C) for Decoder.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, num_layers=3, num_heads=4, mode='normal', stride=4, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.stride = stride
        self.feat_dim = hidden_dim # Interface dim for Decoder
        
        # 1. Local Feature Extractor
        self.backbone = LocalPointNetBackbone(feat_dim=hidden_dim, mode=mode)
        
        # 2. Transformer Encoder (Mixing P and Q)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, norm_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Gravity Prediction Head
        self.gravity_pool = nn.AdaptiveAvgPool1d(1) 
        
        if self.mode == 'equi':
            self.vn_invariant = VNInvariant(hidden_dim)
            self.mu_head = nn.Sequential(VNLinearLeakyReLU(hidden_dim, hidden_dim, dim=3, negative_slope=0.0), VNLinear(hidden_dim, 1))
            self.kappa_mlp = nn.Sequential(nn.Linear(hidden_dim * 3, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Softplus())
        else:
            self.mu_head = nn.Sequential(nn.Linear(hidden_dim, 256), nn.LeakyReLU(0.2), nn.Linear(256, 3))
            self.kappa_mlp = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Softplus())
        
        self.p_normals = None; self.q_normals = None

    def forward(self, p, q, return_feat=False):
        # 1. Extract Local Features (B, C, N)
        f_p, n_p = self.backbone(p); f_q, n_q = self.backbone(q)
        self.p_normals = n_p; self.q_normals = n_q
        
        if self.stride > 1:
            if self.mode == 'equi': f_p = f_p[:, :, :, ::self.stride]; f_q = f_q[:, :, :, ::self.stride]
            else: f_p = f_p[:, :, ::self.stride]; f_q = f_q[:, :, ::self.stride]

        # 2. Early Fusion
        B, C, N_p = f_p.shape if self.mode != 'equi' else (f_p.shape[0], f_p.shape[1], f_p.shape[3])
        
        if self.mode == 'equi':
            pass 
        
        # (B, C, N) -> (B, N, C) for Transformer
        h_p = f_p.permute(0, 2, 1); h_q = f_q.permute(0, 2, 1)
        joint_feat = torch.cat([h_p, h_q], dim=1) # (B, 2N, C)
        
        refined_feat = self.transformer(joint_feat) 
        h_p_out = refined_feat[:, :N_p, :]; h_q_out = refined_feat[:, N_p:, :]
        
        # 3. Predict Gravity
        # (B, N, C) -> (B, C, N) -> Pool
        g_p_vec = self.gravity_pool(h_p_out.permute(0, 2, 1)).squeeze(-1)
        g_q_vec = self.gravity_pool(h_q_out.permute(0, 2, 1)).squeeze(-1)
        
        mu_p, kappa_p = self._predict_head(g_p_vec)
        mu_q, kappa_q = self._predict_head(g_q_vec)
        
        # [Interface Compatibility]
        # Returns Sequence Features (B, N, C)
        if return_feat: return h_p_out, h_q_out, mu_p, mu_q, kappa_p, kappa_q
        return (mu_p, kappa_p), (mu_q, kappa_q)
    
    def _predict_head(self, feat):
        if self.mode == 'equi':
            mu = self.mu_head(feat).mean(dim=1); mu = F.normalize(mu, p=2, dim=1)
            g_inv = self.vn_invariant(feat); kappa = self.kappa_mlp(g_inv) + 1.0
        else:
            mu = self.mu_head(feat); mu = F.normalize(mu, p=2, dim=1)
            kappa = self.kappa_mlp(feat) + 1.0
        return mu, kappa

    def check_correspondence_validity(self, batch_idx, P_indices, Q_indices, g_p, kappa_p, g_q, kappa_q, chi2_thresh=9.0):
        # Implementation identical to PointNet logic
        if self.p_normals is None or self.q_normals is None: return np.ones(len(P_indices), dtype=bool), np.zeros(len(P_indices))
        curr_p_normals = self.p_normals[batch_idx].transpose(0, 1); curr_q_normals = self.q_normals[batch_idx].transpose(0, 1) 
        dev = curr_p_normals.device
        if not isinstance(P_indices, torch.Tensor): P_indices = torch.tensor(P_indices, device=dev)
        if not isinstance(Q_indices, torch.Tensor): Q_indices = torch.tensor(Q_indices, device=dev)
        n_p = curr_p_normals[P_indices]; n_q = curr_q_normals[Q_indices] 
        I_p = torch.matmul(n_p, g_p); I_q = torch.matmul(n_q, g_q)
        sin2_p = torch.clamp(1.0 - I_p**2, min=1e-6); sin2_q = torch.clamp(1.0 - I_q**2, min=1e-6)
        base_variance = 0.01 
        term_p = sin2_p / (kappa_p + 1e-6); term_q = sin2_q / (kappa_q + 1e-6)
        sigma_sq_total = term_p + term_q + base_variance
        diff = torch.abs(I_p) - torch.abs(I_q)
        M2_score = (diff**2) / sigma_sq_total
        return (M2_score < chi2_thresh).cpu().numpy(), M2_score.cpu().numpy()

class GravityTransformerDecoder(nn.Module):
    """
    Standard Transformer Decoder.
    Query: Physics (Gravity + Time)
    Key/Value: Geometry (Sequence Features from Encoder)
    """
    def __init__(self, feat_dim, time_embed_dim=64, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Encoder Feature Projection
        self.geom_proj = nn.Linear(feat_dim, hidden_dim)
        
        # Physics Projection (Query)
        physics_input_dim = 3 + 3 + time_embed_dim
        self.physics_proj = nn.Linear(physics_input_dim, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, norm_first=True, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 6))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        nn.init.constant_(self.head[-1].weight, 0)
        nn.init.constant_(self.head[-1].bias, 0)

    def forward(self, feat_p, feat_q, g_sample_p, g_sample_q, t):
        # feat_p, feat_q: (B, N, C) - Sequence Features
        B = feat_p.shape[0]
        
        h_p = self.geom_proj(feat_p); h_q = self.geom_proj(feat_q)
        memory = torch.cat([h_p, h_q], dim=1) # (B, 2N, H)
        
        t_emb = self.time_embed(t)
        physics_raw = torch.cat([g_sample_p, g_sample_q, t_emb], dim=1)
        tgt = self.physics_proj(physics_raw).unsqueeze(1) # Query (B, 1, H)
        
        out = self.transformer_decoder(tgt=tgt, memory=memory) # (B, 1, H)
        out = self.norm(out).squeeze(1) 
        v_pred = self.head(out)
        return v_pred

# ==============================================================================
# [Agent] Generic Gravity Flow Agent (Dependency Injection)
# ==============================================================================
class GravityFlowAgent(nn.Module):
    """
    Agent accepts ANY encoder and decoder that satisfies the interface.
    Interface:
      Encoder returns: (feat_p, feat_q, mu_p, mu_q, kappa_p, kappa_q)
      Decoder accepts: (feat_p, feat_q, g_sample_p, g_sample_q, t)
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, kappa):
        """
        vMF Reparameterization Trick
        """
        if self.training:
            eps = torch.randn_like(mu)
            scaled_noise = eps / torch.sqrt(kappa + 1e-6)
            z = mu + scaled_noise
            return F.normalize(z, p=2, dim=1)
        else:
            return mu

    def forward(self, x, t, context_q):
        # 1. Perception
        # Works with both PointNet (Global Feat) and Transformer (Seq Feat)
        # as long as the paired Decoder knows how to handle it.
        feat_p, feat_q, mu_p, mu_q, kappa_p, kappa_q = self.encoder(
            x, context_q, return_feat=True
        )
        
        # 2. Sampling
        g_sample_p = self.reparameterize(mu_p, kappa_p)
        g_sample_q = self.reparameterize(mu_q, kappa_q)
        
        # 3. Action
        v_pred = self.decoder(
            feat_p=feat_p, 
            feat_q=feat_q, # Passed directly to decoder (Duck Typing)
            g_sample_p=g_sample_p, 
            g_sample_q=g_sample_q, 
            t=t
        )
        
        return v_pred, (mu_p, kappa_p), (mu_q, kappa_q)

# ==============================================================================
# Main Test Block
# ==============================================================================
if __name__ == '__main__':
    print("=== Modular Agent Test ===\n")

    # --- Option 1: Legacy Configuration (PointNet + MLP) ---
    print("[Config 1] Legacy: PointNet + MLP Decoder")
    legacy_enc = PointNet_VN_Gravity_Bayes_v2(mode='normal', pooling='attentive', stride=4)
    legacy_dec = GravityVelocityDecoder(feat_dim=1024, encoder_mode='normal')
    
    agent_legacy = GravityFlowAgent(encoder=legacy_enc, decoder=legacy_dec)
    
    # Test Legacy
    B, N = 4, 1024
    P = torch.randn(B, 3, N); Q = torch.randn(B, 3, N); t = torch.rand(B)
    v, _, _ = agent_legacy(P, t, Q)
    print(f"  Legacy Output Shape: {v.shape}") # Should be (4, 6)

    print("-" * 30)

    # --- Option 2: New Configuration (Transformer Encoder + Transformer Decoder) ---
    print("[Config 2] New: Transformer Encoder + Transformer Decoder")
    
    # 1. Define Transformer Encoder
    # Note: hidden_dim is the feature dim exchanged between encoder and decoder
    trans_enc = GravityTransformerEncoder(
        input_dim=1024, # PointNet Local Feat Dim
        hidden_dim=256, 
        num_layers=3, 
        num_heads=4,
        mode='normal', 
        stride=4
    )
    
    # 2. Define Transformer Decoder
    trans_dec = GravityTransformerDecoder(
        feat_dim=256, # Must match Encoder's hidden_dim
        hidden_dim=256,
        num_layers=4,
        num_heads=8
    )
    
    # 3. Inject into Agent
    agent_new = GravityFlowAgent(encoder=trans_enc, decoder=trans_dec)
    
    # Test New
    v_new, (mu_p, k_p), (mu_q, k_q) = agent_new(P, t, Q)
    
    print(f"  New Output Shape: {v_new.shape}")
    print(f"  Mu P Shape: {mu_p.shape}, Kappa P Shape: {k_p.shape}")
    
    # Check Gradients
    loss = v_new.sum() + mu_p.sum()
    loss.backward()
    print("  Backward Pass Successful!")
    
    total_params = sum(p.numel() for p in agent_new.parameters() if p.requires_grad)
    print(f"  Total Params (New): {total_params:,}")