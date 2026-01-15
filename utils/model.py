import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.layers import *

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

# -------------------------
# 1) Per-point feature encoder (tiny PointNet-style)
# Input channels: [pos(3), normal outer products(6)] -> 9
# -------------------------
class NormalFeatureEncoder(nn.Module):
    def __init__(self, in_ch=9, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, feat_dim, 1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N)
        return self.net(x)  # (B, feat_dim, N)


# -------------------------
# 2) Overlap-aware pooling
# - subsample by stride to reduce cost
# - compute similarity between P/Q per-point embeddings
# - weight points by best-match confidence (max similarity)
# -------------------------
class OverlapAwarePooling(nn.Module):
    def __init__(self, feat_dim=128, proj_dim=64, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.p_proj = nn.Linear(feat_dim, proj_dim, bias=True)
        self.q_proj = nn.Linear(feat_dim, proj_dim, bias=True)
        self.scale = float(proj_dim) ** 0.5

    def forward(self, f_p: torch.Tensor, f_q: torch.Tensor, stride: int = 16):
        """
        f_p, f_q: (B, C, N)
        returns:
          g_p, g_q: (B, C)
          overlap_conf: (B,)  (a rough confidence score)
        """
        B, C, N = f_p.shape
        if stride < 1:
            stride = 1

        # Subsample
        f_p_s = f_p[:, :, ::stride]  # (B, C, M)
        f_q_s = f_q[:, :, ::stride]  # (B, C, M)
        _, _, M = f_p_s.shape

        # (B, M, C)
        p_pts = f_p_s.permute(0, 2, 1).contiguous()
        q_pts = f_q_s.permute(0, 2, 1).contiguous()

        # Project for similarity (B, M, d)
        p_key = self.p_proj(p_pts)
        q_key = self.q_proj(q_pts)

        # Similarity: (B, M, M)
        sim = torch.matmul(p_key, q_key.transpose(1, 2)) / self.scale

        # Best-match scores
        p_best, _ = sim.max(dim=-1)  # (B, M)
        q_best, _ = sim.max(dim=-2)  # (B, M)

        # Overlap-aware weights (softmax across points)
        w_p = F.softmax(p_best / self.temperature, dim=-1)  # (B, M)
        w_q = F.softmax(q_best / self.temperature, dim=-1)  # (B, M)

        # Weighted global features (B, C)
        g_p = torch.sum(f_p_s * w_p.unsqueeze(1), dim=-1)
        g_q = torch.sum(f_q_s * w_q.unsqueeze(1), dim=-1)

        # Confidence: average of best-match similarity (higher => more overlap evidence)
        overlap_conf = 0.5 * (p_best.mean(dim=-1) + q_best.mean(dim=-1))  # (B,)

        return g_p, g_q, overlap_conf


# -------------------------
# 3) Cross-evidence fusion (light gating)
# - use shared evidence to refine each global feature
# -------------------------
class CrossEvidenceFusion(nn.Module):
    def __init__(self, feat_dim=128, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, feat_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * feat_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, feat_dim),
        )
        self.norm_p = nn.LayerNorm(feat_dim)
        self.norm_q = nn.LayerNorm(feat_dim)

    def forward(self, g_p: torch.Tensor, g_q: torch.Tensor):
        """
        g_p, g_q: (B, C)
        """
        s = self.shared(torch.cat([g_p, g_q], dim=-1))  # (B, C)

        delta_p = self.gate(torch.cat([g_p, g_q, s], dim=-1))
        delta_q = self.gate(torch.cat([g_q, g_p, s], dim=-1))

        g_p_out = self.norm_p(g_p + delta_p)
        g_q_out = self.norm_q(g_q + delta_q)
        return g_p_out, g_q_out, s


# -------------------------
# 4) vMF head
# -------------------------
class VMFHead(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(32, 3)
        self.kappa = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, g: torch.Tensor):
        h = self.trunk(g)
        mu = F.normalize(self.mu(h), p=2, dim=-1)           # (B, 3)
        kappa = self.kappa(h) + 1.0                         # (B, 1) strictly > 1
        return mu, kappa


# -------------------------
# Main model: no VN, <1M params
# -------------------------
class PointNet_Gravity_Bayes_v3(nn.Module):
    """
    - Input: P, Q as (B, 3, N) or (B, 6, N) if normals already included
    - Output: (mu_p, kappa_p), (mu_q, kappa_q)
    - Keeps:
      * NormalEstimator
      * check_correspondence_validity (chi-square test)
    """
    def __init__(self, stride=16, feat_dim=128, proj_dim=64, temperature=0.07):
        super().__init__()
        self.stride = stride
        self.normal_estimator = NormalEstimator(k=30)

        self.encoder = NormalFeatureEncoder(in_ch=9, feat_dim=feat_dim)
        self.pool = OverlapAwarePooling(feat_dim=feat_dim, proj_dim=proj_dim, temperature=temperature)
        self.fusion = CrossEvidenceFusion(feat_dim=feat_dim, hidden=feat_dim)
        self.head = VMFHead(feat_dim=feat_dim)

        self.p_normals = None
        self.q_normals = None

    def extract_feat(self, x: torch.Tensor):
        """
        x: (B, 3, N) or (B, 6, N)  [pos + normal]
        returns:
          f: (B, C, N)
          normals: (B, 3, N)
        """
        B, D, N = x.shape
        if D == 3:
            x = self.normal_estimator(x)  # expected -> (B, 6, N): [pos(3), normal(3)]

        pos = x[:, :3, :]
        n = x[:, 3:, :].contiguous()
        normals = n

        # Normal outer products (6 terms)
        n_xx = n[:, 0:1, :] * n[:, 0:1, :]
        n_yy = n[:, 1:2, :] * n[:, 1:2, :]
        n_zz = n[:, 2:3, :] * n[:, 2:3, :]
        n_xy = n[:, 0:1, :] * n[:, 1:2, :]
        n_xz = n[:, 0:1, :] * n[:, 2:3, :]
        n_yz = n[:, 1:2, :] * n[:, 2:3, :]

        x_in = torch.cat([pos, n_xx, n_yy, n_zz, n_xy, n_xz, n_yz], dim=1)  # (B, 9, N)
        f = self.encoder(x_in)  # (B, feat_dim, N)
        return f, normals

    def forward(self, p: torch.Tensor, q: torch.Tensor, return_feat: bool = False):
        f_p, n_p = self.extract_feat(p)
        f_q, n_q = self.extract_feat(q)

        self.p_normals = n_p
        self.q_normals = n_q

        # Overlap-aware pooling
        g_p, g_q, overlap_conf = self.pool(f_p, f_q, stride=self.stride)

        # Cross evidence fusion (mutual refinement)
        g_p2, g_q2, shared = self.fusion(g_p, g_q)

        # vMF head
        mu_p, kappa_p = self.head(g_p2)
        mu_q, kappa_q = self.head(g_q2)

        if return_feat:
            # return global features too (B, C), plus overlap confidence
            return g_p2, g_q2, mu_p, mu_q, kappa_p, kappa_q, overlap_conf
        return (mu_p, kappa_p), (mu_q, kappa_q)

    # --- Chi-square test (kept, same interface) ---
    def check_correspondence_validity(
        self,
        batch_idx,
        P_indices,
        Q_indices,
        g_p, kappa_p,
        g_q, kappa_q,
        chi2_thresh=9.0
    ):
        if self.p_normals is None or self.q_normals is None:
            return np.ones(len(P_indices), dtype=bool), np.zeros(len(P_indices))

        curr_p_normals = self.p_normals[batch_idx].transpose(0, 1)  # (N,3)
        curr_q_normals = self.q_normals[batch_idx].transpose(0, 1)  # (N,3)
        dev = curr_p_normals.device

        if not isinstance(P_indices, torch.Tensor):
            P_indices = torch.tensor(P_indices, device=dev)
        if not isinstance(Q_indices, torch.Tensor):
            Q_indices = torch.tensor(Q_indices, device=dev)

        n_p = curr_p_normals[P_indices]  # (K,3)
        n_q = curr_q_normals[Q_indices]  # (K,3)

        # g_p, g_q should be (3,) for this batch (e.g., mu_p[batch_idx])
        I_p = torch.matmul(n_p, g_p)
        I_q = torch.matmul(n_q, g_q)

        sin2_p = torch.clamp(1.0 - I_p**2, min=1e-6)
        sin2_q = torch.clamp(1.0 - I_q**2, min=1e-6)

        base_variance = 0.01
        term_p = sin2_p / (kappa_p + 1e-6)
        term_q = sin2_q / (kappa_q + 1e-6)

        sigma_sq_total = term_p + term_q + base_variance
        diff = torch.abs(I_p) - torch.abs(I_q)

        M2_score = (diff**2) / sigma_sq_total
        return (M2_score < chi2_thresh).cpu().numpy(), M2_score.cpu().numpy()


if __name__ == "__main__":

    # Forward Test
    
    model = PointNet_VN_Gravity_Bayes_v2(pooling='max', mode='normal', stride=1)
    model2 = PointNet_Gravity_Bayes_v3(stride=1)
    model.eval()
    model2.eval()
    
    B = 2; N = 1024
    P = torch.rand(B, 3, N)
    Q = torch.rand(B, 3, N)
    g_p = torch.rand(B, 3)
    g_q = torch.rand(B, 3)
    
    (mu_p, kappa_p), (mu_q, kappa_q) = model(P, Q)
    (mu_p2, kappa_p2), (mu_q2, kappa_q2) = model2(P, Q)
    print("P Mu:", mu_p.shape, "Kappa:", kappa_p.shape)
    print("Q Mu:", mu_q.shape, "Kappa:", kappa_q.shape)
    print("P2 Mu:", mu_p2.shape, "Kappa:", kappa_p2.shape)
    print("Q2 Mu:", mu_q2.shape, "Kappa:", kappa_q2.shape)
    
    # Backward Test
    from utils.loss import VMFLoss
    loss_fn = VMFLoss()
    loss = loss_fn(mu_p, kappa_p, g_p)
    loss += loss_fn(mu_q, kappa_q, g_q)
    loss.backward()
    print("Loss:", loss.item())
    print("Backward pass successful.")
    
    loss2 = loss_fn(mu_p2, kappa_p2, g_p)
    loss2 += loss_fn(mu_q2, kappa_q2, g_q)
    loss2.backward()
    print("Loss2:", loss2.item())
    print("Backward pass 2 successful.")
    
    