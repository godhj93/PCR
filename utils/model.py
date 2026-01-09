#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.transform import Rotation
from utils.layers import NormalEstimator

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = x.device  # Use the same device as input tensor

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class PointNet(nn.Module):
    def __init__(self, emb_dims=512, normal=True):
        super(PointNet, self).__init__()
        if normal:
            self.normal_estimator = NormalEstimator()
            self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        else:
            self.normal_estimator = None
            self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        normals = None
        if self.normal_estimator is not None:
            x = self.normal_estimator(x)
            normals = x[:, 3:, :]
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        return x, normals


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        self.U = torch.stack(U, dim=0)
        self.V = torch.stack(V, dim=0)
        self.S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        
        return R, t.view(batch_size, 3)    
  
class GravityLayer(nn.Module):
    def __init__(self, emb_dims):
        super(GravityLayer, self).__init__()
        # 정보이론 관점: P와 Q의 전역 특징을 결합하여 상관관계를 학습
        self.joint_nn = nn.Sequential(
            nn.Linear(emb_dims * 2, emb_dims),
            nn.BatchNorm1d(emb_dims),
            nn.ReLU(),
            nn.Linear(emb_dims, emb_dims // 2),
            nn.ReLU(),
            nn.Linear(emb_dims // 2, 8) # (mu_p, k_p, mu_q, k_q)
        )

    def forward(self, src_emb, tgt_emb):
        # Global Feature (B, C)
        src_g = torch.max(src_emb, dim=-1)[0]
        tgt_g = torch.max(tgt_emb, dim=-1)[0]
        
        # Joint Inference
        combined = torch.cat([src_g, tgt_g], dim=1)
        out = self.joint_nn(combined) # (B, 8)

        # (g_p, k_p), (g_q, k_q) 형태로 반환
        g_p = F.normalize(out[:, 0:3], p=2, dim=1)
        k_p = F.softplus(out[:, 3:4]) + 1.0
        g_q = F.normalize(out[:, 4:7], p=2, dim=1)
        k_q = F.softplus(out[:, 7:8]) + 1.0

        return (g_p, k_p), (g_q, k_q)

class GravityAligner(nn.Module):
    """
    Compute rotation R such that R @ g_src = g_tgt (both unit vectors).
    Batch-safe, handles parallel / anti-parallel cases.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, g_src: torch.Tensor, g_tgt: torch.Tensor) -> torch.Tensor:
        # g_src, g_tgt: (B, 3) assumed normalized (still normalize for safety)
        u = F.normalize(g_src, dim=1, eps=self.eps)
        v = F.normalize(g_tgt, dim=1, eps=self.eps)

        # axis = u x v
        axis = torch.cross(u, v, dim=1)                          # (B,3)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)        # (B,1)
        dot = torch.sum(u * v, dim=1, keepdim=True).clamp(-1.0, 1.0)  # (B,1)

        # If parallel: axis ~ 0
        parallel = (axis_norm < 1e-6)

        # Rodrigues: R = I + sinθ K + (1-cosθ) K^2, where K = skew(k), k = axis/||axis||
        k = axis / (axis_norm + self.eps)                        # (B,3)
        theta = torch.acos(dot)                                   # (B,1)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        K = self._skew(k)                                         # (B,3,3)
        I = torch.eye(3, device=g_src.device, dtype=g_src.dtype).unsqueeze(0)  # (1,3,3)

        R = I + sin_t.view(-1,1,1) * K + (1 - cos_t).view(-1,1,1) * (K @ K)

        # Anti-parallel special handling: dot ~ -1 & axis ~ 0  => pick any orthogonal axis
        antipar = parallel & (dot < 0.0)
        if antipar.any():
            # choose a basis vector not parallel to u: use e_x unless u close to e_x, else e_y
            ex = torch.tensor([1.0, 0.0, 0.0], device=u.device, dtype=u.dtype).view(1,3).repeat(u.size(0),1)
            ey = torch.tensor([0.0, 1.0, 0.0], device=u.device, dtype=u.dtype).view(1,3).repeat(u.size(0),1)

            use_ex = (torch.abs(u[:,0:1]) < 0.9)  # if u not too aligned with x
            basis = torch.where(use_ex, ex, ey)
            axis2 = torch.cross(u, basis, dim=1)
            axis2 = F.normalize(axis2, dim=1, eps=self.eps)
            K2 = self._skew(axis2)
            # 180deg rotation: R = I + 2 K^2 (since sinπ=0, 1-cosπ = 2)
            R_anti = I + 2.0 * (K2 @ K2)
            R = torch.where(antipar.view(-1,1,1), R_anti, R)

        # Pure parallel (dot>0): identity
        R = torch.where(parallel.view(-1,1,1) & (dot > 0.0).view(-1,1,1), I, R)
        return R

    @staticmethod
    def _skew(k: torch.Tensor) -> torch.Tensor:
        # k: (B,3)
        B = k.size(0)
        kx, ky, kz = k[:,0], k[:,1], k[:,2]
        O = torch.zeros(B, device=k.device, dtype=k.dtype)
        K = torch.stack([
            torch.stack([O, -kz,  ky], dim=1),
            torch.stack([kz,  O, -kx], dim=1),
            torch.stack([-ky, kx,  O], dim=1)
        ], dim=1)
        return K


class GravityHypothesisTester(nn.Module):
    """
    test.py의 로직을 DCP 내부에서 쓰기 위한 모듈:
    - gravity align (P->Q frame)
    - centroid shift
    - NN correspondence
    - distance gate + kappa/chi2 inclination gate
    Returns: w_p, w_q, and optionally indices/dist for debugging.
    """
    def __init__(self, chi2_thresh=9.0, dist_scale=3.0, eps=1e-6):
        super().__init__()
        self.chi2_thresh = chi2_thresh
        self.dist_scale = dist_scale
        self.eps = eps
        self.aligner = GravityAligner(eps=eps)

    def forward(self, src, tgt, src_n, tgt_n, g_p, k_p, g_q, k_q, return_debug=False):
        """
        src,tgt: (B,3,N)
        src_n,tgt_n: (B,3,N)
        g_p,g_q: (B,3)
        k_p,k_q: (B,1)
        """
        B, _, N = src.shape

        # 1) gravity align: bring src(+normal) into tgt frame
        R_g = self.aligner(g_p, g_q)                    # (B,3,3)
        src_rot = torch.matmul(R_g, src)                # (B,3,N)
        src_n_rot = torch.matmul(R_g, src_n)            # (B,3,N)

        # 2) centroid shift (same as test.py)
        t_center = tgt.mean(dim=2, keepdim=True) - src_rot.mean(dim=2, keepdim=True)   # (B,3,1)
        src_init = src_rot + t_center

        # 3) NN in aligned frame: dist matrix between src_init and tgt
        dist_pq = self._get_dist_mat(src_init, tgt)     # (B,N,N)
        min_pq, corr_p2q = torch.min(dist_pq, dim=2)    # (B,N)
        min_qp, corr_q2p = torch.min(dist_pq.transpose(1,2), dim=2)  # (B,N)

        # 4) distance gate (tau = dist_scale * median(nn_dist))
        # use sqrt because dist_mat is squared distance
        nn_d_p = torch.sqrt(min_pq.clamp_min(0.0) + self.eps)        # (B,N)
        nn_d_q = torch.sqrt(min_qp.clamp_min(0.0) + self.eps)        # (B,N)

        tau_p = self.dist_scale * nn_d_p.median(dim=1, keepdim=True).values  # (B,1)
        tau_q = self.dist_scale * nn_d_q.median(dim=1, keepdim=True).values  # (B,1)

        geom_p = (nn_d_p <= tau_p).float()   # (B,N)
        geom_q = (nn_d_q <= tau_q).float()   # (B,N)

        # 5) gather matched normals
        tgt_n_matched_for_p = self._gather_by_index(tgt_n, corr_p2q)      # (B,3,N)
        src_n_matched_for_q = self._gather_by_index(src_n_rot, corr_q2p)  # (B,3,N)

        # 6) inclination (frame-consistent: 모두 tgt frame의 g_q로 dot)
        # inc_p: (B,N), inc_p_ref: (B,N)
        inc_p = torch.sum(src_n_rot * g_q.unsqueeze(2), dim=1)
        inc_p_ref = torch.sum(tgt_n_matched_for_p * g_q.unsqueeze(2), dim=1)

        inc_q = torch.sum(tgt_n * g_q.unsqueeze(2), dim=1)
        inc_q_ref = torch.sum(src_n_matched_for_q * g_q.unsqueeze(2), dim=1)

        # 7) chi2/kappa gate
        k_eff = (k_p * k_q) / (k_p + k_q + self.eps)    # (B,1)
        # broadcast k_eff to (B,N)
        k_eff = k_eff.expand(-1, N)

        w_p = torch.sigmoid(self.chi2_thresh - k_eff * (inc_p - inc_p_ref).pow(2)) * geom_p
        w_q = torch.sigmoid(self.chi2_thresh - k_eff * (inc_q - inc_q_ref).pow(2)) * geom_q

        w_p = w_p.unsqueeze(1)  # (B,1,N)
        w_q = w_q.unsqueeze(1)  # (B,1,N)

        if return_debug:
            debug = {
                "R_g": R_g,
                "t_center": t_center,
                "corr_p2q": corr_p2q,
                "corr_q2p": corr_q2p,
                "nn_d_p_median": nn_d_p.median(dim=1).values,
                "nn_d_q_median": nn_d_q.median(dim=1).values,
                "tau_p": tau_p.squeeze(1),
                "tau_q": tau_q.squeeze(1),
                "w_p_mean": w_p.mean(dim=2).squeeze(1),
                "w_q_mean": w_q.mean(dim=2).squeeze(1),
            }
            return w_p, w_q, debug

        return w_p, w_q

    @staticmethod
    def _get_dist_mat(src, tgt):
        # src,tgt: (B,3,N) -> dist: (B,N,N), squared Euclidean
        inner = -2 * torch.matmul(src.transpose(2, 1), tgt)  # (B,N,N)
        xx = torch.sum(src**2, dim=1, keepdim=True).transpose(2, 1)  # (B,N,1)
        yy = torch.sum(tgt**2, dim=1, keepdim=True)                  # (B,1,N)
        return xx + inner + yy

    @staticmethod
    def _gather_by_index(x, idx):
        """
        x: (B,3,Nx), idx: (B,N) in [0, Nx)
        return: (B,3,N)
        """
        B, C, Nx = x.shape
        N = idx.size(1)
        idx_exp = idx.unsqueeze(1).expand(-1, C, -1)  # (B,3,N)
        return torch.gather(x, dim=2, index=idx_exp)


class DCP(nn.Module):
    def __init__(self, args):
        super(DCP, self).__init__()
        
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        self.chi2_thresh = getattr(args, 'chi2_thresh', 9.0)
        self.k_samples = getattr(args, 'k_samples', 128)
        self.hypo_tester = GravityHypothesisTester(
                chi2_thresh=self.chi2_thresh,
                dist_scale=getattr(args, "dist_scale", 3.0),   # test.py의 "3.0 * resolution"에 해당하는 역할
                eps=1e-6
)

        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims, normal=True)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        else:
            raise Exception('Not implemented')

        if getattr(args, 'gravity', False):
            self.gravity_layer = GravityLayer(self.emb_dims) 
        else:
            None
            
        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')
    
    # def _nearest_neighbor(self, src, tgt):
    #     with torch.no_grad():
    #             inner = -2 * torch.matmul(src.transpose(2, 1), tgt)
    #             xx = torch.sum(src**2, dim=1, keepdim=True).transpose(2, 1)
    #             yy = torch.sum(tgt**2, dim=1, keepdim=True)
    #             dist_mat = xx + inner + yy
    #             _, corr_idx = torch.min(dist_mat, dim=2) # (B, N)
    #     return corr_idx
    
    def _get_dist_mat(self, src, tgt):
        inner = -2 * torch.matmul(src.transpose(2, 1), tgt)
        xx = torch.sum(src**2, dim=1, keepdim=True).transpose(2, 1)
        yy = torch.sum(tgt**2, dim=1, keepdim=True)
        return xx + inner + yy
    
    def _geometric_hypothesis_test(self, src, tgt, src_n, tgt_n, g_p, k_p, g_q, k_q):
        """ 
        양방향 기하학적 가설 검정을 수행하여 src용, tgt용 마스크를 각각 반환 
        """
        batch_size, _, num_points = src.size()
        
        # 1. 양방향 NN 인덱스 추출 (P->Q, Q->P)
        with torch.no_grad():
            # P to Q
            dist_pq = self._get_dist_mat(src, tgt)
            _, corr_p2q = torch.min(dist_pq, dim=2)
            # Q to P
            dist_qp = dist_pq.transpose(1, 2)
            _, corr_q2p = torch.min(dist_qp, dim=2)
        
        # 2. 대응되는 법선 벡터 정렬
        def get_matched_n(indices, target_n):
            idx_base = torch.arange(batch_size, device=src.device).view(-1, 1) * num_points
            flat_idx = (indices + idx_base).view(-1)
            matched = target_n.transpose(1, 2).contiguous().view(-1, 3)[flat_idx, :]
            return matched.view(batch_size, num_points, 3).transpose(1, 2)

        src_n_matched = get_matched_n(corr_q2p, src_n) # Q의 대응점인 P의 법선
        tgt_n_matched = get_matched_n(corr_p2q, tgt_n) # P의 대응점인 Q의 법선

        # 3. 통계량 계산
        k_eff = (k_p * k_q) / (k_p + k_q + 1e-6)
        
        # Source용 마스크 (P가 Q와 얼마나 일치하는가)
        inc_p = torch.sum(src_n * g_p.unsqueeze(2), dim=1)
        inc_p_ref = torch.sum(tgt_n_matched * g_q.unsqueeze(2), dim=1)
        w_p = torch.sigmoid(self.chi2_thresh - k_eff * (inc_p - inc_p_ref)**2)

        # Target용 마스크 (Q가 P와 얼마나 일치하는가)
        inc_q = torch.sum(tgt_n * g_q.unsqueeze(2), dim=1)
        inc_q_ref = torch.sum(src_n_matched * g_p.unsqueeze(2), dim=1)
        w_q = torch.sigmoid(self.chi2_thresh - k_eff * (inc_q - inc_q_ref)**2)

        return w_p.unsqueeze(1), w_q.unsqueeze(1)
    
    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        
        src_embedding, src_n = self.emb_nn(src)
        tgt_embedding, tgt_n = self.emb_nn(tgt)

        if self.gravity_layer is not None:
            (g_p, k_p), (g_q, k_q) = self.gravity_layer(src_embedding, tgt_embedding)
            w_p, w_q = self.hypo_tester(src, tgt, src_n, tgt_n, g_p, k_p, g_q, k_q)

            src_embedding = src_embedding * w_p
            tgt_embedding = tgt_embedding * w_q
            
            # Inference 시에만 Top-K 샘플링 적용
            if (not self.training) and (self.k_samples < src_embedding.size(-1)):
                B, C, N = src_embedding.shape           # (B, C, N)
                K = self.k_samples

                # w_p, w_q: (B, 1, N)
                _, top_k_p = torch.topk(w_p.squeeze(1), K, dim=1)  # (B, K)
                _, top_k_q = torch.topk(w_q.squeeze(1), K, dim=1)  # (B, K)

                # 1) Embedding gather: (B, C, N) -> (B, C, K)
                idx_p_emb = top_k_p.unsqueeze(1).expand(-1, C, -1)  # (B, C, K)
                idx_q_emb = top_k_q.unsqueeze(1).expand(-1, C, -1)  # (B, C, K)

                src_embedding = torch.gather(src_embedding, dim=2, index=idx_p_emb)  # (B, C, K)
                tgt_embedding = torch.gather(tgt_embedding, dim=2, index=idx_q_emb)  # (B, C, K)

                # 2) Point gather: (B, 3, N) -> (B, 3, K)
                idx_p_xyz = top_k_p.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, K)
                idx_q_xyz = top_k_q.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, K)

                src_topk = torch.gather(src, dim=2, index=idx_p_xyz)  # (B, 3, K)
                tgt_topk = torch.gather(tgt, dim=2, index=idx_q_xyz)  # (B, 3, K)

            else:
                src_topk = src
                tgt_topk = tgt
                
        # [Step 3] Pointer & Pose Head
        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src_topk, tgt_topk)
        if self.cycle:
            rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt_topk, src_topk)

        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        
        return rotation_ab, translation_ab, rotation_ba, translation_ba, {
            'g_p': g_p if self.gravity_layer is not None else None,
            'k_p': k_p if self.gravity_layer is not None else None,
            'g_q': g_q if self.gravity_layer is not None else None,
            'k_q': k_q if self.gravity_layer is not None else None,
        }
    
if __name__ == '__main__':
    
    import argparse
    from utils.data import data_loader
    from omegaconf import OmegaConf
    from utils.common import count_parameters
    from utils.loss import DCPLoss
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bunny', choices=['modelnet40', 'bunny'])
    parser.add_argument('--bunny_path', type=str, default='data/bunny/reconstruction/bun_zipper.ply')
    parser.add_argument('--method', type=str, default='p2p', choices=['p2p', 'p2l', 'l2l'],
                        help='ICP method: p2p (point-to-point), p2l (point-to-plane), l2l (plane-to-plane)')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum ICP iterations')
    parser.add_argument('--tol', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--dist_thresh', type=float, default=0.1, help='Distance threshold for matching')
    parser.add_argument('--emb_dims', type=int, default=512, help='Dimension of point feature embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of Transformer blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--ff_dims', type=int, default=1024, help='Dimension of feedforward network in Transformer')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads in Transformer')
    parser.add_argument('--emb_nn', type=str, default='pointnet', choices=['pointnet', 'dgcnn'],
                        help='Point cloud embedding network')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'],
                        help='Pointer network')
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'], help='Transformation head')
    parser.add_argument('--cycle', action='store_true', help='Use cycle consistency')
    parser.add_argument('--gravity', action='store_true', help='Use gravity alignment layer')
    args = parser.parse_args()
    
    # Model Forward Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = DCPLoss().to(device)
    
    train_loader, test_loader = data_loader(
        OmegaConf.create({
            'data': {
                'name': 'bunny',
                'bunny_path': args.bunny_path,
                'num_points': 1024,
                'gaussian_noise': True,
                'unseen': False,
                'factor': 1,
                'keep_ratio': 1.0,
                'num_workers': 0,
                'partial_overlap': False,
                'distance_range': 0.1
            },
            'training': {
                'batch_size': 16
            }
        })
    )
    
    sample = next(iter(train_loader))
    model = DCP(args=args).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
     # Data Load
    P = sample['P'].to('cuda')         
    Q = sample['Q'].to('cuda')         
    g_p = sample['g_p'].to('cuda') 
    g_q = sample['g_q'].to('cuda') 
    R_gt = sample['R_gt'].to('cuda')
    t_gt = sample['t_gt'].to('cuda')
    
    R_pq, t_pq, R_qp, t_qp, aux = model(P, Q)
    loss, loss_dict = loss_fn(R_pq, t_pq, R_gt, t_gt, R_qp, t_qp, aux)
    
    print("Loss:", loss.item())
    
    # Backward test
    loss.backward()
    print("Backward pass successful!")
    