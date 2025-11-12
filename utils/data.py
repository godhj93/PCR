"""ModelNet40 dataset loader for point cloud registration."""

from __future__ import annotations

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from typing import Tuple
from pathlib import Path

def load_modelnet40_data(partition: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """Load ModelNet40 data from h5 files."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    all_data = []
    all_label = []
    
    pattern = os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')
    for h5_name in glob.glob(pattern):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)
    
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    
    return all_data, all_label


def jitter_pointcloud(pointcloud: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Add random jitter to point cloud."""
    N, C = pointcloud.shape
    pointcloud = pointcloud + np.clip(sigma * np.random.randn(N, C), -clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        
        self.data, self.label = load_modelnet40_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]  # (N,3)
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)

        if self.partition != 'train':
            np.random.seed(item)

        # ----- 1) 랜덤 회전/이동 샘플링 -----
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx, cosy, cosz = np.cos(anglex), np.cos(angley), np.cos(anglez)
        sinx, siny, sinz = np.sin(anglex), np.sin(angley), np.sin(anglez)

        Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T

        translation_ab = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ])
        translation_ba = -R_ba.dot(translation_ab)

        # ----- 2) 원본 순서에서 P0, Q0 만들기 (GT 1:1 대응) -----
        # P0: (N,3), Q0: (N,3)
        P0 = pointcloud
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        Q0 = rotation_ab.apply(P0) + translation_ab[None, :]  # (N,3)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        N = self.num_points
        orig_idx = np.arange(N)

        # ----- 3) P, Q를 서로 독립적으로 섞되, perm을 저장 -----
        perm_p = np.random.permutation(N)
        perm_q = np.random.permutation(N)

        # 섞인 점군 (모델이 보는 입력)
        P = P0[perm_p]   # (N,3)
        Q = Q0[perm_q]   # (N,3)

        # ----- 4) GT correspondence 인덱스 쌍 (i, j) 만들기 -----
        # P0[k] -> Q0[k]가 GT이므로,
        #    P에서의 위치 i = perm_p^{-1}(k)
        #    Q에서의 위치 j = perm_q^{-1}(k)
        inv_perm_p = np.argsort(perm_p)  # shape (N,)
        inv_perm_q = np.argsort(perm_q)  # shape (N,)

        # k-th 원본 점에 대한 (i_k, j_k) 쌍
        corr = np.stack([inv_perm_p, inv_perm_q], axis=1).astype('int64')  # (N, 2)

        data = {
            'p': P.T.astype('float32'),      # (3,N)
            'q': Q.T.astype('float32'),      # (3,N)
            'R_pq': R_ab.astype('float32'),
            't_pq': translation_ab.astype('float32'),
            'R_qp': R_ba.astype('float32'),
            't_qp': translation_ba.astype('float32'),
            'euler_pq': euler_ab.astype('float32'),
            'euler_qp': euler_ba.astype('float32'),

            # ---- 여기부터 GT correspondence 관련 ----
            # 각 row: [i, j]  (p의 인덱스 i, q의 인덱스 j 가 대응)
            'corr_idx': corr,                # (N, 2), int64

            # 옵션: 나중에 디버깅용으로 perm도 보고 싶으면
            'perm_p': perm_p.astype('int64'),
            'perm_q': perm_q.astype('int64'),
        }
        return data

    def __len__(self):
        return self.data.shape[0]

def data_loader(cfg):
    
    train_loader = torch.utils.data.DataLoader(
        ModelNet40(
            num_points=cfg.data.num_points,
            partition='train',
            gaussian_noise=cfg.data.gaussian_noise,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor
        ),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        ModelNet40(
            num_points=cfg.data.num_points,
            partition='test',
            gaussian_noise=False,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor
        ),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader
    
if __name__ == '__main__':
    dataset = ModelNet40(num_points=1024, partition='train', gaussian_noise=True)
    print(f'Dataset size: {len(dataset)}')

    sample = dataset[0]
    print('Sample keys:', sample.keys())
    print('Pointcloud 1 shape:', sample['p'].shape)   # (3,N)
    print('Pointcloud 2 shape:', sample['q'].shape)   # (3,N)
    print('Rotation matrix shape:', sample['R_pq'].shape)
    print('Translation vector shape:', sample['t_pq'].shape)
    print('Euler angles shape:', sample['euler_pq'].shape)
    print('Corr idx shape:', sample['corr_idx'].shape)

    # ----- GT correspondence 체크 -----
    P = sample['p']        # (3,N)
    Q = sample['q']        # (3,N)
    R = sample['R_pq']     # (3,3)
    t = sample['t_pq']     # (3,)
    corr = sample['corr_idx']  # (N,2)

    N = P.shape[1]

    # 몇 개만 찍어보는 sanity check
    print("\n[Sanity check] First 5 correspondences:")
    for k in range(5):
        i, j = corr[k]
        p_i = P[:, i]              # (3,)
        q_j = Q[:, j]              # (3,)
        q_pred = R @ p_i + t       # (3,)

        err = np.linalg.norm(q_pred - q_j)
        print(f"k={k}, i={i}, j={j}, error={err:.6e}")

    # 전체 correspondence error 통계
    errs = []
    for k in range(N):
        i, j = corr[k]
        p_i = P[:, i]
        q_j = Q[:, j]
        q_pred = R @ p_i + t
        errs.append(np.linalg.norm(q_pred - q_j))

    errs = np.array(errs)
    print("\n[Global correspondence error]")
    print(f"mean = {errs.mean():.6e}")
    print(f"max  = {errs.max():.6e}")
    print(f"min  = {errs.min():.6e}")

    # 인덱스 중복/누락 여부 체크 (full overlap인 경우)
    unique_p = np.unique(corr[:, 0])
    unique_q = np.unique(corr[:, 1])
    print("\n[Index coverage]")
    print(f"unique p indices: {len(unique_p)} / {N}")
    print(f"unique q indices: {len(unique_q)} / {N}")
