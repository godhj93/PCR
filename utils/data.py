"""
Point Cloud Registration Dataset Loader
Supports: ModelNet40, Stanford Bunny
"""

from __future__ import annotations

import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional
from pathlib import Path
import open3d as o3d
# from utils.common import visualize_registration
import matplotlib.pyplot as plt
import hydra

def load_modelnet40_data(partition: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """Load ModelNet40 data from h5 files."""
    # 데이터 경로 설정 (환경에 맞게 수정 필요)
    data_dir = Path(__file__).parent.parent / 'data'
    
    if not data_dir.exists():
        # Fallback for colab/local without specific structure
        data_dir = Path('data') 

    all_data = []
    all_label = []
    
    pattern = os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')
    files = glob.glob(pattern)
    
    if not files:
        print(f"[Warning] No ModelNet40 files found in {pattern}")
        # 빈 배열 반환하여 초기화 에러 방지 (실제 사용시에는 데이터 필요)
        return np.zeros((1, 2048, 3)), np.zeros((1, 1))

    for h5_name in files:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)
    
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    
    return all_data, all_label

def load_bunny_data(file_path: str = 'data/bunny/reconstruction/bun_zipper.ply') -> np.ndarray:
    """Load Stanford Bunny as a normalized point cloud."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Bunny file not found: {file_path}")
    
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    points = np.asarray(pcd.points)
    
    # Normalize to unit sphere
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=-1)))
    points /= furthest_distance 
    
    return points.astype('float32')

def jitter_pointcloud(pointcloud: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Add random jitter to point cloud."""
    N, C = pointcloud.shape
    pointcloud = pointcloud + np.clip(sigma * np.random.randn(N, C), -clip, clip)
    return pointcloud

class RegistrationDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 data_source = None,  # 외부에서 로드된 데이터를 받음 (공유)
                 file_path: Optional[str] = None,  # Bunny 파일 경로 (backward compatibility)
                 num_points: int = 1024, 
                 partition: str = 'train', 
                 gaussian_noise: bool = False, 
                 unseen: bool = False, 
                 factor: float = 1,
                 partial_overlap: bool = True,
                 keep_ratio: float = 0.1,
                 distance_range: float = 10.0):
        
        self.dataset_name = dataset_name.lower()
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.factor = factor
        self.partial_overlap = partial_overlap 
        self.keep_ratio = keep_ratio    
        self.distance_range = distance_range
        # ! Gravity 설정 (World frame에서의 중력 방향)
        # ! 여기서는 z-축 음의 방향을 "중력"으로 정의 (예: g_world = (0, 0, -1))
        # ! 실제 로봇/센서 설정에 맞춰 이 벡터를 바꾸면 됨.
        
        if self.dataset_name == 'modelnet40':
            self.gravity_world = np.array([0.0, -1.0, 0.0], dtype='float32')
            
            self.data, self.label = data_source
            self.label = self.label.squeeze()
            
            # Unseen categories split
            if self.unseen:
                if self.partition == 'test':
                    self.data = self.data[self.label >= 20]
                    self.label = self.label[self.label >= 20]
                elif self.partition == 'train':
                    self.data = self.data[self.label < 20]
                    self.label = self.label[self.label < 20]
                    
        elif self.dataset_name == 'bunny':
            self.gravity_world = np.array([0.0, -1.0, 0.0], dtype='float32')
            
            self.bunny_points = data_source
            assert self.bunny_points is not None, "Bunny points data must be provided."
            # Bunny는 단일 객체이므로 Dataset 길이를 가상으로 설정
            self.virtual_size = 10000 if partition == 'train' else 1000
            print(f"Bunny dataset virtual size set to {self.virtual_size} for partition '{self.partition}'")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    def _generate_rotation(self, seed_idx=None):
        
        if self.partition != 'train' and seed_idx is not None:
            np.random.seed(seed_idx)
        
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
        
        # Euler angle: Rx -> Ry -> Rz 순서로 회전 적용
        R = Rx.dot(Ry).dot(Rz)
        
        euler = np.array([anglez, angley, anglex])  # ZYX 순서
        
        return R.astype('float32'), euler.astype('float32')
      
    def _partial_crop(self, points, seed_idx=None):
        """
        Helper: Randomly crop the point cloud by a plane and resample.
        Keep Ratio: 10% ~ 100% (Cropped 0% ~ 90%)
        """
        if not self.partial_overlap:
            return points

        if self.partition != 'train' and seed_idx is not None:
            np.random.seed(seed_idx)
            
        # 1. 랜덤 방향 벡터 생성
        rand_dir = np.random.randn(3)
        rand_dir /= np.linalg.norm(rand_dir)
        
        # 2. 투영 및 정렬
        proj = points @ rand_dir
        sort_idx = np.argsort(proj)
        
        # 3. 유지할 비율 결정 (0.1 ~ 1.0) -> 즉 0~90% 잘려나감
        
        keep_ratio = np.random.uniform(self.keep_ratio, 1.0)
        num_keep = int(len(points) * keep_ratio)
        
        # 4. Slicing
        keep_idx = sort_idx[:num_keep]
        cropped_points = points[keep_idx]
        
        # 5. Resampling (배치 처리를 위해 원래 점 개수로 복원)
        if len(cropped_points) < self.num_points:
            
            if self.partition != 'train' and seed_idx is not None:
                # 같은 seed_idx를 쓰면 위쪽 패턴과 동기화되어 편향될 수 있으므로 +1을 해줍니다.
                np.random.seed(seed_idx + 1) 

            choice_idx = np.random.choice(len(cropped_points), self.num_points, replace=True)
            resampled_points = cropped_points[choice_idx]
            
            # 중복 샘플링된 점들을 구별하기 위한 jitter (SVD 안정성)
            jitter = np.random.normal(scale=0.02, size=resampled_points.shape)
            resampled_points = resampled_points + jitter.astype('float32')
            
        else:
            if self.partition != 'train' and seed_idx is not None:
                np.random.seed(seed_idx + 2) # 다른 값으로 시드 고정

            choice_idx = np.random.choice(len(cropped_points), self.num_points, replace=False)
            resampled_points = cropped_points[choice_idx]
            
        return resampled_points
        
    def __getitem__(self, item):
        # ---------------------------------------------------------------------
        # 1. Point Cloud 데이터 가져오기 (기존 동일)
        # ---------------------------------------------------------------------
        if self.dataset_name == 'modelnet40':
            pointcloud = self.data[item][:self.num_points]
            
        elif self.dataset_name == 'bunny':
            total_pts = self.bunny_points.shape[0]
            
            if self.partition == 'train':
                idx = np.random.choice(total_pts, self.num_points, replace=False)
            else:
                np.random.seed(item) 
                idx = np.random.choice(total_pts, self.num_points, replace=False)
            pointcloud = self.bunny_points[idx]

        # ---------------------------------------------------------------------
        # 2. Normalization (기존 동일)
        # ---------------------------------------------------------------------
        centroid = np.mean(pointcloud, axis=0)
        pointcloud = pointcloud - centroid
        max_dist = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
        pointcloud = pointcloud / (max_dist + 1e-8)
        
        p_canonical = pointcloud.copy()
        
        # ---------------------------------------------------------------------
        # 3. Augmentation (기존 동일)
        # ---------------------------------------------------------------------
        seed_p = item + 100000 if self.partition != 'train' else None
        seed_q = item if self.partition != 'train' else None
        seed_crop_p = item + 200000 if self.partition != 'train' else None
        seed_crop_q = item + 300000 if self.partition != 'train' else None
        
        R_src, _ = self._generate_rotation(seed_p)
        R_ab, euler_ab = self._generate_rotation(seed_q)
        R_ba = R_ab.T
        
        if self.partition != 'train':
            np.random.seed(item)
        dist = self.distance_range
        translation_ab = np.array([
            np.random.uniform(-dist, dist),
            np.random.uniform(-dist, dist),
            np.random.uniform(-dist, dist)
        ], dtype='float32')
        
        # translation_ba = -R_ba.dot(translation_ab)
        
        if self.partial_overlap:
            P_crop = self._partial_crop(p_canonical, seed_crop_p)
            P0 = (R_src @ P_crop.T).T
            
            Q_crop = self._partial_crop(p_canonical, seed_crop_q)
            Q0_base = (R_src @ Q_crop.T).T
            Q0 = (R_ab @ Q0_base.T).T + translation_ab[None, :]
            
            corr = np.zeros((self.num_points, 2), dtype='int64')
        
        else:
            P0 = (R_src @ p_canonical.T).T
            Q0 = (R_ab @ P0.T).T + translation_ab[None, :]
            
            N = self.num_points
            perm_p = np.random.permutation(N)
            perm_q = np.random.permutation(N)
            
            P0 = P0[perm_p]
            Q0 = Q0[perm_q]
            
            inv_perm_p = np.argsort(perm_p)
            inv_perm_q = np.argsort(perm_q)
            corr = np.stack([inv_perm_p, inv_perm_q], axis=1).astype('int64')

        # Gravity Vectors
        g_p = (R_src @ self.gravity_world).astype('float32')
        g_q = (R_ab @ g_p).astype('float32')

        # Noise
        if self.gaussian_noise:
            P0 = jitter_pointcloud(P0)
            Q0 = jitter_pointcloud(Q0)
        
        return_dict = {
            # === Inputs (All 3xN, Contiguous) ===
            'P': torch.from_numpy(P0).transpose(0, 1).contiguous().to(torch.float32),       # (3, N)
            'Q': torch.from_numpy(Q0).transpose(0, 1).contiguous().to(torch.float32),       # (3, N)
            
            # === Labels ===
            'g_p': torch.from_numpy(g_p).to(torch.float32),                               # (3,)
            'g_q': torch.from_numpy(g_q).to(torch.float32),                               # (3,)
            
            # === Metadata ===
            'R_gt': torch.from_numpy(R_ab).to(torch.float32),         
            't_gt': torch.from_numpy(translation_ab).to(torch.float32),
            'corr_idx': corr,
        }

        return return_dict
        
    def __len__(self):
        if self.dataset_name == 'modelnet40':
            return self.data.shape[0]
        elif self.dataset_name == 'bunny':
            return self.virtual_size
        return 0

def data_loader(cfg):
    """
    cfg: Hydra 또는 OmegaConf 객체
    필수 설정:
    - cfg.data.name: 데이터셋 이름 ('modelnet40' 또는 'bunny')
    - cfg.data.num_points: 샘플링할 점의 개수
    - cfg.training.batch_size: 배치 사이즈
    """
    
    # 1. 필수 설정 존재 여부 체크 (없으면 즉시 에러)
    try:
        dataset_name = cfg.data.name.lower()
    except (AttributeError, KeyError):
        raise KeyError("Config Error: 'cfg.data.name'이 정의되지 않았습니다. (modelnet40 또는 bunny)")

    # 2. 데이터 로드 로직
    if dataset_name == 'modelnet40':
        print(f"Loading ModelNet40 data (num_points: {cfg.data.num_points})...")
        train_data = load_modelnet40_data('train')
        test_data = load_modelnet40_data('test')
        
        train_dataset = RegistrationDataset(
            dataset_name=dataset_name,
            data_source=train_data,
            num_points=cfg.data.num_points,
            partition='train',
            gaussian_noise=cfg.data.gaussian_noise,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor,
            keep_ratio=cfg.data.keep_ratio,
            partial_overlap=cfg.data.partial_overlap,
            distance_range=cfg.data.distance_range,
        )
        test_dataset = RegistrationDataset(
            dataset_name=dataset_name,
            data_source=test_data,
            num_points=cfg.data.num_points,
            partition='test',
            gaussian_noise=False,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor,
            keep_ratio=cfg.data.keep_ratio,
            partial_overlap=cfg.data.partial_overlap,
            distance_range=cfg.data.distance_range,
        )
        
    elif dataset_name == 'bunny':
        # Bunny 경로 체크
        if not hasattr(cfg.data, 'bunny_path'):
            raise KeyError("Config Error: Bunny 데이터셋을 위해 'cfg.data.bunny_path'가 필요합니다.")
            
        print(f"Loading Bunny from {cfg.data.bunny_path}...")
        shared_data = load_bunny_data(cfg.data.bunny_path)
        
        train_dataset = RegistrationDataset(
            dataset_name=dataset_name,
            data_source=shared_data,
            num_points=cfg.data.num_points,
            partition='train',
            gaussian_noise=cfg.data.gaussian_noise,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor,
            keep_ratio=cfg.data.keep_ratio,
            partial_overlap=cfg.data.partial_overlap,
            distance_range=cfg.data.distance_range,
        )
        test_dataset = RegistrationDataset(
            dataset_name=dataset_name,
            data_source=shared_data,
            num_points=cfg.data.num_points,
            partition='test',
            gaussian_noise=False,
            unseen=cfg.data.unseen,
            factor=cfg.data.factor,
            keep_ratio=cfg.data.keep_ratio,
            partial_overlap=cfg.data.partial_overlap,
            distance_range=cfg.data.distance_range,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. 'modelnet40' 또는 'bunny'여야 합니다.")

    # 3. DataLoader 생성 (batch_size 등 필수값 체크)
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            drop_last=False
        )
    except (AttributeError, KeyError) as e:
        raise KeyError(f"Config Error: DataLoader 설정이 누락되었습니다. ({e})")
    
    return train_loader, test_loader

def draw_gravity_arrow(ax, origin, vector, color, label, scale=1.5):
            """
            scale: 1.5로 키워서 점군 밖으로 삐져나오게 함 (잘 보이도록)
            linewidth: 3.0으로 굵게 설정
            zorder: 10으로 설정하여 점들 위에 그려지게 함
            """
            # 벡터 정규화 및 스케일링
            v_norm = vector / (np.linalg.norm(vector) + 1e-6) * scale
            
            x, y, z = origin
            u, v, w = v_norm
            
            # 화살표 그리기 (굵고 진하게)
            ax.quiver(x, y, z, u, v, w, color=color, length=1.0, normalize=False, 
                      linewidth=3.0, arrow_length_ratio=0.2, zorder=100)
            
            # 텍스트 라벨 (약간 띄워서)
            ax.text(x + u*1.1, y + v*1.1, z + w*1.1, label, color=color, 
                    fontsize=12, fontweight='bold', zorder=101)
           
def draw_uncertainty_cone(ax, origin, mu_vec, kappa_val, scale=0.8, color='red'):
    """
    예측된 방향 벡터 주변에 불확실성 원뿔(Wireframe Cone)을 그립니다.
    kappa 값이 클수록 원뿔의 각도가 좁아집니다.
    """
    # 1. Kappa를 각도(Degree)로 변환 (Heuristic 방식)
    # kappa가 매우 크면 각도가 0에 수렴, 작으면 커짐.
    # 시각적 명확성을 위해 최대 각도를 제한합니다 (예: 45도).
    max_angle_deg = 45.0
    # kappa가 0일 때를 대비해 작은 값(1e-1) 추가
    angle_deg = max_angle_deg / np.sqrt(kappa_val + 0.1)
    angle_deg = np.clip(angle_deg, 1.0, 60.0) # 최소 1도, 최대 60도로 제한
    angle_rad = np.radians(angle_deg)

    # 2. 원뿔 기하학 계산
    mu_norm = mu_vec / (np.linalg.norm(mu_vec) + 1e-8)
    
    # mu 벡터에 수직인 기저 벡터(v1, v2) 찾기 (원뿔 밑면 원을 그리기 위해)
    # 임의의 참조 벡터(ref)와 외적을 이용
    if np.isclose(abs(mu_norm[2]), 1.0):
        ref = np.array([1.0, 0.0, 0.0]) # mu가 Z축과 평행할 경우 X축 참조
    else:
        ref = np.array([0.0, 0.0, 1.0]) # 그 외에는 Z축 참조
        
    v1 = np.cross(mu_norm, ref)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(mu_norm, v1) # mu, v1에 모두 수직인 벡터

    # 3. 원뿔 그리기 (Wireframe 방식)
    height = scale * 1.0 # 화살표 길이와 맞춤
    radius = height * np.tan(angle_rad) # 원뿔 밑면 반지름

    # 원뿔 밑면 원주상의 점들 생성
    theta = np.linspace(0, 2*np.pi, 20) # 20개의 선으로 표현
    cone_tip_center = origin + mu_norm * height # 화살표 끝점

    circle_points = []
    for t in theta:
        # 원주상의 점 좌표 계산
        pt_on_circle = cone_tip_center + radius * (np.cos(t)*v1 + np.sin(t)*v2)
        circle_points.append(pt_on_circle)
        
        # 원점(origin)에서 원주상의 점으로 이어지는 선 그리기 (반투명)
        ax.plot([origin[0], pt_on_circle[0]], 
                [origin[1], pt_on_circle[1]], 
                [origin[2], pt_on_circle[2]],
                color=color, alpha=0.2, linewidth=1)

    # 밑면 원 테두리 그리기
    circle_points = np.array(circle_points)
    # 시작점과 끝점을 이어주기 위해 마지막에 첫 점 추가
    circle_points = np.vstack([circle_points, circle_points[0]])
    ax.plot(circle_points[:,0], circle_points[:,1], circle_points[:,2],
            color=color, alpha=0.4, linewidth=2, linestyle='--')
    
if __name__ == '__main__':
    import argparse
    import sys
    
    # ICP 모듈 임포트
    sys.path.append(str(Path(__file__).parent.parent / 'iterative_closet_point'))
    from iterative_closet_point.bunny import run_icp, calculatenormal, build_kdtree
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'bunny'])
    parser.add_argument('--bunny_path', type=str, default='data/bunny/reconstruction/bun_zipper.ply')
    parser.add_argument('--method', type=str, default='p2p', choices=['p2p', 'p2l', 'l2l'],
                        help='ICP method: p2p (point-to-point), p2l (point-to-plane), l2l (plane-to-plane)')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum ICP iterations')
    parser.add_argument('--tol', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--dist_thresh', type=float, default=0.1, help='Distance threshold for matching')
    args = parser.parse_args()

    print(f"Testing RegistrationDataset with {args.dataset}...")
    print(f"ICP Method: {args.method}")
    
    from omegaconf import OmegaConf
    train_loader, test_loader = data_loader(
        OmegaConf.create({
            'data': {
                'name': args.dataset,
                'bunny_path': args.bunny_path,
                'num_points': 1024,
                'gaussian_noise': True,
                'unseen': False,
                'factor': 1,
                'keep_ratio': 1.0,
                'num_workers': 0
            },
            'training': {
                'batch_size': 1
            }
        })
    )
    
    sample = next(iter(test_loader))
    p = sample['P'].numpy()
    q = sample['Q'].numpy()
    R_gt = sample['R_gt']
    t_gt = sample['t_gt']
    g_p = sample['g_p'].numpy()
    g_q = sample['g_q'].numpy()
    
    print(f"Shape of p: {p.shape}, q: {q.shape}, g_p: {g_p.shape}, g_q: {g_q.shape}, R_gt: {R_gt.shape}, t_gt: {t_gt.shape}")
    
    # Visualization of gravity vectors of P and Q
    p_viz = p.squeeze(0)  # (3, N)
    q_viz = q.squeeze(0)  # (3, N)
    
    # Apply GT transformation to p_viz to validate
    answer = ((R_gt.numpy() @ p_viz) + t_gt.numpy().reshape(3, 1)).squeeze()
    print(answer.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p_viz[0,:], p_viz[1,:], p_viz[2,:], c='b', s=5, label='Point Cloud P')
    ax.scatter(q_viz[0,:], q_viz[1,:], q_viz[2,:], c='r', s=5, label='Point Cloud Q')
    ax.scatter(answer[0,:], answer[1,:], answer[2,:], c='g', s=5, label='Transformed P (GT)')
    draw_gravity_arrow(ax, origin=np.mean(p_viz, axis=1), vector=g_p.squeeze(0), color='cyan', label='g_p')
    draw_gravity_arrow(ax, origin=np.mean(q_viz, axis=1), vector=g_q.squeeze(0), color='magenta', label='g_q')
    ax.set_title('Point Clouds with Gravity Vectors')
    ax.legend()
    plt.show()
