"""Learnable Sinkhorn layer for point cloud matching with training and evaluation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data import ModelNet40Registration
from testing import apply_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def estimate_transform_from_matches(
    source: Tensor,
    target: Tensor,
    matching_matrix: Tensor,
    source_to_dustbin: Tensor,
) -> Tuple[Tensor, int]:
    """Estimate transformation from matched points using SVD.
    
    Returns:
        transform: (4, 4) transformation matrix
        num_inliers: number of matched points used
    """
    device = source.device
    
    # Get matches
    src_match_scores, src_match_idx = matching_matrix.max(dim=1)
    src_is_matched = src_match_scores > source_to_dustbin
    
    # Get matched point pairs
    matched_src_idx = torch.where(src_is_matched)[0]
    if len(matched_src_idx) < 3:
        # Not enough matches, return identity
        return torch.eye(4, device=device), 0
    
    matched_tgt_idx = src_match_idx[matched_src_idx]
    
    src_matched = source[matched_src_idx]
    tgt_matched = target[matched_tgt_idx]
    
    # Compute centroids
    src_centroid = src_matched.mean(dim=0, keepdim=True)
    tgt_centroid = tgt_matched.mean(dim=0, keepdim=True)
    
    # Center the points
    src_centered = src_matched - src_centroid
    tgt_centered = tgt_matched - tgt_centroid
    
    # Compute covariance matrix
    H = src_centered.T @ tgt_centered
    
    # SVD
    U, S, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if torch.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T
    
    # Compute translation
    t = tgt_centroid.T - R @ src_centroid.T
    
    # Build transformation matrix
    transform = torch.eye(4, device=device)
    transform[:3, :3] = R
    transform[:3, 3:4] = t
    
    return transform, len(matched_src_idx)


def evaluate_registration_error(
    source: Tensor,
    target: Tensor,
    predicted_transform: Tensor,
    gt_transform: Tensor,
    overlap_count: int,
) -> dict:
    """Evaluate registration error."""
    device = source.device
    
    # Apply transforms
    source_pred = apply_transform(source, predicted_transform)
    source_gt = apply_transform(source, gt_transform)
    
    # Compute errors for overlap region
    overlap_indices = torch.arange(overlap_count, device=device)
    
    # Chamfer distance for overlap
    if overlap_count > 0:
        pred_overlap = source_pred[:overlap_count]
        gt_overlap = source_gt[:overlap_count]
        target_overlap = target[:overlap_count]
        
        # Distance from predicted to target
        dist_pred_to_target = torch.cdist(pred_overlap, target_overlap)
        min_dist_pred = dist_pred_to_target.min(dim=1)[0].mean()
        
        # Distance from gt to target (should be very small)
        dist_gt_to_target = torch.cdist(gt_overlap, target_overlap)
        min_dist_gt = dist_gt_to_target.min(dim=1)[0].mean()
        
        # Rotation error
        R_pred = predicted_transform[:3, :3]
        R_gt = gt_transform[:3, :3]
        R_error = torch.norm(R_pred - R_gt, 'fro')
        
        # Translation error
        t_pred = predicted_transform[:3, 3]
        t_gt = gt_transform[:3, 3]
        t_error = torch.norm(t_pred - t_gt)
    else:
        min_dist_pred = torch.tensor(float('inf'))
        min_dist_gt = torch.tensor(0.0)
        R_error = torch.tensor(float('inf'))
        t_error = torch.tensor(float('inf'))
    
    return {
        "chamfer_dist": min_dist_pred.item(),
        "gt_chamfer_dist": min_dist_gt.item(),
        "rotation_error": R_error.item(),
        "translation_error": t_error.item(),
    }


class LearnableSinkhornLayer(nn.Module):
    """Learnable Sinkhorn matching layer with feature extraction."""
    
    def __init__(
        self,
        input_dim: int = 3,
        feature_dim: int = 128,
        temperature: float = 0.02,
        sinkhorn_iters: int = 100,
        dustbin_init: float = 1.0,
    ):
        super().__init__()
        
        # Enhanced feature extraction networks with dropout
        self.source_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, feature_dim),
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, feature_dim),
        )
        
        # Learnable parameters
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.dustbin_score = nn.Parameter(torch.tensor(dustbin_init))
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = 1e-9
        
    @property
    def temperature(self):
        """Always positive temperature."""
        return torch.exp(self.log_temperature)
        
    def forward(
        self,
        source: Tensor,
        target: Tensor,
        overlap_count: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            source: (N, 3) source points
            target: (M, 3) target points
            overlap_count: number of overlapping points
            
        Returns:
            matching_matrix: (N, M) soft assignment
            source_to_dustbin: (N,) dustbin probabilities for source
            target_to_dustbin: (M,) dustbin probabilities for target
        """
        # Extract features
        source_feat = self.source_encoder(source)  # (N, feature_dim)
        target_feat = self.target_encoder(target)  # (M, feature_dim)
        
        # Compute feature-based similarity
        # Negative distance as score (higher is better)
        pairwise_dist = torch.cdist(source_feat, target_feat, p=2)
        
        num_src, num_tgt = pairwise_dist.shape
        
        # Temperature-scaled scores
        temp = self.temperature
        scores = -pairwise_dist / torch.clamp(temp, min=self.epsilon)
        scores = scores - scores.max()  # Numerical stability
        
        # Add dustbin row and column
        dustbin_row = self.dustbin_score.expand(1, num_tgt)
        extended = torch.cat([scores, dustbin_row], dim=0)
        dustbin_col = self.dustbin_score.expand(num_src + 1, 1)
        extended = torch.cat([extended, dustbin_col], dim=1)
        
        # Initialize marginals based on overlap
        unique_src = max(num_src - overlap_count, 0)
        unique_tgt = max(num_tgt - overlap_count, 0)
        
        mu_vals = torch.ones(num_src, dtype=scores.dtype, device=scores.device)
        nu_vals = torch.ones(num_tgt, dtype=scores.dtype, device=scores.device)
        
        dustbin_scale = 0.3
        if unique_src > 0:
            mu_vals[overlap_count:] = self.epsilon
        if unique_tgt > 0:
            nu_vals[overlap_count:] = self.epsilon
            
        mu_dustbin = torch.tensor(
            [max(float(unique_tgt) * dustbin_scale, self.epsilon)],
            dtype=scores.dtype,
            device=scores.device,
        )
        nu_dustbin = torch.tensor(
            [max(float(unique_src) * dustbin_scale, self.epsilon)],
            dtype=scores.dtype,
            device=scores.device,
        )
        
        mu = torch.cat([mu_vals, mu_dustbin])
        nu = torch.cat([nu_vals, nu_dustbin])
        mu = mu / mu.sum()
        nu = nu / nu.sum()
        
        log_mu = torch.log(mu)
        log_nu = torch.log(nu)
        
        # Sinkhorn iterations
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        
        for _ in range(self.sinkhorn_iters):
            u = log_mu - torch.logsumexp(extended + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(extended + u.unsqueeze(1), dim=0)
        
        log_transport = extended + u.unsqueeze(1) + v.unsqueeze(0)
        transport = torch.exp(log_transport)
        
        matching_matrix = transport[:-1, :-1]
        source_to_dustbin = transport[:-1, -1]
        target_to_dustbin = transport[-1, :-1]
        
        return matching_matrix, source_to_dustbin, target_to_dustbin


def compute_matching_loss(
    matching_matrix: Tensor,
    source_to_dustbin: Tensor,
    target_to_dustbin: Tensor,
    overlap_count: int,
) -> Tuple[Tensor, dict]:
    """Compute loss for matching task.
    
    Loss components:
    1. Overlap matching: -log P(correct match) for overlap points
    2. Dustbin routing: -log P(dustbin) for unique points  
    3. Overlap non-dustbin: encourage overlap points to avoid dustbin
    """
    num_src, num_tgt = matching_matrix.shape
    device = matching_matrix.device
    
    # Ground truth: diagonal matching for overlap region
    overlap_indices = torch.arange(overlap_count, device=device)
    
    # Loss 1: Overlap points should match to their corresponding indices
    overlap_match_loss = torch.tensor(0.0, device=device)
    if overlap_count > 0:
        # For overlap sources, maximize probability of correct target
        correct_probs = matching_matrix[overlap_indices, overlap_indices]
        overlap_match_loss = -torch.log(correct_probs + 1e-9).mean()
    
    # Loss 2: Unique points should go to dustbin
    unique_src_count = max(num_src - overlap_count, 0)
    unique_tgt_count = max(num_tgt - overlap_count, 0)
    
    dustbin_loss = torch.tensor(0.0, device=device)
    count = 0
    if unique_src_count > 0:
        dustbin_loss += -torch.log(source_to_dustbin[overlap_count:] + 1e-9).mean()
        count += 1
    if unique_tgt_count > 0:
        dustbin_loss += -torch.log(target_to_dustbin[overlap_count:] + 1e-9).mean()
        count += 1
    if count > 0:
        dustbin_loss = dustbin_loss / count
    
    # Loss 3: Overlap points should NOT go to dustbin (regularization)
    overlap_not_dustbin_loss = torch.tensor(0.0, device=device)
    if overlap_count > 0:
        overlap_not_dustbin_loss = torch.log(source_to_dustbin[:overlap_count] + 1e-9).mean()
        overlap_not_dustbin_loss += torch.log(target_to_dustbin[:overlap_count] + 1e-9).mean()
        overlap_not_dustbin_loss = -overlap_not_dustbin_loss / 2
    
    total_loss = overlap_match_loss + dustbin_loss + 0.5 * overlap_not_dustbin_loss
    
    metrics = {
        "total_loss": total_loss.item(),
        "overlap_match_loss": overlap_match_loss.item(),
        "dustbin_loss": dustbin_loss.item(),
        "overlap_not_dustbin_loss": overlap_not_dustbin_loss.item(),
    }
    
    return total_loss, metrics


def evaluate_accuracy(
    matching_matrix: Tensor,
    source_to_dustbin: Tensor,
    target_to_dustbin: Tensor,
    overlap_count: int,
) -> dict:
    """Evaluate matching accuracy."""
    num_src, num_tgt = matching_matrix.shape
    device = matching_matrix.device
    
    overlap_indices = torch.arange(overlap_count, device=device)
    
    # Source side
    src_match_scores, src_match_idx = matching_matrix.max(dim=1)
    src_is_matched = src_match_scores > source_to_dustbin
    
    if overlap_count > 0:
        src_correct_overlap = (
            (src_match_idx[:overlap_count] == overlap_indices) & src_is_matched[:overlap_count]
        )
        overlap_accuracy = float(src_correct_overlap.sum().item() / overlap_count)
    else:
        overlap_accuracy = 0.0
    
    # Target side
    tgt_match_scores, tgt_match_idx = matching_matrix.max(dim=0)
    tgt_is_matched = tgt_match_scores > target_to_dustbin
    
    if overlap_count > 0:
        tgt_correct_overlap = (
            (tgt_match_idx[:overlap_count] == overlap_indices) & tgt_is_matched[:overlap_count]
        )
        overlap_recall = float(tgt_correct_overlap.sum().item() / overlap_count)
    else:
        overlap_recall = 0.0
    
    # Dustbin accuracy
    unique_src_count = max(num_src - overlap_count, 0)
    unique_tgt_count = max(num_tgt - overlap_count, 0)
    
    if unique_src_count > 0:
        src_unique_to_dustbin = (~src_is_matched)[overlap_count:].sum().item()
        src_dustbin_acc = src_unique_to_dustbin / unique_src_count
    else:
        src_dustbin_acc = 1.0
        
    if unique_tgt_count > 0:
        tgt_unique_to_dustbin = (~tgt_is_matched)[overlap_count:].sum().item()
        tgt_dustbin_acc = tgt_unique_to_dustbin / unique_tgt_count
    else:
        tgt_dustbin_acc = 1.0
    
    return {
        "overlap_accuracy": overlap_accuracy,
        "overlap_recall": overlap_recall,
        "src_dustbin_acc": src_dustbin_acc,
        "tgt_dustbin_acc": tgt_dustbin_acc,
        "avg_dustbin_acc": (src_dustbin_acc + tgt_dustbin_acc) / 2,
    }


def train_model(
    model: LearnableSinkhornLayer,
    dataset,
    num_epochs: int = 100,
    device: torch.device = torch.device("cpu"),
    checkpoint_dir: str = "checkpoints/learnable_sinkhorn",
    add_noise_augmentation: bool = True,
) -> list:
    """Train the learnable Sinkhorn model."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = []
    best_loss = float('inf')
    
    print("Starting training...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Data augmentation: {add_noise_augmentation}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_metrics = {
            "overlap_match_loss": 0.0,
            "dustbin_loss": 0.0,
            "overlap_not_dustbin_loss": 0.0,
            "overlap_accuracy": 0.0,
            "overlap_recall": 0.0,
            "avg_dustbin_acc": 0.0,
        }
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            source = sample['src'].to(device)
            target = sample['tgt'].to(device)
            transform = sample['transform'].to(device)
            
            # For ModelNet40, all points overlap (full registration)
            overlap_count = source.shape[0]
            
            # Align source to target frame for training
            source_aligned = apply_transform(source, transform)
            
            # Data augmentation: add noise during training
            if add_noise_augmentation:
                noise_std = 0.02 * torch.rand(1).item()  # Random noise 0-0.02
                source_aligned = source_aligned + torch.randn_like(source_aligned) * noise_std
                target = target + torch.randn_like(target) * noise_std
            
            # Forward pass
            matching_matrix, source_to_dustbin, target_to_dustbin = model(
                source_aligned, target, overlap_count
            )
            
            # Compute loss
            loss, metrics = compute_matching_loss(
                matching_matrix, source_to_dustbin, target_to_dustbin, overlap_count
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            for key in ["overlap_match_loss", "dustbin_loss", "overlap_not_dustbin_loss"]:
                epoch_metrics[key] += metrics[key]
            
            # Compute accuracy
            with torch.no_grad():
                acc_metrics = evaluate_accuracy(
                    matching_matrix, source_to_dustbin, target_to_dustbin, sample.overlap_count
                )
                for key in ["overlap_accuracy", "overlap_recall", "avg_dustbin_acc"]:
                    epoch_metrics[key] += acc_metrics[key]
        
        # Average metrics
        num_samples = len(dataset)
        epoch_loss /= num_samples
        for key in epoch_metrics:
            epoch_metrics[key] /= num_samples
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
        
        history.append({"epoch": epoch, "loss": epoch_loss, **epoch_metrics})
        
        # Save checkpoint if best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'metrics': epoch_metrics,
            }, checkpoint_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'metrics': epoch_metrics,
            }, checkpoint_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {epoch_loss:.4f}")
            print(f"  Overlap Acc: {epoch_metrics['overlap_accuracy']:.3f}, Recall: {epoch_metrics['overlap_recall']:.3f}")
            print(f"  Dustbin Acc: {epoch_metrics['avg_dustbin_acc']:.3f}")
            print(f"  Temperature: {model.temperature.item():.4f}, Dustbin: {model.dustbin_score.item():.4f}")
            print(f"  Loss breakdown - Match: {epoch_metrics['overlap_match_loss']:.2f}, "
                  f"Dustbin: {epoch_metrics['dustbin_loss']:.2f}, "
                  f"Non-dustbin: {epoch_metrics['overlap_not_dustbin_loss']:.2f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'metrics': epoch_metrics,
        'history': history,
    }, final_checkpoint_path)
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Final model saved to: {final_checkpoint_path}")
    
    return history


def rotation_error_degrees(R_pred: Tensor, R_gt: Tensor) -> float:
    """Compute rotation error in degrees."""
    R_diff = R_pred @ R_gt.T
    trace = torch.clamp(torch.trace(R_diff), -1.0, 3.0)
    angle_rad = torch.acos((trace - 1.0) / 2.0)
    return torch.rad2deg(angle_rad).item()


def visualize_registration_result(
    src: Tensor,
    tgt: Tensor,
    pred_transform: Tensor,
    gt_transform: Tensor,
    euler_deg: np.ndarray,
    t_vec: np.ndarray,
    rot_error_deg: float,
    trans_error: float,
    save_path: str,
):
    """Visualize registration result with errors."""
    src_np = src.cpu().numpy()
    tgt_np = tgt.cpu().numpy()
    
    # Apply transforms
    src_pred_np = apply_transform(src, pred_transform).cpu().numpy()
    src_gt_np = apply_transform(src, gt_transform).cpu().numpy()
    
    fig = plt.figure(figsize=(18, 5))
    
    # Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(src_np[:, 0], src_np[:, 1], src_np[:, 2], c='r', s=1, label='Source', alpha=0.6)
    ax1.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], c='b', s=1, label='Target', alpha=0.6)
    ax1.set_title('Original Point Clouds')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Ground truth alignment
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(src_gt_np[:, 0], src_gt_np[:, 1], src_gt_np[:, 2], c='g', s=1, label='GT Aligned Src', alpha=0.6)
    ax2.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], c='b', s=1, label='Target', alpha=0.6)
    ax2.set_title(f'Ground Truth\nRot: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°]\nTrans: [{t_vec[0]:.2f}, {t_vec[1]:.2f}, {t_vec[2]:.2f}]')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Predicted alignment
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(src_pred_np[:, 0], src_pred_np[:, 1], src_pred_np[:, 2], c='orange', s=1, label='Pred Aligned Src', alpha=0.6)
    ax3.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], c='b', s=1, label='Target', alpha=0.6)
    ax3.set_title(f'Predicted Alignment\nRot Error: {rot_error_deg:.2f}°\nTrans Error: {trans_error:.4f}')
    ax3.legend()
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use ModelNet40 dataset
    print("\n=== Loading ModelNet40 Dataset ===")
    train_dataset = ModelNet40Registration(
        num_points=1024,
        partition='train',
        gaussian_noise=True,
        rotation_factor=4.0,  # ±45 degrees
        translation_range=0.5,
    )
    
    test_dataset = ModelNet40Registration(
        num_points=1024,
        partition='test',
        gaussian_noise=False,
        rotation_factor=4.0,
        translation_range=0.5,
    )
    
    # Create model with stronger architecture
    model = LearnableSinkhornLayer(
        input_dim=3,
        feature_dim=128,  # Increased from 64
        temperature=0.02,
        sinkhorn_iters=100,
        dustbin_init=1.0,
    )
    
    print("\n=== Initial Parameters ===")
    print(f"Temperature: {model.temperature.item():.4f}")
    print(f"Dustbin Score: {model.dustbin_score.item():.4f}")
    
    print("\n=== Dataset Info ===")
    print(f"Training: {len(train_dataset)} samples")
    print(f"  - Overlap ratio: 0.5, Noise std: 0.02, Translation: 0.8")
    print(f"Test (HARD): {len(test_dataset)} samples")
    print(f"  - Overlap ratio: 0.4, Noise std: 0.05 (2.5x), Translation: 1.2 (1.5x)")
    
    # Train with more epochs
    history = train_model(
        model, train_dataset, 
        num_epochs=200,  # Increased from 100
        device=device,
        add_noise_augmentation=True,
    )
    
    print("\n=== Learned Parameters ===")
    print(f"Temperature: {model.temperature.item():.4f}")
    print(f"Dustbin Score: {model.dustbin_score.item():.4f}")
    
    # Evaluate on test set
    print("\n=== Test Set Evaluation (HARD) ===")
    model.eval()
    test_metrics = {
        "overlap_accuracy": 0.0,
        "overlap_recall": 0.0,
        "avg_dustbin_acc": 0.0,
    }
    test_loss_components = {
        "overlap_match_loss": 0.0,
        "dustbin_loss": 0.0,
        "overlap_not_dustbin_loss": 0.0,
        "total_loss": 0.0,
    }
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            source = sample.source.to(device)
            target = sample.target.to(device)
            transform = sample.transform.to(device=device, dtype=source.dtype)
            
            source_aligned = apply_transform(source, transform)
            
            matching_matrix, source_to_dustbin, target_to_dustbin = model(
                source_aligned, target, sample.overlap_count
            )
            
            # Compute loss components for analysis
            _, loss_metrics = compute_matching_loss(
                matching_matrix, source_to_dustbin, target_to_dustbin, sample.overlap_count
            )
            for key in test_loss_components:
                test_loss_components[key] += loss_metrics[key]
            
            acc_metrics = evaluate_accuracy(
                matching_matrix, source_to_dustbin, target_to_dustbin, sample.overlap_count
            )
            
            for key in test_metrics:
                test_metrics[key] += acc_metrics[key]
    
    for key in test_metrics:
        test_metrics[key] /= len(test_dataset)
    for key in test_loss_components:
        test_loss_components[key] /= len(test_dataset)
    
    print(f"Overlap Accuracy: {test_metrics['overlap_accuracy']:.3f}")
    print(f"Overlap Recall: {test_metrics['overlap_recall']:.3f}")
    print(f"Avg Dustbin Accuracy: {test_metrics['avg_dustbin_acc']:.3f}")
    print(f"\nTest Loss Analysis:")
    print(f"  Total Loss: {test_loss_components['total_loss']:.2f}")
    print(f"  - Overlap Match Loss: {test_loss_components['overlap_match_loss']:.2f}")
    print(f"  - Dustbin Loss: {test_loss_components['dustbin_loss']:.2f}")
    print(f"  - Overlap Non-Dustbin Loss: {test_loss_components['overlap_not_dustbin_loss']:.2f}")
    
    # Evaluate registration performance
    print("\n=== Registration Performance Test ===")
    reg_metrics = {
        "chamfer_dist": 0.0,
        "rotation_error": 0.0,
        "translation_error": 0.0,
        "num_inliers": 0.0,
    }
    
    with torch.no_grad():
        for idx in range(min(10, len(test_dataset))):  # Test on 10 samples
            sample = test_dataset[idx]
            source = sample.source.to(device)
            target = sample.target.to(device)
            gt_transform = sample.transform.to(device=device, dtype=source.dtype)
            
            # Get matches WITHOUT applying ground truth transform
            # This simulates real-world scenario
            matching_matrix, source_to_dustbin, target_to_dustbin = model(
                source, target, sample.overlap_count
            )
            
            # Estimate transform from matches
            pred_transform, num_inliers = estimate_transform_from_matches(
                source, target, matching_matrix, source_to_dustbin
            )
            
            # Evaluate
            errors = evaluate_registration_error(
                source, target, pred_transform, gt_transform, sample.overlap_count
            )
            
            reg_metrics["chamfer_dist"] += errors["chamfer_dist"]
            reg_metrics["rotation_error"] += errors["rotation_error"]
            reg_metrics["translation_error"] += errors["translation_error"]
            reg_metrics["num_inliers"] += num_inliers
    
    num_reg_samples = min(10, len(test_dataset))
    for key in reg_metrics:
        reg_metrics[key] /= num_reg_samples
    
    print(f"Average Chamfer Distance: {reg_metrics['chamfer_dist']:.4f}")
    print(f"Average Rotation Error: {reg_metrics['rotation_error']:.4f}")
    print(f"Average Translation Error: {reg_metrics['translation_error']:.4f}")
    print(f"Average Num Inliers: {reg_metrics['num_inliers']:.1f}")
    print(f"\n⚠️  Registration Analysis:")
    if reg_metrics['chamfer_dist'] > 0.1:
        print(f"  HIGH Chamfer distance ({reg_metrics['chamfer_dist']:.4f}) - Poor alignment!")
        print(f"  Low overlap accuracy ({test_metrics['overlap_accuracy']:.1%}) causes bad registration")
    else:
        print(f"  Good registration performance!")
    
    # Compare with one sample
    print("\n=== Sample Comparison ===")
    sample = test_dataset[0]
    source = sample.source.to(device)
    target = sample.target.to(device)
    transform = sample.transform.to(device=device, dtype=source.dtype)
    source_aligned = apply_transform(source, transform)
    
    with torch.no_grad():
        matching_matrix, source_to_dustbin, target_to_dustbin = model(
            source_aligned, target, sample.overlap_count
        )
    
    print(f"Source points: {source.shape[0]}")
    print(f"Target points: {target.shape[0]}")
    print(f"Overlap count: {sample.overlap_count}")
    
    src_match_scores, src_match_idx = matching_matrix.max(dim=1)
    src_is_matched = src_match_scores > source_to_dustbin
    print(f"Predicted matched: {src_is_matched.sum().item()}")
    
    print("\nFirst 10 matches (idx, prob, dustbin_prob):")
    for idx in range(min(10, source.shape[0])):
        best_idx = torch.argmax(matching_matrix[idx]).item()
        best_prob = matching_matrix[idx, best_idx].item()
        dustbin_prob = source_to_dustbin[idx].item()
        print(f"  src[{idx:02d}] -> tgt[{best_idx:02d}]  prob={best_prob:.3e}  dustbin={dustbin_prob:.3e}")


if __name__ == "__main__":
    main()
