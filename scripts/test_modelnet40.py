"""Test learnable Sinkhorn with ModelNet40 and visualizations."""

import sys
from pathlib import Path
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data import ModelNet40Registration
from testing import apply_transform
from scripts.learnable_sinkhorn import (
    LearnableSinkhornLayer,
    estimate_transform_from_matches,
    evaluate_registration_error,
)


def rotation_error_degrees(R_pred, R_gt):
    """Compute rotation error in degrees."""
    R_diff = R_pred @ R_gt.T
    trace = torch.clamp(torch.trace(R_diff), -1.0, 3.0)
    angle_rad = torch.acos((trace - 1.0) / 2.0)
    return torch.rad2deg(angle_rad).item()


def visualize_registration(
    src, tgt, pred_transform, gt_transform, euler_deg, t_vec,
    rot_error_deg, trans_error, save_path
):
    """Visualize registration result."""
    src_np = src.cpu().numpy()
    tgt_np = tgt.cpu().numpy()
    src_pred_np = apply_transform(src, pred_transform).cpu().numpy()
    src_gt_np = apply_transform(src, gt_transform).cpu().numpy()
    
    fig = plt.figure(figsize=(18, 5))
    
    # Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(src_np[:, 0], src_np[:, 1], src_np[:, 2], c='r', s=1, alpha=0.6)
    ax1.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], c='b', s=1, alpha=0.6)
    ax1.set_title('Original')
    ax1.legend(['Source', 'Target'])
    
    # Ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(src_gt_np[:, 0], src_gt_np[:, 1], src_gt_np[:, 2], c='g', s=1, alpha=0.6)
    ax2.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], c='b', s=1, alpha=0.6)
    ax2.set_title(f'GT\nRot:[{euler_deg[0]:.1f}°,{euler_deg[1]:.1f}°,{euler_deg[2]:.1f}°]\nT:[{t_vec[0]:.2f},{t_vec[1]:.2f},{t_vec[2]:.2f}]')
    ax2.legend(['GT Aligned', 'Target'])
    
    # Predicted
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(src_pred_np[:, 0], src_pred_np[:, 1], src_pred_np[:, 2], c='orange', s=1, alpha=0.6)
    ax3.scatter(tgt_np[:, 0], tgt_np[:, 1], tgt_np[:, 2], c='b', s=1, alpha=0.6)
    ax3.set_title(f'Predicted\nRot Error:{rot_error_deg:.2f}°\nTrans Error:{trans_error:.4f}')
    ax3.legend(['Pred Aligned', 'Target'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load test dataset
    print("\nLoading ModelNet40 test set...")
    test_dataset = ModelNet40Registration(
        num_points=1024,
        partition='test',
        gaussian_noise=False,
        rotation_factor=4.0,
        translation_range=0.5,
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Load trained model
    model = LearnableSinkhornLayer(
        input_dim=3,
        feature_dim=128,
        temperature=0.02,
        sinkhorn_iters=100,
        dustbin_init=1.0,
    )
    
    checkpoint_path = "checkpoints/learnable_sinkhorn/best_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    else:
        print(f"\nNo checkpoint found, using untrained model")
    
    model = model.to(device)
    model.eval()
    
    # Create output directory
    vis_dir = "scripts/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Test on first 5 samples
    print("\n=== Testing on 5 samples ===")
    for idx in range(min(5, len(test_dataset))):
        sample = test_dataset[idx]
        src = sample['src'].to(device)
        tgt = sample['tgt'].to(device)
        gt_transform = sample['transform'].to(device)
        gt_R = sample['R'].to(device)
        gt_t = sample['t'].to(device)
        euler = sample['euler'].numpy()
        euler_deg = np.degrees(euler)
        
        overlap_count = src.shape[0]  # Full overlap
        
        with torch.no_grad():
            # Get matches
            matching_matrix, source_to_dustbin, target_to_dustbin = model(
                src, tgt, overlap_count
            )
            
            # Estimate transform
            pred_transform, num_inliers = estimate_transform_from_matches(
                src, tgt, matching_matrix, source_to_dustbin
            )
        
        # Compute errors
        pred_R = pred_transform[:3, :3]
        pred_t = pred_transform[:3, 3]
        
        rot_error_deg = rotation_error_degrees(pred_R, gt_R)
        trans_error = torch.norm(pred_t - gt_t).item()
        
        print(f"\nSample {idx}:")
        print(f"  GT Rotation (deg): [{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}]")
        print(f"  GT Translation: [{gt_t[0]:.3f}, {gt_t[1]:.3f}, {gt_t[2]:.3f}]")
        print(f"  Rotation Error: {rot_error_deg:.2f}°")
        print(f"  Translation Error: {trans_error:.4f}")
        print(f"  Inliers: {num_inliers}/{overlap_count}")
        
        # Visualize
        save_path = os.path.join(vis_dir, f"sample_{idx:02d}.png")
        visualize_registration(
            src, tgt, pred_transform, gt_transform,
            euler_deg, gt_t.cpu().numpy(),
            rot_error_deg, trans_error, save_path
        )
    
    print(f"\nVisualizations saved to: {vis_dir}/")


if __name__ == "__main__":
    main()
