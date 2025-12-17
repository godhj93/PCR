"""Debug script to check what the flow matching model is actually learning."""

import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data import ModelNet40
from scripts.flow_matching_correspondence import (
    CorrespondenceFlowNetwork,
    build_pi_star_from_corr,
)
import numpy as np

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CorrespondenceFlowNetwork(point_dim=3, hidden_dim=128).to(device)
checkpoint = torch.load("correspondence_flow_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get a test sample
dataset = ModelNet40(num_points=256, partition='test', gaussian_noise=False, unseen=False, factor=4)
sample = dataset[0]

p = sample['p']  # (3, N)
q = sample['q']  # (3, N)
corr_idx = sample['corr_idx']  # (N, 2)

N = p.shape[1]
M = q.shape[1]

# Prepare tensors
P = torch.from_numpy(p).unsqueeze(0).permute(0, 2, 1).to(device)  # (1, N, 3)
Q = torch.from_numpy(q).unsqueeze(0).permute(0, 2, 1).to(device)  # (1, M, 3)
corr = torch.from_numpy(corr_idx).unsqueeze(0).to(device)  # (1, N, 2)

# Ground truth
Pi_1 = build_pi_star_from_corr(corr, N, M)  # (1, N, M)
Pi_0 = torch.full_like(Pi_1, 1.0 / M)

print(f"Pi_0 stats: min={Pi_0.min():.6f}, max={Pi_0.max():.6f}, mean={Pi_0.mean():.6f}")
print(f"Pi_1 stats: min={Pi_1.min():.6f}, max={Pi_1.max():.6f}, mean={Pi_1.mean():.6f}")
print(f"Pi_1 row sums: {Pi_1.sum(dim=-1)[0, :5]}")  # First 5 rows

# Check velocities at different time steps
with torch.no_grad():
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = torch.tensor([t_val], device=device)
        
        # Ground truth interpolation
        Pi_t_gt = (1 - t_val) * Pi_0 + t_val * Pi_1
        
        # Model prediction
        v_theta = model(P, Q, Pi_t_gt, t)
        
        # Target velocity
        dPi_t = Pi_1 - Pi_0
        
        print(f"\n=== t = {t_val:.2f} ===")
        print(f"Pi_t stats: min={Pi_t_gt.min():.6f}, max={Pi_t_gt.max():.6f}, mean={Pi_t_gt.mean():.6f}")
        print(f"v_theta stats: min={v_theta.min():.6f}, max={v_theta.max():.6f}, mean={v_theta.mean():.6f}, std={v_theta.std():.6f}")
        print(f"dPi_t stats: min={dPi_t.min():.6f}, max={dPi_t.max():.6f}, mean={dPi_t.mean():.6f}")
        print(f"MSE loss: {((v_theta - dPi_t) ** 2).mean():.8f}")
        
        # Check a specific row
        row_idx = 0
        print(f"Row {row_idx}: v_theta max={v_theta[0, row_idx].max():.6f} at j={v_theta[0, row_idx].argmax()}")
        print(f"Row {row_idx}: dPi_t max={dPi_t[0, row_idx].max():.6f} at j={dPi_t[0, row_idx].argmax()}")
        print(f"Row {row_idx}: GT match j={corr[0, row_idx, 1].item()}")

# Now check inference
print("\n=== Inference Test (Logit Space) ===")
log_Pi = torch.full((1, N, M), -np.log(M), device=device)
dt = 0.01
num_steps = 100

for step in range(num_steps):
    t = torch.full((1,), step * dt, device=device)
    
    # Convert to probability space
    Pi = torch.softmax(log_Pi, dim=-1)
    
    # Predict velocity
    v = model(P, Q, Pi, t)
    
    # Update in logit space
    log_Pi = log_Pi + dt * v * M
    
    if step % 20 == 0:
        Pi_check = torch.softmax(log_Pi, dim=-1)
        print(f"Step {step}: Pi min={Pi_check.min():.6f}, max={Pi_check.max():.6f}, mean={Pi_check.mean():.6f}")

# Final matching
Pi_final = torch.softmax(log_Pi, dim=-1)
j_pred = Pi_final[0].argmax(dim=-1).cpu().numpy()
j_gt = np.full(N, -1, dtype=np.int64)
for k in range(N):
    i_k, j_k = corr_idx[k]
    j_gt[i_k] = j_k

correct = (j_pred == j_gt)
print(f"\nFinal Accuracy: {correct.mean()*100:.2f}% ({correct.sum()}/{N})")

# Check some specific predictions
print("\nFirst 10 predictions:")
for i in range(10):
    print(f"Point {i}: pred={j_pred[i]}, gt={j_gt[i]}, correct={j_pred[i]==j_gt[i]}, confidence={Pi_final[0, i, j_pred[i]].item():.4f}")
