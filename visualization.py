import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils.data import data_loader, draw_gravity_arrow, draw_uncertainty_cone

def calculate_angular_error(v1, v2):
    """
    두 벡터 사이의 각도 오차(Degree) 계산
    """
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def visualize_gravity_regression(model, loader, device, num_samples=5):
    model.eval()
    
    # 1. 배치 데이터 가져오기
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Test loader is empty.")
        return

    # 2. 데이터 추출 (Dataset __getitem__ 구조 반영)
    # Model은 (B, 3, N) 입력을 기대한다고 가정
    # 'q'는 이미 (3, N)이므로 DataLoader를 거치면 (B, 3, N)이 됨
    input_pc = batch['q'].to(device)       # (B, 3, N) -> 회전된 물체
    gt_gravity = batch['gravity_q'].to(device) # (B, 3) -> 회전된 물체의 로컬 중력

    with torch.no_grad():
        # 3. 모델 추론
        # 가정: 모델 반환값은 (mu, kappa) 튜플 혹은 딕셔너리
        outputs = model(input_pc)
        
        pred_mu = outputs[0]    # (B, 3) 예측된 중력 방향
        pred_kappa = outputs[1] # (B, 1) or (B,) 집중도 (Uncertainty의 역수)

    # 4. CPU 변환 (시각화용)
    # 포인트 클라우드는 시각화를 위해 (B, N, 3)으로 Transpose 필요
    pc_np = input_pc.permute(0, 2, 1).cpu().numpy()  # (B, N, 3)
    gt_np = gt_gravity.cpu().numpy()                 # (B, 3)
    pred_mu_np = pred_mu.cpu().numpy()               # (B, 3)
    pred_kappa_np = pred_kappa.cpu().numpy()         # (B, 1)

    # 5. Plotting
    num_vis = min(num_samples, len(pc_np))
    fig = plt.figure(figsize=(15, 5 * ((num_vis + 2) // 3)))
    
    for i in range(num_vis):
        ax = fig.add_subplot(int(np.ceil(num_vis/3)), 3, i+1, projection='3d')
        
        # 포인트 클라우드 (회전된 상태 Q)
        pts = pc_np[i] # (N, 3)
        centroid = np.mean(pts, axis=0) # 화살표 시작점
        
        # 점군 그리기
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='gray', alpha=0.4, label='Points (Q)')
        
        # 벡터 추출
        v_gt = gt_np[i]
        v_pred = pred_mu_np[i]
        
        # kappa 값 추출 (스칼라)
        kappa_val = pred_kappa_np[i].item() if pred_kappa_np.ndim > 1 else pred_kappa_np[i]
        
        # 화살표 그리기
        draw_gravity_arrow(ax, centroid, v_gt, 'green', 'GT', scale=0.8)
        draw_gravity_arrow(ax, centroid, v_pred, 'red', 'Pred', scale=0.8)
        
        safe_kappa = max(kappa_val, 0.01) 
        draw_uncertainty_cone(ax, centroid, v_pred, safe_kappa, scale=0.8, color='red')
        
        # 오차 계산
        err_deg = calculate_angular_error(v_gt, v_pred)
        
        # 타이틀 정보 표시
        ax.set_title(f"Sample {i}\nError: {err_deg:.2f}°\nKappa (Conf): {kappa_val:.1f}")
        
        # 축 스케일 일정하게 맞추기 (3D 왜곡 방지)
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), 
                              pts[:,1].max()-pts[:,1].min(), 
                              pts[:,2].max()-pts[:,2].min()]).max() / 2.0
        mid_x, mid_y, mid_z = np.mean(pts, axis=0)
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    # Save 
    save_path = os.path.join(args.ckpt_path, "gravity_regression_visualization.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def main(ckpt_path):
    # 1. Config 로드
    yaml_path = os.path.join(os.path.dirname(ckpt_path), '.hydra', 'config.yaml')
        
    print(f"Loading config from: {yaml_path}")
    cfg = OmegaConf.load(yaml_path)

    # 2. 모델 로드
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = instantiate(cfg.model).to(device)
    
    # 3. 체크포인트(가중치) 로드
    print(f"Loading weights from {ckpt_path}")
    ckpt = torch.load(os.path.join(ckpt_path, 'weights', 'ckpt.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 4. 데이터 로더 준비
    # train.py와 동일하게 호출 (보통 (train_loader, test_loader) 반환)
    print("Loading Dataset...")
    _, test_loader = data_loader(cfg)
    
    # 5. 시각화 실행
    print("Visualizing Gravity Regression...")
    visualize_gravity_regression(model, test_loader, device, num_samples=cfg.test.vis_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Gravity Regression Model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()
    
    main(args.ckpt_path)
    
    print("Visualization complete.")