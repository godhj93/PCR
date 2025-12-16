import torch
from tqdm import tqdm
from utils.model import FLOW
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

def train_one_epoch(model: FLOW, train_loader, optimizer, loss_fn, epoch, cfg):
    model.train() 
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
    
    for data in pbar:
        
        # --- 1. 데이터 준비 ---
        p_batch = data['p'].to(cfg.device)         # (B, 3, N)
        q_batch = data['q'].to(cfg.device)         # (B, 3, N)
        corr_idx = data['corr_idx'].to(cfg.device) # (B, N, 2)
        
        B, N = p_batch.shape[0], p_batch.shape[2]
        C_dim = cfg.model.c_dim

        # --- 2. 전역 조건 (Global Context) 생성 ---
        # (B, c_dim * 2)
        c_global = model.encode(p_batch, q_batch) 

        # --- 3. GT 대응 쌍 (x0, x1) 준비 (Gather) ---
        p_idx = corr_idx[..., 0].unsqueeze(1).expand(B, 3, N)
        q_idx = corr_idx[..., 1].unsqueeze(1).expand(B, 3, N)
        
        x0_batch = torch.gather(p_batch, 2, p_idx) # (B, 3, N)
        x1_batch = torch.gather(q_batch, 2, q_idx) # (B, 3, N)

        # --- 4. 배치화 (Flatten) ---
        x_0 = x0_batch.transpose(1, 2).reshape(-1, 3) # (B*N, 3)
        x_1 = x1_batch.transpose(1, 2).reshape(-1, 3) # (B*N, 3)
        
        # (B*N, c_dim * 2)
        c_points = c_global.unsqueeze(1).expand(B, N, -1).reshape(-1, C_dim * 2)

        # --- 5. CFM 손실 계산 ---
        t = torch.rand(B * N, 1, device=cfg.device)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        
        predicted_dx_t = model(t, c_points, x_t) # model.forward() 호출
        
        loss = loss_fn(predicted_dx_t, dx_t)
        
        # --- 6. 역전파 ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.5f}")
    
    train_loss = total_loss / len(train_loader)
    
    # Test
    visualize_now = cfg.test.visualize and (epoch % cfg.test.vis_every_epoch == 0)
    test_loss, test_acc = test_one_epoch(model, train_loader, loss_fn, cfg, epoch=epoch, visualize=visualize_now)
    return {'train_loss': train_loss, 'test_loss': test_loss}, test_acc

def test_one_epoch(model: FLOW, test_loader, loss_fn, cfg, epoch=0, visualize=False):
    model.eval()
    all_pred_corrs = []
    all_gt_corrs = []
    
    total_test_loss = 0.0
    vis_idx = 0  # 시각화 카운터
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for data in pbar:
            p_batch = data['p'].to(cfg.device) # (B, 3, N)
            q_batch = data['q'].to(cfg.device) # (B, 3, M)
            corr_idx_gt = data['corr_idx'].to(cfg.device) # (B, N, 2)
            
            B, N, M = p_batch.shape[0], p_batch.shape[2], q_batch.shape[2]
            C_dim = cfg.model.c_dim

            # --- 1. 전역 조건 생성 ---
            c_global = model.encode(p_batch, q_batch)
            c_points_ode = c_global.unsqueeze(1).expand(B, N, -1).reshape(-1, C_dim * 2)

            # ==================================================
            #   작업 A: 추론 (Inference)
            # ==================================================
            x_t_ode = p_batch.transpose(1, 2).reshape(-1, 3)
            
            n_steps = cfg.test.num_steps
            time_steps = torch.linspace(0, 1.0, n_steps + 1, device=cfg.device)
            
            for i in range(n_steps):
                x_t_ode = model.step(x_t=x_t_ode,
                                 t_start=time_steps[i],
                                 t_end=time_steps[i + 1],
                                 c=c_points_ode)
            
            p_flowed = x_t_ode.reshape(B, N, 3) # (B, N, 3)

            q_batch_trans = q_batch.transpose(1, 2) # (B, M, 3)
            dist_matrix = torch.cdist(p_flowed, q_batch_trans, p=2.0) # (B, N, M)
            corr_j_pred = torch.argmin(dist_matrix, dim=2) # (B, N)
            corr_i_pred = torch.arange(N, device=cfg.device).unsqueeze(0).expand(B, N)
            
            batch_corr_pred = torch.stack([corr_i_pred, corr_j_pred], dim=2) 
            
            # --- 정확도 계산을 위해 CPU로 이동 ---
            batch_corr_pred_cpu = batch_corr_pred.cpu()
            corr_idx_gt_cpu = corr_idx_gt.cpu()
            all_pred_corrs.append(batch_corr_pred_cpu)
            all_gt_corrs.append(corr_idx_gt_cpu)
            
            # ==================================================
            #   작업 B: 테스트 손실 (CFM Loss) 계산
            # ==================================================
            p_idx = corr_idx_gt[..., 0].unsqueeze(1).expand(B, 3, N)
            q_idx = corr_idx_gt[..., 1].unsqueeze(1).expand(B, 3, N)
            x0_batch = torch.gather(p_batch, 2, p_idx)
            x1_batch = torch.gather(q_batch, 2, q_idx)
            x_0 = x0_batch.transpose(1, 2).reshape(-1, 3)
            x_1 = x1_batch.transpose(1, 2).reshape(-1, 3)
            c_points_loss = c_global.unsqueeze(1).expand(B, N, -1).reshape(-1, C_dim * 2)
            t = torch.rand(B * N, 1, device=cfg.device)
            x_t_loss = (1 - t) * x_0 + t * x_1
            dx_t = x_1 - x_0
            predicted_dx_t = model(t, c_points_loss, x_t_loss)
            loss = loss_fn(predicted_dx_t, dx_t)
            total_test_loss += loss.item()

            # ==================================================
            #   작업 C: 시각화 (옵션)
            # ==================================================
            if visualize and vis_idx < cfg.test.vis_count:
                # 배치 중 첫 번째 샘플(b_idx=0)만 시각화
                b_idx = 0
                
                # (N, 3)
                P_np = p_batch[b_idx].transpose(0, 1).cpu().numpy()
                Q_np = q_batch[b_idx].transpose(0, 1).cpu().numpy()
                gt_corr_np = corr_idx_gt_cpu[b_idx].numpy() # (N, 2)
                pred_corr_np = batch_corr_pred_cpu[b_idx].numpy() # (N, 2)
                
                # GT 대응 관계를 딕셔너리로 만듦 (빠른 비교를 위해)
                # { p_idx: gt_q_idx }
                gt_map = {i: j for i, j in gt_corr_np}
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # 점군을 분리하기 위한 오프셋
                vis_offset = np.array([2.0, 0, 0])
                Q_np_offset = Q_np + vis_offset
                
                # 1. 점군 플로팅
                ax.scatter(P_np[:, 0], P_np[:, 1], P_np[:, 2], c='blue', s=1, label='P (Source)')
                ax.scatter(Q_np_offset[:, 0], Q_np_offset[:, 1], Q_np_offset[:, 2], c='orange', s=1, label='Q (Target)')

                # 2. 대응선 플로팅
                for i in range(N):
                    # P의 i번째 점
                    p_pt = P_np[i]
                    
                    # P[i]에 대한 예측된 Q의 인덱스 j
                    j_pred = pred_corr_np[i, 1] 
                    q_pt = Q_np_offset[j_pred]
                    
                    # P[i]에 대한 실제 GT Q의 인덱스 j
                    j_gt = gt_map.get(i, -1) # GT가 없는 경우(partial) -1
                    
                    color = 'g' if j_pred == j_gt else 'r'
                    alpha = 0.5 if color == 'g' else 0.8
                    linewidth = 0.5 if color == 'g' else 0.7
                    
                    ax.plot([p_pt[0], q_pt[0]], 
                            [p_pt[1], q_pt[1]], 
                            [p_pt[2], q_pt[2]], 
                            c=color, alpha=alpha, linewidth=linewidth)

                ax.legend()
                ax.set_title(f"Epoch {epoch} - Vis {vis_idx+1}")
                
                # Accuracy
                correct_count = sum(1 for i in range(N) if pred_corr_np[i, 1] == gt_map.get(i, -1))
                accuracy = correct_count / N * 100.0
                ax.text2D(0.05, 0.95, f"Accuracy: {accuracy:.2f}%", transform=ax.transAxes)                
                
                # (hydra의 run dir에 저장한다고 가정)
                plt.savefig(f"epoch_{epoch}_vis_{vis_idx+1}.png")
                plt.close(fig)
                
                vis_idx += 1


    # --- 5. 결과 취합 ---
    avg_test_loss = total_test_loss / len(test_loader)
    
    all_pred_corrs_tensor = torch.cat(all_pred_corrs, dim=0)
    all_gt_corrs_tensor = torch.cat(all_gt_corrs, dim=0)
    
    # 정확도 계산
    accuracy = (all_pred_corrs_tensor == all_gt_corrs_tensor).all(dim=2).float().mean().item()
    
    return avg_test_loss, accuracy

def visualize_registration(P, Q, R, t, vis, title="Registration Result"):
    """
    P: (3, N) source point cloud
    Q: (3, N) target point cloud
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    """
    # P를 변환하여 Q와 정합 (P_transformed = R * P + t)
    P_transformed = R @ P + t[:, None]

    fig = plt.figure(figsize=(12, 6))

    # 1. 정합 전 (Before Registration)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(P[0], P[1], P[2], c='blue', s=2, label='Source (P)')
    ax1.scatter(Q[0], Q[1], Q[2], c='red', s=2, label='Target (Q)')
    ax1.set_title("Before Registration")
    ax1.legend()

    # 2. 정합 후 (After Registration)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(P_transformed[0], P_transformed[1], P_transformed[2], c='blue', s=2, label='Transformed P')
    ax2.scatter(Q[0], Q[1], Q[2], c='red', s=2, label='Target (Q)')
    ax2.set_title("After Registration (GT)")
    ax2.legend()

    if vis:
        plt.show()
    
    