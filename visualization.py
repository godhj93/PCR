import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from flow_matching.flow_matching.utils.manifolds.se3 import SE3

# --- Matplotlib 3D 좌표계 그리기 헬퍼 함수 (이전과 동일) ---
def draw_frame(ax, T, scale=0.2, alpha=1.0, label=None):
    if isinstance(T, torch.Tensor):
        T = T.cpu().numpy()
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ['r', 'g', 'b']
    axes_vectors = [R[:, 0], R[:, 1], R[:, 2]]
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],
                  axes_vectors[i][0], axes_vectors[i][1], axes_vectors[i][2],
                  length=scale, color=colors[i], alpha=alpha)
    if label:
        ax.text(origin[0], origin[1], origin[2], label, fontsize=12)

# --- 유틸리티 함수 ---
def hat3(omega: torch.Tensor) -> torch.Tensor:
    ox, oy, oz = omega[..., 0], omega[..., 1], omega[..., 2]
    O = torch.zeros(omega.shape[:-1] + (3, 3), dtype=omega.dtype, device=omega.device)
    O[..., 0, 1] = -oz; O[..., 0, 2] =  oy
    O[..., 1, 0] =  oz; O[..., 1, 2] = -ox
    O[..., 2, 0] = -oy; O[..., 2, 1] =  ox
    return O

# ***** 여기가 수정된 부분 *****
def axis_angle_to_R(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    se3.py 테스트 코드에 있던 올바르고 수치적으로 안정적인 버전.
    """
    # 입력이 스칼라 텐서일 수 있으므로 expand를 위해 1차원으로 만듭니다.
    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    if axis.dim() == 1:
        axis = axis.unsqueeze(0)

    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
    th = angle.squeeze(-1)
    omega = axis * th[..., None]
    Omega = hat3(omega)
    
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(Omega)

    # 작은 각도에서 0으로 나누는 것을 방지하는 수치적으로 안정적인 공식
    # sin(th)/th
    sin_by_th = (torch.sin(th) / th).where(th.abs() > 1e-12, torch.ones_like(th))
    # (1-cos(th))/th^2
    one_minus_cos_by_th2 = ((1 - torch.cos(th)) / (th * th)).where(
        th.abs() > 1e-12, 0.5 * torch.ones_like(th)
    )
    
    sin_by_th = sin_by_th[..., None, None]
    one_minus_cos_by_th2 = one_minus_cos_by_th2[..., None, None]
    
    R = I + sin_by_th * Omega + one_minus_cos_by_th2 * (Omega @ Omega)

    # 배치 차원이 추가되었다면 다시 squeeze
    if R.shape[0] == 1:
        R = R.squeeze(0)
        
    return R

def make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# --- 메인 애니메이션 스크립트 (이전과 동일) ---
if __name__ == "__main__":
    
    torch.set_printoptions(precision=4, sci_mode=False)
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    se3 = SE3().to(device)

    # 1. 시작 포즈 T1과 목표 포즈 T2 정의
    R1 = torch.eye(3, dtype=dtype, device=device)
    t1 = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    T1 = make_T(R1, t1)
    
    R2_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    R2_angle = torch.tensor(torch.pi / 2.0, dtype=dtype, device=device)
    R2 = axis_angle_to_R(R2_axis, R2_angle) # 수정된 함수 호출
    t2 = torch.tensor([1.0, 1.0, 0.5], dtype=dtype, device=device)
    T2 = make_T(R2, t2)
    
    # ... (이하 애니메이션 생성 및 저장 코드는 이전과 동일) ...
    U = se3.logmap(T1, T2)
    num_frames = 100
    trajectory_origins = []
    intermediate_poses = []
    for s in np.linspace(0.0, 1.0, num_frames):
        T_s = se3.expmap(T1, U * s)
        trajectory_origins.append(T_s[:3, 3].cpu().numpy())
        intermediate_poses.append(T_s)
    trajectory = np.array(trajectory_origins)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_range = np.array([trajectory[:, 0].max()-trajectory[:, 0].min(), 
                          trajectory[:, 1].max()-trajectory[:, 1].min(), 
                          trajectory[:, 2].max()-trajectory[:, 2].min()]).max() / 2.0
    mid_x = (trajectory[:, 0].max()+trajectory[:, 0].min()) * 0.5
    mid_y = (trajectory[:, 1].max()+trajectory[:, 1].min()) * 0.5
    mid_z = (trajectory[:, 2].max()+trajectory[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range - 0.2, mid_x + max_range + 0.2)
    ax.set_ylim(mid_y - max_range - 0.2, mid_y + max_range + 0.2)
    ax.set_zlim(mid_z - max_range - 0.2, mid_z + max_range + 0.2)

    def update(frame):
        ax.cla()
        draw_frame(ax, T1, scale=0.2, label='$T_1$ (start)')
        draw_frame(ax, T2, scale=0.2, label='$T_2$ (end)')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'k--', label='Path of Origin (Geodesic)')
        current_pose = intermediate_poses[frame]
        draw_frame(ax, current_pose, scale=0.15, alpha=0.8)
        ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
        ax.set_title(f'SE(3) Geodesic Interpolation (Frame {frame}/{num_frames})')
        ax.legend(loc='upper left'); ax.grid(True)
        ax.set_xlim(mid_x - max_range - 0.2, mid_x + max_range + 0.2)
        ax.set_ylim(mid_y - max_range - 0.2, mid_y + max_range + 0.2)
        ax.set_zlim(mid_z - max_range - 0.2, mid_z + max_range + 0.2)
        return fig,

    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

    output_filename_gif = 'se3_interpolation_corrected.gif'
    print(f"Saving animation to {output_filename_gif}...")
    ani.save(output_filename_gif, writer='imagemagick', fps=20)
    print("GIF saving complete.")