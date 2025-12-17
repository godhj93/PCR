import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_interpolation_timing_on_sphere(angle_degrees=170):
    """
    (최종 수정 코드) SLERP와 NLERP의 경로 위 '시간에 따른 위치' 차이를 명확히 시각화하고,
    정확한 각도 오차를 그래프로 보여줍니다.
    """
    # 1. 데이터 설정
    angle_radians = np.deg2rad(angle_degrees)
    v_a = np.array([0, 0, 1.0])
    v_b = np.array([np.sin(angle_radians), 0, np.cos(angle_radians)])
    
    # 전체 경로를 그리기 위한 t 값
    t_path = np.linspace(0, 1, 100)
    # 특정 시점의 위치를 표시하기 위한 t 값
    t_points = np.array([0, 0.25, 0.5, 0.75, 1.0])

    # 2. 경로 및 특정 지점 계산
    dot = np.dot(v_a, v_b)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)

    def slerp(t_val):
        return (np.sin((1 - t_val) * omega) / sin_omega) * v_a + \
               (np.sin(t_val * omega) / sin_omega) * v_b

    def nlerp(t_val):
        lerp_val = (1 - t_val) * v_a + t_val * v_b
        return lerp_val / np.linalg.norm(lerp_val)

    # 전체 경로 계산 (SLERP 경로 하나만 그려도 됨)
    slerp_full_path = np.array([slerp(t) for t in t_path])

    # 특정 시점(t_points)에서의 위치 계산
    slerp_specific_points = np.array([slerp(t) for t in t_points])
    nlerp_specific_points = np.array([nlerp(t) for t in t_points])

    # 3. 전체 t에 대한 각도 오차 계산
    angular_error_deg = np.rad2deg(np.arccos(np.clip(np.sum(slerp_full_path * np.array([nlerp(t) for t in t_path]), axis=1), -1.0, 1.0)))

    # 4. 시각화
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(f'Interpolation Timing Analysis (Initial Angle: {angle_degrees}°)', fontsize=20)

    # 왼쪽: 3D 시간차 시각화
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Timing Difference on the Path", fontsize=16)
    
    # 구 와이어프레임
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x, y, z = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    ax1.plot_wireframe(x, y, z, color="lightgray", alpha=0.2)

    # 경로
    ax1.plot(slerp_full_path[:, 0], slerp_full_path[:, 1], slerp_full_path[:, 2], 'g-', label='Geodesic Path', linewidth=3, alpha=0.5)
    
    # 시간별 위치 표시
    ax1.scatter(slerp_specific_points[:, 0], slerp_specific_points[:, 1], slerp_specific_points[:, 2],
                c='green', s=150, ec='black', label='SLERP points (t=0, .25, .5, .75, 1)', zorder=10)
    ax1.scatter(nlerp_specific_points[:, 0], nlerp_specific_points[:, 1], nlerp_specific_points[:, 2],
                c='red', s=150, marker='X', label='NLERP points (t=0, .25, .5, .75, 1)', zorder=10)

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_box_aspect([1,1,1])
    ax1.view_init(elev=20, azim=-70) # 오차가 잘 보이는 측면 뷰
    ax1.legend()

    # 오른쪽: 2D 오차 그래프
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Angular Error vs. Time', fontsize=16)
    ax2.plot(t_path, angular_error_deg, 'm-', linewidth=3)
    max_error_idx = np.argmax(angular_error_deg)
    max_error_val = angular_error_deg[max_error_idx]
    ax2.plot(t_path[max_error_idx], max_error_val, 'bo', markersize=8, label=f'Max Error: {max_error_val:.4f}°')
    ax2.set_xlabel('Interpolation factor (t)')
    ax2.set_ylabel('Error in degrees (°)')
    ax2.grid(True)
    ax2.set_ylim(bottom=0)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    visualize_interpolation_timing_on_sphere(angle_degrees=90)