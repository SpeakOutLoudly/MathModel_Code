import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colour
# BT2020数据
BT2020_R = (0.708, 0.292)
BT2020_G = (0.170, 0.797)
BT2020_B = (0.131, 0.046)

# 显示屏RGB三基色
DP_R = (0.64, 0.33)
DP_G = (0.30, 0.60)
DP_B = (0.15, 0.06)

BT2020 = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]
DP = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]
DP2 = [[0.67, 0.33], [0.21, 0.71], [0.14, 0.08]]
M_sRGB_to_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

def DP_xy_to_xyz(xy, Y=10.0):
    xy = np.array(xy)
    x, y = xy[:, 0], xy[:, 1]
    X = x * Y / y
    Z = (1 - x - y) * Y / y
    return np.stack([X, Y*np.ones_like(x), Z], axis=1)

def xyz_to_xy(XYZ):
    XYZ = np.array(XYZ)
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    denom = X + Y + Z
    denom = np.clip(denom, 1e-6, None)
    x = X / denom
    y = Y / denom
    return np.stack([x, y], axis=1)

def rgb_to_xy(rgb, M_rgb_to_xyz):
    xyz = rgb @ M_rgb_to_xyz.T
    x = xyz[:, 0] / (xyz[:, 0] + xyz[:, 1] + xyz[:, 2])
    y = xyz[:, 1] / (xyz[:, 0] + xyz[:, 1] + xyz[:, 2])
    return np.stack([x, y], axis=1)

def sample_bt2020_or_DP(BT2020, n_samples=1):
    """
    在BT.2020色域三角形内进行均匀随机采样。
    
    参数:
        BT2020 (list): 包含三个顶点坐标的列表，格式为[[x_r, y_r], [x_g, y_g], [x_b, y_b]]。
        n_samples (int): 需要生成的样本数量，默认为1。
        
    返回:
        np.array: 采样结果，形状为(n_samples, 2)。
    """
    # 将顶点转换为NumPy数组
    R = np.array(BT2020[0])
    G = np.array(BT2020[1])
    B = np.array(BT2020[2])
    
    # 生成均匀分布的u和v
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)
    
    # 调整u和v使得u + v <= 1
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    
    # 计算采样点的坐标
    w = 1 - u - v
    x = w * R[0] + u * G[0] + v * B[0]
    y = w * R[1] + u * G[1] + v * B[1]
    
    # 组合结果并返回
    return np.column_stack((x, y))

def chromaticity_to_xyz_matrix(primaries, whitepoint):
    M = []
    for x, y in primaries:
        z = 1 - x - y
        M.append([x / y, 1.0, z / y])
    M = np.array(M).T
    Xw, Yw, Zw = whitepoint
    S = np.linalg.inv(M) @ np.array([Xw, Yw, Zw])
    return M * S

def delta_e_00_batch(lab1, lab2):
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    avg_L = 0.5 * (L1 + L2)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = 0.5 * (C1 + C2)

    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    avg_Cp = 0.5 * (C1p + C2p)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    deltahp = h2p - h1p
    deltahp = np.where(deltahp > 180, deltahp - 360, deltahp)
    deltahp = np.where(deltahp < -180, deltahp + 360, deltahp)

    delta_Hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(deltahp / 2))
    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p

    avg_hp = np.where(np.abs(h1p - h2p) > 180, (h1p + h2p + 360) / 2, (h1p + h2p) / 2)
    T = 1 - 0.17 * np.cos(np.radians(avg_hp - 30)) + 0.24 * np.cos(np.radians(2 * avg_hp)) \
        + 0.32 * np.cos(np.radians(3 * avg_hp + 6)) - 0.20 * np.cos(np.radians(4 * avg_hp - 63))

    delta_theta = 30 * np.exp(-((avg_hp - 275) / 25)**2)
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    Sl = 1 + (0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2)
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_theta)) * Rc

    delta_E = np.sqrt(
        (delta_Lp / Sl)**2 +
        (delta_Cp / Sc)**2 +
        (delta_Hp / Sh)**2 +
        Rt * (delta_Cp / Sc) * (delta_Hp / Sh)
    )

    return delta_E

def rgb_bt2020_to_dp(rgb_bt2020, M):
    return np.clip(rgb_bt2020 @ M.T, 0, 1)  # 映射 + 限制在 [0,1]


def f(t):
    delta = 6/29
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4/29))

def xyz_to_lab_batch(xyz, white_point=(0.95047, 1.00000, 1.08883)):
    Xn, Yn, Zn = white_point
    X = xyz[:, 0] / Xn
    Y = xyz[:, 1] / Yn
    Z = xyz[:, 2] / Zn

    fx = f(X)
    fy = f(Y)
    fz = f(Z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=1)

def geometric_loss(M_flat, DP, BT2020):
    DP = np.array(DP)                            # (3, 2)
    DP = np.hstack([DP, np.ones((3, 1))])        # → (3, 3)
    BT2020 = np.array(BT2020)                    # (3, 2)

    M = M_flat.reshape(3, 3)                     # (3, 3)
    mapped = (M @ DP.T).T                        # → (3, 3)
    mapped_xy = mapped[:, :2]                    # → (3, 2)

    loss = np.linalg.norm(mapped_xy - BT2020, axis=1).mean()
    return loss


def E00_loss_fn(M_flat, rgb_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch):
    M = M_flat.reshape(3, 3)
    
    # 映射 BT2020 RGB 到 DP RGB（拟合矩阵 M）
    rgb_dp = rgb_samples @ M.T
    rgb_dp = np.clip(rgb_dp, 0, 1)  # 限制在 [0,1]

    # DP RGB → XYZ → Lab（预测值）
    xyz_pred = rgb_dp @ M_dp_to_xyz.T
    lab_pred = xyz_to_lab_batch(xyz_pred)

    # BT2020 RGB → XYZ → Lab（真实值）
    xyz_true = rgb_samples @ M_bt2020_to_xyz.T
    lab_true = xyz_to_lab_batch(xyz_true)

    # ΔE00
    deltaE = delta_e_00_batch(lab_true, lab_pred)
    return np.mean(deltaE)

def combined_loss(M_flat, rgb_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch, DP_xy, BT2020_xy, alpha=0.9):
    # 变换矩阵
    M = M_flat.reshape(3, 3)
    
    # ===== ΔE00 感知损失 =====
    rgb_dp = rgb_samples @ M.T
    rgb_dp = np.clip(rgb_dp, 0, 1)

    xyz_pred = rgb_dp @ M_dp_to_xyz.T
    lab_pred = xyz_to_lab_batch(xyz_pred)

    xyz_true = rgb_samples @ M_bt2020_to_xyz.T
    lab_true = xyz_to_lab_batch(xyz_true)

    deltaE = delta_e_00_batch(lab_true, lab_pred)
    color_loss = np.mean(deltaE)
    color_loss /= 100
    # ===== 几何误差（顶点 xy 色度匹配）=====
    DP_xyz = DP_xy_to_xyz(DP_xy)  # shape: (3, 3)
    BT2020_xyz = DP_xy_to_xyz(BT2020_xy)

    DP_mapped = M @ DP_xyz  # 映射后的三基色在 BT2020 空间
    DP_mapped_xy = xyz_to_xy(DP_mapped)

    geo_loss = np.linalg.norm(DP_mapped_xy - BT2020_xy, axis=1).mean()

    return alpha * color_loss + (1 - alpha) * geo_loss


def plot_triangles(examples):
    """
    将多个 3x2 的点集合（表示三角形的三个顶点）绘制在同一平面上。
    每个三角形用不同颜色边框，不填充颜色。
    """
    fig, ax = plt.subplots()
    
    # 使用 matplotlib 的内置颜色循环
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, triangle in enumerate(examples):
        triangle = np.array(triangle)  # shape (3, 2)
        
        # 形成闭合路径 (3 + 1) 个点
        polygon = np.vstack([triangle, triangle[0]])
        
        # 绘制边框线
        ax.plot(polygon[:, 0], polygon[:, 1], color=colors[i % len(colors)], label=f'Triangle {i+1}')
    
    ax.set_aspect('equal')  # 保持比例
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Overlayed Triangles')
    ax.legend()
    plt.grid(True)
    plt.show()

def plot_chromaticity_with_triangles(examples):
    """
    在 CIE 1931 xy 色度图上叠加多个 RGB 三角形。
    每个三角形用不同颜色边框，不填充。
    """
    # 初始化底图：CIE 1931 Chromaticity Diagram
    figure, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
    
    # matplotlib 默认颜色循环
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, triangle in enumerate(examples):
        triangle = np.array(triangle)
        polygon = np.vstack([triangle, triangle[0]])  # 连接回起点形成闭环
        axes.plot(polygon[:, 0], polygon[:, 1], color=colors[i % len(colors)],
                  linewidth=2, label=f'Triangle {i+1}')
    
    axes.legend()
    axes.set_title("CIE 1931 Chromaticity Diagram with RGB Triangles")
    plt.grid(True)
    plt.show()


# D65 whitepoint in XYZ
whitepoint = (0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290)

# Step 1: BT2020 → XYZ
M_bt2020_to_xyz = chromaticity_to_xyz_matrix(BT2020, whitepoint)

# Step 2: DP/sRGB → XYZ
M_dp_to_xyz = chromaticity_to_xyz_matrix(DP, whitepoint)

# Step 3: XYZ → DP
M_xyz_to_dp = np.linalg.inv(M_dp_to_xyz)

# Step 4: BT2020 → DP RGB
M_bt2020_to_dp = M_xyz_to_dp @ M_bt2020_to_xyz


# 1. 采样一组 BT.2020 RGB 样本 {c_i}
N = 1000
bt2020_rgb_samples = np.random.rand(N, 3)
#bt2020_rgb_samples = sample_bt2020_or_DP(BT2020, 100)

data = sample_bt2020_or_DP(DP, 3)
#print(data.T)
#print("\n")
#print(M_bt2020_to_xyz)

# 2. 映射至 XYZ → Lab_true
xyz_true = M_bt2020_to_xyz @ bt2020_rgb_samples.T
lab_true = xyz_to_lab_batch(xyz_true.T)

# 3. 应用拟合的映射函数 f() → 显示器RGB
dp_rgb_est = f(bt2020_rgb_samples)

# 4. DP RGB → XYZ → Lab_pred
xyz_pred = M_dp_to_xyz @ dp_rgb_est.T
lab_pred = xyz_to_lab_batch(xyz_pred.T)

# 5. ΔE00损失
delta_E = delta_e_00_batch(lab_true, lab_pred)
loss = np.mean(delta_E)
print(loss)

M0 = np.eye(3).flatten()
res = minimize(
    #E00_loss_fn,
    combined_loss,
    M0,
    args=(bt2020_rgb_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch, DP, BT2020, 0.95),
    #args=(bt2020_rgb_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch),
    #args=(DP, BT2020),
    method='L-BFGS-B',
    options={'maxiter': 500, 'disp': True}
)          # E00损失函数和geometric损失函数

M_opt = res.x.reshape(3, 3)
print(M_opt)

R = [1, 0, 0]
G = [0, 1, 0]
B = [0, 0, 1]
DP_RGB = np.array([R, G, B])
DP_mapped = (M_opt @ DP_RGB.T).T  # shape (3, 3)

DP_xy_mapped = rgb_to_xy(DP_mapped, M_sRGB_to_XYZ)
print(DP_xy_mapped)
examples = [
        BT2020,
        DP,
        DP2,
        DP_xy_mapped
    ]
plot_triangles(examples)
plot_chromaticity_with_triangles(examples)