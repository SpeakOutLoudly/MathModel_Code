import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import colour
from colormath.color_objects import LabColor, XYZColor
from colormath.color_conversions import convert_color

BT2020 = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]]
sRGB_DP = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]
NTSC = [[0.67, 0.33], [0.21, 0.71], [0.14, 0.08]]

M_sRGB_to_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

def lab_to_xyz_batch(lab_array):
    result = []
    for lab in lab_array:
        lab_color = LabColor(*lab)
        xyz_color = convert_color(lab_color, XYZColor)
        result.append([xyz_color.xyz_x, xyz_color.xyz_y, xyz_color.xyz_z])
    return np.array(result)

def rgb_to_xy(rgb, M_rgb_to_xyz):
    xyz = rgb @ M_rgb_to_xyz.T
    x = xyz[:, 0] / (xyz[:, 0] + xyz[:, 1] + xyz[:, 2])
    y = xyz[:, 1] / (xyz[:, 0] + xyz[:, 1] + xyz[:, 2])
    return np.stack([x, y], axis=1)

def xyz_to_xy_test(M_opt, RGB_basic, M_bt2020_to_xyz):
    # BT2020 to DP
    M_opt_inv = np.linalg.inv(M_opt)
    dp_rgb_mapped = (M_opt_inv @ RGB_basic.T).T  # shape (3, 3)
    BT2020_to_DP_mapped = rgb_to_xy(dp_rgb_mapped, M_bt2020_to_xyz)
    return BT2020_to_DP_mapped

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
    lab1 = np.array(lab1)
    lab2 = np.array(lab2)
    
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

def combined_loss(M_flat, rgb_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch):

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

    return color_loss 

def optimize_model_N_times(whitepoint, sRGB_DP, M_flat_init, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch,
                           N=10, method='DE', random_seed_offset=31):
    """
    对比不同优化器在 N 轮随机样本下的表现

    参数：
    - M_flat_init: 初始 M（flatten）
    - M_bt2020_to_xyz: BT2020 → XYZ 变换矩阵
    - M_dp_to_xyz: DP → XYZ 变换矩阵
    - xyz_to_lab_batch: XYZ → Lab 转换函数（批量）
    - N: 循环次数
    - method: 优化方法选择，'L-BFGS-B' 或 'DE'
    - random_seed_offset: 随机种子偏移量，确保每轮样本不同但可复现

    返回：
    - losses: ndarray[N]，每轮优化得到的 loss
    """

    losses = []
    area_diffs = []
    RGB_basic = np.eye(3)
    ref_area = triangle_area(sRGB_DP)

    for i in range(N):
        seed = i + random_seed_offset
        np.random.seed(seed)
        test_samples = np.random.rand(100, 3)

        def loss_fn(M_flat):
            return combined_loss(M_flat, test_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch)

        if method == 'DE':
            bounds = [(-2, 2)] * 9
            res = differential_evolution(
                loss_fn,
                bounds,
                strategy='best1bin',
                maxiter=1000,
                polish=True,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'DE'")

        M_opt = res.x.reshape(3, 3)
        # ===========================
        BT2020_to_DP_mapped = xyz_to_xy_test(M_opt, RGB_basic, M_bt2020_to_xyz)
        BT_mapped_xyz = chromaticity_to_xyz_matrix(BT2020_to_DP_mapped, whitepoint)
        BT_mapped_lab = xyz_to_lab_batch(BT_mapped_xyz)
        BT_lab = xyz_to_lab_batch(M_sRGB_to_XYZ)
        loss = np.mean(delta_e_00_batch(BT_mapped_lab, BT_lab))
        # ===========================
        # final_loss = loss_fn(res.x)
        losses.append(loss)

        triangle_xy = xyz_to_xy_test(M_opt, RGB_basic, M_bt2020_to_xyz)
        area = triangle_area(triangle_xy)
        area_diff = abs(area - ref_area)
        area_diffs.append(area_diff)

    return np.array(losses), np.array(area_diffs)

def triangle_area(pts):
    """
    计算三角形面积：pts 是 3x2 的 xy 坐标矩阵
    使用 Shoelace formula（鞋带公式）
    """
    pts = np.array(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))


def plot_chromaticity_with_triangles(example_dict):
    """
    在 CIE 1931 xy 色度图上叠加多个 RGB 三角形。
    前两个三角形为实线，后续为虚线，图例使用变量名。
    """
    figure, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_solid = '-'
    linestyle_dashed = '--'

    for i, (label, triangle) in enumerate(example_dict.items()):
        triangle = np.array(triangle)
        polygon = np.vstack([triangle, triangle[0]])
        linestyle = linestyle_solid if i < 2 else linestyle_dashed

        axes.plot(polygon[:, 0], polygon[:, 1],
                  color=colors[i % len(colors)],
                  linewidth=2,
                  linestyle=linestyle,
                  label=label)

    axes.legend()
    axes.set_title("CIE 1931 Chromaticity Diagram with RGB Triangles")
    plt.grid(True)
    plt.show()


def plot_loss_statistics(losses, title='Loss Distribution', method_name='L-BFGS-B'):
    """
    绘制柱状图并显示统计信息。

    参数：
    - losses: 一维 ndarray，优化 N 次的 loss 值
    - title: 图表标题
    - method_name: 优化方法名称，用于图表显示
    """

    # 计算统计量
    mean_loss = np.mean(losses)
    min_loss = np.min(losses)
    std_loss = np.std(losses)

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(losses)), losses, color='skyblue', edgecolor='black')

    # 高亮最小值
    min_index = np.argmin(losses)
    bars[min_index].set_color('orange')

    # 标注统计量
    plt.axhline(mean_loss, color='red', linestyle='--', label=f'Mean: {mean_loss:.4f}')
    plt.axhline(min_loss, color='green', linestyle='--', label=f'Min: {min_loss:.4f}')
    plt.text(len(losses) - 1, mean_loss + 0.05, f'σ: {std_loss:.4f}', color='red', fontsize=10, ha='right')

    # 图形美化
    plt.title(f'{title} ({method_name})', fontsize=14)
    plt.xlabel('Trial Index')
    plt.ylabel('Loss Value')
    plt.xticks(range(len(losses)))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    # D65 whitepoint in XYZ
    whitepoint = (0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290)
    # BT2020 → XYZ
    M_bt2020_to_xyz = chromaticity_to_xyz_matrix(BT2020, whitepoint)
    # DP/sRGB → XYZ
    M_dp_to_xyz = chromaticity_to_xyz_matrix(sRGB_DP, whitepoint)

    # =============== 训练部分 ===============
    # 采样一组 BT.2020 RGB 样本 {c_i}
    M0 = np.eye(3).flatten()
    M0_flat = np.eye(3).flatten()

    # L-BFGS-B 优化 50 次
    losses_lbfgs, area_diffs = optimize_model_N_times(
        whitepoint,
        sRGB_DP,
        M0_flat,
        M_bt2020_to_xyz,
        M_dp_to_xyz,
        xyz_to_lab_batch,
        N=50,
        method='DE'
    )
    print("DE Losses:", losses_lbfgs)

    plot_loss_statistics(losses_lbfgs, title='DE Distribution', method_name='Differential Evolution')
    plot_loss_statistics(area_diffs, title='Chromaticity Area Difference', method_name='Differential Evolution')
    # ============== 单独测试 ==============
    np.random.seed(35)
    test_samples = np.random.rand(100, 3)
    bounds = [(-2, 2)] * 9
    def loss_fn(M_flat):
        return combined_loss(M_flat, test_samples, M_bt2020_to_xyz, M_dp_to_xyz, xyz_to_lab_batch)
    
    res1 = differential_evolution(
        loss_fn,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        polish=True,
        seed=35,
        )
    M_opt = res1.x.reshape(3, 3)
    # 映射到色度图上
    # DB to BT2020
    RGB_basic = np.eye(3)
    # BT2020 to DP
    BT2020_to_DP_mapped = xyz_to_xy_test(M_opt, RGB_basic, M_bt2020_to_xyz)

    examples = {
        "BT2020": BT2020,
        "sRGB_DP": sRGB_DP,
        "BT2020_to_DP_mapped": BT2020_to_DP_mapped
    }

    plot_chromaticity_with_triangles(examples)
