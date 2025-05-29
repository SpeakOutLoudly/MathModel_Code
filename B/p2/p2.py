import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull
import colour 

def rgb_to_xyz_torch(rgb):
    device = rgb.device
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=device, dtype=torch.float32)
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask,
                             torch.pow((rgb + 0.055) / 1.055, 2.4),
                             rgb / 12.92)
    return torch.matmul(rgb_linear, M.T)

def xyz_to_lab_torch(xyz):
    device = xyz.device
    white = torch.tensor([0.95047, 1.00000, 1.08883], device=device, dtype=torch.float32)
    xyz_scaled = xyz / white
    delta = 6.0 / 29.0
    def f(t):
        return torch.where(t > delta**3,
                           torch.pow(t, 1.0/3.0),
                           t / (3.0 * delta**2) + 4.0 / 29.0)
    f_xyz = f(xyz_scaled)
    L = 116.0 * f_xyz[..., 1] - 16.0
    a = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return torch.stack([L, a, b], dim=-1)

def rgb_to_lab_torch(rgb):
    # 添加一个小的epsilon防止log(0)或者除以0的情况
    return xyz_to_lab_torch(rgb_to_xyz_torch(rgb.clamp(min=1e-8)))

def deltaE2000_torch(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    k_L, k_C, k_H = 1.0, 1.0, 1.0
    
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    
    avg_C = (C1 + C2) / 2.0
    
    G = 0.5 * (1 - torch.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    
    C1p = torch.sqrt(a1p**2 + b1**2)
    C2p = torch.sqrt(a2p**2 + b2**2)
    
    h1p = torch.rad2deg(torch.atan2(b1, a1p))
    h1p = torch.where(h1p < 0, h1p + 360, h1p)
    h2p = torch.rad2deg(torch.atan2(b2, a2p))
    h2p = torch.where(h2p < 0, h2p + 360, h2p)
    
    avg_L = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0
    
    h_diff = h2p - h1p
    delta_hp = torch.where(torch.abs(h_diff) <= 180, h_diff, h_diff - 360 * torch.sign(h_diff))
    
    Delta_Lp = L2 - L1
    Delta_Cp = C2p - C1p
    Delta_hp = 2 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(delta_hp / 2.0))
    
    h_sum = h1p + h2p
    avg_hp = torch.where(torch.abs(h_diff) > 180, (h_sum + 360) / 2, h_sum / 2)
    
    T = (1 - 0.17 * torch.cos(torch.deg2rad(avg_hp - 30)) +
         0.24 * torch.cos(torch.deg2rad(2 * avg_hp)) +
         0.32 * torch.cos(torch.deg2rad(3 * avg_hp + 6)) -
         0.20 * torch.cos(torch.deg2rad(4 * avg_hp - 63)))
    
    delta_ro = 30 * torch.exp(-((avg_hp - 275) / 25)**2)
    
    R_C = 2 * torch.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
    S_L = 1 + (0.015 * (avg_L - 50)**2) / torch.sqrt(20 + (avg_L - 50)**2)
    S_C = 1 + 0.045 * avg_Cp
    S_H = 1 + 0.015 * avg_Cp * T
    R_T = -torch.sin(torch.deg2rad(2 * delta_ro)) * R_C
    
    delta_E = torch.sqrt(
        (Delta_Lp / (k_L * S_L))**2 +
        (Delta_Cp / (k_C * S_C))**2 +
        (Delta_hp / (k_H * S_H))**2 +
        R_T * (Delta_Cp / (k_C * S_C)) * (Delta_hp / (k_H * S_H))
    )
    
    return delta_E

class CombinedLoss(nn.Module):

    def __init__(self, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_rgbcx, target_rgbcx):
        # MSE Loss on all 5 channels
        loss_mse = self.mse_loss(pred_rgbcx, target_rgbcx)
        
        # DeltaE2000 Loss on the first 3 (RGB) channels
        pred_rgb = pred_rgbcx[:, :3]
        target_rgb = target_rgbcx[:, :3]
        
        pred_lab = rgb_to_lab_torch(pred_rgb)
        target_lab = rgb_to_lab_torch(target_rgb)
        
        loss_delta_e = torch.mean(deltaE2000_torch(pred_lab, target_lab))
    
        # Combine the losses
        total_loss = self.alpha * loss_mse + self.beta * loss_delta_e
        return total_loss


def rgb_to_xyz(rgb):
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055)/1.055)**2.4, rgb / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return np.dot(rgb_linear, M.T)

def xyz_to_lab(xyz):
    white = np.array([0.95047, 1.00000, 1.08883])
    xyz_scaled = xyz / white
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, np.cbrt(t), t/(3*delta**2) + 4/29)
    f_xyz = f(xyz_scaled)
    L = 116*f_xyz[...,1] - 16
    a = 500*(f_xyz[...,0] - f_xyz[...,1])
    b = 200*(f_xyz[...,1] - f_xyz[...,2])
    return np.stack([L,a,b], axis=-1)

def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(np.clip(rgb, 0, 1)))

def deltaE2000(Lab1, Lab2):
    L1, a1, b1 = Lab1[...,0], Lab1[...,1], Lab1[...,2]
    L2, a2, b2 = Lab2[...,0], Lab2[...,1], Lab2[...,2]
    avg_L = 0.5 * (L1 + L2)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt((avg_C**7) / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(np.abs(dhp) > 180, dhp - 360 * np.sign(dhp), dhp)
    delta_hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    avg_Lp = (L1 + L2) / 2
    avg_Cp = (C1p + C2p) / 2
    hp_sum = h1p + h2p
    avg_hp = np.where(np.abs(h1p - h2p) > 180, (hp_sum + 360) / 2, hp_sum / 2)
    T = 1 - 0.17*np.cos(np.radians(avg_hp - 30)) + \
        0.24*np.cos(np.radians(2*avg_hp)) + \
        0.32*np.cos(np.radians(3*avg_hp + 6)) - \
        0.20*np.cos(np.radians(4*avg_hp - 63))
    delta_ro = 30 * np.exp(-((avg_hp - 275)/25)**2)
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    Sl = 1 + ((0.015 * (avg_Lp - 50)**2) / np.sqrt(20 + (avg_Lp - 50)**2))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_ro)) * Rc
    delta_E = np.sqrt(
        (delta_Lp / Sl)**2 +
        (delta_Cp / Sc)**2 +
        (delta_hp / Sh)**2 +
        Rt * (delta_Cp / Sc) * (delta_hp / Sh))
    return delta_E

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def generate_train_data(n_samples=2000):
    # X是相机输入 RGBV (4通道)
    X = np.random.rand(n_samples, 4).astype(np.float32) 
    

    W = np.array([
        [0.9, 0.05, 0.03, 0.02],    # R_out = 0.9*R_in + 0.05*G_in + ...
        [0.05, 0.85, 0.05, 0.05],   # G_out
        [0.02, 0.03, 0.9, 0.05],    # B_out
        [0.01, 0.02, 0.03, 0.9],    # C_out (受V通道影响较大)
        [0.02, 0.05, 0.02, 0.91]    # X_out (受V通道影响较大)
    ], dtype=np.float32)
    
    Y_linear = X.dot(W.T)
    Y_nonlinear = Y_linear + 0.02 * np.sin(5 * np.pi * X[:, 0:1]) # 加入非线性扰动
    Y_nonlinear = np.clip(Y_nonlinear, 0, 1) # 确保颜色值在0-1范围内
    return X, Y_nonlinear.astype(np.float32)

def train_model(X, Y, epochs=100, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_losses, val_losses = [], []
    

    loss_fn = CombinedLoss(alpha=0.1, beta=1.0).to(device)
    
    for epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(len(X_train))
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = torch.tensor(X_train[indices], dtype=torch.float32, device=device)
            batch_y = torch.tensor(Y_train[indices], dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = loss_fn(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(indices)
        avg_train_loss = epoch_loss / len(X_train)
    
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float32, device=device)
            val_y = torch.tensor(Y_val, dtype=torch.float32, device=device)
            val_pred = model(val_x)
            
            val_loss = loss_fn(val_pred, val_y).item()
    
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
    return model, train_losses, val_losses, X_val, Y_val

def visualize_errors(model, X_val, Y_val):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_val, dtype=torch.float32, device=device)
        outputs = model(inputs).cpu().numpy()
        targets = Y_val
    

    pred_rgb = outputs[:, :3] 
    target_rgb = targets[:, :3] 
    
    pred_lab = rgb_to_lab(pred_rgb)
    target_lab = rgb_to_lab(target_rgb)
    delta_e = deltaE2000(pred_lab, target_lab)
    
    plt.figure(figsize=(6,4))
    plt.hist(delta_e, bins=50, color='skyblue', range=(0, max(5, np.max(delta_e))))
    plt.title('ΔE2000 Error Histogram (Trained with Hybrid Loss)')
    plt.xlabel('ΔE2000')
    plt.ylabel('Pixel Count')
    plt.show()
    
    sorted_de = np.sort(delta_e)
    cdf = np.arange(len(sorted_de)) / float(len(sorted_de))
    
    plt.figure(figsize=(6,4))
    plt.plot(sorted_de, cdf, color='green')
    plt.title('CDF of ΔE2000 Error (Trained with Hybrid Loss)')
    plt.xlabel('ΔE2000')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.show()

def plot_chromaticity_with_triangles(example_dict):

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
        
        for j, point in enumerate(triangle):
            axes.scatter(point[0], point[1], color=colors[i % len(colors)], s=70, zorder=5, edgecolors='black')
            axes.text(point[0] + 0.005, point[1] + 0.005, f'{label[j]}', fontsize=10, color='black') # Label points as R, G, B, V, C, X as appropriate
    
    axes.legend()
    axes.set_title("CIE 1931 Chromaticity Diagram with Multi-Primary Gamuts")
    plt.grid(True)
    plt.show()

def _rgbv_to_rgb_display(rgbv):

    if rgbv.ndim == 1: # Handle single color (1D array)
        r, g, b, v = rgbv
    else: # Handle multiple colors (2D array)
        r, g, b, v = rgbv[:, 0], rgbv[:, 1], rgbv[:, 2], rgbv[:, 3]
    
    r_display = np.clip(r + v * 0.1, 0, 1)
    g_display = np.clip(g, 0, 1)
    b_display = np.clip(b + v * 0.2, 0, 1)
    
    return np.stack([r_display, g_display, b_display], axis=-1)

def _rgbcx_to_rgb_display(rgbcx):

    if rgbcx.ndim == 1: # Handle single color (1D array)
        r, g, b, c, x = rgbcx
    else: # Handle multiple colors (2D array)
        r, g, b, c, x = rgbcx[:, 0], rgbcx[:, 1], rgbcx[:, 2], rgbcx[:, 3], rgbcx[:, 4]
    
    # 简化的C和X通道融合
    # C (Cyan) 影响 G 和 B
    # X (Extra Red) 影响 R
    r_display = np.clip(r + x * 0.3, 0, 1) # X通道增加红色
    g_display = np.clip(g + c * 0.2, 0, 1) # C通道增加绿色
    b_display = np.clip(b + c * 0.3, 0, 1) # C通道增加蓝色
    
    return np.stack([r_display, g_display, b_display], axis=-1)

def visualize_sample_predictions(model, X_val, Y_val, num_samples=5):

    device = next(model.parameters()).device
    model.eval()
    
    # 随机选择num_samples个样本
    indices = np.random.choice(len(X_val), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 2 * num_samples))
    fig.suptitle('Sample Color Predictions (Input RGBV -> Output RGBCX)', fontsize=16)
    
    for i, idx in enumerate(indices):
        input_rgbv = X_val[idx]
        target_rgbcx = Y_val[idx]
        
        # 模型预测
        with torch.no_grad():
            pred_rgbcx_tensor = model(torch.tensor(input_rgbv, dtype=torch.float32, device=device).unsqueeze(0))
            pred_rgbcx = pred_rgbcx_tensor.squeeze(0).cpu().numpy()
        
        # 将RGBV转换为RGB用于显示（简化处理）
        display_input_rgb = _rgbv_to_rgb_display(input_rgbv)
        
        # 将RGBCX转换为RGB用于显示（简化处理）
        display_target_rgb = _rgbcx_to_rgb_display(target_rgbcx)
        display_pred_rgb = _rgbcx_to_rgb_display(pred_rgbcx)
    
        delta_e = deltaE2000(rgb_to_lab(display_pred_rgb), rgb_to_lab(display_target_rgb))
        
        # 绘制
        ax = axes[i, 0]
        ax.imshow([[display_input_rgb]]) # imshow需要2D数组，所以用[[color]]
        ax.set_title(f'Input (RGBV)\nSample {idx}', fontsize=8)
        ax.axis('off')
    
        ax = axes[i, 1]
        ax.imshow([[display_target_rgb]])
        ax.set_title(f'Target (RGBCX->RGB)', fontsize=8)
        ax.axis('off')
    
        ax = axes[i, 2]
        ax.imshow([[display_pred_rgb]])
        ax.set_title(f'Predicted (RGBCX->RGB)\nΔE2000: {delta_e:.2f}', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('sample_predictions.png', dpi=300)
    plt.show()

# 主程序 

if __name__ == '__main__':
    X, Y = generate_train_data(n_samples=4000)

    model, train_losses, val_losses, X_val, Y_val = train_model(X, Y, epochs=200, lr=5e-4)
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Loss (0.1*MSE + 1.0*ΔE2000)')
    plt.legend()
    plt.title('Training Loss Curve (Hybrid Loss)')
    plt.show()

    # 定义标准sRGB的R,G,B基色坐标
    PRIMARY_R = [0.64, 0.33]
    PRIMARY_G = [0.30, 0.60]
    PRIMARY_B = [0.15, 0.06]
    
    # 定义相机新增的 'V' (Violet/紫色) 基色坐标
    # 选择一个在蓝色和光谱轨迹紫色区域之间的点
    PRIMARY_V = [0.16, 0.03] 
    
    # 定义LED屏新增的 'C' (Cyan/青色) 和 'X' (假设为一种更深的红色) 基色坐标
    # 选择一个能扩展蓝绿边界的青色点
    PRIMARY_C = [0.18, 0.45] 
    # 选择一个比sRGB的R更红的点，以扩展红色边界
    PRIMARY_X = [0.70, 0.30]
    
    # 组合成输入和输出系统的基色字典
    input_system_primaries_coords = {
        'RGBV Input Gamut': [PRIMARY_R, PRIMARY_G, PRIMARY_B, PRIMARY_V]
    }
    
    output_system_primaries_coords = {
        'RGBCX Output Gamut': [PRIMARY_R, PRIMARY_G, PRIMARY_C, PRIMARY_B, PRIMARY_X]
    }
    
    all_gamuts_for_plotting = {**input_system_primaries_coords, **output_system_primaries_coords}
    
    visualize_errors(model, X_val, Y_val)
    
    plot_chromaticity_with_triangles(all_gamuts_for_plotting)
    
    visualize_sample_predictions(model, X_val, Y_val, num_samples=8) 