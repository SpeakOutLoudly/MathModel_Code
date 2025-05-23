import colour
import matplotlib.pyplot as plt

# 获取 CIE 1931 标准观察者数据
cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
wavelengths = cmfs.wavelengths
x_bar = cmfs.values[:, 0]  # X
y_bar = cmfs.values[:, 1]  # Y (亮度)
z_bar = cmfs.values[:, 2]  # Z


# 绘制 CIE 1931 2° Standard Observer xy 色度图
colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=True)

# 绘制曲线
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, x_bar, "r", label="$\overline{x}(\lambda)$")
plt.plot(wavelengths, y_bar, "g", label="$\overline{y}(\lambda)$")
plt.plot(wavelengths, z_bar, "b", label="$\overline{z}(\lambda)$")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Sensitivity")
plt.title("CIE 1931 XYZ Color Matching Functions")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CIE 1931 XYZ Color System and Horseshoe Chromaticity Diagram', fontsize=16)

# CIE 1931 Standard Observer Color Matching Functions
# Wavelength range from 380nm to 780nm
wavelengths = np.arange(380, 781, 5)  # Every 5nm

# Simplified CIE 1931 color matching functions (approximated)
def cie_color_matching_functions(wavelengths):
    """
    Approximate CIE 1931 XYZ color matching functions
    These are simplified versions for demonstration
    """
    x_bar = np.zeros_like(wavelengths, dtype=float)
    y_bar = np.zeros_like(wavelengths, dtype=float)
    z_bar = np.zeros_like(wavelengths, dtype=float)
    
    for i, wl in enumerate(wavelengths):
        # X color matching function (approximated)
        if 380 <= wl <= 780:
            x_bar[i] = (
                0.362 * np.exp(-0.5 * ((wl - 442) / 36.5) ** 2) +
                1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9) ** 2) +
                -0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.4) ** 2)
            )
        
        # Y color matching function (approximated)
        if 380 <= wl <= 780:
            y_bar[i] = (
                0.821 * np.exp(-0.5 * ((wl - 568.8) / 46.9) ** 2) +
                0.286 * np.exp(-0.5 * ((wl - 530.9) / 16.3) ** 2)
            )
        
        # Z color matching function (approximated)
        if 380 <= wl <= 780:
            z_bar[i] = (
                1.217 * np.exp(-0.5 * ((wl - 437) / 11.8) ** 2) +
                0.681 * np.exp(-0.5 * ((wl - 459) / 26) ** 2)
            )
    
    # Ensure non-negative values
    x_bar = np.maximum(x_bar, 0)
    y_bar = np.maximum(y_bar, 0)
    z_bar = np.maximum(z_bar, 0)
    
    return x_bar, y_bar, z_bar

# Get color matching functions
x_bar, y_bar, z_bar = cie_color_matching_functions(wavelengths)

# Plot 1: CIE XYZ Color Matching Functions
ax1 = axes[0, 0]
ax1.plot(wavelengths, x_bar, 'r-', linewidth=2, label='x̄(λ) - X tristimulus')
ax1.plot(wavelengths, y_bar, 'g-', linewidth=2, label='ȳ(λ) - Y tristimulus')
ax1.plot(wavelengths, z_bar, 'b-', linewidth=2, label='z̄(λ) - Z tristimulus')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Tristimulus Value')
ax1.set_title('CIE 1931 XYZ Color Matching Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(380, 780)

# Calculate chromaticity coordinates for spectral locus
def calculate_chromaticity(X, Y, Z):
    """Calculate chromaticity coordinates x, y from XYZ"""
    total = X + Y + Z
    # Avoid division by zero
    total = np.where(total == 0, 1e-10, total)
    x = X / total
    y = Y / total
    z = Z / total  # z = 1 - x - y
    return x, y, z

# Spectral locus chromaticity coordinates
x_spectral, y_spectral, z_spectral = calculate_chromaticity(x_bar, y_bar, z_bar)

# Plot 2: Horseshoe Chromaticity Diagram
ax2 = axes[0, 1]

# Plot spectral locus (horseshoe curve)
ax2.plot(x_spectral, y_spectral, 'k-', linewidth=3, label='Spectral Locus')

# Add purple line (connecting 380nm to 780nm)
purple_x = [x_spectral[0], x_spectral[-1]]
purple_y = [y_spectral[0], y_spectral[-1]]
ax2.plot(purple_x, purple_y, 'purple', linewidth=2, linestyle='--', label='Purple Line')

# Fill the gamut area
gamut_x = np.concatenate([x_spectral, [x_spectral[0]]])
gamut_y = np.concatenate([y_spectral, [y_spectral[0]]])
ax2.fill(gamut_x, gamut_y, alpha=0.2, color='lightblue', label='Visible Gamut')

# Mark important points
# White point (D65 illuminant approximation)
white_x, white_y = 0.3127, 0.3290
ax2.plot(white_x, white_y, 'wo', markersize=8, markeredgecolor='black', 
         markeredgewidth=2, label='White Point (D65)')

# Mark some wavelength points
wavelength_markers = [380, 450, 500, 550, 600, 650, 700, 780]
for i, wl in enumerate(wavelength_markers):
    if wl <= 780:
        idx = int((wl - 380) / 5)
        if idx < len(x_spectral):
            ax2.annotate(f'{wl}nm', (x_spectral[idx], y_spectral[idx]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.set_xlabel('x chromaticity coordinate')
ax2.set_ylabel('y chromaticity coordinate')
ax2.set_title('CIE 1931 Chromaticity Diagram (Horseshoe)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 0.8)
ax2.set_ylim(0, 0.9)
ax2.set_aspect('equal')

# Plot 3: RGB to XYZ Transformation Matrix
ax3 = axes[1, 0]
ax3.axis('off')

# CIE RGB to XYZ transformation matrix (approximate)
rgb_to_xyz_matrix = np.array([
    [0.5767, 0.1856, 0.1881],
    [0.2973, 0.6273, 0.0752],
    [0.0270, 0.0706, 0.9911]
])

# Display transformation equations
equations_text = """
CIE RGB to XYZ Transformation:

[X]   [0.5767, 0.1856, 0.1881] [R]
[Y] = [0.2973, 0.6273, 0.0752] [G]
[Z]   [0.0270, 0.0706, 0.9911] [B]

Chromaticity Coordinates:
x = X / (X + Y + Z)
y = Y / (X + Y + Z)
z = Z / (X + Y + Z) = 1 - x - y

Key Properties:
• X, Y, Z are always positive
• Y represents luminance
• Horseshoe shape represents all
  visible colors in xy chromaticity
• Purple line connects spectrum
  endpoints (380nm ↔ 780nm)
"""

ax3.text(0.05, 0.95, equations_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

# Plot 4: Color Temperature Locus and Common Illuminants
ax4 = axes[1, 1]

# Planckian locus (black body radiation) - simplified approximation
def planckian_locus_xy(T_range):
    """Calculate approximate xy coordinates for Planckian locus"""
    T = np.array(T_range)
    
    # Approximate formulas for Planckian locus in xy chromaticity
    # These are simplified approximations
    x_planck = np.zeros_like(T, dtype=float)
    y_planck = np.zeros_like(T, dtype=float)
    
    for i, temp in enumerate(T):
        if temp >= 1667 and temp <= 25000:
            # Approximation formulas
            if temp <= 4000:
                x_planck[i] = -0.2661239e9/temp**3 - 0.2343580e6/temp**2 + 0.8776956e3/temp + 0.179910
            else:
                x_planck[i] = -3.0258469e9/temp**3 + 2.1070379e6/temp**2 + 0.2226347e3/temp + 0.240390
            
            if temp <= 2222:
                y_planck[i] = -1.1063814*x_planck[i]**3 - 1.34811020*x_planck[i]**2 + 2.18555832*x_planck[i] - 0.20219683
            elif temp <= 4000:
                y_planck[i] = -0.9549476*x_planck[i]**3 - 1.37418593*x_planck[i]**2 + 2.09137015*x_planck[i] - 0.16748867
            else:
                y_planck[i] = 3.0817580*x_planck[i]**3 - 5.87338670*x_planck[i]**2 + 3.75112997*x_planck[i] - 0.37001483
    
    return x_planck, y_planck

# Temperature range for Planckian locus
temperatures = np.logspace(np.log10(1000), np.log10(10000), 100)
x_planck, y_planck = planckian_locus_xy(temperatures)

# Plot spectral locus again
ax4.plot(x_spectral, y_spectral, 'k-', linewidth=2, label='Spectral Locus')
ax4.plot(purple_x, purple_y, 'purple', linewidth=2, linestyle='--')

# Plot Planckian locus
ax4.plot(x_planck, y_planck, 'r-', linewidth=2, label='Planckian Locus')

# Mark common illuminants
illuminants = {
    'A (2856K)': (0.4476, 0.4074),
    'D50 (5003K)': (0.3457, 0.3585),
    'D65 (6504K)': (0.3127, 0.3290),
    'E (Equal Energy)': (0.3333, 0.3333)
}

for name, (x, y) in illuminants.items():
    ax4.plot(x, y, 'o', markersize=6, label=name)

ax4.set_xlabel('x chromaticity coordinate')
ax4.set_ylabel('y chromaticity coordinate')
ax4.set_title('Chromaticity Diagram with Illuminants')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 0.8)
ax4.set_ylim(0, 0.9)
ax4.set_aspect('equal')

plt.tight_layout()
plt.show()

# Additional information
print("CIE 1931 XYZ Color System Information:")
print("="*50)
print(f"Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm")
print(f"Number of data points: {len(wavelengths)}")
print(f"Spectral locus x range: {x_spectral.min():.3f} to {x_spectral.max():.3f}")
print(f"Spectral locus y range: {y_spectral.min():.3f} to {y_spectral.max():.3f}")
print("\nKey Features of the Horseshoe Diagram:")
print("• The curved edge represents monochromatic light (pure spectral colors)")
print("• The straight edge (purple line) represents non-spectral purples")
print("• All visible colors lie within this horseshoe-shaped boundary")
print("• The Y tristimulus value represents luminance (brightness)")
print("• White light appears near the center of the diagram")

# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BT2020_R = (0.708, 0.292)
BT2020_G = (0.170, 0.797)
BT2020_B = (0.131, 0.046)

# 显示屏RGB三基色
DP_R = (0.64, 0.33)
DP_G = (0.30, 0.60)
DP_B = (0.15, 0.06)

BT2020 = [BT2020_R, BT2020_G, BT2020_B]
DP = [DP_R, DP_G, DP_B]

# ------------------------------
# 1. 路径验证与数据加载
# ------------------------------
def load_data():
    """加载RGB数据和目标XYZ数据"""
    # 检查文件路径
    file_path = Path('../data/origin/xlsx/B题附件：RGB数值.xlsx')
    print(f"绝对路径: {file_path.absolute()}")
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取Excel数据（假设数据格式为：R, G, B, X_target, Y_target, Z_target）
    df = pd.read_excel(file_path)
    rgb_data = df[['R', 'G', 'B']].values.astype(np.float32)  # 输入RGB
    xyz_target = df[['X', 'Y', 'Z']].values.astype(np.float32)  # 目标XYZ

    # 数据归一化（可选，根据数据范围调整）
    rgb_data /= 255.0  # 假设RGB原始范围是0-255
    return rgb_data, xyz_target

# ------------------------------
# 2. 定义神经网络模型
# ------------------------------
class ColorTransformModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个可训练的3x3矩阵（即RGB→XYZ的变换矩阵M）
        self.M = nn.Parameter(torch.randn(3, 3, dtype=torch.float32))

    def forward(self, rgb):
        """输入形状: (batch_size, 3)"""
        return torch.mm(rgb, self.M)  # 矩阵乘法: RGB @ M → XYZ

# ------------------------------
# 3. 训练过程
# ------------------------------
def train_model(rgb_data, xyz_target, epochs=1000, lr=0.01):
    # 转换为PyTorch张量
    rgb_tensor = torch.from_numpy(rgb_data)
    xyz_tensor = torch.from_numpy(xyz_target)

    # 初始化模型、损失函数和优化器
    model = ColorTransformModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 记录训练损失
    losses = []

    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        xyz_pred = model(rgb_tensor)
        loss = criterion(xyz_pred, xyz_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:4d}, Loss: {loss.item():.6f}')

    # 提取优化后的矩阵
    M_optimized = model.M.detach().numpy()
    return M_optimized, losses

# ------------------------------
# 4. 主程序
# ------------------------------
if __name__ == "__main__":
    try:
        # 加载数据
        rgb_data, xyz_target = load_data()
        print(f"数据加载成功！样本数量: {rgb_data.shape[0]}")

        # 训练模型
        M_optimized, losses = train_model(rgb_data, xyz_target, epochs=1000)

        # 输出优化后的矩阵
        print("\n优化后的RGB→XYZ变换矩阵M:")
        print(np.array_str(M_optimized, precision=6, suppress_small=True))

        # 可视化训练损失
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('RGB to XYZ Matrix Optimization')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"错误发生: {str(e)}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

def triangle_area_cross_product(points):
    """
    Calculate triangle area using cross product method
    Input: [[x1, y1], [x2, y2], [x3, y3]]
    Output: area (float)
    """
    # Convert to numpy array for easier manipulation
    points = np.array(points)
    
    # Extract coordinates
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    
    # Using cross product formula: |det([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])| / 2
    area = abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2)
    
    return area

def triangle_area_shoelace(points):
    """
    Calculate triangle area using shoelace formula
    Input: [[x1, y1], [x2, y2], [x3, y3]]
    Output: area (float)
    """
    points = np.array(points)
    
    # Shoelace formula: 1/2 * |sum of (xi * yi+1 - xi+1 * yi)|
    n = len(points)
    area = 0
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2

def triangle_area_vectors(points):
    """
    Calculate triangle area using vector cross product
    Input: [[x1, y1], [x2, y2], [x3, y3]]
    Output: area (float)
    """
    points = np.array(points)
    
    # Create vectors from point 1 to points 2 and 3
    v1 = points[1] - points[0]  # Vector from p1 to p2
    v2 = points[2] - points[0]  # Vector from p1 to p3
    
    # Cross product magnitude gives twice the area
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    area = abs(cross_product) / 2
    
    return area

def triangle_area_heron(points):
    """
    Calculate triangle area using Heron's formula
    Input: [[x1, y1], [x2, y2], [x3, y3]]
    Output: area (float)
    """
    points = np.array(points)
    
    # Calculate side lengths
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    a = distance(points[0], points[1])
    b = distance(points[1], points[2])
    c = distance(points[2], points[0])
    
    # Semi-perimeter
    s = (a + b + c) / 2
    
    # Heron's formula: sqrt(s(s-a)(s-b)(s-c))
    # Handle potential negative values under square root
    discriminant = s * (s - a) * (s - b) * (s - c)
    if discriminant < 0:
        return 0  # Degenerate triangle
    
    area = math.sqrt(discriminant)
    return area

def triangle_area_numpy(points):
    """
    Calculate triangle area using numpy's built-in functions
    Input: [[x1, y1], [x2, y2], [x3, y3]]
    Output: area (float)
    """
    points = np.array(points)
    
    # Using numpy's cross product
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    
    # For 2D vectors, cross product gives a scalar
    area = abs(np.cross(v1, v2)) / 2
    return area

# Main function (recommended)
def triangle_area(points):
    """
    Calculate the area of a triangle given three 2D coordinate points.
    
    Parameters:
    points: list or array-like, [[x1, y1], [x2, y2], [x3, y3]]
    
    Returns:
    float: area of the triangle
    
    Example:
    >>> triangle_area([[0, 0], [4, 0], [2, 3]])
    6.0
    """
    return triangle_area_cross_product(points)

# Test the functions
def test_triangle_area_functions():
    """Test all triangle area calculation methods"""
    
    # Test cases
    test_cases = [
        [[0, 0], [4, 0], [2, 3]],      # Simple triangle
        [[1, 1], [4, 5], [7, 2]],      # General triangle
        [[0, 0], [3, 4], [0, 4]],      # Right triangle
        [[-1, -1], [1, -1], [0, 1]],   # Triangle with negative coordinates
        [[0, 0], [1, 0], [0, 1]],      # Unit right triangle
    ]
    
    methods = [
        ("Cross Product", triangle_area_cross_product),
        ("Shoelace Formula", triangle_area_shoelace),
        ("Vector Method", triangle_area_vectors),
        ("Heron's Formula", triangle_area_heron),
        ("NumPy Method", triangle_area_numpy)
    ]
    
    print("Triangle Area Calculation Test Results:")
    print("=" * 60)
    
    for i, points in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {points}")
        print("-" * 40)
        
        results = []
        for method_name, method_func in methods:
            try:
                area = method_func(points)
                results.append(area)
                print(f"{method_name:15}: {area:.6f}")
            except Exception as e:
                print(f"{method_name:15}: Error - {e}")
        
        # Check if all methods give the same result (within tolerance)
        if results and all(abs(r - results[0]) < 1e-10 for r in results):
            print(f"✓ All methods agree: Area = {results[0]:.6f}")
        else:
            print("⚠ Methods disagree!")

# Visualization function
def visualize_triangle(points, title="Triangle Area Calculation"):
    """Visualize the triangle and display its area"""
    
    points = np.array(points)
    area = triangle_area(points)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create triangle polygon
    triangle = Polygon(points, fill=True, alpha=0.3, color='lightblue', 
                      edgecolor='blue', linewidth=2)
    ax.add_patch(triangle)
    
    # Plot vertices
    ax.scatter(points[:, 0], points[:, 1], color='red', s=100, zorder=5)
    
    # Label vertices
    labels = ['A', 'B', 'C']
    for i, (x, y) in enumerate(points):
        ax.annotate(f'{labels[i]}({x}, {y})', (x, y), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold')
    
    # Calculate and display side lengths
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    sides = [
        distance(points[0], points[1]),
        distance(points[1], points[2]),
        distance(points[2], points[0])
    ]
    
    # Add side length labels
    midpoints = [
        (points[0] + points[1]) / 2,
        (points[1] + points[2]) / 2,
        (points[2] + points[0]) / 2
    ]
    
    for i, (midpoint, side_length) in enumerate(zip(midpoints, sides)):
        ax.annotate(f'{side_length:.2f}', midpoint, 
                   xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{title}\nArea = {area:.4f} square units', fontsize=14)
    
    # Set axis limits with some padding
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    padding = max(x_max - x_min, y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout()
    plt.show()
    
    return area

# Example usage and testing
if __name__ == "__main__":
    # Run tests
    test_triangle_area_functions()
    
    # Example usage
    print("\n" + "="*60)
    print("EXAMPLE USAGE:")
    print("="*60)
    
    # Example triangles
    examples = [
        [[0.708, 0.292],
         [0.170, 0.797],
         [0.131, 0.046]],
        [[8.75494411e-01, 1.91006348e-01],
         [2.58905246e-01, 6.38137297e-01],
         [1.26019286e-01, 7.86688870e-04]],
        [[0.26007957, 0.51852447], 
         [0.35171858, 0.34421927], 
         [0.38425413, 0.23895326]],
        [[0.76161057, 0.25960421],
         [0.24413979, 0.78372638],
         [0.15,       0.06      ]]
    ]
    examples2 = [
        [[0.708, 0.292],
         [0.170, 0.797],
         [0.131, 0.046]],
        [[0.64, 0.33],
         [0.30, 0.60],
         [0.15, 0.06]],
        [[8.75494411e-01, 1.91006348e-01],
         [2.58905246e-01, 6.38137297e-01],
         [1.26019286e-01, 7.86688870e-04]],
        [[0.76161057, 0.25960421],
         [0.24413979, 0.78372638],
         [0.15,       0.06      ]]
    ]
    for i, points in enumerate(examples):
        print(f"\nExample {i+1}: Triangle with vertices {points}")
        area = triangle_area(points)
        print(f"Area: {area:.4f} square units")
        
        # Visualize the first example
        if i == 3:
            visualize_triangle(points, f"Example {i+1} Triangle")

print("\nAll triangle area calculation methods implemented successfully!")