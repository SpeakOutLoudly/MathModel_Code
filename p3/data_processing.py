import pandas as pd
import numpy as np
import os
from pathlib import Path
"""
将附录数据中的R_R, R_G, R_B等转化为 3X3 矩阵,便于后续计算
target:
 [[220   0       0]
 [ 0     220     0]
 [ 0     0     220]]

data:
 [[219.47073413   9.51538086   9.40454102]
 [  9.42808219 219.44717262   9.38500977]
 [  9.44848633   9.5        219.04761905]]
"""
# BT2020数据
BT2020_R = (0.708, 0.292)
BT2020_G = (0.170, 0.797)
BT2020_B = (0.131, 0.046)

# 显示屏RGB三基色
DP_R = (0.64, 0.33)
DP_G = (0.30, 0.60)
DP_B = (0.15, 0.06)

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir, "\n")

# Fix: Use consistent filename
file_name = 'B题附件：RGB数值.xlsx'
file_path = os.path.join(current_dir, '..', 'data', 'origin', 'xlsx', file_name)

# Check if file exists and load it
if os.path.exists(file_path):
    xlsx = pd.ExcelFile(file_path)
    print("文件读取成功")
else:
    print("文件未找到：", file_path)
    # Try alternative path
    alt_path = Path('../data/origin/xlsx/B题附件：RGB数值.xlsx')
    if alt_path.exists():
        xlsx = pd.ExcelFile(alt_path)
        print("使用备用路径读取成功")
    else:
        print("备用路径也未找到文件")
        exit(1)

# 1) 目标码值——按需线性化（例如去 gamma 2.2）
rgb_tar = xlsx.parse('RGB目标值')
rgb_tar = rgb_tar.apply(pd.to_numeric, errors='coerce').dropna(how='all')
rgb_tar_lin = (rgb_tar/255.0) ** 2.2   # sRGB → 线性光强

# 2) 摄像机三通道对 LED 原色的采样，拼成 3×3×N
try:
    R_sheets = [xlsx.parse(s) for s in ['R_R','R_G','R_B']]
    G_sheets = [xlsx.parse(s) for s in ['G_R','G_G','G_B']]
    B_sheets = [xlsx.parse(s) for s in ['B_R','B_G','B_B']]
except Exception as e:
    print(f"读取工作表时出错: {e}")
    print("可用的工作表名称:", xlsx.sheet_names)
    exit(1)

# 清洗缺失行、列，再 reshape → (N,3) 形式
def clean(m):
    m = m.apply(pd.to_numeric, errors='coerce')
    m = m.dropna(how='all').dropna(axis=1, how='all')
    return m.values.flatten()

# Fix: Apply clean function to each sheet separately
R_cleaned = [clean(sheet) for sheet in R_sheets]
G_cleaned = [clean(sheet) for sheet in G_sheets]
B_cleaned = [clean(sheet) for sheet in B_sheets]

# 得到 M_cam ≈ [[<R_R均值>,<R_G均值>,<R_B均值>],[…]]
M_cam = np.array([[np.mean(a) for a in R_cleaned],
                  [np.mean(a) for a in G_cleaned],
                  [np.mean(a) for a in B_cleaned]])   # 3×3

print("M_cam shape:", M_cam.shape)
print("M_cam:\n", M_cam)