import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import pywt

def count_numbers_in_range(s):
    match = re.search(r'-0003-([\d_]+)-.*?;', s)
    if match:
        number_part = match.group(1)  # 提取如 "44_45"
        return len(number_part.split('_'))
    else:
        return 0



def melt_alleles(df):
    records = []

    df["count"] = df["Sample File"].apply(count_numbers_in_range)

    for _, row in df.iterrows():
        sample_name = row["Sample File"]
        marker = row["Marker"]
        dye = row["Dye"]
        for i in range(1,30):
            alleles = row.get(f"Allele {i}")
            size = row.get(f"Size {i}")
            height = row.get(f"Height {i}")
            count = row.get(f"count")
            if pd.notna(alleles) and pd.notna(size) and pd.notna(height):
                records.append({
                    "SampleFile": str(sample_name),
                    "Marker": marker,
                    "Dye": dye,
                    "Allele": str(alleles),
                    "Size": float(size),
                    "Height": float(height),
                    "count": count
                })
    return pd.DataFrame(records)

def Preliminary_processing(df_long):
    # df_long = melt_alleles(df)
    # print(df.columns.tolist())
    # print(df.head())

    palette = {
        'B': '#1f77b4', 'G': '#2ca02c', 'Y': '#ff7f0e', 'R': '#d62728'
    }

    sample_name = df_long['SampleFile'].unique()
    #markers = df_long['SampleFile'].unique()
    #n_cols = min(len(sample_name), 5)
    #n_rows = int(np.ceil(len(sample_name) / n_cols))

    
    for idx, Sample in enumerate(sample_name):
        if idx > 5:
            break
        plt.figure(figsize=(6,4))
        subset = df_long[df_long['SampleFile'] == Sample]
        for _, row in subset.iterrows():
            color = palette.get(row['Dye'], 'gray')
            linestyle = '--' if row['Allele'].upper() == 'OL' else '-'
            plt.vlines(x=row['Size'], ymin=0, ymax=row['Height'],
                       colors=color, linewidth=2, linestyles=linestyle)
            if row['Height'] > 5:
                plt.text(row['Size'], row['Height'] + 5, row['Allele'],
                         ha='center', va='bottom', fontsize=8)
        
        plt.title(f"Sample: {Sample}")
        plt.xlabel("Size (bp)")
        plt.ylabel("Height")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# ---------- Step 2: 信号平滑方法 ----------
def sliding_window_min(signal, window_size=5):
    return pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).min().values

def savgol_smooth(signal, window_length=11, polyorder=3):
    if len(signal) < window_length:
        window_length = len(signal) // 2 * 2 + 1  # 保证奇数
    return savgol_filter(signal, window_length, polyorder)

def wavelet_denoise(signal, wavelet='db4', level=2):
    coeffs = pywt.wavedec(signal, wavelet, mode="per")
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, mode="per")[:len(signal)]

# ---------- Step 3: 绘图 ----------
def plot_marker(marker_df, smooth_type="savgol"):
    size = marker_df['Size'].values
    height = marker_df['Height'].values
    dye = marker_df['Dye'].iloc[0]
    alleles = marker_df['Allele'].values

    # 按大小排序
    sorted_idx = np.argsort(size)
    size = size[sorted_idx]
    height = height[sorted_idx]
    alleles = alleles[sorted_idx]

    # 平滑曲线
    if smooth_type == "savgol":
        smoothed = savgol_smooth(height)
    elif smooth_type == "min":
        smoothed = sliding_window_min(height)
    elif smooth_type == "wavelet":
        smoothed = wavelet_denoise(height)
    else:
        smoothed = height

    color_map = {'B': '#1f77b4', 'G': '#2ca02c', 'Y': '#ff7f0e', 'R': '#d62728'}
    plt.figure(figsize=(10, 4))
    plt.plot(size, smoothed, label=f"Smoothed ({smooth_type})", color=color_map.get(dye, 'gray'), lw=2)
    plt.vlines(size, 0, height, color=color_map.get(dye, 'gray'), lw=1, linestyle='--', alpha=0.6)

    # 注释等位基因
    for x, h, allele in zip(size, height, alleles):
        if h > 5:
            plt.text(x, h + 5, allele, ha='center', va='bottom', fontsize=8)

    plt.title(f"STR SampleFile: {marker_df['SampleFile'].iloc[0]} ({dye})")
    plt.xlabel("Size (bp)")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.show()

# ---------- Step 4: 主函数 ----------
def main():
    # 替换为你自己的路径
    path = "MathModel_Code/data/origin/xlsx/附件1：不同人数的STR图谱数据.xlsx"
    df = pd.read_excel(path)
    df_long = melt_alleles(df)

    # 测试图
    #for Sample in df_long['SampleFile'].unique()[:5]:  # 可改为全部 or 指定 Sample
    #    marker_df = df_long[df_long['SampleFile'] == Sample]
    #    plot_marker(marker_df, smooth_type="wavelet")  # "min", "savgol", or "wavelet"
    #Preliminary_processing(df_long)

    df_long.to_csv("MathModel_Code/data/preprocess/预处理文件.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    main()

    