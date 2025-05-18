"""
gmm_contributor_estimation.py
=============================
依赖：
    pip install pandas numpy scikit-learn matplotlib seaborn scipy pywavelets openpyxl

功能：
1. 读取 STR 峰高数据 → melt_alleles → df_long（长表）
2. 对每个 SampleFile 的所有 Height 做 GMM 拟合（N = 1..max_components）
3. 计算 AIC / BIC，选取信息准则最小的 N 作为“估计贡献者人数”
4. 输出结果 DataFrame，可保存为 CSV；可选画出 N-BIC 折线图做直观检查
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ---------- 你的预处理代码（原样保留，可略作删减） ----------
import re, seaborn as sns, pywt
from scipy.signal import savgol_filter
import Data_pre
# ……（此处省略，你已有的 count_numbers_in_range/melt_alleles 等函数）……
# 请确保把你之前贴出的所有辅助函数一并放进来
# ----------------------------------------------------------------

# ---------- 核心函数：估计贡献者人数 ----------
def estimate_contributors(heights,
                          max_components: int = 5,
                          criterion: str = "BIC",
                          random_state: int = 42):
    """
    对一组峰高向量 heights 进行 GMM 拟合，返回最佳成分数 N 及每个 N 的准则值
    Parameters
    ----------
    heights : array-like, shape (n_peaks,)
        某个样本全部峰高
    max_components : int
        尝试的最大贡献者人数（即最大高斯成分数）
    criterion : {"AIC","BIC"}
        信息准则类型
    random_state : int
        复现实验
    Returns
    -------
    best_n : int
        信息准则最小对应的成分数
    scores : dict
        {N: AIC/BIC 数值}
    """
    heights = np.asarray(heights).reshape(-1, 1)
    scores = {}
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n,
                              covariance_type="full",
                              random_state=random_state)
        gmm.fit(heights)
        score = gmm.bic(heights) if criterion.upper() == "BIC" else gmm.aic(heights)
        scores[n] = score
    best_n = min(scores, key=scores.get)
    return best_n, scores


def run_estimation(df_long,
                   max_components: int = 5,
                   criterion: str = "BIC",
                   plot: bool = True):
    """
    对 df_long 中每个 SampleFile 估计贡献者人数
    Returns
    -------
    result_df : pd.DataFrame
        列: SampleFile, BestN, {N=1 的准则值}, …, {N=max_components 的准则值}
    """
    records = []
    score_cols = [f"{criterion}_{n}" for n in range(1, max_components + 1)]

    for sample, sub in df_long.groupby("SampleFile", sort=False):
        heights = sub["Height"].values
        # 建议可先过滤极小噪声峰，如 heights > 50，否则弱峰会拉大方差
        heights = heights[heights > 0]      # 视需求调整阈值
        best_n, scores = estimate_contributors(heights,
                                               max_components=max_components,
                                               criterion=criterion)
        row = {"SampleFile": sample, "BestN": best_n}
        row.update({f"{criterion}_{n}": scores[n] for n in scores})
        records.append(row)

        # 可选：逐样本绘制 N-BIC 曲线
        if plot:
            xs, ys = zip(*sorted(scores.items()))
            plt.figure()
            plt.plot(xs, ys, marker='o')
            plt.title(f"{sample} — {criterion}")
            plt.xlabel("Number of Contributors (N)")
            plt.ylabel(criterion)
            plt.xticks(xs)
            plt.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()

    result_df = pd.DataFrame(records)
    # 保证所有 N 的列都存在
    for col in score_cols:
        if col not in result_df:
            result_df[col] = np.nan
    return result_df

def accuracy_rate(df_long):
    Count = df_long["count"].values
    BestN = df_long["BestN"].values

    total = [0, 0, 0, 0]  # [2人, 3人, 4人, 其他]
    correct = [0, 0, 0, 0]

    for i in range(len(Count)):
        true_n = Count[i]
        pred_n = BestN[i]
        if true_n == 2:
            total[0] += 1
            if pred_n == true_n:
                correct[0] += 1
        elif true_n == 3:
            total[1] += 1
            if pred_n == true_n:
                correct[1] += 1
        elif true_n == 4:
            total[2] += 1
            if pred_n == true_n:
                correct[2] += 1
        else:
            total[3] += 1
            if pred_n == true_n:
                correct[3] += 1

    # 打印准确率
    for i, n in enumerate([2, 3, 4, 5]):
        if total[i] > 0:
            acc = correct[i] / total[i]
            print(f"{n}人样本识别准确率：{acc:.2%} （{correct[i]}/{total[i]}）")
        else:
            print(f"{n}人样本：无样本")

    # 总体准确率
    total_all = sum(total)
    correct_all = sum(correct)
    if total_all > 0:
        print(f"总体准确率：{correct_all / total_all:.2%} （{correct_all}/{total_all}）")
       

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


# ---------- 示例主函数 ----------
def main():
    # 1) 数据读取 & melt
    path = "D:/Work_Code/math_latex/25数模/法医物证多人身份鉴定问题数据集/附件1：不同人数的STR图谱数据.xlsx"
    df_raw = pd.read_excel(path)
    df_long = Data_pre.melt_alleles(df_raw)          # 你的函数
    # df_long["Height"] = savgol_smooth(df_long["Height"].values)
    # df_long = Data_pre.Preliminary_processing(df_long)  # 若想做平滑/可视化

    # 2) GMM+AIC/BIC 估计
    result_df = run_estimation(df_long,
                               max_components=5,  # 视场景可调高
                               criterion="BIC",   # 或 "AIC"
                               plot=False)         # 若不想画图设 False

    # 3) 输出结果
    result_df["count"] = result_df["SampleFile"].apply(Data_pre.count_numbers_in_range)
    print(result_df.head())

    out_path = "贡献者人数估计结果.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    accuracy_rate(result_df)
    print(f"已保存结果至 {out_path}")

if __name__ == "__main__":
    main()
