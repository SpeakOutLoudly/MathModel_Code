import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, differential_evolution

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LEDColorCorrection:
    """
    基于三基色原理和CIE Lab色彩空间的颜色校正
    使用差分进化算法优化校正矩阵
    """
    
    def __init__(self):
        self.correction_matrix = None
        self.correction_bias = None
        self.gamma_correction = None
        self.measured_data = None
        self.target_data = None
        
    def load_excel_data(self, excel_path):
        """从Excel文件加载数据"""
        print(f"正在加载Excel文件: {excel_path}")
        
        sheets = ['R', 'G', 'B', 'target_R', 'target_G', 'target_B']
        data_dict = {}
        
        for sheet_name in sheets:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None).iloc[0:64,0:64]
            data_dict[sheet_name] = df.values
            print(f"已加载工作表 '{sheet_name}': {df.shape}")
        
        # 组织数据
        self.measured_data = np.stack([
            data_dict['R'],
            data_dict['G'], 
            data_dict['B']
        ], axis=-1)
        
        self.target_data = np.stack([
            data_dict['target_R'],
            data_dict['target_G'],
            data_dict['target_B']
        ], axis=-1)
        
        print(f"测量数据形状: {self.measured_data.shape}")
        print(f"目标数据形状: {self.target_data.shape}")
    
    def rgb_to_xyz(self, rgb):
        """RGB转XYZ色彩空间"""
        rgb_norm = rgb / 255.0
        
        # Gamma校正
        rgb_linear = np.where(rgb_norm <= 0.04045,
                             rgb_norm / 12.92,
                             np.power((rgb_norm + 0.055) / 1.055, 2.4))
        
        # sRGB到XYZ的转换矩阵
        transform_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = np.dot(rgb_linear, transform_matrix.T)
        return xyz
    
    def xyz_to_lab(self, xyz):
        """XYZ转CIE Lab色彩空间"""
        # D65白点
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        
        x = xyz[..., 0] / Xn
        y = xyz[..., 1] / Yn
        z = xyz[..., 2] / Zn
        
        # 立方根变换
        fx = np.where(x > 0.008856, np.power(x, 1/3), (7.787 * x + 16/116))
        fy = np.where(y > 0.008856, np.power(y, 1/3), (7.787 * y + 16/116))
        fz = np.where(z > 0.008856, np.power(z, 1/3), (7.787 * z + 16/116))
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.stack([L, a, b], axis=-1)
    
    def calculate_color_difference(self, lab1, lab2):
        """计算CIE Delta E 2000色差"""
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_bar = 0.5 * (C1 + C2)

        G = 0.5 * (1 - np.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2

        C1p = np.sqrt(a1p**2 + b1**2)
        C2p = np.sqrt(a2p**2 + b2**2)

        h1p = np.degrees(np.arctan2(b1, a1p)) % 360
        h2p = np.degrees(np.arctan2(b2, a2p)) % 360

        dLp = L2 - L1
        dCp = C2p - C1p

        dhp = h2p - h1p
        dhp = dhp - 360 * (dhp > 180) + 360 * (dhp < -180)
        dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

        L_bar = 0.5 * (L1 + L2)
        C_bar_p = 0.5 * (C1p + C2p)

        h_bar_p = (h1p + h2p + 360 * (np.abs(h1p - h2p) > 180)) / 2
        h_bar_p %= 360

        T = (1
            - 0.17 * np.cos(np.radians(h_bar_p - 30))
            + 0.24 * np.cos(np.radians(2 * h_bar_p))
            + 0.32 * np.cos(np.radians(3 * h_bar_p + 6))
            - 0.20 * np.cos(np.radians(4 * h_bar_p - 63)))

        Sl = 1 + (0.015 * (L_bar - 50)**2) / np.sqrt(20 + (L_bar - 50)**2)
        Sc = 1 + 0.045 * C_bar_p
        Sh = 1 + 0.015 * C_bar_p * T

        delta_theta = 30 * np.exp(-((h_bar_p - 275)/25)**2)
        Rc = 2 * np.sqrt(C_bar_p**7 / (C_bar_p**7 + 25**7))
        Rt = -np.sin(np.radians(2 * delta_theta)) * Rc

        dE = np.sqrt(
            (dLp / Sl)**2 +
            (dCp / Sc)**2 +
            (dHp / Sh)**2 +
            Rt * (dCp / Sc) * (dHp / Sh)
        )

        return dE
    
    def estimate_gamma_parameters(self):
        """估计LED的Gamma参数（保留线性比例偏移）"""
        print("正在估计Gamma参数...")
        gamma_params = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            meas = self.measured_data[..., i].flatten() / 255.0
            targ = self.target_data[..., i].flatten() / 255.0
            mask = (targ >= 0) & (targ <= 1)
            m = meas[mask]
            t = targ[mask]
            if len(m) > 0:
                # 拟合 log(m) = gamma * log(t) + offset
                A = np.vstack([np.log(t + 1e-8), np.ones_like(t)]).T
                gamma, offset = np.linalg.lstsq(A, np.log(m + 1e-8), rcond=None)[0]
                gamma = float(np.clip(gamma, 0.0, 3.0))
                scale = float(np.exp(offset))
            else:
                gamma, scale = 1.0, 1.0
            gamma_params[channel] = {'gamma': gamma, 'scale': scale}
            print(f"{channel}通道 Gamma: {gamma:.3f}, Scale: {scale:.3f}")
        self.gamma_correction = gamma_params
        return gamma_params

    def apply_gamma_correction(self, rgb_data, inverse=False):
        """应用Gamma校正：在归一化 [0,1] 空间先应用线性比例，再做幂运算"""
        if self.gamma_correction is None:
            return rgb_data
        data = rgb_data.astype(np.float32) / 255.0
        out = np.zeros_like(data)
        for i, channel in enumerate(['R', 'G', 'B']):
            gamma = self.gamma_correction[channel]['gamma']
            scale = self.gamma_correction[channel]['scale']
            ch = data[..., i]
            if not inverse:
                # 前向：先比例，再幂
                tmp = ch * scale
                tmp = np.clip(tmp, 0.0, 1.0)
                out_ch = np.power(tmp, gamma)
            else:
                # 反向：开幂，再去比例
                tmp = np.power(ch, 1.0 / gamma)
                out_ch = tmp / np.maximum(scale, 1e-8)
            out[..., i] = np.clip(out_ch, 0.0, 1.0)
        # 恢复到 [0,255]
        return (out * 255.0).astype(rgb_data.dtype)
    
    def correction_function(self, params, measured_lin, target_lin):
        """
        优化函数：线性校正矩阵 M 和偏置 b，params 长度 12。
        corrected = clip(M @ measured + b, [0,1])
        计算 ΔE₀₀ + 正则化。
        """
        M = params[:9].reshape(3,3)
        b = params[9:].reshape(1,3)

        # 应用矩阵和偏置
        corr = np.dot(measured_lin, M.T) + b
        corr = np.clip(corr, 0.0, 1.0)

        # 转到 XYZ → Lab
        transform = np.array([[0.4124564,0.3575761,0.1804375],
                              [0.2126729,0.7151522,0.0721750],
                              [0.0193339,0.1191920,0.9503041]])
        tgt_xyz = np.dot(target_lin, transform.T)
        corr_xyz = np.dot(corr, transform.T)
        tgt_lab = self.xyz_to_lab(tgt_xyz.reshape(-1,3)).reshape(corr.shape)
        corr_lab = self.xyz_to_lab(corr_xyz.reshape(-1,3)).reshape(corr.shape)

        # 色差
        deltaE = self.calculate_color_difference(tgt_lab, corr_lab)
        loss = np.mean(deltaE)

        # 矩阵正则 + 偏置正则
        loss += 0.001 * (np.sum((M - np.eye(3))**2) + np.sum(b**2))
        det = np.linalg.det(M)
        if det <= 0 or abs(det) < 0.1:
            loss += 1000.0
        return loss
    
    def calibrate_correction_matrix(self):
        print("开始校正：矩阵 + 偏置...")
        self.estimate_gamma_parameters()
        # 预处理：线性化
        meas = self.apply_gamma_correction(self.measured_data.astype(np.float32), inverse=True)/255.0
        targ = self.apply_gamma_correction(self.target_data.astype(np.float32), inverse=True)/255.0
        meas_flat = meas.reshape(-1,3)
        targ_flat = targ.reshape(-1,3)
        # 差分进化优化 12 参数
        bounds = [(-2,2)]*9 + [(-0.1,0.1)]*3
        res = differential_evolution(
            self.correction_function, bounds,
            args=(meas_flat, targ_flat), maxiter=200, popsize=15, seed=42
        )
        x0 = res.x
        # 局部 L-BFGS-B
        local = minimize(
            self.correction_function, x0, args=(meas_flat,targ_flat),
            method='L-BFGS-B', options={'maxiter':500}
        )
        M_opt = local.x[:9].reshape(3,3)
        b_opt = local.x[9:].reshape(3)
        self.correction_matrix = M_opt
        self.correction_bias = b_opt
        print("校正完成；矩阵行列式：", np.linalg.det(M_opt))
        print("偏置：", b_opt)
        return M_opt, b_opt

    def apply_correction(self, input_rgb):
        """应用带偏置的线性校正"""
        lin = self.apply_gamma_correction(input_rgb.astype(np.float32), inverse=True)/255.0
        flat = lin.reshape(-1,3)
        corr = np.dot(flat, self.correction_matrix.T) + self.correction_bias
        corr = np.clip(corr, 0.0, 1.0).reshape(input_rgb.shape)
        out = (corr * 255.0).astype(np.float32)
        final = self.apply_gamma_correction(out, inverse=False)
        return final.astype(np.uint8)
    
    def evaluate_correction(self):
        """评估校正效果"""
        corrected = self.apply_correction(self.measured_data.astype(np.float32))
        
        measured_xyz = self.rgb_to_xyz(self.measured_data.astype(np.float32))
        corrected_xyz = self.rgb_to_xyz(corrected.astype(np.float32))
        target_xyz = self.rgb_to_xyz(self.target_data.astype(np.float32))
        
        measured_lab = self.xyz_to_lab(measured_xyz)
        corrected_lab = self.xyz_to_lab(corrected_xyz)
        target_lab = self.xyz_to_lab(target_xyz)
        
        diff_before = self.calculate_color_difference(measured_lab, target_lab)
        diff_after = self.calculate_color_difference(corrected_lab, target_lab)
        
        print("="*50)
        print("校正效果评估报告")
        print("="*50)
        print(f"校正前平均色差: {np.mean(diff_before):.3f}")
        print(f"校正后平均色差: {np.mean(diff_after):.3f}")
        print(f"色差改善: {np.mean(diff_before) - np.mean(diff_after):.3f}")
        print(f"改善百分比: {((np.mean(diff_before) - np.mean(diff_after)) / np.mean(diff_before) * 100):.1f}%")
        print(f"校正前最大色差: {np.max(diff_before):.3f}")
        print(f"校正后最大色差: {np.max(diff_after):.3f}")
        print(f"色差<1.0的像素比例: 校正前{np.mean(diff_before < 1.0)*100:.1f}%, 校正后{np.mean(diff_after < 1.0)*100:.1f}%")
        print("="*50)
        
        return corrected, diff_before, diff_after
    
    def visualize_results(self):
        """可视化校正结果"""
        corrected_data = self.apply_correction(self.measured_data.astype(np.float32))
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # 第一行：测量数据
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['Reds', 'Greens', 'Blues'])):
            im = axes[0, i].imshow(self.measured_data[:, :, i], cmap=color, vmin=0, vmax=255)
            axes[0, i].set_title(f'测量值 - {channel} 通道')
            axes[0, i].axis('off')
            plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        measured_rgb = np.clip(self.measured_data / 255.0, 0, 1)
        axes[0, 3].imshow(measured_rgb)
        axes[0, 3].set_title('测量值 - RGB合成')
        axes[0, 3].axis('off')
        
        # 第二行：目标数据
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['Reds', 'Greens', 'Blues'])):
            im = axes[1, i].imshow(self.target_data[:, :, i], cmap=color, vmin=0, vmax=255)
            axes[1, i].set_title(f'目标值 - {channel} 通道')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        target_rgb = np.clip(self.target_data / 255.0, 0, 1)
        axes[1, 3].imshow(target_rgb)
        axes[1, 3].set_title('目标值 - RGB合成')
        axes[1, 3].axis('off')
        
        # 第三行：校正后数据
        for i, (channel, color) in enumerate(zip(['R', 'G', 'B'], ['Reds', 'Greens', 'Blues'])):
            im = axes[2, i].imshow(corrected_data[:, :, i], cmap=color, vmin=0, vmax=255)
            axes[2, i].set_title(f'校正后 - {channel} 通道')
            axes[2, i].axis('off')
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        corrected_rgb = np.clip(corrected_data / 255.0, 0, 1)
        axes[2, 3].imshow(corrected_rgb)
        axes[2, 3].set_title('校正后 - RGB合成')
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.show()


# 主函数
if __name__ == "__main__":
    files = ["MathModel_Code\\data\\preprocess\\RedPicture.xlsx", "MathModel_Code\\data\\preprocess\\GreenPicture.xlsx", "MathModel_Code\\data\\preprocess\\BluePicture.xlsx"]
    
    corrector = LEDColorCorrection()
    
    for filepath in files:
        corrector.load_excel_data(filepath)
        correction_matrix = corrector.calibrate_correction_matrix()
        
        print("\n评估校正效果:")
        corrected_display, diff_before, diff_after = corrector.evaluate_correction()
        
        corrector.visualize_results()
        
        print("\n校正完成！")