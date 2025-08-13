import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def main():
    """比较FSE和no-FSE的结果"""
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    print("=== Data Overview ===")
    print(f"Total data points: {len(df_full)}")
    print(f"L values: {sorted(df_full['L'].unique())}")
    print(f"U range: {df_full['U'].min():.3f} to {df_full['U'].max():.3f}")
    print(f"Y range: {df_full['Y'].min():.3f} to {df_full['Y'].max():.3f}")
    
    # 1. FSE方法（使用所有数据）
    print("\n=== FSE Method (All data) ===")
    data_fse = df_full[["L","U","Y"]].to_numpy(float)
    err_fse = df_full["sigma"].to_numpy(float)
    
    try:
        (params_fse, errs_fse) = fit_data_collapse_fse(data_fse, err_fse, 8.40, 1.4, 0.8, -0.3, 
                                                      n_knots=10, lam=1e-3, n_boot=10,
                                                      bounds=((8.30, 9.00), (1.2, 3.0), (0.0, 3.0), (-1.5, -0.05)),
                                                      normalize=True)
        print(f"FSE fitted parameters:")
        print(f"  U_c = {params_fse[0]:.6f} ± {errs_fse[0]:.6f}")
        print(f"  a = {params_fse[1]:.6f} ± {errs_fse[1]:.6f}")
        print(f"  b = {params_fse[2]:.6f} ± {errs_fse[2]:.6f}")
        print(f"  c = {params_fse[3]:.6f} ± {errs_fse[3]:.6f}")
        
        # 计算FSE坍缩质量
        x_fse, Ycorr_fse = collapse_transform(data_fse, params_fse, normalize=True)
        x_range_fse = x_fse.max() - x_fse.min()
        y_ranges_fse = []
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            y_range = df_full["Y"][m].max() - df_full["Y"][m].min()
            y_ranges_fse.append(y_range)
        collapse_quality_fse = x_range_fse / np.mean(y_ranges_fse)
        print(f"  Collapse quality: {collapse_quality_fse:.2f}")
        
    except Exception as e:
        print(f"FSE method failed: {e}")
        params_fse = None
    
    # 2. 经典方法（去掉L=7）
    print("\n=== Classic Method (L=7 removed) ===")
    df_no_fse = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data_classic = df_no_fse[["L","U","Y"]].to_numpy(float)
    err_classic = df_no_fse["sigma"].to_numpy(float)
    
    try:
        (params_classic, errs_classic) = fit_data_collapse(data_classic, err_classic, 8.40, 1.4, 
                                                         n_knots=10, lam=1e-3, n_boot=10,
                                                         bounds=((8.30, 8.70), (0.8, 2.0)))
        print(f"Classic fitted parameters:")
        print(f"  U_c = {params_classic[0]:.6f} ± {errs_classic[0]:.6f}")
        print(f"  a = {params_classic[1]:.6f} ± {errs_classic[1]:.6f}")
        
        # 计算经典方法坍缩质量
        x_classic, Ycorr_classic = collapse_transform(data_classic, params_classic)
        x_range_classic = x_classic.max() - x_classic.min()
        y_ranges_classic = []
        for L in sorted(df_no_fse["L"].unique()):
            m = (df_no_fse["L"]==L).to_numpy()
            y_range = df_no_fse["Y"][m].max() - df_no_fse["Y"][m].min()
            y_ranges_classic.append(y_range)
        collapse_quality_classic = x_range_classic / np.mean(y_ranges_classic)
        print(f"  Collapse quality: {collapse_quality_classic:.2f}")
        
    except Exception as e:
        print(f"Classic method failed: {e}")
        params_classic = None
    
    # 3. 绘制比较图
    if params_fse is not None and params_classic is not None:
        print("\n=== Generating Comparison Plots ===")
        
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 原始数据对比
        ax1 = axes[0, 0]
        for L in sorted(df_full["L"].unique()):
            sub = df_full[df_full["L"]==L]
            line, = ax1.plot(sub["U"], sub["Y"], "-o", lw=1.2, ms=3, label=f"L={L}")
            ax1.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        ax1.set_xlabel("U"); ax1.set_ylabel("Y"); ax1.set_title("Raw data (All L values)")
        ax1.grid(True, alpha=0.25); ax1.legend()
        
        # 去掉L=7后的数据
        ax2 = axes[0, 1]
        for L in sorted(df_no_fse["L"].unique()):
            sub = df_no_fse[df_no_fse["L"]==L]
            line, = ax2.plot(sub["U"], sub["Y"], "-o", lw=1.2, ms=3, label=f"L={L}")
            ax2.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        ax2.set_xlabel("U"); ax2.set_ylabel("Y"); ax2.set_title("Raw data (L=7 removed)")
        ax2.grid(True, alpha=0.25); ax2.legend()
        
        # FSE坍缩结果
        ax3 = axes[1, 0]
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_fse[m]; ys = Ycorr_fse[m]
            # FSE误差传播
            Lvals = df_full["L"][m].to_numpy(float)
            b, c = params_fse[2], params_fse[3]
            Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
            S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
            ss = (df_full["sigma"][m].to_numpy() / S)
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = ax3.plot(xs, ys, "-o", lw=1.2, ms=3, label=f"L={L}")
            ax3.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        ax3.set_xlabel("(U - Uc) * L^a"); ax3.set_ylabel("Y / normalized (1 + b L^c)")
        ax3.set_title(f"FSE Collapse: Uc={params_fse[0]:.4f}, a={params_fse[1]:.3f}")
        ax3.grid(True, alpha=0.25); ax3.legend()
        
        # 经典方法坍缩结果
        ax4 = axes[1, 1]
        for L in sorted(df_no_fse["L"].unique()):
            m = (df_no_fse["L"]==L).to_numpy()
            xs = x_classic[m]; ys = Ycorr_classic[m]
            ss = df_no_fse["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = ax4.plot(xs, ys, "-o", lw=1.2, ms=3, label=f"L={L}")
            ax4.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        ax4.set_xlabel("(U - Uc) * L^a"); ax4.set_ylabel("Y")
        ax4.set_title(f"Classic Collapse: Uc={params_classic[0]:.4f}, a={params_classic[1]:.3f}")
        ax4.grid(True, alpha=0.25); ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "fse_vs_no_fse_comparison.png"), dpi=180)
        plt.close()
        
        print("Comparison plot saved as: fse_vs_no_fse_comparison.png")
    
    # 4. 总结和建议
    print("\n=== Summary and Recommendations ===")
    if params_fse is not None and params_classic is not None:
        print(f"FSE method (all data):")
        print(f"  U_c = {params_fse[0]:.6f} ± {errs_fse[0]:.6f}")
        print(f"  a = {params_fse[1]:.6f} ± {errs_fse[1]:.6f}")
        print(f"  Collapse quality: {collapse_quality_fse:.2f}")
        print(f"  Data points: {len(df_full)}")
        
        print(f"\nClassic method (L=7 removed):")
        print(f"  U_c = {params_classic[0]:.6f} ± {errs_classic[0]:.6f}")
        print(f"  a = {params_classic[1]:.6f} ± {errs_classic[1]:.6f}")
        print(f"  Collapse quality: {collapse_quality_classic:.2f}")
        print(f"  Data points: {len(df_no_fse)}")
        
        print(f"\nKey observations:")
        print(f"  - FSE includes finite-size corrections (b, c parameters)")
        print(f"  - Classic method is simpler but may miss finite-size effects")
        print(f"  - Removing L=7 reduces data but may improve collapse quality")
        print(f"  - Compare collapse quality ratios to decide which method is better")
        
        print(f"\nRecommendations:")
        if collapse_quality_fse > collapse_quality_classic:
            print(f"  - FSE method shows better collapse quality ({collapse_quality_fse:.2f} vs {collapse_quality_classic:.2f})")
            print(f"  - Keep using FSE method for final analysis")
        else:
            print(f"  - Classic method shows better collapse quality ({collapse_quality_classic:.2f} vs {collapse_quality_fse:.2f})")
            print(f"  - Consider using classic method or investigate why FSE is not helping")
        
        print(f"  - Check the generated plots to visually assess collapse quality")
        print(f"  - Consider the physical meaning of finite-size corrections in your system")

if __name__ == "__main__":
    main() 