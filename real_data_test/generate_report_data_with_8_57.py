import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def generate_all_plots():
    """生成所有需要的图表，包括新的U_c=8.57 FSE解"""
    
    print("Generating all plots including the new U_c=8.57 FSE solution...")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print(f"Data loaded: {len(df_full)} points, L values: {sorted(df_full['L'].unique())}")
    
    # 1. Raw Curves (All L)
    print("\n=== Generating Raw Curves (All L) ===")
    plt.figure(figsize=(10, 6))
    for L in sorted(df_full["L"].unique()):
        m = (df_full["L"]==L).to_numpy()
        U_vals = df_full["U"][m].to_numpy(float)
        Y_vals = df_full["Y"][m].to_numpy(float)
        sigma_vals = df_full["sigma"][m].to_numpy(float)
        order = np.argsort(U_vals)
        U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
        plt.errorbar(U_vals, Y_vals, yerr=sigma_vals, fmt="o-", lw=1.2, ms=4, capsize=3, 
                    label=f"L={L}", elinewidth=1)
    plt.xlabel("U"); plt.ylabel("Y")
    plt.title("Raw Data Curves - All L Values")
    plt.grid(True, alpha=0.25); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "raw_curves_all_L.png"), dpi=180)
    plt.close()
    
    # 2. No FSE - All L
    print("=== Generating No FSE - All L ===")
    try:
        (params_no_fse_all, errs_no_fse_all) = fit_data_collapse(data, err, 8.66, 1.025, 
                                                                n_knots=10, lam=1e-3, n_boot=5,
                                                                bounds=((8.60, 8.80), (0.8, 1.3)))
        
        x_no_fse_all, Ycorr_no_fse_all = collapse_transform(data, params_no_fse_all)
        
        plt.figure(figsize=(10, 6))
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_no_fse_all[m]; ys = Ycorr_no_fse_all[m]; ss = df_full["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE - All L: Uc={params_no_fse_all[0]:.4f}, ν^(-1)={params_no_fse_all[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_all_L.png"), dpi=180)
        plt.close()
        
        print(f"  U_c = {params_no_fse_all[0]:.6f} ± {errs_no_fse_all[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse_all[1]:.6f} ± {errs_no_fse_all[1]:.6f}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        params_no_fse_all, errs_no_fse_all = None, None
    
    # 3. No FSE - Drop L=7
    print("=== Generating No FSE - Drop L=7 ===")
    df_no_L7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data_no_L7 = df_no_L7[["L","U","Y"]].to_numpy(float)
    err_no_L7 = df_no_L7["sigma"].to_numpy(float)
    
    try:
        (params_no_fse_L7, errs_no_fse_L7) = fit_data_collapse(data_no_L7, err_no_L7, 8.66, 1.025, 
                                                              n_knots=10, lam=1e-3, n_boot=5,
                                                              bounds=((8.60, 8.80), (0.8, 1.3)))
        
        x_no_fse_L7, Ycorr_no_fse_L7 = collapse_transform(data_no_L7, params_no_fse_L7)
        
        plt.figure(figsize=(10, 6))
        for L in sorted(df_no_L7["L"].unique()):
            m = (df_no_L7["L"]==L).to_numpy()
            xs = x_no_fse_L7[m]; ys = Ycorr_no_fse_L7[m]; ss = df_no_L7["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE - Drop L=7: Uc={params_no_fse_L7[0]:.4f}, ν^(-1)={params_no_fse_L7[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_drop_L7.png"), dpi=180)
        plt.close()
        
        print(f"  U_c = {params_no_fse_L7[0]:.6f} ± {errs_no_fse_L7[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse_L7[1]:.6f} ± {errs_no_fse_L7[1]:.6f}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        params_no_fse_L7, errs_no_fse_L7 = None, None
    
    # 4. No FSE - Drop L=7,9
    print("=== Generating No FSE - Drop L=7,9 ===")
    df_no_L7_9 = df_full[~df_full["L"].isin([7, 9])].copy().reset_index(drop=True)
    data_no_L7_9 = df_no_L7_9[["L","U","Y"]].to_numpy(float)
    err_no_L7_9 = df_no_L7_9["sigma"].to_numpy(float)
    
    try:
        (params_no_fse_L7_9, errs_no_fse_L7_9) = fit_data_collapse(data_no_L7_9, err_no_L7_9, 8.66, 1.025, 
                                                                   n_knots=10, lam=1e-3, n_boot=5,
                                                                   bounds=((8.50, 8.80), (0.8, 1.3)))
        
        x_no_fse_L7_9, Ycorr_no_fse_L7_9 = collapse_transform(data_no_L7_9, params_no_fse_L7_9)
        
        plt.figure(figsize=(10, 6))
        for L in sorted(df_no_L7_9["L"].unique()):
            m = (df_no_L7_9["L"]==L).to_numpy()
            xs = x_no_fse_L7_9[m]; ys = Ycorr_no_fse_L7_9[m]; ss = df_no_L7_9["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE - Drop L=7,9: Uc={params_no_fse_L7_9[0]:.4f}, ν^(-1)={params_no_fse_L7_9[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_drop_L7_9.png"), dpi=180)
        plt.close()
        
        print(f"  U_c = {params_no_fse_L7_9[0]:.6f} ± {errs_no_fse_L7_9[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse_L7_9[1]:.6f} ± {errs_no_fse_L7_9[1]:.6f}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        params_no_fse_L7_9, errs_no_fse_L7_9 = None, None
    
    # 5. FSE-normalized - All L (Original high-quality solution)
    print("=== Generating FSE-normalized - All L (Original U_c=8.39) ===")
    try:
        (params_fse_orig, errs_fse_orig) = fit_data_collapse_fse(data, err, 8.40, 1.4, 0.8, -0.3, 
                                                               n_knots=10, lam=1e-3, n_boot=5,
                                                               bounds=((8.30, 9.00), (1.2, 3.0), (0.0, 3.0), (-1.5, -0.05)),
                                                               normalize=True)
        
        x_fse_orig, Ycorr_fse_orig = collapse_transform(data, params_fse_orig, normalize=True)
        
        plt.figure(figsize=(10, 6))
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_fse_orig[m]; ys = Ycorr_fse_orig[m]
            # FSE误差传播
            Lvals = df_full["L"][m].to_numpy(float)
            b, c = params_fse_orig[2], params_fse_orig[3]
            Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
            S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
            ss = (df_full["sigma"][m].to_numpy() / S)
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y / normalized (1 + b L^c)")
        plt.title(f"FSE-normalized - All L (High Quality): Uc={params_fse_orig[0]:.4f}, ν^(-1)={params_fse_orig[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "fse_normalized_all_L_high_quality.png"), dpi=180)
        plt.close()
        
        print(f"  U_c = {params_fse_orig[0]:.6f} ± {errs_fse_orig[0]:.6f}")
        print(f"  ν^(-1) = {params_fse_orig[1]:.6f} ± {errs_fse_orig[1]:.6f}")
        print(f"  b = {params_fse_orig[2]:.6f} ± {errs_fse_orig[2]:.6f}")
        print(f"  c = {params_fse_orig[3]:.6f} ± {errs_fse_orig[3]:.6f}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        params_fse_orig, errs_fse_orig = None, None
    
    # 6. FSE-normalized - All L (New physically reasonable solution U_c=8.57)
    print("=== Generating FSE-normalized - All L (New U_c=8.57) ===")
    try:
        (params_fse_new, errs_fse_new) = fit_data_collapse_fse(data, err, 8.58, 1.025, 0.6, -0.5, 
                                                             n_knots=10, lam=1e-3, n_boot=5,
                                                             bounds=((8.50, 8.80), (0.8, 1.3), (0.0, 2.0), (-1.0, -0.1)),
                                                             normalize=True)
        
        x_fse_new, Ycorr_fse_new = collapse_transform(data, params_fse_new, normalize=True)
        
        plt.figure(figsize=(10, 6))
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_fse_new[m]; ys = Ycorr_fse_new[m]
            # FSE误差传播
            Lvals = df_full["L"][m].to_numpy(float)
            b, c = params_fse_new[2], params_fse_new[3]
            Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
            S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
            ss = (df_full["sigma"][m].to_numpy() / S)
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y / normalized (1 + b L^c)")
        plt.title(f"FSE-normalized - All L (Physically Reasonable): Uc={params_fse_new[0]:.4f}, ν^(-1)={params_fse_new[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "fse_normalized_all_L_physically_reasonable.png"), dpi=180)
        plt.close()
        
        print(f"  U_c = {params_fse_new[0]:.6f} ± {errs_fse_new[0]:.6f}")
        print(f"  ν^(-1) = {params_fse_new[1]:.6f} ± {errs_fse_new[1]:.6f}")
        print(f"  b = {params_fse_new[2]:.6f} ± {errs_fse_new[2]:.6f}")
        print(f"  c = {params_fse_new[3]:.6f} ± {errs_fse_new[3]:.6f}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        params_fse_new, errs_fse_new = None, None
    
    # 7. Comparison plot: Original FSE vs New FSE vs No FSE
    print("=== Generating Comparison Plot ===")
    try:
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Original FSE (High Quality)
        plt.subplot(2, 3, 1)
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_fse_orig[m]; ys = Ycorr_fse_orig[m]
            Lvals = df_full["L"][m].to_numpy(float)
            b, c = params_fse_orig[2], params_fse_orig[3]
            Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
            S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
            ss = (df_full["sigma"][m].to_numpy() / S)
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=2, capsize=1, elinewidth=0.8, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y / normalized")
        plt.title(f"FSE High Quality\nUc={params_fse_orig[0]:.4f}, ν^(-1)={params_fse_orig[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend(fontsize=8)
        
        # Subplot 2: New FSE (Physically Reasonable)
        plt.subplot(2, 3, 2)
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_fse_new[m]; ys = Ycorr_fse_new[m]
            Lvals = df_full["L"][m].to_numpy(float)
            b, c = params_fse_new[2], params_fse_new[3]
            Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
            S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
            ss = (df_full["sigma"][m].to_numpy() / S)
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=2, capsize=1, elinewidth=0.8, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y / normalized")
        plt.title(f"FSE Physically Reasonable\nUc={params_fse_new[0]:.4f}, ν^(-1)={params_fse_new[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend(fontsize=8)
        
        # Subplot 3: No FSE
        plt.subplot(2, 3, 3)
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_no_fse_all[m]; ys = Ycorr_no_fse_all[m]; ss = df_full["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=2, capsize=1, elinewidth=0.8, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE\nUc={params_no_fse_all[0]:.4f}, ν^(-1)={params_no_fse_all[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend(fontsize=8)
        
        # Subplot 4: Raw data for reference
        plt.subplot(2, 3, 4)
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            U_vals = df_full["U"][m].to_numpy(float)
            Y_vals = df_full["Y"][m].to_numpy(float)
            sigma_vals = df_full["sigma"][m].to_numpy(float)
            order = np.argsort(U_vals)
            U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
            plt.errorbar(U_vals, Y_vals, yerr=sigma_vals, fmt="o-", lw=1.2, ms=3, capsize=2, 
                        label=f"L={L}", elinewidth=1)
        plt.xlabel("U"); plt.ylabel("Y")
        plt.title("Raw Data Curves")
        plt.grid(True, alpha=0.25); plt.legend(fontsize=8)
        
        # Subplot 5: Parameter comparison
        plt.subplot(2, 3, 5)
        methods = ['FSE High Quality', 'FSE Physically\nReasonable', 'No FSE']
        Uc_values = [params_fse_orig[0], params_fse_new[0], params_no_fse_all[0]]
        Uc_errors = [errs_fse_orig[0], errs_fse_new[0], errs_no_fse_all[0]]
        colors = ['red', 'blue', 'green']
        
        bars = plt.bar(methods, Uc_values, yerr=Uc_errors, capsize=5, color=colors, alpha=0.7)
        plt.ylabel("U_c value")
        plt.title("Critical Point U_c Comparison")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, Uc_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Subplot 6: Collapse quality comparison
        plt.subplot(2, 3, 6)
        # Calculate collapse quality for each method
        qualities = []
        for method_name, x_data, y_data in [("FSE High Quality", x_fse_orig, Ycorr_fse_orig),
                                           ("FSE Physically\nReasonable", x_fse_new, Ycorr_fse_new),
                                           ("No FSE", x_no_fse_all, Ycorr_no_fse_all)]:
            x_range = x_data.max() - x_data.min()
            y_ranges = []
            for L in sorted(df_full["L"].unique()):
                m = (df_full["L"]==L).to_numpy()
                y_range = y_data[m].max() - y_data[m].min()
                y_ranges.append(y_range)
            quality = x_range / np.mean(y_ranges)
            qualities.append(quality)
        
        bars = plt.bar(methods, qualities, color=colors, alpha=0.7)
        plt.ylabel("Collapse Quality")
        plt.title("Data Collapse Quality Comparison")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, quality in zip(bars, qualities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{quality:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "comprehensive_comparison.png"), dpi=180)
        plt.close()
        
        print("  Comparison plot saved as: comprehensive_comparison.png")
        
    except Exception as e:
        print(f"  Comparison plot failed: {e}")
    
    # 返回所有结果用于生成表格
    results = {
        'No FSE - All L': (params_no_fse_all, errs_no_fse_all, len(df_full)),
        'No FSE - Drop L=7': (params_no_fse_L7, errs_no_fse_L7, len(df_no_L7)),
        'No FSE - Drop L=7,9': (params_no_fse_L7_9, errs_no_fse_L7_9, len(df_no_L7_9)),
        'FSE-high-quality': (params_fse_orig, errs_fse_orig, len(df_full)),
        'FSE-physically-reasonable': (params_fse_new, errs_fse_new, len(df_full))
    }
    
    return results

def calculate_collapse_quality(df, x_collapsed, Y_collapsed):
    """计算坍缩质量"""
    x_range = x_collapsed.max() - x_collapsed.min()
    y_ranges = []
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
        y_ranges.append(y_range)
    return x_range / np.mean(y_ranges)

def main():
    """主函数"""
    print("Generating comprehensive report data with U_c=8.57 FSE solution...")
    
    # 生成所有图表
    results = generate_all_plots()
    
    # 计算坍缩质量
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print("\n=== Calculating Collapse Quality ===")
    
    qualities = {}
    
    # No FSE - All L
    if results['No FSE - All L'][0] is not None:
        params, errs, n_points = results['No FSE - All L']
        x_no_fse, Ycorr_no_fse = collapse_transform(data, params)
        quality = calculate_collapse_quality(df_full, x_no_fse, Ycorr_no_fse)
        qualities['No FSE - All L'] = quality
        print(f"No FSE - All L: Collapse quality = {quality:.2f}")
    
    # No FSE - Drop L=7
    if results['No FSE - Drop L=7'][0] is not None:
        params, errs, n_points = results['No FSE - Drop L=7']
        df_no_L7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
        data_no_L7 = df_no_L7[["L","U","Y"]].to_numpy(float)
        err_no_L7 = df_no_L7["sigma"].to_numpy(float)
        x_no_fse_L7, Ycorr_no_fse_L7 = collapse_transform(data_no_L7, params)
        quality = calculate_collapse_quality(df_no_L7, x_no_fse_L7, Ycorr_no_fse_L7)
        qualities['No FSE - Drop L=7'] = quality
        print(f"No FSE - Drop L=7: Collapse quality = {quality:.2f}")
    
    # No FSE - Drop L=7,9
    if results['No FSE - Drop L=7,9'][0] is not None:
        params, errs, n_points = results['No FSE - Drop L=7,9']
        df_no_L7_9 = df_full[~df_full["L"].isin([7, 9])].copy().reset_index(drop=True)
        data_no_L7_9 = df_no_L7_9[["L","U","Y"]].to_numpy(float)
        err_no_L7_9 = df_no_L7_9["sigma"].to_numpy(float)
        x_no_fse_L7_9, Ycorr_no_fse_L7_9 = collapse_transform(data_no_L7_9, params)
        quality = calculate_collapse_quality(df_no_L7_9, x_no_fse_L7_9, Ycorr_no_fse_L7_9)
        qualities['No FSE - Drop L=7,9'] = quality
        print(f"No FSE - Drop L=7,9: Collapse quality = {quality:.2f}")
    
    # FSE-high-quality
    if results['FSE-high-quality'][0] is not None:
        params, errs, n_points = results['FSE-high-quality']
        x_fse_orig, Ycorr_fse_orig = collapse_transform(data, params, normalize=True)
        quality = calculate_collapse_quality(df_full, x_fse_orig, Ycorr_fse_orig)
        qualities['FSE-high-quality'] = quality
        print(f"FSE-high-quality: Collapse quality = {quality:.2f}")
    
    # FSE-physically-reasonable
    if results['FSE-physically-reasonable'][0] is not None:
        params, errs, n_points = results['FSE-physically-reasonable']
        x_fse_new, Ycorr_fse_new = collapse_transform(data, params, normalize=True)
        quality = calculate_collapse_quality(df_full, x_fse_new, Ycorr_fse_new)
        qualities['FSE-physically-reasonable'] = quality
        print(f"FSE-physically-reasonable: Collapse quality = {quality:.2f}")
    
    print("\n=== Results Summary ===")
    for method, (params, errs, n_points) in results.items():
        if params is not None:
            if len(params) == 2:  # No FSE
                print(f"{method}:")
                print(f"  U_c = {params[0]:.6f} ± {errs[0]:.6f}")
                print(f"  ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
                print(f"  Collapse quality = {qualities.get(method, 'N/A')}")
                print(f"  Data points = {n_points}")
            else:  # FSE
                print(f"{method}:")
                print(f"  U_c = {params[0]:.6f} ± {errs[0]:.6f}")
                print(f"  ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
                print(f"  b = {params[2]:.6f} ± {errs[2]:.6f}")
                print(f"  c = {params[3]:.6f} ± {errs[3]:.6f}")
                print(f"  Collapse quality = {qualities.get(method, 'N/A')}")
                print(f"  Data points = {n_points}")
            print()
    
    print("All plots and data generated successfully!")
    print("Files created:")
    print("  - raw_curves_all_L.png")
    print("  - no_fse_all_L.png")
    print("  - no_fse_drop_L7.png")
    print("  - no_fse_drop_L7_9.png")
    print("  - fse_normalized_all_L_high_quality.png")
    print("  - fse_normalized_all_L_physically_reasonable.png")
    print("  - comprehensive_comparison.png")

if __name__ == "__main__":
    main() 