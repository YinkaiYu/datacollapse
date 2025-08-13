import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def get_objective_value_simple(data, params, err, method='no_fse'):
    """简化版本：直接计算加权残差平方和"""
    
    L, U, Y = data[:, 0], data[:, 1], data[:, 2]
    sigma = err if err is not None else np.full_like(Y, np.std(Y)*0.05 + 1e-6)
    w = 1.0/(sigma**2)  # 权重
    
    if method == 'fse' and len(params) >= 4:
        # FSE方法：考虑有限尺寸修正
        Uc, a, b, c = params[:4]
        # 计算归一化因子
        Lr = float(np.exp(np.mean(np.log(L))))
        s = (1.0 + b*(L**c)) / (1.0 + b*(Lr**c))
        Yc = Y / s
        w = w * (s**2)  # 方差传播
    else:
        # 经典方法：无有限尺寸修正
        Uc, a = params[:2]
        Yc = Y
    
    # 计算坍缩坐标
    x = (U - Uc) * (L**a)
    
    # 简单的质量评估：计算坍缩后的方差
    # 这是datacollapse库目标函数的一个近似
    x_range = x.max() - x.min()
    y_ranges = []
    for L_val in np.unique(L):
        m = (L == L_val)
        y_range = Yc[m].max() - Yc[m].min()
        y_ranges.append(y_range)
    
    # 坍缩质量指标：x范围与y范围的比例
    collapse_quality = x_range / np.mean(y_ranges)
    
    # 加权残差平方和的近似（相对于整体平均）
    y_mean = np.mean(Yc)
    weighted_residual = np.sum(w * (Yc - y_mean)**2) / len(Y)
    
    return weighted_residual, collapse_quality

def generate_all_plots():
    """生成所有需要的图表"""
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    print(f"Data loaded: {len(df_full)} points, L values: {sorted(df_full['L'].unique())}")
    
    results = {}
    
    # 1. Raw curves - All L
    print("\n=== Generating Raw Curves (All L) ===")
    plt.figure(figsize=(10, 6))
    for L in sorted(df_full["L"].unique()):
        sub = df_full[df_full["L"]==L]
        line, = plt.plot(sub["U"], sub["Y"], "-", lw=1.2, label=f"L={L}")
        plt.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
    plt.xlabel("U"); plt.ylabel("Y"); plt.title("Raw Curves - All L")
    plt.grid(True, alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "raw_curves_all_L.png"), dpi=180); plt.close()
    
    # 2. No FSE - All L
    print("=== Generating No FSE - All L ===")
    data_all = df_full[["L","U","Y"]].to_numpy(float)
    err_all = df_full["sigma"].to_numpy(float)
    
    try:
        (params_no_fse_all, errs_no_fse_all) = fit_data_collapse(data_all, err_all, 8.40, 1.4, 
                                                                n_knots=10, lam=1e-3, n_boot=10,
                                                                bounds=((8.30, 9.00), (0.8, 2.0)))
        
        # 获取目标函数值和坍缩质量
        objective_no_fse_all, quality_no_fse_all = get_objective_value_simple(data_all, params_no_fse_all, err_all, 'no_fse')
        
        results['No FSE - All L'] = {
            'params': params_no_fse_all,
            'errors': errs_no_fse_all,
            'objective': objective_no_fse_all,
            'quality': quality_no_fse_all,
            'data_size': len(df_full)
        }
        
        print(f"  U_c = {params_no_fse_all[0]:.6f} ± {errs_no_fse_all[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse_all[1]:.6f} ± {errs_no_fse_all[1]:.6f}")
        print(f"  Objective value = {objective_no_fse_all:.6f}")
        print(f"  Collapse quality = {quality_no_fse_all:.2f}")
        
        # 绘制坍缩结果
        x_no_fse_all, Ycorr_no_fse_all = collapse_transform(data_all, params_no_fse_all)
        plt.figure(figsize=(10, 6))
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_no_fse_all[m]; ys = Ycorr_no_fse_all[m]
            ss = df_full["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE - All L: Uc={params_no_fse_all[0]:.4f}, ν^(-1)={params_no_fse_all[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_all_L.png"), dpi=180); plt.close()
        
    except Exception as e:
        print(f"  Failed: {e}")
        results['No FSE - All L'] = None
    
    # 3. No FSE - Drop L=7
    print("=== Generating No FSE - Drop L=7 ===")
    df_no_7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data_no_7 = df_no_7[["L","U","Y"]].to_numpy(float)
    err_no_7 = df_no_7["sigma"].to_numpy(float)
    
    try:
        (params_no_fse_no_7, errs_no_fse_no_7) = fit_data_collapse(data_no_7, err_no_7, 8.40, 1.4, 
                                                                  n_knots=10, lam=1e-3, n_boot=10,
                                                                  bounds=((8.30, 8.70), (0.8, 2.0)))
        
        # 获取目标函数值和坍缩质量
        objective_no_fse_no_7, quality_no_fse_no_7 = get_objective_value_simple(data_no_7, params_no_fse_no_7, err_no_7, 'no_fse')
        
        results['No FSE - Drop L=7'] = {
            'params': params_no_fse_no_7,
            'errors': errs_no_fse_no_7,
            'objective': objective_no_fse_no_7,
            'quality': quality_no_fse_no_7,
            'data_size': len(df_no_7)
        }
        
        print(f"  U_c = {params_no_fse_no_7[0]:.6f} ± {errs_no_fse_no_7[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse_no_7[1]:.6f} ± {errs_no_fse_no_7[1]:.6f}")
        print(f"  Objective value = {objective_no_fse_no_7:.6f}")
        print(f"  Collapse quality = {quality_no_fse_no_7:.2f}")
        
        # 绘制坍缩结果
        x_no_fse_no_7, Ycorr_no_fse_no_7 = collapse_transform(data_no_7, params_no_fse_no_7)
        plt.figure(figsize=(10, 6))
        for L in sorted(df_no_7["L"].unique()):
            m = (df_no_7["L"]==L).to_numpy()
            xs = x_no_fse_no_7[m]; ys = Ycorr_no_fse_no_7[m]
            ss = df_no_7["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE - Drop L=7: Uc={params_no_fse_no_7[0]:.4f}, ν^(-1)={params_no_fse_no_7[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_drop_L7.png"), dpi=180); plt.close()
        
    except Exception as e:
        print(f"  Failed: {e}")
        results['No FSE - Drop L=7'] = None
    
    # 4. No FSE - Drop L=7,9
    print("=== Generating No FSE - Drop L=7,9 ===")
    df_no_7_9 = df_full[(df_full["L"] != 7) & (df_full["L"] != 9)].copy().reset_index(drop=True)
    data_no_7_9 = df_no_7_9[["L","U","Y"]].to_numpy(float)
    err_no_7_9 = df_no_7_9["sigma"].to_numpy(float)
    
    try:
        (params_no_fse_no_7_9, errs_no_fse_no_7_9) = fit_data_collapse(data_no_7_9, err_no_7_9, 8.40, 1.4, 
                                                                       n_knots=10, lam=1e-3, n_boot=10,
                                                                       bounds=((8.30, 8.70), (0.8, 2.0)))
        
        # 获取目标函数值和坍缩质量
        objective_no_fse_no_7_9, quality_no_fse_no_7_9 = get_objective_value_simple(data_no_7_9, params_no_fse_no_7_9, err_no_7_9, 'no_fse')
        
        results['No FSE - Drop L=7,9'] = {
            'params': params_no_fse_no_7_9,
            'errors': errs_no_fse_no_7_9,
            'objective': objective_no_fse_no_7_9,
            'quality': quality_no_fse_no_7_9,
            'data_size': len(df_no_7_9)
        }
        
        print(f"  U_c = {params_no_fse_no_7_9[0]:.6f} ± {errs_no_fse_no_7_9[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse_no_7_9[1]:.6f} ± {errs_no_fse_no_7_9[1]:.6f}")
        print(f"  Objective value = {objective_no_fse_no_7_9:.6f}")
        print(f"  Collapse quality = {quality_no_fse_no_7_9:.2f}")
        
        # 绘制坍缩结果
        x_no_fse_no_7_9, Ycorr_no_fse_no_7_9 = collapse_transform(data_no_7_9, params_no_fse_no_7_9)
        plt.figure(figsize=(10, 6))
        for L in sorted(df_no_7_9["L"].unique()):
            m = (df_no_7_9["L"]==L).to_numpy()
            xs = x_no_fse_no_7_9[m]; ys = Ycorr_no_fse_no_7_9[m]
            ss = df_no_7_9["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"No FSE - Drop L=7,9: Uc={params_no_fse_no_7_9[0]:.4f}, ν^(-1)={params_no_fse_no_7_9[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_drop_L7_9.png"), dpi=180); plt.close()
        
    except Exception as e:
        print(f"  Failed: {e}")
        results['No FSE - Drop L=7,9'] = None
    
    # 5. FSE-normalized - All L
    print("=== Generating FSE-normalized - All L ===")
    try:
        (params_fse, errs_fse) = fit_data_collapse_fse(data_all, err_all, 8.40, 1.4, 0.8, -0.3, 
                                                      n_knots=10, lam=1e-3, n_boot=10,
                                                      bounds=((8.30, 9.00), (1.2, 3.0), (0.0, 3.0), (-1.5, -0.05)),
                                                      normalize=True)
        
        # 获取目标函数值和坍缩质量
        objective_fse, quality_fse = get_objective_value_simple(data_all, params_fse, err_all, 'fse')
        
        results['FSE-normalized - All L'] = {
            'params': params_fse,
            'errors': errs_fse,
            'objective': objective_fse,
            'quality': quality_fse,
            'data_size': len(df_full)
        }
        
        print(f"  U_c = {params_fse[0]:.6f} ± {errs_fse[0]:.6f}")
        print(f"  ν^(-1) = {params_fse[1]:.6f} ± {errs_fse[1]:.6f}")
        print(f"  b = {params_fse[2]:.6f} ± {errs_fse[2]:.6f}")
        print(f"  c = {params_fse[3]:.6f} ± {errs_fse[3]:.6f}")
        print(f"  Objective value = {objective_fse:.6f}")
        print(f"  Collapse quality = {quality_fse:.2f}")
        
        # 绘制坍缩结果
        x_fse, Ycorr_fse = collapse_transform(data_all, params_fse, normalize=True)
        plt.figure(figsize=(10, 6))
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
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y / normalized (1 + b L^c)")
        plt.title(f"FSE-normalized - All L: Uc={params_fse[0]:.4f}, ν^(-1)={params_fse[1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fse_normalized_all_L.png"), dpi=180); plt.close()
        
    except Exception as e:
        print(f"  Failed: {e}")
        results['FSE-normalized - All L'] = None
    
    return results

def main():
    """主函数"""
    print("Generating all plots and data for the report...")
    
    # 生成所有图表和数据
    results = generate_all_plots()
    
    # 打印结果摘要
    print("\n=== Results Summary ===")
    for method, result in results.items():
        if result is not None:
            print(f"\n{method}:")
            print(f"  U_c = {result['params'][0]:.6f} ± {result['errors'][0]:.6f}")
            print(f"  ν^(-1) = {result['params'][1]:.6f} ± {result['errors'][1]:.6f}")
            if len(result['params']) > 2:
                print(f"  b = {result['params'][2]:.6f} ± {result['errors'][2]:.6f}")
                print(f"  c = {result['params'][3]:.6f} ± {result['errors'][3]:.6f}")
            print(f"  Objective value = {result['objective']:.6f}")
            print(f"  Collapse quality = {result['quality']:.2f}")
            print(f"  Data points: {result['data_size']}")
    
    print(f"\nAll plots generated successfully!")
    print(f"Files created:")
    print(f"  - raw_curves_all_L.png")
    print(f"  - no_fse_all_L.png")
    print(f"  - no_fse_drop_L7.png")
    print(f"  - no_fse_drop_L7_9.png")
    print(f"  - fse_normalized_all_L.png")
    
    print(f"\nNote: Objective value is the weighted residual sum of squares (approximation)")
    print(f"Collapse quality is x_range/y_range ratio (higher = better collapse)")

if __name__ == "__main__":
    main() 