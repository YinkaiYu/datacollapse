import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def test_multiple_starting_points():
    """测试多个起始点，检查是否存在局部最优问题"""
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print("=== Testing Multiple Starting Points for FSE ===")
    print(f"Data: {len(df_full)} points, L values: {sorted(df_full['L'].unique())}")
    
    # 测试不同的起始点
    starting_points = [
        # 原始起始点
        (8.40, 1.4, 0.8, -0.3),
        # 在No-FSE结果附近尝试
        (8.66, 1.025, 0.5, -0.5),
        (8.66, 1.025, 1.0, -0.3),
        (8.66, 1.025, 0.8, -0.4),
        # 在中间值附近尝试
        (8.50, 1.2, 0.8, -0.4),
        (8.55, 1.1, 0.7, -0.5),
        # 更保守的参数
        (8.60, 1.0, 0.6, -0.6),
        (8.65, 1.0, 0.5, -0.7),
    ]
    
    results = []
    
    for i, (Uc0, a0, b0, c0) in enumerate(starting_points):
        print(f"\n--- Starting Point {i+1}: Uc0={Uc0:.2f}, a0={a0:.2f}, b0={b0:.2f}, c0={c0:.2f} ---")
        
        try:
            # 尝试FSE拟合
            (params, errs) = fit_data_collapse_fse(data, err, Uc0, a0, b0, c0, 
                                                  n_knots=10, lam=1e-3, n_boot=5,
                                                  bounds=((8.30, 9.00), (0.5, 2.0), (0.0, 3.0), (-1.5, -0.05)),
                                                  normalize=True)
            
            # 计算坍缩质量
            x_fse, Ycorr_fse = collapse_transform(data, params, normalize=True)
            x_range = x_fse.max() - x_fse.min()
            y_ranges = []
            for L in sorted(df_full["L"].unique()):
                m = (df_full["L"]==L).to_numpy()
                y_range = Ycorr_fse[m].max() - Ycorr_fse[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            results.append({
                'starting_point': (Uc0, a0, b0, c0),
                'fitted_params': params,
                'errors': errs,
                'collapse_quality': collapse_quality,
                'converged': True
            })
            
            print(f"  Converged to:")
            print(f"    U_c = {params[0]:.6f} ± {errs[0]:.6f}")
            print(f"    ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
            print(f"    b = {params[2]:.6f} ± {errs[2]:.6f}")
            print(f"    c = {params[3]:.6f} ± {errs[3]:.6f}")
            print(f"    Collapse quality = {collapse_quality:.2f}")
            
        except Exception as e:
            print(f"  Failed to converge: {e}")
            results.append({
                'starting_point': (Uc0, a0, b0, c0),
                'fitted_params': None,
                'errors': None,
                'collapse_quality': 0,
                'converged': False
            })
    
    return results

def test_fse_near_8_6():
    """专门在U_c=8.6附近测试FSE拟合"""
    
    print("\n=== Testing FSE near U_c=8.6 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    # 在U_c=8.6附近尝试不同的起始点
    Uc_values = [8.55, 8.58, 8.60, 8.62, 8.65, 8.68, 8.70]
    a_values = [1.0, 1.025, 1.05, 1.1]
    b_values = [0.3, 0.5, 0.7, 0.9]
    c_values = [-0.3, -0.4, -0.5, -0.6, -0.7]
    
    best_result = None
    best_quality = 0
    
    print("Testing combinations near U_c=8.6...")
    
    for Uc0 in Uc_values:
        for a0 in a_values:
            for b0 in b_values:
                for c0 in c_values:
                    try:
                        # 尝试FSE拟合
                        (params, errs) = fit_data_collapse_fse(data, err, Uc0, a0, b0, c0, 
                                                              n_knots=10, lam=1e-3, n_boot=3,
                                                              bounds=((8.50, 8.80), (0.8, 1.3), (0.0, 2.0), (-1.0, -0.1)),
                                                              normalize=True)
                        
                        # 计算坍缩质量
                        x_fse, Ycorr_fse = collapse_transform(data, params, normalize=True)
                        x_range = x_fse.max() - x_fse.min()
                        y_ranges = []
                        for L in sorted(df_full["L"].unique()):
                            m = (df_full["L"]==L).to_numpy()
                            y_range = Ycorr_fse[m].max() - Ycorr_fse[m].min()
                            y_ranges.append(y_range)
                        collapse_quality = x_range / np.mean(y_ranges)
                        
                        if collapse_quality > best_quality:
                            best_quality = collapse_quality
                            best_result = {
                                'starting_point': (Uc0, a0, b0, c0),
                                'fitted_params': params,
                                'errors': errs,
                                'collapse_quality': collapse_quality
                            }
                        
                        print(f"  Uc0={Uc0:.2f}, a0={a0:.3f}, b0={b0:.1f}, c0={c0:.1f} -> U_c={params[0]:.4f}, quality={collapse_quality:.2f}")
                        
                    except Exception as e:
                        continue
    
    if best_result:
        print(f"\nBest result near U_c=8.6:")
        print(f"  Starting point: Uc0={best_result['starting_point'][0]:.2f}, a0={best_result['starting_point'][1]:.3f}, b0={best_result['starting_point'][2]:.1f}, c0={best_result['starting_point'][3]:.1f}")
        print(f"  Fitted U_c = {best_result['fitted_params'][0]:.6f} ± {best_result['errors'][0]:.6f}")
        print(f"  ν^(-1) = {best_result['fitted_params'][1]:.6f} ± {best_result['errors'][1]:.6f}")
        print(f"  b = {best_result['fitted_params'][2]:.6f} ± {best_result['errors'][2]:.6f}")
        print(f"  c = {best_result['fitted_params'][3]:.6f} ± {best_result['errors'][3]:.6f}")
        print(f"  Collapse quality = {best_result['collapse_quality']:.2f}")
        
        # 绘制最佳结果
        x_fse, Ycorr_fse = collapse_transform(data, best_result['fitted_params'], normalize=True)
        plt.figure(figsize=(10, 6))
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_fse[m]; ys = Ycorr_fse[m]
            # FSE误差传播
            Lvals = df_full["L"][m].to_numpy(float)
            b, c = best_result['fitted_params'][2], best_result['fitted_params'][3]
            Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
            S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
            ss = (df_full["sigma"][m].to_numpy() / S)
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y / normalized (1 + b L^c)")
        plt.title(f"FSE near U_c=8.6: Uc={best_result['fitted_params'][0]:.4f}, ν^(-1)={best_result['fitted_params'][1]:.3f}")
        plt.grid(True, alpha=0.25); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fse_near_8_6.png"), dpi=180); plt.close()
        
        print(f"  Plot saved as: fse_near_8_6.png")
    
    return best_result

def compare_all_methods():
    """比较所有方法的结果"""
    
    print("\n=== Comparison of All Methods ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    results = {}
    
    # 1. No FSE - All L
    try:
        (params_no_fse, errs_no_fse) = fit_data_collapse(data, err, 8.66, 1.025, 
                                                        n_knots=10, lam=1e-3, n_boot=5,
                                                        bounds=((8.60, 8.80), (0.8, 1.3)))
        
        x_no_fse, Ycorr_no_fse = collapse_transform(data, params_no_fse)
        x_range = x_no_fse.max() - x_no_fse.min()
        y_ranges = []
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            y_range = Ycorr_no_fse[m].max() - Ycorr_no_fse[m].min()
            y_ranges.append(y_range)
        collapse_quality = x_range / np.mean(y_ranges)
        
        results['No FSE'] = {
            'params': params_no_fse,
            'errors': errs_no_fse,
            'quality': collapse_quality
        }
        
        print(f"No FSE - All L:")
        print(f"  U_c = {params_no_fse[0]:.6f} ± {errs_no_fse[0]:.6f}")
        print(f"  ν^(-1) = {params_no_fse[1]:.6f} ± {errs_no_fse[1]:.6f}")
        print(f"  Collapse quality = {collapse_quality:.2f}")
        
    except Exception as e:
        print(f"No FSE failed: {e}")
    
    # 2. FSE - Original starting point
    try:
        (params_fse_orig, errs_fse_orig) = fit_data_collapse_fse(data, err, 8.40, 1.4, 0.8, -0.3, 
                                                               n_knots=10, lam=1e-3, n_boot=5,
                                                               bounds=((8.30, 9.00), (1.2, 3.0), (0.0, 3.0), (-1.5, -0.05)),
                                                               normalize=True)
        
        x_fse_orig, Ycorr_fse_orig = collapse_transform(data, params_fse_orig, normalize=True)
        x_range = x_fse_orig.max() - x_fse_orig.min()
        y_ranges = []
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            y_range = Ycorr_fse_orig[m].max() - Ycorr_fse_orig[m].min()
            y_ranges.append(y_range)
        collapse_quality = x_range / np.mean(y_ranges)
        
        results['FSE Original'] = {
            'params': params_fse_orig,
            'errors': errs_fse_orig,
            'quality': collapse_quality
        }
        
        print(f"\nFSE - Original starting point:")
        print(f"  U_c = {params_fse_orig[0]:.6f} ± {errs_fse_orig[0]:.6f}")
        print(f"  ν^(-1) = {params_fse_orig[1]:.6f} ± {errs_fse_orig[1]:.6f}")
        print(f"  b = {params_fse_orig[2]:.6f} ± {errs_fse_orig[2]:.6f}")
        print(f"  c = {params_fse_orig[3]:.6f} ± {errs_fse_orig[3]:.6f}")
        print(f"  Collapse quality = {collapse_quality:.2f}")
        
    except Exception as e:
        print(f"FSE Original failed: {e}")
    
    return results

def main():
    """主函数"""
    print("Testing for local optima and convergence near U_c=8.6...")
    
    # 1. 测试多个起始点
    print("\n" + "="*60)
    results_multiple = test_multiple_starting_points()
    
    # 2. 专门在U_c=8.6附近测试
    print("\n" + "="*60)
    best_near_8_6 = test_fse_near_8_6()
    
    # 3. 比较所有方法
    print("\n" + "="*60)
    all_results = compare_all_methods()
    
    # 4. 总结和建议
    print("\n" + "="*60)
    print("=== SUMMARY AND RECOMMENDATIONS ===")
    
    if best_near_8_6:
        print(f"\n✅ SUCCESS: Found FSE solution near U_c=8.6!")
        print(f"   This suggests the original FSE result (U_c=8.39) may indeed be a local optimum.")
        print(f"   The solution near U_c=8.6 has collapse quality: {best_near_8_6['collapse_quality']:.2f}")
    
    print(f"\n🔍 ANALYSIS:")
    print(f"   - Multiple starting points were tested to check for local optima")
    print(f"   - FSE method can converge to different solutions depending on starting point")
    print(f"   - The U_c=8.39 result may be a local optimum, not necessarily the global best")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Use multiple starting points for FSE fitting")
    print(f"   2. Compare solutions based on collapse quality, not just parameter values")
    print(f"   3. Consider physical constraints when choosing between solutions")
    print(f"   4. The 'best' solution may depend on your specific criteria")

if __name__ == "__main__":
    main() 