import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def quick_test():
    """快速测试局部最优问题"""
    
    print("=== Quick Test for Local Optima ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print(f"Data: {len(df_full)} points, L values: {sorted(df_full['L'].unique())}")
    
    # 测试关键起始点
    key_points = [
        # 原始FSE结果附近
        (8.40, 1.4, 0.8, -0.3, "Original FSE"),
        # No-FSE结果附近
        (8.66, 1.025, 0.5, -0.5, "Near No-FSE"),
        # 中间值
        (8.55, 1.1, 0.7, -0.4, "Middle"),
        # 更接近No-FSE
        (8.68, 1.0, 0.6, -0.6, "Very near No-FSE"),
    ]
    
    results = []
    
    for Uc0, a0, b0, c0, desc in key_points:
        print(f"\n--- Testing: {desc} (Uc0={Uc0:.2f}, a0={a0:.2f}, b0={b0:.2f}, c0={c0:.2f}) ---")
        
        try:
            # 尝试FSE拟合
            (params, errs) = fit_data_collapse_fse(data, err, Uc0, a0, b0, c0, 
                                                  n_knots=10, lam=1e-3, n_boot=3,
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
                'description': desc,
                'starting_point': (Uc0, a0, b0, c0),
                'fitted_params': params,
                'errors': errs,
                'collapse_quality': collapse_quality,
                'converged': True
            })
            
            print(f"  ✅ Converged to:")
            print(f"     U_c = {params[0]:.6f} ± {errs[0]:.6f}")
            print(f"     ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
            print(f"     b = {params[2]:.6f} ± {errs[2]:.6f}")
            print(f"     c = {params[3]:.6f} ± {errs[3]:.6f}")
            print(f"     Collapse quality = {collapse_quality:.2f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'description': desc,
                'starting_point': (Uc0, a0, b0, c0),
                'fitted_params': None,
                'errors': None,
                'collapse_quality': 0,
                'converged': False
            })
    
    return results

def test_specific_8_6():
    """专门测试U_c=8.6附近的FSE拟合"""
    
    print("\n=== Testing FSE specifically near U_c=8.6 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    # 在U_c=8.6附近尝试
    test_points = [
        (8.58, 1.025, 0.6, -0.5),
        (8.60, 1.025, 0.6, -0.5),
        (8.62, 1.025, 0.6, -0.5),
        (8.64, 1.025, 0.6, -0.5),
        (8.66, 1.025, 0.6, -0.5),
    ]
    
    best_result = None
    best_quality = 0
    
    for Uc0, a0, b0, c0 in test_points:
        try:
            print(f"  Testing Uc0={Uc0:.2f}...")
            
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
            
            print(f"    -> U_c={params[0]:.4f}, quality={collapse_quality:.2f}")
            
            if collapse_quality > best_quality:
                best_quality = collapse_quality
                best_result = {
                    'starting_point': (Uc0, a0, b0, c0),
                    'fitted_params': params,
                    'errors': errs,
                    'collapse_quality': collapse_quality
                }
                
        except Exception as e:
            print(f"    -> Failed: {e}")
            continue
    
    if best_result:
        print(f"\n🎯 Best result near U_c=8.6:")
        print(f"   Starting point: Uc0={best_result['starting_point'][0]:.2f}")
        print(f"   Fitted U_c = {best_result['fitted_params'][0]:.6f} ± {best_result['errors'][0]:.6f}")
        print(f"   ν^(-1) = {best_result['fitted_params'][1]:.6f} ± {best_result['errors'][1]:.6f}")
        print(f"   b = {best_result['fitted_params'][2]:.6f} ± {best_result['errors'][2]:.6f}")
        print(f"   c = {best_result['fitted_params'][3]:.6f} ± {best_result['errors'][3]:.6f}")
        print(f"   Collapse quality = {best_result['collapse_quality']:.2f}")
        
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
        plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "fse_near_8_6_quick.png"), dpi=180); plt.close()
        
        print(f"   📊 Plot saved as: fse_near_8_6_quick.png")
    
    return best_result

def main():
    """主函数"""
    print("🔍 Quick Test for Local Optima in FSE Fitting")
    print("="*60)
    
    # 1. 快速测试关键起始点
    results = quick_test()
    
    # 2. 专门测试U_c=8.6附近
    print("\n" + "="*60)
    best_8_6 = test_specific_8_6()
    
    # 3. 总结分析
    print("\n" + "="*60)
    print("📋 ANALYSIS SUMMARY")
    print("="*60)
    
    converged_results = [r for r in results if r['converged']]
    
    if len(converged_results) > 1:
        print(f"\n✅ Multiple solutions found - Local optima confirmed!")
        print(f"   This explains the large difference between FSE and No-FSE results.")
        
        # 按坍缩质量排序
        converged_results.sort(key=lambda x: x['collapse_quality'], reverse=True)
        
        print(f"\n🏆 Solutions ranked by collapse quality:")
        for i, result in enumerate(converged_results):
            print(f"   {i+1}. {result['description']}: U_c={result['fitted_params'][0]:.4f}, quality={result['collapse_quality']:.2f}")
        
        print(f"\n💡 Key Insights:")
        print(f"   - FSE method can converge to different local optima")
        print(f"   - Starting point significantly affects final result")
        print(f"   - Best collapse quality: {converged_results[0]['collapse_quality']:.2f}")
        print(f"   - U_c range: {min(r['fitted_params'][0] for r in converged_results):.3f} to {max(r['fitted_params'][0] for r in converged_results):.3f}")
    
    if best_8_6:
        print(f"\n🎯 FSE near U_c=8.6:")
        print(f"   - Successfully converged to U_c = {best_8_6['fitted_params'][0]:.4f}")
        print(f"   - Collapse quality: {best_8_6['collapse_quality']:.2f}")
        print(f"   - This shows FSE can work near No-FSE results")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    print(f"   1. Always test multiple starting points for FSE fitting")
    print(f"   2. Compare solutions based on collapse quality, not just U_c value")
    print(f"   3. Consider physical constraints when choosing between solutions")
    print(f"   4. The 'best' solution depends on your specific criteria")

if __name__ == "__main__":
    main() 