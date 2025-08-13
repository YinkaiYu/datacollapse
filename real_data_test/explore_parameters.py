import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def load_data():
    """加载合并后的真实数据"""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)
    return df, data, err

def test_manual_collapse(df, data, err, U_c_test, a_test):
    """手动测试特定的U_c和a值，看看数据坍缩效果"""
    
    # 手动计算坍缩坐标，转换为numpy数组
    x_manual = ((df["U"] - U_c_test) * (df["L"] ** a_test)).to_numpy()
    
    plt.figure(figsize=(12, 8))
    
    # 原始数据
    plt.subplot(2, 2, 1)
    for L in sorted(df["L"].unique()):
        sub = df[df["L"]==L]
        line, = plt.plot(sub["U"], sub["Y"], "-o", lw=1.2, ms=3, label=f"L={L}")
        plt.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
    plt.axvline(U_c_test, color='red', linestyle='--', alpha=0.7, label=f'U_c={U_c_test}')
    plt.xlabel("U"); plt.ylabel("Y"); plt.title("Raw data")
    plt.grid(True, alpha=0.25); plt.legend()
    
    # 手动坍缩结果
    plt.subplot(2, 2, 2)
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        xs = x_manual[m]; ys = df["Y"][m].to_numpy()
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        plt.plot(xs, ys, "-o", lw=1.2, ms=3, label=f"L={L}")
    plt.xlabel(f"(U - {U_c_test}) * L^{a_test}"); plt.ylabel("Y")
    plt.title(f"Manual collapse: U_c={U_c_test}, a={a_test}")
    plt.grid(True, alpha=0.25); plt.legend()
    
    # 计算坍缩质量指标
    x_range = x_manual.max() - x_manual.min()
    y_ranges = []
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        y_range = df["Y"][m].max() - df["Y"][m].min()
        y_ranges.append(y_range)
    
    collapse_quality = x_range / np.mean(y_ranges)
    
    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.8, f"Collapse Quality Metrics:", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"X range: {x_range:.3f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Avg Y range: {np.mean(y_ranges):.3f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Quality ratio: {collapse_quality:.2f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Higher ratio = better collapse", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    # 参数扫描建议
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f"Parameter Scan Suggestions:", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"U_c range: {U_c_test-0.1:.2f} to {U_c_test+0.1:.2f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"a range: {a_test-0.3:.2f} to {a_test+0.3:.2f}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Try: U_c ± [0.05, 0.1, 0.15]", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Try: a ± [0.1, 0.2, 0.3]", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f"manual_collapse_test_Uc{U_c_test}_a{a_test}.png"), dpi=180)
    plt.close()
    
    return collapse_quality

def test_without_fse(df, data, err):
    """测试不使用FSE的情况"""
    print("Testing without FSE (classic collapse)...")
    
    # 尝试不同的U_c和a值
    U_c_values = [8.38, 8.40, 8.42, 8.44, 8.46]
    a_values = [1.2, 1.4, 1.6, 1.8, 2.0]
    
    best_chi2 = float('inf')
    best_params = None
    
    results = []
    
    for U_c in U_c_values:
        for a in a_values:
            try:
                # 使用经典方法（无FSE）
                (params, errs) = fit_data_collapse(data, err, U_c, a, 
                                                 n_knots=10, lam=1e-3, n_boot=2,
                                                 bounds=((U_c-0.1, U_c+0.1), (a-0.2, a+0.2)))
                
                # 计算chi2（这里需要手动计算，因为fit_data_collapse可能不返回chi2）
                x_test = (data[:, 1] - params[0]) * (data[:, 0] ** params[1])
                y_test = data[:, 2]
                # 简单的chi2计算
                chi2 = np.mean((y_test - np.mean(y_test))**2)
                
                results.append({
                    'U_c': U_c,
                    'a': a,
                    'fitted_U_c': params[0],
                    'fitted_a': params[1],
                    'chi2': chi2
                })
                
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = (U_c, a)
                    
                print(f"U_c={U_c:.2f}, a={a:.2f} -> fitted: U_c={params[0]:.4f}, a={params[1]:.4f}, chi2={chi2:.6f}")
                
            except Exception as e:
                print(f"Failed for U_c={U_c:.2f}, a={a:.2f}: {e}")
    
    # 绘制结果
    if results:
        results_df = pd.DataFrame(results)
        
        plt.figure(figsize=(15, 5))
        
        # Chi2热图
        plt.subplot(1, 3, 1)
        pivot = results_df.pivot(index='a', columns='U_c', values='chi2')
        im = plt.imshow(pivot.values, cmap='viridis', aspect='auto', 
                       extent=[pivot.columns.min(), pivot.columns.max(), 
                               pivot.index.min(), pivot.index.max()])
        plt.colorbar(im, label='Chi2')
        plt.xlabel('U_c'); plt.ylabel('a'); plt.title('Chi2 without FSE')
        
        # 拟合参数偏差
        plt.subplot(1, 3, 2)
        plt.scatter(results_df['U_c'], results_df['fitted_U_c'] - results_df['U_c'], 
                   c=results_df['chi2'], cmap='viridis')
        plt.colorbar(label='Chi2')
        plt.xlabel('Initial U_c'); plt.ylabel('Fitted - Initial U_c')
        plt.title('U_c convergence')
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.subplot(1, 3, 3)
        plt.scatter(results_df['a'], results_df['fitted_a'] - results_df['a'], 
                   c=results_df['chi2'], cmap='viridis')
        plt.colorbar(label='Chi2')
        plt.xlabel('Initial a'); plt.ylabel('Fitted - Initial a')
        plt.title('a convergence')
        plt.axhline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "no_fse_analysis.png"), dpi=180)
        plt.close()
        
        print(f"\nBest initial values without FSE: U_c={best_params[0]:.2f}, a={best_params[1]:.2f}")
        print(f"Best chi2: {best_chi2:.6f}")
    
    return best_params

def test_fse_vs_no_fse(df, data, err):
    """比较FSE和no-FSE的结果"""
    print("\nComparing FSE vs no-FSE...")
    
    # 测试不同的数据集
    datasets = {
        'All data': (data, err),
        'Without L=7': (data[df['L'] != 7], err[df['L'] != 7]),
        'Without L=7,9': (data[(df['L'] != 7) & (df['L'] != 9)], err[(df['L'] != 7) & (df['L'] != 9)])
    }
    
    results = {}
    
    for name, (data_sub, err_sub) in datasets.items():
        print(f"\n--- {name} ---")
        
        if len(data_sub) == 0:
            print("No data!")
            continue
            
        # 尝试经典方法
        try:
            (params_classic, errs_classic) = fit_data_collapse(data_sub, err_sub, 8.40, 1.4, 
                                                             n_knots=10, lam=1e-3, n_boot=2)
            print(f"Classic: U_c={params_classic[0]:.4f}, a={params_classic[1]:.4f}")
        except Exception as e:
            print(f"Classic failed: {e}")
            params_classic = None
        
        # 尝试FSE方法
        try:
            (params_fse, errs_fse) = fit_data_collapse_fse(data_sub, err_sub, 8.40, 1.4, 0.8, -0.3, 
                                                          n_knots=10, lam=1e-3, n_boot=2,
                                                          bounds=((8.30, 9.00), (1.2, 3.0), (0.0, 3.0), (-1.5, -0.05)),
                                                          normalize=True)
            print(f"FSE: U_c={params_fse[0]:.4f}, a={params_fse[1]:.4f}, b={params_fse[2]:.4f}, c={params_fse[3]:.4f}")
        except Exception as e:
            print(f"FSE failed: {e}")
            params_fse = None
        
        results[name] = {
            'classic': params_classic,
            'fse': params_fse,
            'data_size': len(data_sub)
        }
    
    return results

def main():
    """主函数"""
    print("Loading data...")
    df, data, err = load_data()
    
    print(f"Data loaded: {len(df)} points, L values: {sorted(df['L'].unique())}")
    
    # 1. 手动测试参数
    print("\n=== Manual Parameter Testing ===")
    U_c_guess = 8.38  # 从之前的拟合结果估计
    a_guess = 1.4
    
    quality = test_manual_collapse(df, data, err, U_c_guess, a_guess)
    print(f"Manual collapse quality: {quality:.2f}")
    
    # 2. 测试不同参数组合
    print("\n=== Testing Different Parameters ===")
    for U_c in [8.36, 8.38, 8.40, 8.42]:
        for a in [1.2, 1.4, 1.6]:
            quality = test_manual_collapse(df, data, err, U_c, a)
            print(f"U_c={U_c:.2f}, a={a:.1f} -> quality: {quality:.2f}")
    
    # 3. 测试无FSE情况
    print("\n=== Testing without FSE ===")
    best_no_fse = test_without_fse(df, data, err)
    
    # 4. 比较FSE vs no-FSE
    print("\n=== FSE vs No-FSE Comparison ===")
    fse_comparison = test_fse_vs_no_fse(df, data, err)
    
    print("\n=== Summary ===")
    print("1. Manual testing completed - check generated plots")
    print("2. Best no-FSE initial values:", best_no_fse)
    print("3. FSE comparison completed")
    print("\nRecommendations:")
    print("- Use the manual collapse plots to estimate good U_c and a ranges")
    print("- Compare FSE vs no-FSE results to decide if FSE is needed")
    print("- Use the best parameters as starting points for run_real.py")

if __name__ == "__main__":
    main() 