import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import time
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

# 设置matplotlib支持中文[[memory:5669012]]
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_no_fse_drop_l7():
    """分析No-FSE Drop L=7的最佳结果"""
    
    print("🔍 分析No-FSE Drop L=7 (基于改进的初始值策略)")
    print("-" * 50)
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    df_drop_l7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    
    data = df_drop_l7[["L","U","Y"]].to_numpy(float)
    err = df_drop_l7["sigma"].to_numpy(float)
    
    print(f"数据集：{len(df_drop_l7)}个数据点，L={sorted(df_drop_l7['L'].unique())}")
    
    # 基于大规模分析确定的最优策略：使用较高的初始值
    initial_values = [
        (8.67, 1.2, "高a值1"),
        (8.67, 1.3, "高a值2"), 
        (8.66, 1.25, "中等高a值"),
        (8.68, 1.15, "保守高a值"),
        (8.65, 1.35, "激进高a值")
    ]
    
    print("\n测试优化的初始值...")
    best_result = None
    best_quality = 0
    results = []
    
    # 宽松的边界允许充分探索
    bounds = ((8.0, 9.0), (0.8, 2.0))
    
    for Uc0, a0, desc in initial_values:
        print(f"  测试 {desc}: Uc0={Uc0:.2f}, a0={a0:.2f}")
        
        try:
            start_time = time.time()
            (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                             n_knots=10, lam=1e-3, n_boot=10,
                                             bounds=bounds)
            elapsed = time.time() - start_time
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data, params)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_drop_l7["L"].unique()):
                m = (df_drop_l7["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            result = {
                'description': desc,
                'initial': (Uc0, a0),
                'params': params,
                'errors': errs,
                'quality': collapse_quality,
                'time': elapsed,
                'x_collapsed': x_collapsed,
                'Y_collapsed': Y_collapsed
            }
            results.append(result)
            
            print(f"    → U_c={params[0]:.4f}±{errs[0]:.4f}, ν^(-1)={params[1]:.4f}±{errs[1]:.4f}")
            print(f"    → 质量={collapse_quality:.1f}, 耗时={elapsed:.1f}s")
            
            if collapse_quality > best_quality:
                best_quality = collapse_quality
                best_result = result.copy()
                
        except Exception as e:
            print(f"    → 失败: {e}")
    
    if best_result:
        print(f"\n✅ No-FSE Drop L=7 最佳结果:")
        print(f"   U_c = {best_result['params'][0]:.6f} ± {best_result['errors'][0]:.6f}")
        print(f"   ν^(-1) = {best_result['params'][1]:.6f} ± {best_result['errors'][1]:.6f}")
        print(f"   坍缩质量 = {best_result['quality']:.2f}")
        print(f"   来自策略: {best_result['description']}")
    
    return best_result, results, df_drop_l7

def analyze_fse_all_l():
    """分析FSE All-L的结果（带进度条）"""
    
    print(f"\n🔬 分析FSE All-L (谨慎初始值策略)")
    print("-" * 50)
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print(f"数据集：{len(df_full)}个数据点，L={sorted(df_full['L'].unique())}")
    
    # 基于经验的初始值策略（FSE参数多，更复杂）
    initial_values = [
        (8.67, 1.2, 0.5, -0.5, "保守策略"),
        (8.40, 1.4, 0.8, -0.6, "已知高质量解"),
        (8.57, 1.0, 0.6, -1.0, "物理合理解"),
    ]
    
    print(f"\n测试{len(initial_values)}组FSE初始值...")
    best_result = None
    best_quality = 0
    results = []
    
    # FSE的边界设置
    bounds = ((8.0, 9.0), (0.8, 2.0), (0.0, 2.0), (-1.5, -0.1))
    
    # 使用进度条
    for i, (Uc0, a0, b0, c0, desc) in enumerate(tqdm(initial_values, desc="FSE拟合进度")):
        print(f"\n  策略 {i+1}/{len(initial_values)}: {desc}")
        print(f"    初始值: Uc0={Uc0:.2f}, a0={a0:.2f}, b0={b0:.2f}, c0={c0:.2f}")
        
        try:
            start_time = time.time()
            
            # FSE拟合（normalize=True）
            (params, errs) = fit_data_collapse_fse(data, err, Uc0, a0, b0, c0,
                                                 n_knots=10, lam=1e-3, n_boot=5,
                                                 bounds=bounds, normalize=True)
            elapsed = time.time() - start_time
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data, params, normalize=True)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_full["L"].unique()):
                m = (df_full["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            result = {
                'description': desc,
                'initial': (Uc0, a0, b0, c0),
                'params': params,
                'errors': errs,
                'quality': collapse_quality,
                'time': elapsed,
                'x_collapsed': x_collapsed,
                'Y_collapsed': Y_collapsed
            }
            results.append(result)
            
            print(f"    → U_c={params[0]:.4f}±{errs[0]:.4f}, ν^(-1)={params[1]:.4f}±{errs[1]:.4f}")
            print(f"    → b={params[2]:.4f}±{errs[2]:.4f}, c={params[3]:.4f}±{errs[3]:.4f}")
            print(f"    → 质量={collapse_quality:.1f}, 耗时={elapsed:.1f}s")
            
            if collapse_quality > best_quality:
                best_quality = collapse_quality
                best_result = result.copy()
                
        except Exception as e:
            print(f"    → 失败: {e}")
            continue
    
    if best_result:
        print(f"\n✅ FSE All-L 最佳结果:")
        print(f"   U_c = {best_result['params'][0]:.6f} ± {best_result['errors'][0]:.6f}")
        print(f"   ν^(-1) = {best_result['params'][1]:.6f} ± {best_result['errors'][1]:.6f}")
        print(f"   b = {best_result['params'][2]:.6f} ± {best_result['errors'][2]:.6f}")
        print(f"   c = {best_result['params'][3]:.6f} ± {best_result['errors'][3]:.6f}")
        print(f"   坍缩质量 = {best_result['quality']:.2f}")
        print(f"   来自策略: {best_result['description']}")
    
    return best_result, results, df_full

def create_individual_plots(nofse_result, fse_result, df_drop_l7, df_full):
    """生成用于插入报告的单独小图表"""
    
    print(f"\n🎨 生成报告用的单独图表...")
    
    # 图表保存目录
    plot_dir = os.path.dirname(__file__)
    
    # 配色
    colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71']
    markers = ['o', 's', '^', 'd']
    
    # 1. No-FSE Drop L=7 原始数据
    plt.figure(figsize=(10, 6))
    L_values = sorted(df_drop_l7["L"].unique())
    for i, L in enumerate(L_values):
        m = (df_drop_l7["L"]==L).to_numpy()
        U_vals = df_drop_l7["U"][m].to_numpy()
        Y_vals = df_drop_l7["Y"][m].to_numpy()
        sigma_vals = df_drop_l7["sigma"][m].to_numpy()
        order = np.argsort(U_vals)
        U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
        
        plt.errorbar(U_vals, Y_vals, yerr=sigma_vals, 
                    fmt=f"{markers[i]}-", color=colors[i], lw=2, ms=5, 
                    capsize=3, label=f"L={L}", alpha=0.8, elinewidth=1.5)
    
    plt.xlabel("U", fontsize=14, fontweight='bold')
    plt.ylabel("Y", fontsize=14, fontweight='bold')
    plt.title("No-FSE Drop L=7: Original Data", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "nofse_drop_l7_raw.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. No-FSE Drop L=7 坍缩结果
    plt.figure(figsize=(10, 6))
    for i, L in enumerate(L_values):
        m = (df_drop_l7["L"]==L).to_numpy()
        xs = nofse_result['x_collapsed'][m]
        ys = nofse_result['Y_collapsed'][m]
        ss = df_drop_l7["sigma"][m].to_numpy()
        order = np.argsort(xs)
        xs, ys, ss = xs[order], ys[order], ss[order]
        
        plt.plot(xs, ys, "-", color=colors[i], lw=2.5, alpha=0.9, label=f"L={L}")
        plt.errorbar(xs, ys, yerr=ss, fmt=markers[i], color=colors[i], 
                    ms=4, capsize=2, elinewidth=1.2, alpha=0.7)
    
    plt.xlabel("(U - Uc) × L^(1/ν)", fontsize=14, fontweight='bold')
    plt.ylabel("Y", fontsize=14, fontweight='bold')
    plt.title(f"No-FSE Drop L=7: Data Collapse\nν^(-1) = {nofse_result['params'][1]:.3f}, Quality = {nofse_result['quality']:.1f}", 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "nofse_drop_l7_collapse.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. FSE All-L 原始数据
    plt.figure(figsize=(10, 6))
    L_values_all = sorted(df_full["L"].unique())
    for i, L in enumerate(L_values_all):
        m = (df_full["L"]==L).to_numpy()
        U_vals = df_full["U"][m].to_numpy()
        Y_vals = df_full["Y"][m].to_numpy()
        sigma_vals = df_full["sigma"][m].to_numpy()
        order = np.argsort(U_vals)
        U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
        
        plt.errorbar(U_vals, Y_vals, yerr=sigma_vals, 
                    fmt=f"{markers[i]}-", color=colors[i], lw=2, ms=5, 
                    capsize=3, label=f"L={L}", alpha=0.8, elinewidth=1.5)
    
    plt.xlabel("U", fontsize=14, fontweight='bold')
    plt.ylabel("Y", fontsize=14, fontweight='bold')
    plt.title("FSE All-L: Original Data", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fse_all_l_raw.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. FSE All-L 坍缩结果
    plt.figure(figsize=(10, 6))
    for i, L in enumerate(L_values_all):
        m = (df_full["L"]==L).to_numpy()
        xs = fse_result['x_collapsed'][m]
        ys = fse_result['Y_collapsed'][m]
        
        # FSE误差传播
        Lvals = df_full["L"][m].to_numpy(float)
        b, c = fse_result['params'][2], fse_result['params'][3]
        Lr = float(np.exp(np.mean(np.log(df_full['L'].to_numpy(float)))))
        S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
        ss = (df_full["sigma"][m].to_numpy() / S)
        
        order = np.argsort(xs)
        xs, ys, ss = xs[order], ys[order], ss[order]
        
        plt.plot(xs, ys, "-", color=colors[i], lw=2.5, alpha=0.9, label=f"L={L}")
        plt.errorbar(xs, ys, yerr=ss, fmt=markers[i], color=colors[i], 
                    ms=4, capsize=2, elinewidth=1.2, alpha=0.7)
    
    plt.xlabel("(U - Uc) × L^(1/ν)", fontsize=14, fontweight='bold')
    plt.ylabel("Y / normalized", fontsize=14, fontweight='bold')
    plt.title(f"FSE All-L: Data Collapse\nν^(-1) = {fse_result['params'][1]:.3f}, Quality = {fse_result['quality']:.1f}", 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fse_all_l_collapse.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 参数对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # U_c对比
    methods = ['No-FSE\nDrop L=7', 'FSE\nAll-L']
    uc_values = [nofse_result['params'][0], fse_result['params'][0]]
    uc_errors = [nofse_result['errors'][0], fse_result['errors'][0]]
    
    bars1 = ax1.bar(methods, uc_values, yerr=uc_errors, capsize=5, 
                   color=['#3498DB', '#E74C3C'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel("U_c", fontsize=14, fontweight='bold')
    ax1.set_title("Critical Point Comparison", fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for bar, value, error in zip(bars1, uc_values, uc_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + error + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # ν^(-1)对比
    nu_inv_values = [nofse_result['params'][1], fse_result['params'][1]]
    nu_inv_errors = [nofse_result['errors'][1], fse_result['errors'][1]]
    
    bars2 = ax2.bar(methods, nu_inv_values, yerr=nu_inv_errors, capsize=5,
                   color=['#3498DB', '#E74C3C'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel("ν^(-1)", fontsize=14, fontweight='bold')
    ax2.set_title("Critical Exponent Comparison", fontsize=16, fontweight='bold')
    ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.8, label='ν^(-1) = 1')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for bar, value, error in zip(bars2, nu_inv_values, nu_inv_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "parameter_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 质量对比图
    plt.figure(figsize=(10, 6))
    qualities = [nofse_result['quality'], fse_result['quality']]
    colors_qual = ['#3498DB', '#E74C3C']
    
    bars = plt.bar(methods, qualities, color=colors_qual, alpha=0.7, edgecolor='black')
    plt.ylabel("Collapse Quality", fontsize=14, fontweight='bold')
    plt.title("Data Collapse Quality Comparison", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for bar, quality in zip(bars, qualities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{quality:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 添加质量评级线
    plt.axhline(100, color='green', linestyle='--', alpha=0.6, label='Excellent (>100)')
    plt.axhline(70, color='orange', linestyle='--', alpha=0.6, label='Good (>70)')
    plt.axhline(50, color='red', linestyle='--', alpha=0.6, label='Acceptable (>50)')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "quality_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 已生成报告用图表:")
    print("  - nofse_drop_l7_raw.png")
    print("  - nofse_drop_l7_collapse.png") 
    print("  - fse_all_l_raw.png")
    print("  - fse_all_l_collapse.png")
    print("  - parameter_comparison.png")
    print("  - quality_comparison.png")

def main():
    print("🎯 No-FSE Drop L=7 vs FSE All-L 专门对比分析")
    print("基于改进的初始值理解，重新评估两种方法")
    print("="*60)
    
    # 1. 分析No-FSE Drop L=7
    nofse_result, nofse_results, df_drop_l7 = analyze_no_fse_drop_l7()
    
    # 2. 分析FSE All-L
    fse_result, fse_results, df_full = analyze_fse_all_l()
    
    # 3. 生成对比图表
    if nofse_result and fse_result:
        create_individual_plots(nofse_result, fse_result, df_drop_l7, df_full)
        
        # 4. 打印对比总结
        print(f"\n" + "="*60)
        print(f"📊 最终对比总结")
        print(f"="*60)
        
        print(f"\n【No-FSE Drop L=7】:")
        print(f"  U_c = {nofse_result['params'][0]:.6f} ± {nofse_result['errors'][0]:.6f}")
        print(f"  ν^(-1) = {nofse_result['params'][1]:.6f} ± {nofse_result['errors'][1]:.6f}")
        print(f"  坍缩质量 = {nofse_result['quality']:.2f}")
        print(f"  数据点 = {len(df_drop_l7)}")
        
        print(f"\n【FSE All-L】:")
        print(f"  U_c = {fse_result['params'][0]:.6f} ± {fse_result['errors'][0]:.6f}")
        print(f"  ν^(-1) = {fse_result['params'][1]:.6f} ± {fse_result['errors'][1]:.6f}")
        print(f"  b = {fse_result['params'][2]:.6f} ± {fse_result['errors'][2]:.6f}")
        print(f"  c = {fse_result['params'][3]:.6f} ± {fse_result['errors'][3]:.6f}")
        print(f"  坍缩质量 = {fse_result['quality']:.2f}")
        print(f"  数据点 = {len(df_full)}")
        
        print(f"\n【关键差异】:")
        uc_diff = abs(nofse_result['params'][0] - fse_result['params'][0])
        nu_inv_diff = abs(nofse_result['params'][1] - fse_result['params'][1])
        quality_diff = fse_result['quality'] - nofse_result['quality']
        
        print(f"  ΔU_c = {uc_diff:.4f}")
        print(f"  Δν^(-1) = {nu_inv_diff:.4f}")
        print(f"  Δ质量 = {quality_diff:+.1f} (FSE相对No-FSE)")
        
        if quality_diff > 0:
            print(f"  💡 FSE方法坍缩质量更优")
        else:
            print(f"  💡 No-FSE方法坍缩质量更优")
            
        return nofse_result, fse_result
    else:
        print("❌ 分析失败，无法生成对比")
        return None, None

if __name__ == "__main__":
    main() 