import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

def debug_no_fse_analysis():
    """调试No-FSE分析，找出问题所在"""
    
    print("=== 调试No-FSE分析 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    print(f"数据加载完成: {len(df_full)}个数据点")
    print(f"L值: {sorted(df_full['L'].unique())}")
    print(f"U范围: {df_full['U'].min():.3f} 到 {df_full['U'].max():.3f}")
    print(f"Y范围: {df_full['Y'].min():.3f} 到 {df_full['Y'].max():.3f}")
    print(f"sigma范围: {df_full['sigma'].min():.6f} 到 {df_full['sigma'].max():.6f}")
    
    # 准备数据
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print(f"\n数据准备完成:")
    print(f"  data.shape: {data.shape}")
    print(f"  err.shape: {err.shape}")
    
    # 1. 使用最简单的No-FSE拟合
    print(f"\n=== 1. 简单No-FSE拟合 (All L) ===")
    
    # 尝试不同的起始参数
    starting_points = [
        (8.40, 1.4, "ChatGPT类似"),
        (8.60, 1.0, "保守估计"),
        (8.70, 1.0, "中等估计"), 
        (8.80, 0.9, "激进估计"),
        (8.50, 1.2, "宽范围1"),
        (8.75, 0.8, "宽范围2")
    ]
    
    results = []
    
    for Uc0, a0, desc in starting_points:
        print(f"\n  测试起始点: {desc} (Uc0={Uc0:.2f}, a0={a0:.1f})")
        
        try:
            # 设置宽松的边界
            bounds = ((8.0, 9.0), (0.5, 2.0))
            
            # 执行拟合
            (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                             n_knots=10, lam=1e-3, n_boot=5,
                                             bounds=bounds)
            
            print(f"    拟合成功:")
            print(f"      U_c = {params[0]:.6f} ± {errs[0]:.6f}")
            print(f"      ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data, params)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_full["L"].unique()):
                m = (df_full["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            print(f"      坍缩质量 = {collapse_quality:.2f}")
            
            results.append({
                'desc': desc,
                'start': (Uc0, a0),
                'params': params,
                'errors': errs,
                'quality': collapse_quality,
                'x_collapsed': x_collapsed,
                'Y_collapsed': Y_collapsed
            })
            
        except Exception as e:
            print(f"    拟合失败: {e}")
            continue
    
    # 找到最佳结果
    if results:
        best_result = max(results, key=lambda x: x['quality'])
        print(f"\n  ✅ 最佳结果:")
        print(f"     起始点: {best_result['desc']}")
        print(f"     U_c = {best_result['params'][0]:.6f} ± {best_result['errors'][0]:.6f}")
        print(f"     ν^(-1) = {best_result['params'][1]:.6f} ± {best_result['errors'][1]:.6f}")
        print(f"     坍缩质量 = {best_result['quality']:.2f}")
    
    # 2. 特别测试ChatGPT的参数范围
    print(f"\n=== 2. 特别测试ChatGPT参数范围 ===")
    
    # ChatGPT报告: U_c=7517(38), ν^(-1)=1.0763(54)
    # 这个U_c=7517看起来有问题，可能是8.7517？
    chatgpt_tests = [
        (8.7517, 1.0763, "ChatGPT直接翻译"),
        (7.517, 1.0763, "ChatGPT字面值(不太可能)"),  
        (8.517, 1.0763, "ChatGPT可能遗漏8"),
        (8.7500, 1.0700, "ChatGPT近似值")
    ]
    
    for Uc0, a0, desc in chatgpt_tests:
        print(f"\n  测试: {desc} (Uc0={Uc0:.4f}, a0={a0:.4f})")
        
        try:
            # 根据参数设置合理边界
            if Uc0 < 8.0:
                bounds = ((7.0, 8.5), (0.8, 1.5))
            else:
                bounds = ((8.0, 9.0), (0.8, 1.5))
            
            (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                             n_knots=10, lam=1e-3, n_boot=5,
                                             bounds=bounds)
            
            print(f"    成功:")
            print(f"      U_c = {params[0]:.6f} ± {errs[0]:.6f}")
            print(f"      ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data, params)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_full["L"].unique()):
                m = (df_full["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            print(f"      坍缩质量 = {collapse_quality:.2f}")
            
        except Exception as e:
            print(f"    失败: {e}")
    
    # 3. 检查CORRECTED报告使用的参数
    print(f"\n=== 3. 检查CORRECTED报告的问题 ===")
    
    # CORRECTED报告声称: U_c=8.6625, ν^(-1)=1.0250
    print(f"  测试CORRECTED报告的参数: U_c=8.6625, ν^(-1)=1.0250")
    
    try:
        # 看看是否能重现这个结果
        (params, errs) = fit_data_collapse(data, err, 8.6625, 1.0250, 
                                         n_knots=10, lam=1e-3, n_boot=5,
                                         bounds=((8.60, 8.70), (1.02, 1.03)))  # 非常窄的边界
        
        print(f"    用窄边界重现:")
        print(f"      U_c = {params[0]:.6f} ± {errs[0]:.6f}")
        print(f"      ν^(-1) = {params[1]:.6f} ± {errs[1]:.6f}")
        
        # 这可能解释了为什么CORRECTED报告的所有结果都是1.0250
        # 可能用了过窄的边界导致参数被困在边界上
        
    except Exception as e:
        print(f"    重现失败: {e}")
    
    # 4. 生成最佳结果的图表
    if results and best_result:
        print(f"\n=== 4. 生成最佳结果图表 ===")
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 原始数据
        plt.subplot(2, 2, 1)
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            U_vals = df_full["U"][m].to_numpy()
            Y_vals = df_full["Y"][m].to_numpy()
            sigma_vals = df_full["sigma"][m].to_numpy()
            order = np.argsort(U_vals)
            U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
            plt.errorbar(U_vals, Y_vals, yerr=sigma_vals, fmt="o-", lw=1.2, ms=3, 
                       capsize=2, label=f"L={L}", elinewidth=1)
        plt.xlabel("U"); plt.ylabel("Y")
        plt.title("Raw Data - All L")
        plt.legend(); plt.grid(True, alpha=0.3)
        
        # 子图2: 最佳坍缩结果
        plt.subplot(2, 2, 2)
        x_collapsed = best_result['x_collapsed']
        Y_collapsed = best_result['Y_collapsed']
        for L in sorted(df_full["L"].unique()):
            m = (df_full["L"]==L).to_numpy()
            xs = x_collapsed[m]; ys = Y_collapsed[m]; ss = df_full["sigma"][m].to_numpy()
            order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
            line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, 
                       elinewidth=1, color=line.get_color())
        plt.xlabel("(U - Uc) * L^(1/ν)"); plt.ylabel("Y")
        plt.title(f"Best No-FSE Collapse\nUc={best_result['params'][0]:.4f}, ν^(-1)={best_result['params'][1]:.3f}")
        plt.legend(); plt.grid(True, alpha=0.3)
        
        # 子图3: 所有结果的参数对比
        plt.subplot(2, 2, 3)
        Uc_vals = [r['params'][0] for r in results]
        a_vals = [r['params'][1] for r in results]
        qualities = [r['quality'] for r in results]
        colors = plt.cm.viridis([q/max(qualities) for q in qualities])
        
        scatter = plt.scatter(Uc_vals, a_vals, c=qualities, s=100, cmap='viridis')
        plt.colorbar(scatter, label='Collapse Quality')
        plt.xlabel("U_c"); plt.ylabel("ν^(-1)")
        plt.title("Parameter Space Exploration")
        plt.grid(True, alpha=0.3)
        
        # 子图4: 质量对比
        plt.subplot(2, 2, 4)
        descriptions = [r['desc'] for r in results]
        plt.bar(range(len(qualities)), qualities, color=[plt.cm.viridis(q/max(qualities)) for q in qualities])
        plt.xlabel("Starting Point"); plt.ylabel("Collapse Quality")
        plt.title("Quality Comparison")
        plt.xticks(range(len(descriptions)), descriptions, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "debug_no_fse_analysis.png"), dpi=180)
        plt.close()
        print(f"     图表保存为: debug_no_fse_analysis.png")
    
    return results

def main():
    """主函数"""
    print("🔍 调试No-FSE分析，找出问题根源")
    print("="*60)
    
    results = debug_no_fse_analysis()
    
    # 总结
    print(f"\n📋 总结:")
    print(f"✅ 完成了详细的No-FSE调试分析")
    print(f"✅ 测试了多个起始点和参数范围")  
    print(f"✅ 特别检查了ChatGPT的结果")
    print(f"✅ 分析了CORRECTED报告的问题可能来源")
    print(f"✅ 生成了调试图表")
    
    if results:
        best = max(results, key=lambda x: x['quality'])
        print(f"\n🎯 最佳No-FSE结果:")
        print(f"   U_c = {best['params'][0]:.6f} ± {best['errors'][0]:.6f}")
        print(f"   ν^(-1) = {best['params'][1]:.6f} ± {best['errors'][1]:.6f}")
        print(f"   坍缩质量 = {best['quality']:.2f}")
        print(f"   起始点: {best['desc']}")

if __name__ == "__main__":
    main() 