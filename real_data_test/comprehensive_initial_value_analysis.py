import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# import seaborn as sns  # 不需要seaborn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

# 设置matplotlib支持中文[[memory:5669012]]
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def systematic_initial_value_analysis():
    """系统性分析不同初始值对ν^(-1)拟合结果的影响"""
    
    print("🔬 系统性初始值分析：全面探索ν^(-1)的可能取值")
    print("目标：测试大量初始值，分析结果分布，找到最可靠的解")
    print("="*70)
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    datasets = {
        'All L': df_full.copy(),
        'Drop L=7': df_full[df_full["L"] != 7].copy().reset_index(drop=True),
        'Drop L=7,9': df_full[~df_full["L"].isin([7, 9])].copy().reset_index(drop=True)
    }
    
    results = {}
    
    for dataset_name, df in datasets.items():
        print(f"\n=== 分析数据集: {dataset_name} ===")
        data = df[["L","U","Y"]].to_numpy(float)
        err = df["sigma"].to_numpy(float)
        
        print(f"数据: {len(df)}点, L={sorted(df['L'].unique())}")
        
        # 1. 系统性网格搜索初始值
        print("1. 网格搜索初始值空间...")
        
        # 创建初始值网格
        Uc_range = np.linspace(8.50, 8.80, 10)  # 10个Uc初始值
        a_range = np.linspace(0.7, 1.8, 15)     # 15个a初始值
        
        grid_results = []
        success_count = 0
        total_count = len(Uc_range) * len(a_range)
        
        print(f"   测试 {total_count} 个初始值组合...")
        
        for i, Uc0 in enumerate(Uc_range):
            for j, a0 in enumerate(a_range):
                try:
                    # 设置宽松的边界
                    bounds = ((8.0, 9.0), (0.5, 2.5))
                    
                    (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                                     n_knots=10, lam=1e-3, n_boot=3,
                                                     bounds=bounds)
                    
                    # 计算坍缩质量
                    x_collapsed, Y_collapsed = collapse_transform(data, params)
                    x_range = x_collapsed.max() - x_collapsed.min()
                    y_ranges = []
                    for L in sorted(df["L"].unique()):
                        m = (df["L"]==L).to_numpy()
                        y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                        y_ranges.append(y_range)
                    collapse_quality = x_range / np.mean(y_ranges)
                    
                    grid_results.append({
                        'Uc0': Uc0,
                        'a0': a0,
                        'Uc_final': params[0],
                        'a_final': params[1],
                        'Uc_err': errs[0],
                        'a_err': errs[1],
                        'quality': collapse_quality,
                        'converged': True
                    })
                    success_count += 1
                    
                except Exception as e:
                    grid_results.append({
                        'Uc0': Uc0,
                        'a0': a0,
                        'Uc_final': np.nan,
                        'a_final': np.nan,
                        'Uc_err': np.nan,
                        'a_err': np.nan,
                        'quality': np.nan,
                        'converged': False
                    })
        
        grid_df = pd.DataFrame(grid_results)
        converged_df = grid_df[grid_df['converged']].copy()
        
        print(f"   成功收敛: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if len(converged_df) > 0:
            print(f"   ν^(-1)范围: {converged_df['a_final'].min():.3f} ~ {converged_df['a_final'].max():.3f}")
            print(f"   U_c范围: {converged_df['Uc_final'].min():.3f} ~ {converged_df['Uc_final'].max():.3f}")
            print(f"   坍缩质量范围: {converged_df['quality'].min():.1f} ~ {converged_df['quality'].max():.1f}")
        
        # 2. 随机采样补充分析
        print("2. 随机采样补充分析...")
        
        np.random.seed(42)
        n_random = 200
        random_results = []
        
        for _ in range(n_random):
            # 随机初始值
            Uc0 = np.random.uniform(8.3, 8.9)
            a0 = np.random.uniform(0.5, 2.0)
            
            try:
                bounds = ((8.0, 9.0), (0.4, 3.0))
                (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                                 n_knots=10, lam=1e-3, n_boot=3,
                                                 bounds=bounds)
                
                # 计算坍缩质量
                x_collapsed, Y_collapsed = collapse_transform(data, params)
                x_range = x_collapsed.max() - x_collapsed.min()
                y_ranges = []
                for L in sorted(df["L"].unique()):
                    m = (df["L"]==L).to_numpy()
                    y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                    y_ranges.append(y_range)
                collapse_quality = x_range / np.mean(y_ranges)
                
                random_results.append({
                    'Uc0': Uc0,
                    'a0': a0,
                    'Uc_final': params[0],
                    'a_final': params[1],
                    'Uc_err': errs[0],
                    'a_err': errs[1],
                    'quality': collapse_quality,
                    'converged': True
                })
                
            except:
                continue
        
        random_df = pd.DataFrame(random_results)
        print(f"   随机采样成功: {len(random_df)}/{n_random} ({len(random_df)/n_random*100:.1f}%)")
        
        # 3. 合并所有结果进行分析
        all_results = pd.concat([converged_df, random_df], ignore_index=True)
        
        if len(all_results) > 0:
            print(f"\n📊 总体统计 (n={len(all_results)}):")
            print(f"   ν^(-1): 均值={all_results['a_final'].mean():.3f}, 标准差={all_results['a_final'].std():.3f}")
            print(f"   U_c: 均值={all_results['Uc_final'].mean():.3f}, 标准差={all_results['Uc_final'].std():.3f}")
            print(f"   坍缩质量: 均值={all_results['quality'].mean():.1f}, 标准差={all_results['quality'].std():.1f}")
            
            # 4. 质量筛选分析
            quality_thresholds = [50, 70, 90, 110]
            for threshold in quality_thresholds:
                high_quality = all_results[all_results['quality'] >= threshold]
                if len(high_quality) > 0:
                    print(f"\n   质量≥{threshold}的解 (n={len(high_quality)}):")
                    print(f"     ν^(-1): {high_quality['a_final'].mean():.3f}±{high_quality['a_final'].std():.3f}")
                    print(f"     U_c: {high_quality['Uc_final'].mean():.3f}±{high_quality['Uc_final'].std():.3f}")
            
            # 5. 找到最佳解
            best_idx = all_results['quality'].idxmax()
            best_result = all_results.loc[best_idx]
            
            print(f"\n🏆 最高质量解:")
            print(f"   U_c = {best_result['Uc_final']:.6f} ± {best_result['Uc_err']:.6f}")
            print(f"   ν^(-1) = {best_result['a_final']:.6f} ± {best_result['a_err']:.6f}")
            print(f"   坍缩质量 = {best_result['quality']:.2f}")
            print(f"   (起始值: Uc0={best_result['Uc0']:.3f}, a0={best_result['a0']:.3f})")
        
        results[dataset_name] = {
            'all_results': all_results,
            'best_result': best_result if len(all_results) > 0 else None,
            'grid_df': grid_df,
            'random_df': random_df
        }
    
    return results

def create_comprehensive_visualization(results):
    """创建全面的可视化图表"""
    
    print(f"\n🎨 生成综合可视化图表...")
    
    # 设置颜色方案
    colors = ['#2E86C1', '#E74C3C', '#F39C12']
    dataset_names = ['All L', 'Drop L=7', 'Drop L=7,9']
    
    # 创建大型图表
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(6, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 参数分布直方图
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
            
        all_results = results[dataset_name]['all_results']
        
        # ν^(-1)分布
        ax1 = fig.add_subplot(gs[0, i])
        ax1.hist(all_results['a_final'], bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        ax1.axvline(1.0, color='red', linestyle='--', alpha=0.8, label='ν^(-1)=1')
        ax1.set_xlabel('ν^(-1)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{dataset_name}\nν^(-1) Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # U_c分布
        ax2 = fig.add_subplot(gs[1, i])
        ax2.hist(all_results['Uc_final'], bins=30, alpha=0.7, color=colors[i], edgecolor='black')
        ax2.set_xlabel('U_c')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'U_c Distribution')
        ax2.grid(True, alpha=0.3)
    
    # 2. 参数相关性散点图
    ax3 = fig.add_subplot(gs[0, 3])
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
        all_results = results[dataset_name]['all_results']
        ax3.scatter(all_results['Uc_final'], all_results['a_final'], 
                   alpha=0.6, c=colors[i], label=dataset_name, s=20)
    ax3.set_xlabel('U_c')
    ax3.set_ylabel('ν^(-1)')
    ax3.set_title('Parameter Correlation')
    ax3.axhline(1.0, color='red', linestyle='--', alpha=0.8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. 质量分布
    ax4 = fig.add_subplot(gs[1, 3])
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
        all_results = results[dataset_name]['all_results']
        ax4.hist(all_results['quality'], bins=20, alpha=0.6, 
                color=colors[i], label=dataset_name, edgecolor='black')
    ax4.set_xlabel('Collapse Quality')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Quality Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. 初始值vs最终值的关系
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
            
        all_results = results[dataset_name]['all_results']
        
        # 初始a vs 最终a
        ax5 = fig.add_subplot(gs[2, i])
        scatter = ax5.scatter(all_results['a0'], all_results['a_final'], 
                            c=all_results['quality'], cmap='viridis', 
                            alpha=0.7, s=30)
        ax5.plot([0.5, 2.0], [0.5, 2.0], 'r--', alpha=0.5, label='y=x')
        ax5.set_xlabel('Initial a₀')
        ax5.set_ylabel('Final ν^(-1)')
        ax5.set_title(f'{dataset_name}\nInitial vs Final a')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Quality')
    
    # 5. 质量vs参数关系
    ax6 = fig.add_subplot(gs[2, 3])
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
        all_results = results[dataset_name]['all_results']
        ax6.scatter(all_results['a_final'], all_results['quality'], 
                   alpha=0.6, c=colors[i], label=dataset_name, s=20)
    ax6.set_xlabel('ν^(-1)')
    ax6.set_ylabel('Collapse Quality')
    ax6.set_title('Quality vs ν^(-1)')
    ax6.axvline(1.0, color='red', linestyle='--', alpha=0.8)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 6. 最佳解的数据坍缩可视化
    for i, dataset_name in enumerate(dataset_names):
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
            
        # 准备数据
        if dataset_name == 'All L':
            df_plot = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
        elif dataset_name == 'Drop L=7':
            df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
            df_plot = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
        else:  # Drop L=7,9
            df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
            df_plot = df_full[~df_full["L"].isin([7, 9])].copy().reset_index(drop=True)
            
        data = df_plot[["L","U","Y"]].to_numpy(float)
        best_result = results[dataset_name]['best_result']
        best_params = [best_result['Uc_final'], best_result['a_final']]
        
        # 原始数据
        ax7 = fig.add_subplot(gs[3, i])
        for L in sorted(df_plot["L"].unique()):
            m = (df_plot["L"]==L).to_numpy()
            U_vals = df_plot["U"][m].to_numpy()
            Y_vals = df_plot["Y"][m].to_numpy()
            sigma_vals = df_plot["sigma"][m].to_numpy()
            order = np.argsort(U_vals)
            U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
            ax7.errorbar(U_vals, Y_vals, yerr=sigma_vals, fmt="o-", lw=1.2, ms=3, 
                        capsize=2, label=f"L={L}", alpha=0.8)
        ax7.set_xlabel("U")
        ax7.set_ylabel("Y")
        ax7.set_title(f"{dataset_name}\nRaw Data")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 坍缩后数据
        ax8 = fig.add_subplot(gs[4, i])
        x_collapsed, Y_collapsed = collapse_transform(data, best_params)
        for L in sorted(df_plot["L"].unique()):
            m = (df_plot["L"]==L).to_numpy()
            xs = x_collapsed[m]
            ys = Y_collapsed[m]
            ss = df_plot["sigma"][m].to_numpy()
            order = np.argsort(xs)
            xs, ys, ss = xs[order], ys[order], ss[order]
            line, = ax8.plot(xs, ys, "-", lw=1.5, alpha=0.8, label=f"L={L}")
            ax8.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, 
                        elinewidth=1, color=line.get_color(), alpha=0.6)
        ax8.set_xlabel("(U - Uc) × L^(1/ν)")
        ax8.set_ylabel("Y")
        ax8.set_title(f"Best Collapse\nν^(-1)={best_result['a_final']:.3f}, Q={best_result['quality']:.1f}")
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 7. 总结统计表格
    ax9 = fig.add_subplot(gs[5, :])
    ax9.axis('off')
    
    # 创建总结表格
    summary_data = []
    for dataset_name in dataset_names:
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
            
        all_results = results[dataset_name]['all_results']
        best_result = results[dataset_name]['best_result']
        
        # 高质量解的统计
        high_quality = all_results[all_results['quality'] >= 80]
        
        if len(high_quality) > 0:
            summary_data.append([
                dataset_name,
                f"{len(all_results)}",
                f"{all_results['a_final'].mean():.3f}±{all_results['a_final'].std():.3f}",
                f"{high_quality['a_final'].mean():.3f}±{high_quality['a_final'].std():.3f}" if len(high_quality) > 0 else "N/A",
                f"{best_result['a_final']:.3f}±{best_result['a_err']:.3f}",
                f"{best_result['quality']:.1f}",
                "✓" if best_result['a_final'] > 1.0 else "✗"
            ])
    
    if summary_data:
        table = ax9.table(cellText=summary_data,
                         colLabels=['Dataset', 'N_total', 'Mean ν^(-1)', 'High-Q ν^(-1)', 'Best ν^(-1)', 'Best Quality', 'ν^(-1)>1'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.12, 0.08, 0.15, 0.15, 0.15, 0.12, 0.08])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(summary_data) + 1):
            for j in range(7):
                if i == 0:  # 表头
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j % 2 == 0:
                        table[(i, j)].set_facecolor('#F5F5F5')
    
    ax9.set_title('Summary Statistics of Initial Value Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Initial Value Analysis for ν^(-1) Parameter\nSystematic Exploration of Parameter Space', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), "comprehensive_initial_value_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 综合分析图表已保存: comprehensive_initial_value_analysis.png")
    return output_path

def generate_final_recommendations(results):
    """生成最终推荐方案"""
    
    print(f"\n" + "="*70)
    print(f"📋 最终推荐方案基于大规模初始值分析")
    print(f"="*70)
    
    print(f"\n🔍 分析总结:")
    
    for dataset_name in ['All L', 'Drop L=7', 'Drop L=7,9']:
        if dataset_name not in results or results[dataset_name]['best_result'] is None:
            continue
            
        all_results = results[dataset_name]['all_results']
        best_result = results[dataset_name]['best_result']
        
        print(f"\n--- {dataset_name} ---")
        print(f"总测试次数: {len(all_results)}")
        
        # 分质量层次分析
        quality_levels = [
            (120, "卓越"),
            (100, "优秀"), 
            (80, "良好"),
            (60, "中等"),
            (0, "全部")
        ]
        
        for min_quality, level_name in quality_levels:
            subset = all_results[all_results['quality'] >= min_quality]
            if len(subset) > 0:
                mean_a = subset['a_final'].mean()
                std_a = subset['a_final'].std()
                mean_uc = subset['Uc_final'].mean()
                std_uc = subset['Uc_final'].std()
                
                print(f"  {level_name}质量解(Q≥{min_quality}, n={len(subset)}):")
                print(f"    ν^(-1) = {mean_a:.3f} ± {std_a:.3f}")
                print(f"    U_c = {mean_uc:.3f} ± {std_uc:.3f}")
                
                if min_quality >= 80:  # 高质量解的额外分析
                    above_one = len(subset[subset['a_final'] > 1.0])
                    print(f"    ν^(-1)>1的比例: {above_one}/{len(subset)} ({above_one/len(subset)*100:.1f}%)")
                break
        
        print(f"  最佳解:")
        print(f"    U_c = {best_result['Uc_final']:.6f} ± {best_result['Uc_err']:.6f}")
        print(f"    ν^(-1) = {best_result['a_final']:.6f} ± {best_result['a_err']:.6f}")
        print(f"    坍缩质量 = {best_result['quality']:.2f}")
    
    # 交叉验证ChatGPT结果
    print(f"\n🤖 与ChatGPT结果的对比:")
    chatgpt_uc = 8.670
    chatgpt_a = 1.056
    
    if 'Drop L=7' in results and results['Drop L=7']['best_result'] is not None:
        our_best = results['Drop L=7']['best_result']
        print(f"ChatGPT (Drop L=7): U_c={chatgpt_uc:.3f}, ν^(-1)={chatgpt_a:.3f}")
        print(f"我们最佳 (Drop L=7): U_c={our_best['Uc_final']:.3f}, ν^(-1)={our_best['a_final']:.3f}")
        print(f"差异: ΔU_c={abs(our_best['Uc_final']-chatgpt_uc):.4f}, Δν^(-1)={abs(our_best['a_final']-chatgpt_a):.4f}")
        
        # 检查ChatGPT的结果在我们的分布中的位置
        drop_l7_results = results['Drop L=7']['all_results']
        a_percentile = (drop_l7_results['a_final'] <= chatgpt_a).mean() * 100
        uc_percentile = (drop_l7_results['Uc_final'] <= chatgpt_uc).mean() * 100
        
        print(f"ChatGPT结果在我们分布中的分位数:")
        print(f"  ν^(-1)={chatgpt_a:.3f} 位于第{a_percentile:.1f}百分位")
        print(f"  U_c={chatgpt_uc:.3f} 位于第{uc_percentile:.1f}百分位")
    
    print(f"\n🎯 最终推荐:")
    
    # 找到所有数据集中质量最高的解
    all_best_results = []
    for dataset_name in results:
        if results[dataset_name]['best_result'] is not None:
            best = results[dataset_name]['best_result'].copy()
            best['dataset'] = dataset_name
            all_best_results.append(best)
    
    if all_best_results:
        overall_best = max(all_best_results, key=lambda x: x['quality'])
        
        print(f"1. 最高质量解 ({overall_best['dataset']}):")
        print(f"   U_c = {overall_best['Uc_final']:.6f} ± {overall_best['Uc_err']:.6f}")
        print(f"   ν^(-1) = {overall_best['a_final']:.6f} ± {overall_best['a_err']:.6f}")
        print(f"   坍缩质量 = {overall_best['quality']:.2f}")
        
        print(f"\n2. 建议的保守估计 (基于高质量解的均值):")
        high_quality_all = pd.concat([
            results[name]['all_results'][results[name]['all_results']['quality'] >= 80]
            for name in results if results[name]['best_result'] is not None
        ])
        
        if len(high_quality_all) > 0:
            conservative_uc = high_quality_all['Uc_final'].mean()
            conservative_a = high_quality_all['a_final'].mean()
            conservative_uc_std = high_quality_all['Uc_final'].std()
            conservative_a_std = high_quality_all['a_final'].std()
            
            print(f"   U_c = {conservative_uc:.6f} ± {conservative_uc_std:.6f}")
            print(f"   ν^(-1) = {conservative_a:.6f} ± {conservative_a_std:.6f}")
            print(f"   (基于{len(high_quality_all)}个高质量解)")
        
        print(f"\n3. 可靠性评估:")
        print(f"   - 参数稳定性: {'好' if overall_best['a_err'] < 0.05 else '中等'}")
        print(f"   - 物理合理性: {'是' if overall_best['a_final'] > 1.0 else '否'} (ν^(-1) > 1)")
        print(f"   - 坍缩质量: {'卓越' if overall_best['quality'] > 120 else '优秀' if overall_best['quality'] > 100 else '良好'}")

def main():
    print("🔬 全面初始值分析：探索ν^(-1)参数的真实取值")
    print("策略：大规模测试不同初始值，分析结果分布，提供可视化验证")
    print("="*70)
    
    # 1. 系统性初始值分析
    results = systematic_initial_value_analysis()
    
    # 2. 创建综合可视化
    create_comprehensive_visualization(results)
    
    # 3. 生成最终推荐
    generate_final_recommendations(results)
    
    print(f"\n📊 分析完成!")
    print(f"✅ 已测试数百个不同初始值组合")
    print(f"✅ 生成了详细的可视化图表供您验证坍缩质量")
    print(f"✅ 提供了基于统计分析的可靠推荐方案")
    print(f"✅ 图表文件: comprehensive_initial_value_analysis.png")

if __name__ == "__main__":
    main() 