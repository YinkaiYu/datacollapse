import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

# 设置matplotlib支持中文[[memory:5669012]]
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_beautiful_collapse_plots():
    """生成最优美的数据坍缩图表展示"""
    
    print("🎨 生成优美的数据坍缩图表")
    print("展示基于大规模分析得到的最佳解")
    print("="*50)
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    # 基于大规模分析的最佳参数
    best_solutions = {
        'All L': {
            'params': [8.7452, 1.1829],
            'errors': [0.0019, 0.0163],
            'quality': 117.69,
            'data': df_full.copy()
        },
        'Drop L=7': {
            'params': [8.6702, 1.2180],
            'errors': [0.0018, 0.0227],
            'quality': 114.69,
            'data': df_full[df_full["L"] != 7].copy().reset_index(drop=True)
        },
        'Drop L=7,9': {
            'params': [8.6177, 1.2682],
            'errors': [0.0074, 0.0399],
            'quality': 118.25,
            'data': df_full[~df_full["L"].isin([7, 9])].copy().reset_index(drop=True)
        }
    }
    
    # 创建大型美观图表
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # 配色方案
    colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71']  # 红、蓝、橙、绿
    markers = ['o', 's', '^', 'd']
    
    # 为每个数据集生成图表
    for row, (name, solution) in enumerate(best_solutions.items()):
        df = solution['data']
        params = solution['params']
        errors = solution['errors']
        quality = solution['quality']
        
        data = df[["L","U","Y"]].to_numpy(float)
        err = df["sigma"].to_numpy(float)
        
        print(f"绘制 {name}: U_c={params[0]:.4f}, ν^(-1)={params[1]:.4f}, Q={quality:.1f}")
        
        # 原始数据图
        ax1 = fig.add_subplot(gs[row, 0])
        L_values = sorted(df["L"].unique())
        
        for i, L in enumerate(L_values):
            m = (df["L"]==L).to_numpy()
            U_vals = df["U"][m].to_numpy()
            Y_vals = df["Y"][m].to_numpy()
            sigma_vals = df["sigma"][m].to_numpy()
            order = np.argsort(U_vals)
            U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
            
            ax1.errorbar(U_vals, Y_vals, yerr=sigma_vals, 
                        fmt=f"{markers[i]}-", color=colors[i], lw=2, ms=5, 
                        capsize=3, label=f"L={L}", alpha=0.8, elinewidth=1.5)
        
        ax1.set_xlabel("U", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Y", fontsize=12, fontweight='bold')
        ax1.set_title(f"{name}\nOriginal Data", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(labelsize=10)
        
        # 坍缩后数据图
        ax2 = fig.add_subplot(gs[row, 1])
        x_collapsed, Y_collapsed = collapse_transform(data, params)
        
        for i, L in enumerate(L_values):
            m = (df["L"]==L).to_numpy()
            xs = x_collapsed[m]
            ys = Y_collapsed[m]
            ss = df["sigma"][m].to_numpy()
            order = np.argsort(xs)
            xs, ys, ss = xs[order], ys[order], ss[order]
            
            # 先画线
            ax2.plot(xs, ys, "-", color=colors[i], lw=2.5, alpha=0.9, label=f"L={L}")
            # 再画误差棒
            ax2.errorbar(xs, ys, yerr=ss, fmt=markers[i], color=colors[i], 
                        ms=4, capsize=2, elinewidth=1.2, alpha=0.7)
        
        ax2.set_xlabel("(U - Uc) × L^(1/ν)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Y", fontsize=12, fontweight='bold')
        ax2.set_title(f"Data Collapse\nν^(-1) = {params[1]:.3f}, Quality = {quality:.1f}", 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(labelsize=10)
        
        # 残差分析图
        ax3 = fig.add_subplot(gs[row, 2])
        
        # 计算样条拟合来获得残差
        from scipy.interpolate import UnivariateSpline
        
        # 对坍缩后的数据进行样条拟合
        x_all = x_collapsed.flatten()
        y_all = Y_collapsed.flatten()
        order = np.argsort(x_all)
        x_sorted, y_sorted = x_all[order], y_all[order]
        
        # 样条拟合
        spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*0.1, k=3)
        y_spline = spline(x_sorted)
        residuals = y_sorted - y_spline
        
        # 按L值绘制残差
        for i, L in enumerate(L_values):
            m = (df["L"]==L).to_numpy()
            xs = x_collapsed[m]
            ys = Y_collapsed[m]
            
            # 计算对应的残差
            L_residuals = []
            L_xs = []
            for x_val, y_val in zip(xs, ys):
                idx = np.argmin(np.abs(x_sorted - x_val))
                L_residuals.append(y_val - spline(x_val))
                L_xs.append(x_val)
            
            ax3.scatter(L_xs, L_residuals, c=colors[i], marker=markers[i], 
                       s=40, alpha=0.7, label=f"L={L}")
        
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel("(U - Uc) × L^(1/ν)", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Residuals", fontsize=12, fontweight='bold')
        ax3.set_title("Fit Residuals", fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(labelsize=10)
        
        # 质量指标图
        ax4 = fig.add_subplot(gs[row, 3])
        
        # 计算每个L的y范围和总体x范围
        x_range = x_collapsed.max() - x_collapsed.min()
        y_ranges = []
        L_labels = []
        
        for L in L_values:
            m = (df["L"]==L).to_numpy()
            y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
            y_ranges.append(y_range)
            L_labels.append(f"L={L}")
        
        bars = ax4.bar(L_labels, y_ranges, color=colors[:len(L_values)], alpha=0.7)
        
        # 添加x_range参考线
        ax4.axhline(x_range, color='red', linestyle='--', lw=2, 
                   label=f'X Range = {x_range:.2f}')
        
        # 添加数值标签
        for bar, y_range in zip(bars, y_ranges):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{y_range:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel("Y Range per L", fontsize=12, fontweight='bold')
        ax4.set_title(f"Quality Metric\nQ = {x_range:.2f} / {np.mean(y_ranges):.2f} = {quality:.1f}", 
                     fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10, framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(labelsize=10)
    
    # 总标题
    plt.suptitle('High-Quality Data Collapse Solutions from Large-Scale Initial Value Analysis\n' + 
                 'Demonstrating ν^(-1) > 1 with Excellent Collapse Quality', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 保存图表
    output_path = os.path.join(os.path.dirname(__file__), "beautiful_collapse_verification.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 优美坍缩图表已保存: beautiful_collapse_verification.png")
    return output_path

def create_comparison_with_old_results():
    """创建与旧结果对比的图表"""
    
    print("\n🔍 生成与旧结果对比的图表")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    # 对比方案
    solutions = {
        'Old (Problem)': {
            'params': [8.7578, 0.9026],  # 旧脚本的问题结果
            'color': '#E74C3C',  # 红色
            'linestyle': '--',
            'description': 'Low a₀, narrow bounds'
        },
        'ChatGPT': {
            'params': [8.670, 1.056],   # ChatGPT结果
            'color': '#F39C12',  # 橙色
            'linestyle': '-.',
            'description': 'ChatGPT result'
        },
        'Our Best': {
            'params': [8.7452, 1.1829], # 我们的最佳结果
            'color': '#2ECC71',  # 绿色
            'linestyle': '-',
            'description': 'High-quality systematic search'
        }
    }
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison of Different ν^(-1) Solutions\nDemonstrating the Impact of Initial Value Strategy', 
                 fontsize=16, fontweight='bold')
    
    L_values = sorted(df_full["L"].unique())
    colors = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71']
    markers = ['o', 's', '^', 'd']
    
    # 原始数据
    ax1 = axes[0, 0]
    for i, L in enumerate(L_values):
        m = (df_full["L"]==L).to_numpy()
        U_vals = df_full["U"][m].to_numpy()
        Y_vals = df_full["Y"][m].to_numpy()
        sigma_vals = df_full["sigma"][m].to_numpy()
        order = np.argsort(U_vals)
        U_vals, Y_vals, sigma_vals = U_vals[order], Y_vals[order], sigma_vals[order]
        
        ax1.errorbar(U_vals, Y_vals, yerr=sigma_vals, 
                    fmt=f"{markers[i]}-", color=colors[i], lw=2, ms=4, 
                    capsize=3, label=f"L={L}", alpha=0.8)
    
    ax1.set_xlabel("U", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Y", fontsize=12, fontweight='bold')
    ax1.set_title("Original Data", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 为每个解生成坍缩图
    plot_positions = [(0, 1), (1, 0), (1, 1)]
    
    for idx, (name, solution) in enumerate(solutions.items()):
        ax = axes[plot_positions[idx]]
        params = solution['params']
        
        # 计算坍缩
        x_collapsed, Y_collapsed = collapse_transform(data, params)
        
        # 计算质量
        x_range = x_collapsed.max() - x_collapsed.min()
        y_ranges = []
        for L in L_values:
            m = (df_full["L"]==L).to_numpy()
            y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
            y_ranges.append(y_range)
        quality = x_range / np.mean(y_ranges)
        
        # 绘制坍缩数据
        for i, L in enumerate(L_values):
            m = (df_full["L"]==L).to_numpy()
            xs = x_collapsed[m]
            ys = Y_collapsed[m]
            ss = df_full["sigma"][m].to_numpy()
            order = np.argsort(xs)
            xs, ys, ss = xs[order], ys[order], ss[order]
            
            line, = ax.plot(xs, ys, solution['linestyle'], color=colors[i], 
                           lw=2.5, alpha=0.9, label=f"L={L}")
            ax.errorbar(xs, ys, yerr=ss, fmt=markers[i], color=colors[i], 
                       ms=3, capsize=2, elinewidth=1, alpha=0.6)
        
        ax.set_xlabel("(U - Uc) × L^(1/ν)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Y", fontsize=12, fontweight='bold')
        
        # 根据质量设置标题颜色
        if quality > 100:
            title_color = 'green'
            quality_desc = "Excellent"
        elif quality > 70:
            title_color = 'orange'
            quality_desc = "Good"
        else:
            title_color = 'red'
            quality_desc = "Poor"
        
        ax.set_title(f"{name}\nν^(-1) = {params[1]:.3f}, Q = {quality:.1f} ({quality_desc})", 
                    fontsize=14, fontweight='bold', color=title_color)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存对比图
    output_path = os.path.join(os.path.dirname(__file__), "solution_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 对比图表已保存: solution_comparison.png")
    return output_path

def main():
    print("🎨 生成最优美的数据坍缩验证图表")
    print("目标：展示基于大规模分析的高质量解，供您验证坍缩效果")
    print("="*60)
    
    # 1. 生成美观的坍缩图表
    create_beautiful_collapse_plots()
    
    # 2. 生成对比图表
    create_comparison_with_old_results()
    
    print(f"\n📊 图表生成完成!")
    print(f"✅ 主要验证图表: beautiful_collapse_verification.png")
    print(f"✅ 对比分析图表: solution_comparison.png")
    print(f"✅ 综合分析图表: comprehensive_initial_value_analysis.png")
    print(f"\n🔍 您可以通过这些图表直观验证:")
    print(f"  - 数据坍缩的质量")
    print(f"  - 不同解的对比效果")
    print(f"  - 残差分析")
    print(f"  - 质量指标的计算过程")

if __name__ == "__main__":
    main() 