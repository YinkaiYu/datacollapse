import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

def verify_chatgpt_drop_l7():
    """专门验证ChatGPT的Drop L=7结果"""
    
    print("=== 验证ChatGPT的Drop L=7结果 ===")
    print("ChatGPT报告: U_c=8.670(1), ν^(-1)=1.056(2), 来源于No-FSE Drop L=7")
    print("")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    # 准备Drop L=7数据集
    df_drop_l7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data_drop_l7 = df_drop_l7[["L","U","Y"]].to_numpy(float)
    err_drop_l7 = df_drop_l7["sigma"].to_numpy(float)
    
    print(f"Drop L=7数据集:")
    print(f"  数据点数: {len(df_drop_l7)}")
    print(f"  L值: {sorted(df_drop_l7['L'].unique())}")
    print(f"  U范围: {df_drop_l7['U'].min():.3f} - {df_drop_l7['U'].max():.3f}")
    print("")
    
    # 1. 用ChatGPT的值作为起始点
    print("1. 用ChatGPT的参数作为起始点:")
    chatgpt_uc = 8.670
    chatgpt_a = 1.056
    
    try:
        bounds = ((8.0, 9.0), (0.5, 2.0))
        (params1, errs1) = fit_data_collapse(data_drop_l7, err_drop_l7, 
                                           chatgpt_uc, chatgpt_a, 
                                           n_knots=10, lam=1e-3, n_boot=5,
                                           bounds=bounds)
        
        # 计算坍缩质量
        x_collapsed, Y_collapsed = collapse_transform(data_drop_l7, params1)
        x_range = x_collapsed.max() - x_collapsed.min()
        y_ranges = []
        for L in sorted(df_drop_l7["L"].unique()):
            m = (df_drop_l7["L"]==L).to_numpy()
            y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
            y_ranges.append(y_range)
        collapse_quality1 = x_range / np.mean(y_ranges)
        
        print(f"  起始点: U_c={chatgpt_uc:.3f}, ν^(-1)={chatgpt_a:.3f}")
        print(f"  结果: U_c={params1[0]:.6f}±{errs1[0]:.6f}, ν^(-1)={params1[1]:.6f}±{errs1[1]:.6f}")
        print(f"  坍缩质量: {collapse_quality1:.2f}")
        print(f"  与ChatGPT差异: ΔU_c={abs(params1[0]-chatgpt_uc):.4f}, Δν^(-1)={abs(params1[1]-chatgpt_a):.4f}")
        
        # 检查是否接近ChatGPT的结果
        if abs(params1[0] - chatgpt_uc) < 0.01 and abs(params1[1] - chatgpt_a) < 0.1:
            print(f"  ✅ 非常接近ChatGPT的结果！")
        else:
            print(f"  ❌ 与ChatGPT结果有明显差异")
        
    except Exception as e:
        print(f"  失败: {e}")
        params1, errs1, collapse_quality1 = None, None, 0
    
    print("")
    
    # 2. 系统性测试不同起始点
    print("2. 系统性测试不同起始点:")
    
    starting_points = [
        (8.60, 1.0, "标准1"),
        (8.65, 1.0, "标准2"),
        (8.67, 1.05, "接近ChatGPT"),
        (8.670, 1.056, "ChatGPT精确"),
        (8.68, 1.1, "稍高"),
        (8.70, 0.9, "低ν"),
        (8.75, 0.8, "很低ν"),
    ]
    
    results = []
    
    for Uc0, a0, desc in starting_points:
        print(f"  测试 {desc}: Uc0={Uc0:.3f}, a0={a0:.3f}")
        
        try:
            bounds = ((8.0, 9.0), (0.5, 2.0))
            (params, errs) = fit_data_collapse(data_drop_l7, err_drop_l7, 
                                             Uc0, a0, 
                                             n_knots=10, lam=1e-3, n_boot=3,
                                             bounds=bounds)
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data_drop_l7, params)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_drop_l7["L"].unique()):
                m = (df_drop_l7["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            print(f"    结果: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
            print(f"    坍缩质量: {collapse_quality:.2f}")
            
            results.append({
                'desc': desc,
                'start': (Uc0, a0),
                'params': params,
                'errors': errs,
                'quality': collapse_quality
            })
            
        except Exception as e:
            print(f"    失败: {e}")
        
        print("")
    
    # 3. 找到最佳结果并分析
    if results:
        print("3. 结果分析:")
        
        # 按质量排序
        results_by_quality = sorted(results, key=lambda x: x['quality'], reverse=True)
        
        print("  按坍缩质量排序:")
        for i, r in enumerate(results_by_quality):
            print(f"    {i+1}. {r['desc']}: U_c={r['params'][0]:.4f}, ν^(-1)={r['params'][1]:.4f}, 质量={r['quality']:.2f}")
        
        best_result = results_by_quality[0]
        print(f"\n  🏆 最佳结果: {best_result['desc']}")
        print(f"     U_c = {best_result['params'][0]:.6f} ± {best_result['errors'][0]:.6f}")
        print(f"     ν^(-1) = {best_result['params'][1]:.6f} ± {best_result['errors'][1]:.6f}")
        print(f"     坍缩质量 = {best_result['quality']:.2f}")
        
        # 检查最佳结果是否与ChatGPT一致
        chatgpt_uc_precise = 8.670
        chatgpt_a_precise = 1.056
        
        print(f"\n  🔍 与ChatGPT精确对比:")
        print(f"     ChatGPT: U_c={chatgpt_uc_precise:.3f}, ν^(-1)={chatgpt_a_precise:.3f}")
        print(f"     我们最佳: U_c={best_result['params'][0]:.3f}, ν^(-1)={best_result['params'][1]:.3f}")
        print(f"     差异: ΔU_c={abs(best_result['params'][0]-chatgpt_uc_precise):.4f}, Δν^(-1)={abs(best_result['params'][1]-chatgpt_a_precise):.4f}")
        
        if (abs(best_result['params'][0] - chatgpt_uc_precise) < 0.02 and 
            abs(best_result['params'][1] - chatgpt_a_precise) < 0.1):
            print(f"     ✅ 结果基本一致！")
        else:
            print(f"     ❌ 结果不一致，需要进一步调查")
        
        return best_result
    
    return None

def compare_all_l_vs_drop_l7():
    """比较All L和Drop L=7的结果"""
    
    print("\n" + "="*60)
    print("=== 比较All L vs Drop L=7的No-FSE结果 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    datasets = {
        'All L': df_full.copy(),
        'Drop L=7': df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    }
    
    results = {}
    
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        data = df[["L","U","Y"]].to_numpy(float)
        err = df["sigma"].to_numpy(float)
        
        print(f"数据: {len(df)}点, L={sorted(df['L'].unique())}")
        
        # 测试多个起始点
        starting_points = [
            (8.70, 1.0),
            (8.67, 1.05),
            (8.75, 0.9),
        ]
        
        best_quality = 0
        best_result = None
        
        for Uc0, a0 in starting_points:
            try:
                bounds = ((8.0, 9.0), (0.5, 2.0))
                (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                                 n_knots=10, lam=1e-3, n_boot=5,
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
                
                if collapse_quality > best_quality:
                    best_quality = collapse_quality
                    best_result = (params, errs, collapse_quality)
                
            except Exception as e:
                continue
        
        if best_result:
            params, errs, quality = best_result
            print(f"最佳结果: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
            print(f"坍缩质量: {quality:.2f}")
            results[name] = (params, errs, quality)
    
    # 对比分析
    if 'All L' in results and 'Drop L=7' in results:
        print(f"\n🔍 对比分析:")
        all_l_params, _, all_l_quality = results['All L']
        drop_l7_params, _, drop_l7_quality = results['Drop L=7']
        
        print(f"  All L:    U_c={all_l_params[0]:.4f}, ν^(-1)={all_l_params[1]:.4f}, 质量={all_l_quality:.1f}")
        print(f"  Drop L=7: U_c={drop_l7_params[0]:.4f}, ν^(-1)={drop_l7_params[1]:.4f}, 质量={drop_l7_quality:.1f}")
        
        print(f"  差异:     ΔU_c={drop_l7_params[0]-all_l_params[0]:.4f}, Δν^(-1)={drop_l7_params[1]-all_l_params[1]:.4f}")
        
        # 检查ChatGPT声称的Drop L=7结果
        print(f"\n  ChatGPT声称的Drop L=7: U_c=8.670, ν^(-1)=1.056")
        print(f"  我们的Drop L=7:        U_c={drop_l7_params[0]:.3f}, ν^(-1)={drop_l7_params[1]:.3f}")
        print(f"  与ChatGPT差异:         ΔU_c={abs(drop_l7_params[0]-8.670):.4f}, Δν^(-1)={abs(drop_l7_params[1]-1.056):.4f}")

def main():
    print("🔍 验证ChatGPT的Drop L=7 No-FSE结果")
    print("ChatGPT声称: U_c=8.670(1), ν^(-1)=1.056(2), 来源于Drop L=7")
    print("="*60)
    
    # 验证ChatGPT的Drop L=7结果
    best_drop_l7 = verify_chatgpt_drop_l7()
    
    # 比较All L和Drop L=7
    compare_all_l_vs_drop_l7()
    
    print(f"\n📋 总结:")
    print(f"✅ 完成了ChatGPT Drop L=7结果的详细验证")
    print(f"✅ 系统测试了多个起始点")
    print(f"✅ 对比了All L和Drop L=7的差异")
    
    if best_drop_l7:
        print(f"\n🎯 我们的最佳Drop L=7结果:")
        print(f"   U_c = {best_drop_l7['params'][0]:.6f} ± {best_drop_l7['errors'][0]:.6f}")
        print(f"   ν^(-1) = {best_drop_l7['params'][1]:.6f} ± {best_drop_l7['errors'][1]:.6f}")
        print(f"   坍缩质量 = {best_drop_l7['quality']:.2f}")
        
        # 判断与ChatGPT的一致性
        chatgpt_diff_uc = abs(best_drop_l7['params'][0] - 8.670)
        chatgpt_diff_a = abs(best_drop_l7['params'][1] - 1.056)
        
        if chatgpt_diff_uc < 0.02 and chatgpt_diff_a < 0.1:
            print(f"   ✅ 与ChatGPT结果一致！")
        else:
            print(f"   ❌ 与ChatGPT结果不一致")
            print(f"   需要进一步调查差异原因")

if __name__ == "__main__":
    main() 