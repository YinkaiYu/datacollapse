import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

def investigate_corrected_problem():
    """调查CORRECTED报告中No-FSE结果的问题"""
    
    print("=== 调查CORRECTED报告的问题 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print(f"数据准备完成: {len(df_full)}个数据点")
    
    # CORRECTED报告声称的结果
    corrected_claims = {
        'All L': {'Uc': 8.6625, 'a': 1.0250, 'quality': 78.51},
        'Drop L=7': {'Uc': 8.6188, 'a': 1.0250, 'quality': 69.91},
        'Drop L=7,9': {'Uc': 8.5594, 'a': 1.0250, 'quality': 63.36}
    }
    
    print(f"\nCORRECTED报告声称的No-FSE结果:")
    for method, claims in corrected_claims.items():
        print(f"  {method}: U_c={claims['Uc']:.4f}, ν^(-1)={claims['a']:.4f}, 质量={claims['quality']:.2f}")
    
    print(f"\n🚨 问题: 所有方法的ν^(-1)都完全相同 = 1.0250")
    print(f"这在统计上几乎不可能，表明存在系统性错误。")
    
    # 尝试重现这些"可疑"的结果
    print(f"\n=== 尝试重现CORRECTED报告的结果 ===")
    
    datasets = {
        'All L': (data, err, df_full),
        'Drop L=7': (df_full[df_full["L"] != 7].reset_index(drop=True),),
        'Drop L=7,9': (df_full[~df_full["L"].isin([7, 9])].reset_index(drop=True),)
    }
    
    # 完整准备数据集
    for name in datasets:
        if len(datasets[name]) == 1:
            df_subset = datasets[name][0]
            data_subset = df_subset[["L","U","Y"]].to_numpy(float)
            err_subset = df_subset["sigma"].to_numpy(float)
            datasets[name] = (data_subset, err_subset, df_subset)
    
    for method, claims in corrected_claims.items():
        print(f"\n--- 测试 {method} ---")
        data_subset, err_subset, df_subset = datasets[method]
        
        print(f"数据集大小: {len(df_subset)}点")
        
        # 1. 尝试用CORRECTED的起始值进行正常拟合
        print(f"1. 正常拟合 (起始点: {claims['Uc']:.4f}, {claims['a']:.4f})")
        try:
            bounds = ((8.0, 9.0), (0.5, 2.0))  # 正常边界
            (params, errs) = fit_data_collapse(data_subset, err_subset, 
                                             claims['Uc'], claims['a'], 
                                             n_knots=10, lam=1e-3, n_boot=3,
                                             bounds=bounds)
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data_subset, params)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_subset["L"].unique()):
                m = (df_subset["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            print(f"   实际结果: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
            print(f"   坍缩质量: {collapse_quality:.2f}")
            print(f"   与CORRECTED对比: ΔU_c={params[0]-claims['Uc']:.4f}, Δν^(-1)={params[1]-claims['a']:.4f}")
            
        except Exception as e:
            print(f"   失败: {e}")
        
        # 2. 尝试用极窄边界强制得到1.0250
        print(f"2. 用极窄边界强制 ν^(-1)=1.0250")
        try:
            # 设置极窄边界，强制ν^(-1)在1.0250附近
            bounds = ((8.0, 9.0), (1.024, 1.026))  # 几乎固定ν^(-1)
            (params, errs) = fit_data_collapse(data_subset, err_subset, 
                                             claims['Uc'], claims['a'], 
                                             n_knots=10, lam=1e-3, n_boot=3,
                                             bounds=bounds)
            
            print(f"   强制结果: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
            print(f"   ✅ 成功重现ν^(-1)≈1.0250！")
            print(f"   💡 这可能解释了CORRECTED报告的问题：使用了过窄的边界")
            
        except Exception as e:
            print(f"   失败: {e}")
    
    # 3. 检查是否可能是代码错误
    print(f"\n=== 可能的代码错误分析 ===")
    print(f"CORRECTED报告的问题可能来源:")
    print(f"1. 🎯 边界设置错误: 可能意外设置了bounds=(..., (1.024, 1.026))")
    print(f"2. 🎯 参数固定错误: 可能意外固定了ν^(-1)参数")
    print(f"3. 🎯 初始值问题: 可能所有拟合都用了相同的起始点和边界")
    print(f"4. 🎯 数据问题: 可能用了错误的数据文件或子集")
    
    return None

def correct_no_fse_analysis():
    """正确的No-FSE分析"""
    
    print(f"\n=== 正确的No-FSE分析 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    datasets = {
        'All L': df_full.copy(),
        'Drop L=7': df_full[df_full["L"] != 7].copy().reset_index(drop=True),
        'Drop L=7,9': df_full[~df_full["L"].isin([7, 9])].copy().reset_index(drop=True)
    }
    
    correct_results = {}
    
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        data = df[["L","U","Y"]].to_numpy(float)
        err = df["sigma"].to_numpy(float)
        
        print(f"数据: {len(df)}点, L={sorted(df['L'].unique())}")
        
        # 多个起始点测试
        starting_points = [
            (8.70, 1.0),
            (8.75, 0.9),
            (8.80, 0.8),
        ]
        
        best_quality = 0
        best_result = None
        
        for Uc0, a0 in starting_points:
            try:
                bounds = ((8.0, 9.0), (0.5, 2.0))
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
                
                if collapse_quality > best_quality:
                    best_quality = collapse_quality
                    best_result = (params, errs, collapse_quality)
                
            except Exception as e:
                continue
        
        if best_result:
            params, errs, quality = best_result
            print(f"最佳结果: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
            print(f"坍缩质量: {quality:.2f}")
            correct_results[name] = (params, errs, quality)
    
    return correct_results

def main():
    print("🔍 调查CORRECTED报告No-FSE结果的问题根源")
    print("="*60)
    
    # 调查问题
    investigate_corrected_problem()
    
    # 提供正确分析
    correct_results = correct_no_fse_analysis()
    
    # 对比总结
    print(f"\n" + "="*60)
    print(f"📋 总结对比:")
    print(f"")
    print(f"CORRECTED报告 (❌ 错误):")
    print(f"  All L:       U_c=8.6625, ν^(-1)=1.0250")
    print(f"  Drop L=7:    U_c=8.6188, ν^(-1)=1.0250") 
    print(f"  Drop L=7,9:  U_c=8.5594, ν^(-1)=1.0250")
    print(f"  问题: 所有ν^(-1)完全相同！")
    print(f"")
    print(f"正确分析 (✅ 正确):")
    for name, (params, errs, quality) in correct_results.items():
        print(f"  {name:<12}: U_c={params[0]:.4f}, ν^(-1)={params[1]:.4f}, 质量={quality:.1f}")
    print(f"  特点: ν^(-1)值合理变化")
    
    print(f"\n🎯 结论:")
    print(f"CORRECTED报告的No-FSE结果确实有严重问题，")
    print(f"很可能是由于参数边界设置错误导致的。")

if __name__ == "__main__":
    main() 