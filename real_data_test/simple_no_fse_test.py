import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

def test_no_fse():
    """简单测试No-FSE拟合"""
    
    print("=== 简单No-FSE测试 ===")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    print(f"数据: {len(df_full)}点, L={sorted(df_full['L'].unique())}")
    
    # 测试几个关键起始点
    tests = [
        (8.70, 1.0, "标准"),
        (8.75, 0.9, "高Uc"),
        (8.6625, 1.0250, "CORRECTED"),  # 试试CORRECTED报告的值
        (8.7517, 1.0763, "ChatGPT"),    # ChatGPT的值
    ]
    
    results = []
    
    for Uc0, a0, name in tests:
        print(f"\n测试 {name}: Uc0={Uc0:.4f}, a0={a0:.4f}")
        
        try:
            bounds = ((8.0, 9.0), (0.5, 2.0))
            (params, errs) = fit_data_collapse(data, err, Uc0, a0, 
                                             n_knots=10, lam=1e-3, n_boot=3,
                                             bounds=bounds)
            
            # 计算坍缩质量
            x_collapsed, Y_collapsed = collapse_transform(data, params)
            x_range = x_collapsed.max() - x_collapsed.min()
            y_ranges = []
            for L in sorted(df_full["L"].unique()):
                m = (df_full["L"]==L).to_numpy()
                y_range = Y_collapsed[m].max() - Y_collapsed[m].min()
                y_ranges.append(y_range)
            collapse_quality = x_range / np.mean(y_ranges)
            
            print(f"  成功: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
            print(f"  坍缩质量: {collapse_quality:.2f}")
            
            results.append((name, params, errs, collapse_quality))
            
        except Exception as e:
            print(f"  失败: {e}")
    
    # 找最佳结果
    if results:
        best = max(results, key=lambda x: x[3])
        print(f"\n✅ 最佳结果: {best[0]}")
        print(f"   U_c = {best[1][0]:.6f} ± {best[2][0]:.6f}")
        print(f"   ν^(-1) = {best[1][1]:.6f} ± {best[2][1]:.6f}")
        print(f"   坍缩质量 = {best[3]:.2f}")
    
    return results

def analyze_chatgpt_result():
    """分析ChatGPT的结果"""
    print(f"\n=== 分析ChatGPT结果 ===")
    print(f"ChatGPT报告: U_c=7517(38), ν^(-1)=1.0763(54)")
    print(f"")
    print(f"可能的解释:")
    print(f"1. U_c=7517 -> 可能是 8.7517 (遗漏了8.)")
    print(f"2. 误差(38) -> 可能是 0.0038")
    print(f"3. ν^(-1)=1.0763(54) -> 误差可能是 0.0054")
    print(f"")
    print(f"我们的最接近结果:")
    
    # 加载数据测试
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df_full[["L","U","Y"]].to_numpy(float)
    err = df_full["sigma"].to_numpy(float)
    
    try:
        # 测试8.7517
        bounds = ((8.0, 9.0), (0.8, 1.5))
        (params, errs) = fit_data_collapse(data, err, 8.7517, 1.0763, 
                                         n_knots=10, lam=1e-3, n_boot=3,
                                         bounds=bounds)
        
        print(f"测试8.7517起始点:")
        print(f"  结果: U_c={params[0]:.6f}±{errs[0]:.6f}, ν^(-1)={params[1]:.6f}±{errs[1]:.6f}")
        
        # 检查是否接近ChatGPT的值
        if abs(params[0] - 8.7517) < 0.1 and abs(params[1] - 1.0763) < 0.2:
            print(f"  ✅ 接近ChatGPT的结果！")
        else:
            print(f"  ❌ 与ChatGPT结果差异较大")
            
    except Exception as e:
        print(f"测试失败: {e}")

def main():
    print("🔍 No-FSE简单测试 + ChatGPT结果分析")
    print("="*50)
    
    # 测试No-FSE
    results = test_no_fse()
    
    # 分析ChatGPT结果
    analyze_chatgpt_result()
    
    print(f"\n📋 总结:")
    print(f"✅ 完成No-FSE测试")
    print(f"✅ 分析了ChatGPT结果的可能性")

if __name__ == "__main__":
    main() 