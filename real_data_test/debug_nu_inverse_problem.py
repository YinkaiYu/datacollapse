import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def debug_nu_inverse_problem():
    """调试ν^(-1)总是小于1的问题"""
    
    print("=== 调试ν^(-1) < 1问题 ===")
    print("目标：找出为什么我们总是得到ν^(-1) < 1，而ChatGPT得到ν^(-1) > 1")
    print("")
    
    # 加载数据
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    
    print(f"数据检查:")
    print(f"  总数据点: {len(df_full)}")
    print(f"  L值: {sorted(df_full['L'].unique())}")
    print(f"  U范围: {df_full['U'].min():.3f} - {df_full['U'].max():.3f}")
    print(f"  Y范围: {df_full['Y'].min():.3f} - {df_full['Y'].max():.3f}")
    
    # 1. 检查数据格式和单位
    print(f"\n1. 数据格式检查:")
    print(f"前5行数据:")
    print(df_full.head())
    
    # 2. 检查拟合函数的使用方式
    print(f"\n2. 拟合函数测试:")
    
    # 准备Drop L=7数据（ChatGPT说这个给出最好结果）
    df_drop_l7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data_drop_l7 = df_drop_l7[["L","U","Y"]].to_numpy(float)
    err_drop_l7 = df_drop_l7["sigma"].to_numpy(float)
    
    print(f"Drop L=7数据: {len(df_drop_l7)}点, L={sorted(df_drop_l7['L'].unique())}")
    
    # 3. 测试不同的参数理解
    print(f"\n3. 参数含义检查:")
    print(f"fit_data_collapse(data, err, Uc0, a0, ...)")
    print(f"其中a0是什么？")
    print(f"- 我理解的: a0 = ν^(-1) = 1/ν")
    print(f"- 标度关系: x = (U - Uc) * L^a")
    print(f"- 如果a = 1/ν，那么a > 1意味着ν < 1")
    print(f"- 如果a = ν，那么a > 1意味着ν > 1")
    print(f"")
    print(f"🤔 问题可能在于参数定义的混淆!")
    
    # 4. 系统测试不同的起始参数范围
    print(f"\n4. 系统测试不同起始参数:")
    
    test_cases = [
        # (Uc0, a0, description, expectation)
        (8.67, 0.5, "低a值1", "如果a=1/ν，则ν=2"),
        (8.67, 0.8, "低a值2", "如果a=1/ν，则ν=1.25"),
        (8.67, 1.0, "a=1", "如果a=1/ν，则ν=1"),
        (8.67, 1.2, "高a值1", "如果a=1/ν，则ν=0.83"),
        (8.67, 1.5, "高a值2", "如果a=1/ν，则ν=0.67"),
        (8.67, 2.0, "很高a值", "如果a=1/ν，则ν=0.5"),
    ]
    
    results = []
    
    for Uc0, a0, desc, expect in test_cases:
        print(f"\n  测试 {desc}: Uc0={Uc0:.2f}, a0={a0:.1f}")
        print(f"    预期: {expect}")
        
        try:
            # 设置宽松边界来看参数的自然收敛
            bounds = ((8.0, 9.0), (0.3, 3.0))
            
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
            
            print(f"    结果: U_c={params[0]:.4f}±{errs[0]:.4f}, a={params[1]:.4f}±{errs[1]:.4f}")
            print(f"    如果a=1/ν: ν={1/params[1]:.4f}")
            print(f"    如果a=ν: ν={params[1]:.4f}")
            print(f"    坍缩质量: {collapse_quality:.2f}")
            
            results.append({
                'desc': desc,
                'start_a': a0,
                'final_a': params[1],
                'final_Uc': params[0],
                'quality': collapse_quality,
                'nu_if_a_is_inv_nu': 1/params[1],
                'nu_if_a_is_nu': params[1]
            })
            
        except Exception as e:
            print(f"    失败: {e}")
    
    # 5. 分析结果模式
    if results:
        print(f"\n5. 结果模式分析:")
        print(f"{'起始a':<8} {'最终a':<8} {'ν(a=1/ν)':<12} {'ν(a=ν)':<10} {'质量':<8}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['start_a']:<8.1f} {r['final_a']:<8.4f} {r['nu_if_a_is_inv_nu']:<12.4f} {r['nu_if_a_is_nu']:<10.4f} {r['quality']:<8.1f}")
        
        # 找到质量最高的结果
        best = max(results, key=lambda x: x['quality'])
        print(f"\n🏆 最佳质量结果: {best['desc']}")
        print(f"   最终参数: U_c={best['final_Uc']:.4f}, a={best['final_a']:.4f}")
        print(f"   如果a=1/ν: ν={best['nu_if_a_is_inv_nu']:.4f}")
        print(f"   如果a=ν: ν={best['nu_if_a_is_nu']:.4f}")
        print(f"   坍缩质量: {best['quality']:.2f}")
        
        # 与ChatGPT对比
        chatgpt_a = 1.056  # ChatGPT的ν^(-1)
        print(f"\n🔍 与ChatGPT对比:")
        print(f"   ChatGPT: ν^(-1)=1.056 → ν=0.947")
        print(f"   如果我们的a=ν^(-1): 最佳a={best['final_a']:.3f} → ν={1/best['final_a']:.3f}")
        print(f"   如果我们的a=ν: 最佳a={best['final_a']:.3f} → ν={best['final_a']:.3f}")
        
        if abs(best['final_a'] - chatgpt_a) < 0.1:
            print(f"   ✅ 我们的a接近ChatGPT的ν^(-1)，参数定义一致")
        elif abs(1/best['final_a'] - 1/chatgpt_a) < 0.1:
            print(f"   ✅ 我们的1/a接近ChatGPT的ν，参数定义相反")
        else:
            print(f"   ❌ 参数不匹配，需要进一步调查")
    
    return results

def check_scaling_relationship():
    """检查标度关系的定义"""
    
    print(f"\n" + "="*60)
    print(f"=== 检查标度关系定义 ===")
    
    print(f"标准有限尺寸标度理论:")
    print(f"Y(U,L) = L^(d-z) * F((U-Uc)*L^(1/ν))")
    print(f"")
    print(f"简化的数据坍缩形式:")
    print(f"Y ≈ f(x)，其中 x = (U - Uc) * L^a")
    print(f"")
    print(f"关键问题: a的定义是什么？")
    print(f"选项1: a = 1/ν  (我一直假设的)")
    print(f"选项2: a = ν    (ChatGPT可能假设的)")
    print(f"")
    
    # 加载数据做一个直观检查
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    df_drop_l7 = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    
    print(f"直观检查 - 看看哪个定义更合理:")
    
    # 使用ChatGPT的参数
    Uc_chatgpt = 8.670
    a_chatgpt = 1.056  # ChatGPT说这是ν^(-1)
    
    print(f"ChatGPT参数: Uc={Uc_chatgpt}, ν^(-1)={a_chatgpt}")
    print(f"")
    
    # 计算两种理解下的坍缩坐标
    for L in sorted(df_drop_l7["L"].unique()):
        L_data = df_drop_l7[df_drop_l7["L"] == L]
        U_vals = L_data["U"].values
        Y_vals = L_data["Y"].values
        
        # 假设a = 1/ν
        x1 = (U_vals - Uc_chatgpt) * (L ** a_chatgpt)
        
        # 假设a = ν  
        x2 = (U_vals - Uc_chatgpt) * (L ** (1/a_chatgpt))
        
        print(f"L={L}:")
        print(f"  如果a=1/ν: x范围 = [{x1.min():.2f}, {x1.max():.2f}]")
        print(f"  如果a=ν:   x范围 = [{x2.min():.2f}, {x2.max():.2f}]")
    
    print(f"\n💡 观察: 如果不同L的x范围重叠良好，说明定义正确")

def main():
    print("🔍 调试ν^(-1) < 1的系统性问题")
    print("="*60)
    
    # 调试参数问题
    results = debug_nu_inverse_problem()
    
    # 检查标度关系定义
    check_scaling_relationship()
    
    print(f"\n📋 诊断总结:")
    print(f"1. 检查了数据格式和范围")
    print(f"2. 系统测试了不同起始参数")
    print(f"3. 分析了参数定义的可能混淆")
    print(f"4. 与ChatGPT结果进行了对比")
    
    if results:
        best = max(results, key=lambda x: x['quality'])
        print(f"\n🎯 建议:")
        print(f"最佳拟合参数: a={best['final_a']:.4f}")
        print(f"需要明确a的物理意义来与ChatGPT对比")

if __name__ == "__main__":
    main() 