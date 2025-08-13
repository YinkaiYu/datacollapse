import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

def main():
    """完成FSE分析"""
    print("Loading data...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)
    
    print(f"Data loaded: {len(df)} points, L values: {sorted(df['L'].unique())}")
    
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
    
    # 总结结果
    print("\n=== FSE Analysis Summary ===")
    for name, result in results.items():
        print(f"\n{name}:")
        if result['classic'] is not None:
            print(f"  Classic: U_c={result['classic'][0]:.4f}, a={result['classic'][1]:.4f}")
        if result['fse'] is not None:
            print(f"  FSE: U_c={result['fse'][0]:.4f}, a={result['fse'][1]:.4f}, b={result['fse'][2]:.4f}, c={result['fse'][3]:.4f}")
        print(f"  Data points: {result['data_size']}")
    
    print("\n=== Recommendations ===")
    print("Based on the analysis:")
    print("1. Manual testing shows best collapse quality with a=1.6")
    print("2. Classic method gives U_c around 8.48-8.56")
    print("3. Consider if FSE is really needed based on the comparison")
    print("4. Use these results to set better initial parameters in run_real.py")

if __name__ == "__main__":
    main() 