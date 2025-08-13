import os
import pandas as pd
import numpy as np

def process_real_data():
    """处理Data_scale目录中的数据，将它们合并成sample_data.csv格式"""
    
    # 数据目录路径
    data_dir = os.path.join(os.path.dirname(__file__), "Data_scale")
    
    # 存储所有数据的列表
    all_data = []
    
    # 遍历所有L*目录
    for item in os.listdir(data_dir):
        if item.startswith('L') and os.path.isdir(os.path.join(data_dir, item)):
            # 提取L值
            L = int(item[1:])  # 去掉'L'前缀
            
            # 读取R/R01文件
            r01_file = os.path.join(data_dir, item, "R", "R01")
            
            if os.path.exists(r01_file):
                print(f"Processing {item}/R/R01...")
                
                # 读取数据文件（空格分隔的3列：U, Y, sigma）
                try:
                    # 使用pandas读取，假设是空格分隔
                    df_temp = pd.read_csv(r01_file, sep=r'\s+', header=None, 
                                        names=['U', 'Y', 'sigma'])
                    
                    # 添加L列
                    df_temp['L'] = L
                    
                    # 重新排列列顺序
                    df_temp = df_temp[['L', 'U', 'Y', 'sigma']]
                    
                    all_data.append(df_temp)
                    print(f"  Loaded {len(df_temp)} data points for L={L}")
                    
                except Exception as e:
                    print(f"  Error reading {r01_file}: {e}")
            else:
                print(f"Warning: {r01_file} not found")
    
    if all_data:
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 按L和U排序
        combined_df = combined_df.sort_values(['L', 'U']).reset_index(drop=True)
        
        # 保存为CSV文件
        output_file = os.path.join(os.path.dirname(__file__), "real_data_combined.csv")
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully processed data:")
        print(f"Total data points: {len(combined_df)}")
        print(f"L values: {sorted(combined_df['L'].unique())}")
        print(f"U range: {combined_df['U'].min():.3f} to {combined_df['U'].max():.3f}")
        print(f"Output saved to: {output_file}")
        
        return combined_df
    else:
        print("No data was processed!")
        return None

if __name__ == "__main__":
    process_real_data() 