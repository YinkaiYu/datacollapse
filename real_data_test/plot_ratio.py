import os
import numpy as np
import matplotlib.pyplot as plt

# ==== 配置 ====
L_list = [7, 9, 11, 13]       # 系统尺寸
R_list = ["R10", "R01", "R11"]  # 需要绘制的类型
data_root = "Data_scale"      # 数据主目录
save_dir = "photo/ratio"      # 保存图片的目录

# ==== 创建保存目录 ====
os.makedirs(save_dir, exist_ok=True)

for Rtype in R_list:
    plt.figure(figsize=(6, 4))

    for L in L_list:
        data_path = os.path.join(data_root, f"L{L}", "R", Rtype)
        if not os.path.isfile(data_path):
            print(f"Warning: {data_path} 不存在，跳过 L={L}")
            continue

        # 读取数据
        data = np.loadtxt(data_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)  # 只有一行时调整形状

        U_vals = data[:, 0]
        y_vals = data[:, 1]
        err_vals = data[:, 2]  # 误差列

        # 过滤 8~9 范围的数据
        mask = (U_vals >= 8.1) & (U_vals <= 9.3)
        U_vals = U_vals[mask]
        y_vals = y_vals[mask]
        err_vals = err_vals[mask]

        # 绘制带误差棒的曲线
        plt.errorbar(U_vals, y_vals, yerr=err_vals, fmt="o-", capsize=4, label=f"L={L}")

    plt.xlabel("U")
    plt.ylabel(Rtype)
    plt.title(Rtype)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{Rtype}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"图已保存到 {save_path}")
