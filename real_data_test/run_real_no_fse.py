import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

# 使用合并后的真实数据文件，但去掉L=7的数据
df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
df = df_full[df_full["L"] != 7].copy().reset_index(drop=True)  # 去掉L=7的数据

print(f"Original data: {len(df_full)} points")
print(f"Data without L=7: {len(df)} points")
print(f"Remaining L values: {sorted(df['L'].unique())}")

data = df[["L","U","Y"]].to_numpy(float)
err = df["sigma"].to_numpy(float)

# 基于之前分析结果，使用更好的初始参数
# 经典方法（无FSE）只需要U_c和a两个参数
theta0 = (8.40, 1.4)  # U_c, a
bounds = ((8.30, 8.70), (0.8, 2.0))  # U_c范围, a范围

print(f"\nFitting parameters:")
print(f"Initial guess: U_c={theta0[0]}, a={theta0[1]}")
print(f"Bounds: U_c in {bounds[0]}, a in {bounds[1]}")

# 使用经典方法拟合（无FSE）
(params, errs) = fit_data_collapse(data, err, *theta0, n_knots=10, lam=1e-3, n_boot=10,
                                   bounds=bounds)

print(f"\nFitted parameters:")
print(f"U_c = {params[0]:.6f} ± {errs[0]:.6f}")
print(f"a = {params[1]:.6f} ± {errs[1]:.6f}")

# BEFORE - 原始数据曲线（去掉L=7后）
plt.figure(figsize=(10, 6))
for L in sorted(df["L"].unique()):
    sub = df[df["L"]==L]
    line, = plt.plot(sub["U"], sub["Y"], "-", lw=1.2, label=f"L={L}")
    plt.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
plt.xlabel("U"); plt.ylabel("Y"); plt.title("Raw curves - No FSE (L=7 removed)")
plt.grid(True, alpha=0.25); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "plot_before_no_fse.png"), dpi=180); plt.close()

# AFTER - 数据坍缩后的结果（经典方法）
x, Ycorr = collapse_transform(data, params)
plt.figure(figsize=(10, 6))
for L in sorted(df["L"].unique()):
    m = (df["L"]==L).to_numpy()
    xs = x[m]; ys = Ycorr[m]
    # 经典方法没有FSE修正，直接使用原始误差
    ss = df["sigma"][m].to_numpy()
    order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
    line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
    plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
plt.xlabel("(U - Uc) * L^a"); plt.ylabel("Y")
plt.title(f"Collapsed (classic, no FSE): Uc={params[0]:.4f}, a={params[1]:.3f}")
plt.grid(True, alpha=0.25); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "plot_after_no_fse.png"), dpi=180); plt.close()

# 比较：包含L=7的FSE结果 vs 不包含L=7的经典结果
print(f"\n=== Comparison Summary ===")
print(f"Classic method (no FSE, L=7 removed):")
print(f"  U_c = {params[0]:.6f} ± {errs[0]:.6f}")
print(f"  a = {params[1]:.6f} ± {errs[1]:.6f}")
print(f"  Data points: {len(df)}")
print(f"  L values: {sorted(df['L'].unique())}")

# 计算坍缩质量指标
x_range = x.max() - x.min()
y_ranges = []
for L in sorted(df["L"].unique()):
    m = (df["L"]==L).to_numpy()
    y_range = df["Y"][m].max() - df["Y"][m].min()
    y_ranges.append(y_range)

collapse_quality = x_range / np.mean(y_ranges)
print(f"\nCollapse quality metrics:")
print(f"  X range: {x_range:.3f}")
print(f"  Avg Y range: {np.mean(y_ranges):.3f}")
print(f"  Quality ratio: {collapse_quality:.2f} (higher = better collapse)")

print(f"\nFiles generated:")
print(f"  - plot_before_no_fse.png: Raw data curves")
print(f"  - plot_after_no_fse.png: Collapsed data")
print(f"\nRecommendations:")
print(f"  - Compare these results with the FSE version")
print(f"  - Check if removing L=7 improves the collapse")
print(f"  - Consider if FSE is really necessary for your data") 