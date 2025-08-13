
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse_fse, collapse_transform

# 使用合并后的真实数据文件
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
data = df[["L","U","Y"]].to_numpy(float)
err  = df["sigma"].to_numpy(float)

# 根据真实数据的范围调整初始参数和边界
theta0 = (8.64, 1.8, 0.8, -0.3)
bounds = ((8.30, 9.00), (1.2, 3.0), (0.0, 3.0), (-1.5, -0.05))

(params, errs) = fit_data_collapse_fse(data, err, *theta0, n_knots=10, lam=1e-3, n_boot=4,
                                       bounds=bounds, normalize=True, L_ref="geom")

# BEFORE - 原始数据曲线
plt.figure(figsize=(10, 6))
for L in sorted(df["L"].unique()):
    sub = df[df["L"]==L]
    line, = plt.plot(sub["U"], sub["Y"], "-", lw=1.2, label=f"L={L}")
    plt.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
plt.xlabel("U"); plt.ylabel("Y"); plt.title("Raw curves - Real data")
plt.grid(True, alpha=0.25); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "plot_before_real.png"), dpi=180); plt.close()

# AFTER (normalized FSE) - 数据坍缩后的结果
x, Ycorr = collapse_transform(data, params, normalize=True, L_ref="geom")
plt.figure(figsize=(10, 6))
for L in sorted(df["L"].unique()):
    m = (df["L"]==L).to_numpy()
    xs = x[m]; ys = Ycorr[m]
    # propagate sigma with the same normalized scale
    Lvals = df["L"][m].to_numpy(float)
    b, c = params[2], params[3]
    Lr = float(np.exp(np.mean(np.log(df['L'].to_numpy(float)))))
    S = (1.0 + b*(Lvals**c)) / (1.0 + b*(Lr**c))
    ss = (df["sigma"][m].to_numpy() / S)
    order = np.argsort(xs); xs, ys, ss = xs[order], ys[order], ss[order]
    line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={L}")
    plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
plt.xlabel("(U - Uc) * L^a"); plt.ylabel("Y / normalized (1 + b L^c)")
plt.title(f"Collapsed (fitted, normalized FSE): Uc={params[0]:.4f}, a={params[1]:.3f}, b={params[2]:.3f}, c={params[3]:.3f}")
plt.grid(True, alpha=0.25); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "plot_after_real.png"), dpi=180); plt.close()

print("Fitted params:", params, "+/-", errs)
print(f"Data summary:")
print(f"  L values: {sorted(df['L'].unique())}")
print(f"  Total data points: {len(df)}")
print(f"  U range: {df['U'].min():.3f} to {df['U'].max():.3f}")
print(f"  Y range: {df['Y'].min():.3f} to {df['Y'].max():.3f}")
