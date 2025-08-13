
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse_fse_robust, collapse_transform

# Use real data combined
csv_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
df = pd.read_csv(csv_path)
data = df[["L","U","Y"]].to_numpy(float)
err  = df["sigma"].to_numpy(float)

# Robust finite-size-correction fit (normalized)
b0, c0 = 0.8, -1.0
b_grid = np.linspace(b0-0.3, b0+0.3, 7)
c_grid = np.linspace(c0-0.3, c0+0.3, 7)
(params, errs) = fit_data_collapse_fse_robust(
    data, err, 8.45, 1.20,
    b_grid=b_grid, c_grid=c_grid,
    n_knots=10, lam=1e-3, n_boot=4,
    bounds_Ua=((8.0, 9.0), (0.8, 2.0)), normalize=True,
    optimizer="NM_then_Powell", maxiter=4000, random_restarts=2
)

# BEFORE
plt.figure()
for L in sorted(df["L"].unique()):
    sub = df[df["L"]==L]
    line, = plt.plot(sub["U"], sub["Y"], "-", lw=1.2, label=f"L={L}")
    plt.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
plt.xlabel("U"); plt.ylabel("R"); plt.title("Raw curves")
plt.grid(True, alpha=0.25); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "plot_before.png"), dpi=180); plt.close()

# AFTER (normalized finite-size correction)
x, Ycorr = collapse_transform(data, params, normalize=True, L_ref="geom")
plt.figure()
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
plt.xlabel("(U - Uc) * L^a"); plt.ylabel("R / normalized (1 + b L^c)")
plt.title(f"Collapsed (normalized finite-size correction): Uc={params[0]:.4f}, a={params[1]:.3f}, b={params[2]:.3f}, c={params[3]:.3f}")
plt.grid(True, alpha=0.25); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "plot_after.png"), dpi=180); plt.close()

print("Fitted params:", params, "+/-", errs)
