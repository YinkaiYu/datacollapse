import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse_fse_robust, collapse_transform

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_quality(df, x_collapsed, Y_collapsed):
    x_range = x_collapsed.max() - x_collapsed.min()
    y_ranges = []
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        if not np.any(m):
            continue
        yL = Y_collapsed[m]
        if len(yL) == 0:
            continue
        y_ranges.append(yL.max() - yL.min())
    return float(x_range / np.mean(y_ranges)) if len(y_ranges) else np.nan


def run(n_trials=60, seed=0):
    rng = np.random.default_rng(seed)
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, 'real_data_combined.csv'))
    data_base = df[["L","U","Y"]].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)

    results = []
    for opt in ["NM_then_Powell", "Powell"]:
        for t in tqdm(range(n_trials), desc=f"FSE optimizer={opt}"):
            Uc0 = float(rng.uniform(8.30, 8.80))
            a0  = float(rng.uniform(0.9, 1.5))
            b0  = float(rng.uniform(0.3, 1.4))
            c0  = float(rng.uniform(-1.45, -0.35))
            b_grid = np.unique(np.clip(np.linspace(b0-0.20, b0+0.20, 5), 0.0, 2.0))
            c_grid = np.unique(np.clip(np.linspace(c0-0.20, c0+0.20, 5), -1.5, -0.05))

            Yb = data_base[:,2] + rng.normal(0.0, sigma)
            data = data_base.copy(); data[:,2] = Yb
            try:
                params, errs = fit_data_collapse_fse_robust(
                    data, sigma, Uc0, a0,
                    b_grid=b_grid, c_grid=c_grid,
                    n_knots=10, lam=1e-3, n_boot=0,
                    bounds_Ua=((8.0, 9.0), (0.8, 2.0)),
                    normalize=True,
                    optimizer=opt, maxiter=4000, random_restarts=0
                )
                xC, YC = collapse_transform(data, params, normalize=True)
                Q = compute_quality(df, xC, YC)
                results.append({
                    'optimizer': opt,
                    'Uc': params[0], 'a': params[1], 'b': params[2], 'c': params[3],
                    'Q': Q
                })
            except Exception:
                continue

    res = pd.DataFrame(results)
    out_csv = os.path.join(base, 'compare_fse_optimizers.csv')
    res.to_csv(out_csv, index=False)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    # Q distributions
    for opt, g in res.groupby('optimizer'):
        axes[0].hist(g['Q'], bins=30, alpha=0.6, label=f"{opt} (n={len(g)})")
    axes[0].set_xlabel('Quality'); axes[0].set_ylabel('Count'); axes[0].set_title('Q distributions by optimizer'); axes[0].legend()
    # Uc scatter
    for opt, g in res.groupby('optimizer'):
        axes[1].scatter(g['Uc'], g['Q'], s=12, alpha=0.6, label=opt)
    axes[1].set_xlabel('U_c'); axes[1].set_ylabel('Quality'); axes[1].set_title('Q vs U_c by optimizer'); axes[1].legend()
    out_png = os.path.join(base, 'compare_fse_optimizers.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
    print('Saved:', os.path.basename(out_csv), os.path.basename(out_png))

if __name__ == '__main__':
    run() 