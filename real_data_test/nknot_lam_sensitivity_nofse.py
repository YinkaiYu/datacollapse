import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_quality(df, x_collapsed, Y_collapsed):
    x_range = x_collapsed.max() - x_collapsed.min()
    y_ranges = []
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        yL = Y_collapsed[m]
        if len(yL) == 0:
            continue
        y_ranges.append(yL.max() - yL.min())
    return float(x_range / np.mean(y_ranges)) if len(y_ranges) else np.nan


def run():
    base = os.path.dirname(__file__)
    df_full = pd.read_csv(os.path.join(base, "real_data_combined.csv"))
    df = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data = df[["L","U","Y"]].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)

    n_knots_list = [8, 10, 12, 14]
    lam_list = [1e-4, 1e-3, 1e-2]
    # 有代表性的多起点（围绕最佳区）
    inits = [(8.67, 1.10), (8.67, 1.20), (8.67, 1.25), (8.66, 1.18)]
    bounds = ((8.3, 9.0), (0.8, 1.8))

    recs = []
    for nk in tqdm(n_knots_list, desc="n_knots"):
        for lam in lam_list:
            best = None
            vals = []
            for Uc0, a0 in inits:
                try:
                    (params, errs) = fit_data_collapse(data, sigma, Uc0, a0,
                                                        n_knots=nk, lam=lam, n_boot=0, bounds=bounds)
                    xC, YC = collapse_transform(data, params)
                    Q = compute_quality(df, xC, YC)
                    vals.append((params, Q))
                    if (best is None) or (Q > best[1]):
                        best = (params, Q)
                except Exception:
                    continue
            if best is not None:
                # 记录最优以及多起点的统计
                qs = np.array([q for _, q in vals]) if vals else np.array([])
                a_mean = np.mean([p[1] for p,_ in vals]) if vals else np.nan
                Uc_mean = np.mean([p[0] for p,_ in vals]) if vals else np.nan
                recs.append({
                    'n_knots': nk, 'lam': lam,
                    'Uc_best': best[0][0], 'a_best': best[0][1], 'Q_best': best[1],
                    'Uc_mean': Uc_mean, 'a_mean': a_mean,
                    'Q_mean': float(np.mean(qs)) if qs.size else np.nan,
                    'Q_std': float(np.std(qs)) if qs.size else np.nan,
                })
    res = pd.DataFrame(recs)
    out_csv = os.path.join(base, 'nknot_lam_sensitivity_nofse.csv')
    res.to_csv(out_csv, index=False)

    # 绘图：热力图（Q_best）与箱线/误差条
    piv = res.pivot_table(index='n_knots', columns='lam', values='Q_best')
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    ax = fig.add_subplot(gs[0,0])
    if piv.size:
        im = ax.imshow(piv.values, origin='lower', aspect='auto', cmap='viridis',
                       extent=[min(piv.columns), max(piv.columns), min(piv.index), max(piv.index)])
        cbar = fig.colorbar(im, ax=ax, label='Q_best')
    ax.set_xlabel('lam'); ax.set_ylabel('n_knots'); ax.set_title('Q_best heatmap (No-FSE Drop L=7)')

    ax2 = fig.add_subplot(gs[0,1])
    # 误差条：对每个(nk,lam)的多起点Q统计
    xs = np.arange(len(res))
    ax2.errorbar(xs, res['Q_mean'], yerr=res['Q_std'], fmt='o', alpha=0.7)
    ax2.set_xlabel('config idx'); ax2.set_ylabel('Q mean ± std'); ax2.set_title('Init variability by config')

    out_png = os.path.join(base, 'nknot_lam_sensitivity_nofse.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
    print('Saved:', os.path.basename(out_csv), os.path.basename(out_png))

if __name__ == '__main__':
    run() 