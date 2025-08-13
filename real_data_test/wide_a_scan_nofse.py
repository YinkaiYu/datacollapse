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
        if not np.any(m):
            continue
        yL = Y_collapsed[m]
        if len(yL) == 0:
            continue
        y_ranges.append(yL.max() - yL.min())
    return float(x_range / np.mean(y_ranges)) if len(y_ranges) else np.nan


def run_scan():
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    df = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)

    # Very wide scan
    Uc0_list = np.linspace(8.40, 8.90, 21)
    a0_list  = np.linspace(0.30, 2.00, 22)
    bounds = ((8.0, 9.0), (0.30, 2.50))

    records = []
    for Uc0 in tqdm(Uc0_list, desc="Uc0 grid"):
        for a0 in a0_list:
            try:
                (params, errs) = fit_data_collapse(
                    data, err, float(Uc0), float(a0),
                    n_knots=10, lam=1e-3, n_boot=0, bounds=bounds)
                xC, YC = collapse_transform(data, params)
                Q = compute_quality(df, xC, YC)
                records.append({
                    'Uc0': Uc0, 'a0': a0,
                    'Uc': params[0], 'a': params[1],
                    'Q': Q
                })
            except Exception:
                records.append({'Uc0': Uc0, 'a0': a0, 'Uc': np.nan, 'a': np.nan, 'Q': np.nan})
                continue

    res = pd.DataFrame(records)
    res = res[np.isfinite(res['Uc']) & np.isfinite(res['a']) & np.isfinite(res['Q'])].copy()

    # Save CSV
    out_csv = os.path.join(os.path.dirname(__file__), 'wide_a_scan_nofse_dropL7.csv')
    res.to_csv(out_csv, index=False)

    # Print top solutions
    res_sorted = res.sort_values('Q', ascending=False)
    top5 = res_sorted.head(5)
    print("Top-5 by Quality (No-FSE Drop L=7, wide a scan):")
    print(top5[['Uc','a','Q']].to_string(index=False))
    print(f"Best ν^(-1) ~ {float(top5.iloc[0]['a']):.4f}, Q={float(top5.iloc[0]['Q']):.1f}")

    # Split by a<1 and a>=1 groups
    grp_lo = res[res['a'] < 1.0]
    grp_hi = res[res['a'] >= 1.0]
    def stats(g):
        if len(g)==0: return (np.nan, np.nan, 0)
        return (float(g['Q'].mean()), float(g['Q'].max()), int(len(g)))
    m_lo, mx_lo, n_lo = stats(grp_lo)
    m_hi, mx_hi, n_hi = stats(grp_hi)
    print(f"Group a<1:    n={n_lo}, meanQ={m_lo:.1f}, maxQ={mx_lo:.1f}")
    print(f"Group a>=1:   n={n_hi}, meanQ={m_hi:.1f}, maxQ={mx_hi:.1f}")

    # Plots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Quality vs final a
    ax = fig.add_subplot(gs[0,0])
    ax.scatter(res['a'], res['Q'], s=10, alpha=0.6)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}')
    ax.set_ylabel('Quality')
    ax.set_title('Quality vs final ν^{-1}')

    # Heatmap Q vs (Uc0,a0)
    ax = fig.add_subplot(gs[0,1])
    pivot_Q = res.pivot_table(index='a0', columns='Uc0', values='Q')
    if pivot_Q.size > 0:
        x0, x1 = float(np.min(pivot_Q.columns.values)), float(np.max(pivot_Q.columns.values))
        y0, y1 = float(np.min(pivot_Q.index.values)), float(np.max(pivot_Q.index.values))
        im = ax.imshow(pivot_Q.values, origin='lower', aspect='auto',
                       extent=[x0, x1, y0, y1], cmap='viridis')
        fig.colorbar(im, ax=ax, label='Quality')
    ax.set_xlabel('Initial U_c0'); ax.set_ylabel('Initial a0')
    ax.set_title('Quality heatmap vs initial')

    # Histogram of final a
    ax = fig.add_subplot(gs[0,2])
    ax.hist(res['a'], bins=60, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}')
    ax.set_ylabel('Count')

    # Best collapse visual
    best = res_sorted.iloc[0]
    params_best = (float(best['Uc']), float(best['a']))
    xC, YC = collapse_transform(df[["L","U","Y"]].to_numpy(float), params_best)
    ax = fig.add_subplot(gs[1, :])
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        xs = xC[m]; ys = YC[m]
        ss = df["sigma"][m].to_numpy()
        order = np.argsort(xs)
        xs, ys, ss = xs[order], ys[order], ss[order]
        line, = ax.plot(xs, ys, '-', lw=1.5, label=f'L={L}')
        ax.errorbar(xs, ys, yerr=ss, fmt='o', ms=3, capsize=2, elinewidth=1, color=line.get_color(), alpha=0.7)
    ax.set_xlabel('(U - Uc) × L^(1/ν)')
    ax.set_ylabel('R_{01}')
    ax.set_title(f'Best collapse: Uc={params_best[0]:.4f}, ν^(-1)={params_best[1]:.3f}, Q={best["Q"]:.1f}')
    ax.legend()

    out_png = os.path.join(os.path.dirname(__file__), 'wide_a_scan_nofse_dropL7.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    print(f"Saved: {os.path.basename(out_csv)}, {os.path.basename(out_png)}; n={len(res)}")
    return res


if __name__ == '__main__':
    run_scan() 