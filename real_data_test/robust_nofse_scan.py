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


def run_scan(uc_center=8.696, uc_span=0.07, a_min=1.0, a_max=1.4,
             n_uc=21, n_a=17, bounds=((8.0, 9.0), (0.8, 2.0))):
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    df = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)

    Uc0_list = np.linspace(uc_center-uc_span, uc_center+uc_span, n_uc)
    a0_list  = np.linspace(a_min, a_max, n_a)

    records = []
    for Uc0 in tqdm(Uc0_list, desc="No-FSE Uc0 grid"):
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
    out_csv = os.path.join(os.path.dirname(__file__), 'robust_nofse_dropL7_scan.csv')
    res.to_csv(out_csv, index=False)

    # Plots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Heatmap Q vs (Uc0,a0)
    ax = fig.add_subplot(gs[0,0])
    pivot_Q = res.pivot_table(index='a0', columns='Uc0', values='Q')
    if pivot_Q.size > 0:
        x0, x1 = float(np.min(pivot_Q.columns.values)), float(np.max(pivot_Q.columns.values))
        y0, y1 = float(np.min(pivot_Q.index.values)), float(np.max(pivot_Q.index.values))
        im = ax.imshow(pivot_Q.values, origin='lower', aspect='auto',
                       extent=[x0, x1, y0, y1], cmap='viridis')
        fig.colorbar(im, ax=ax, label='Quality')
    ax.set_xlabel('Initial U_c0'); ax.set_ylabel('Initial a0')
    ax.set_title('No-FSE Drop L=7: Quality heatmap vs initial')

    # Scatter final a vs initial a (colored by Q)
    ax = fig.add_subplot(gs[0,1])
    sc = ax.scatter(res['a0'], res['a'], c=res['Q'], cmap='viridis', s=20)
    fig.colorbar(sc, ax=ax, label='Quality')
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Initial a0'); ax.set_ylabel('Final ν^{-1}')
    ax.set_title('Final a vs initial a')

    # Scatter final Uc vs initial Uc (colored by Q)
    ax = fig.add_subplot(gs[0,2])
    sc = ax.scatter(res['Uc0'], res['Uc'], c=res['Q'], cmap='viridis', s=20)
    fig.colorbar(sc, ax=ax, label='Quality')
    ax.set_xlabel('Initial U_c0'); ax.set_ylabel('Final U_c')
    ax.set_title('Final U_c vs initial U_c')

    # Histograms
    ax = fig.add_subplot(gs[1,0])
    ax.hist(res['a'], bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}'); ax.set_ylabel('Count')

    ax = fig.add_subplot(gs[1,1])
    ax.hist(res['Uc'], bins=40, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final U_c'); ax.set_ylabel('Count')

    # Quality vs final a
    ax = fig.add_subplot(gs[1,2])
    ax.scatter(res['a'], res['Q'], s=20, alpha=0.6)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}'); ax.set_ylabel('Quality')

    out_png = os.path.join(os.path.dirname(__file__), 'robust_nofse_dropL7_scan.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    print(f"Saved: {os.path.basename(out_csv)}, {os.path.basename(out_png)}; n={len(res)}")
    return res


def main():
    run_scan()

if __name__ == '__main__':
    main() 