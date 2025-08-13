import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import (
    fit_data_collapse_fse,
    fit_data_collapse_fse_robust,
    collapse_transform,
)

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


def scan_grid(b_vals, c_vals, init_Uc_list, init_a_list, bounds, n_knots=10, lam=1e-3):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)

    records = []
    for c0 in tqdm(c_vals, desc="c grid"):
        for b0 in tqdm(b_vals, desc=f"b grid (c0={c0:.2f})", leave=False):
            best = None
            for Uc0 in init_Uc_list:
                for a0 in init_a_list:
                    try:
                        (params, errs) = fit_data_collapse_fse(
                            data, err, float(Uc0), float(a0), float(b0), float(c0),
                            n_knots=n_knots, lam=lam, n_boot=0, bounds=bounds, normalize=True)
                        xC, YC = collapse_transform(data, params, normalize=True)
                        Q = compute_quality(df, xC, YC)
                        rec = {
                            'Uc0': Uc0, 'a0': a0, 'b0': b0, 'c0': c0,
                            'Uc': params[0], 'a': params[1], 'b': params[2], 'c': params[3],
                            'Q': Q
                        }
                        if best is None or Q > best['Q']:
                            best = rec
                    except Exception:
                        continue
            if best is not None:
                records.append(best)
    res = pd.DataFrame(records)
    return df, res


def bootstrap_top_cells(df, res, b_vals, c_vals, bounds, top_k=5, n_boot=6, n_knots=10, lam=1e-3):
    # pick top cells by Q
    top = res.sort_values('Q', ascending=False).head(top_k).copy()
    boot_records = []
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)

    for _, row in top.iterrows():
        # center small grid around (b,c)
        b0 = float(row['b']); c0 = float(row['c'])
        # ensure c negative
        b_grid = np.unique(np.clip(np.linspace(b0-0.2, b0+0.2, 5), 0.0, 2.0))
        c_grid = np.unique(np.clip(np.linspace(c0-0.2, c0+0.2, 5), -1.5, -0.05))
        try:
            (params, errs) = fit_data_collapse_fse_robust(
                data, err, float(row['Uc']), float(row['a']),
                b_grid=b_grid, c_grid=c_grid,
                n_knots=n_knots, lam=lam, n_boot=n_boot, bounds_Ua=((bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])),
                normalize=True
            )
            xC, YC = collapse_transform(data, params, normalize=True)
            Q = compute_quality(df, xC, YC)
            boot_records.append({
                'Uc': params[0], 'a': params[1], 'b': params[2], 'c': params[3],
                'Uc_err': errs[0], 'a_err': errs[1], 'b_err': errs[2], 'c_err': errs[3],
                'Q': Q,
                'b_center': b0, 'c_center': c0
            })
        except Exception:
            continue
    boot_df = pd.DataFrame(boot_records)
    return boot_df


def main():
    # Grid and initial sets
    b_vals = np.linspace(0.4, 1.2, 9)   # 0.4..1.2 step 0.1
    c_vals = np.linspace(-1.4, -0.4, 11)  # -1.4..-0.4 step 0.1
    init_Uc_list = [8.40, 8.57, 8.67]
    init_a_list  = [1.0, 1.2, 1.4]
    bounds = ((8.0, 9.0), (0.8, 2.0), (0.0, 2.0), (-1.5, -0.1))

    # Scan grid (progress bars inside)
    df, res = scan_grid(b_vals, c_vals, init_Uc_list, init_a_list, bounds)

    # Save grid scan CSV
    out_csv = os.path.join(os.path.dirname(__file__), 'robust_fse_grid_scan.csv')
    res.to_csv(out_csv, index=False)

    # Heatmap of best Q per (b,c)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    piv = res.pivot_table(index='c0', columns='b0', values='Q', aggfunc='max')
    if piv.size > 0:
        x0, x1 = float(np.min(piv.columns.values)), float(np.max(piv.columns.values))
        y0, y1 = float(np.min(piv.index.values)), float(np.max(piv.index.values))
        im = ax.imshow(piv.values, origin='lower', aspect='auto',
                       extent=[x0, x1, y0, y1], cmap='viridis')
        cbar = fig.colorbar(im, ax=ax, label='Quality (best per cell)')
    ax.set_xlabel('b'); ax.set_ylabel('c')
    ax.set_title('FSE All-L: best quality over (b,c) grid')
    out_png = os.path.join(os.path.dirname(__file__), 'robust_fse_grid_heatmap.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    # Bootstrap top cells
    boot_df = bootstrap_top_cells(df, res, b_vals, c_vals, bounds, top_k=5, n_boot=6)
    out_boot = os.path.join(os.path.dirname(__file__), 'robust_fse_bootstrap_top.csv')
    boot_df.to_csv(out_boot, index=False)

    # Print quick summary
    if len(res):
        top = res.sort_values('Q', ascending=False).head(5)
        print('Top-5 grid scan (no boot):')
        print(top[['Uc','a','b','c','Q']].to_string(index=False))
    if len(boot_df):
        print('Bootstrap (top cells):')
        print(boot_df[['Uc','a','b','c','Uc_err','a_err','b_err','c_err','Q']].to_string(index=False))

    print(f"Saved: {os.path.basename(out_csv)}, {os.path.basename(out_png)}, {os.path.basename(out_boot)}")


if __name__ == '__main__':
    main() 