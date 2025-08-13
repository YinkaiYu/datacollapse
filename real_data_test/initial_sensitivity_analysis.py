import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, collapse_transform

# Plot config: avoid Chinese labels in plots [[memory:5669012]]
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(0)


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


def analyze_nofse_drop_l7():
    print("=== Sensitivity: No-FSE (Drop L=7) ===")
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    df = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)

    # Grid of initial values (moderate size)
    Uc0_list = np.linspace(8.62, 8.76, 10)
    a0_list  = np.linspace(0.9, 1.4, 11)
    bounds = ((8.3, 9.0), (0.6, 1.8))

    records = []
    for Uc0 in Uc0_list:
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
                    # errs may be NaN when n_boot=0; keep but do not filter on them
                    'Uc_err': errs[0], 'a_err': errs[1],
                    'Q': Q
                })
            except Exception as e:
                records.append({'Uc0': Uc0, 'a0': a0, 'Uc': np.nan, 'a': np.nan, 'Uc_err': np.nan, 'a_err': np.nan, 'Q': np.nan})
                continue

    df_res = pd.DataFrame(records)
    # Keep rows where core outputs are finite (do not drop by errs)
    df_res = df_res[np.isfinite(df_res['Uc']) & np.isfinite(df_res['a']) & np.isfinite(df_res['Q'])].copy()

    # Correlations between initial and final
    rho_a = spearmanr(df_res['a0'], df_res['a']).statistic if len(df_res)>2 else np.nan
    rho_Uc = spearmanr(df_res['Uc0'], df_res['Uc']).statistic if len(df_res)>2 else np.nan

    print(f"Spearman corr: a_final vs a0 = {rho_a:.3f}; Uc_final vs Uc0 = {rho_Uc:.3f}")

    # Plots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Heatmap of Q vs (Uc0,a0)
    ax = fig.add_subplot(gs[0,0])
    pivot_Q = df_res.pivot_table(index='a0', columns='Uc0', values='Q')
    if pivot_Q.size > 0:
        x0, x1 = float(np.min(pivot_Q.columns.values)), float(np.max(pivot_Q.columns.values))
        y0, y1 = float(np.min(pivot_Q.index.values)), float(np.max(pivot_Q.index.values))
        im = ax.imshow(pivot_Q.values, origin='lower', aspect='auto',
                       extent=[x0, x1, y0, y1], cmap='viridis')
        fig.colorbar(im, ax=ax, label='Quality')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_xlabel('Initial U_c0')
    ax.set_ylabel('Initial a0')
    ax.set_title('No-FSE: Quality heatmap vs initial')

    # Scatter final a vs initial a (colored by Q)
    ax = fig.add_subplot(gs[0,1])
    sc = ax.scatter(df_res['a0'], df_res['a'], c=df_res['Q'], cmap='viridis', s=25)
    fig.colorbar(sc, ax=ax, label='Quality')
    if len(df_res) > 0:
        ax.plot([df_res['a0'].min(), df_res['a0'].max()], [df_res['a0'].min(), df_res['a0'].max()], 'r--', alpha=0.6)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.6)
    ax.axhline(1.0, color='orange', ls='--', alpha=0.6)
    ax.set_xlabel('Initial a0')
    ax.set_ylabel('Final ν^{-1}')
    ax.set_title(f'Final a vs initial a (Spearman={rho_a:.2f})')

    # Scatter final Uc vs initial Uc (colored by Q)
    ax = fig.add_subplot(gs[0,2])
    sc = ax.scatter(df_res['Uc0'], df_res['Uc'], c=df_res['Q'], cmap='viridis', s=25)
    fig.colorbar(sc, ax=ax, label='Quality')
    if len(df_res) > 0:
        lb, ub = df_res['Uc0'].min(), df_res['Uc0'].max()
        ax.plot([lb,ub],[lb,ub],'r--', alpha=0.6)
    ax.set_xlabel('Initial U_c0')
    ax.set_ylabel('Final U_c')
    ax.set_title(f'Final U_c vs initial U_c (Spearman={rho_Uc:.2f})')

    # Histogram of final a
    ax = fig.add_subplot(gs[1,0])
    ax.hist(df_res['a'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of final ν^{-1}')

    # Histogram of final Uc
    ax = fig.add_subplot(gs[1,1])
    ax.hist(df_res['Uc'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final U_c')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of final U_c')

    # Quality vs final a
    ax = fig.add_subplot(gs[1,2])
    ax.scatter(df_res['a'], df_res['Q'], s=20, alpha=0.6)
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}')
    ax.set_ylabel('Quality')
    ax.set_title('Quality vs final ν^{-1}')

    out = os.path.join(os.path.dirname(__file__), 'sensitivity_nofse_dropL7.png')
    plt.tight_layout(); plt.savefig(out, dpi=220); plt.close()
    print(f"Saved plot: {os.path.basename(out)}; n={len(df_res)}")

    # Summary
    if len(df_res) > 0:
        print("No-FSE summary:")
        print(df_res[['Uc','a','Q']].describe().to_string())

    return df_res


def analyze_fse_all_l():
    print("=== Sensitivity: FSE (All L) ===")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float)

    # Small grid to keep runtime reasonable
    Uc0_list = np.linspace(8.35, 8.75, 6)
    a0_list  = np.linspace(1.0, 1.4, 5)
    bc_list  = [(0.6, -0.6), (0.8, -0.6), (0.6, -1.0)]  # modest variations
    bounds = ((8.0, 9.0), (0.8, 2.0), (0.0, 2.0), (-1.5, -0.1))

    records = []
    for (b0, c0) in bc_list:
        for Uc0 in Uc0_list:
            for a0 in a0_list:
                try:
                    (params, errs) = fit_data_collapse_fse(
                        data, err, float(Uc0), float(a0), float(b0), float(c0),
                        n_knots=10, lam=1e-3, n_boot=0, bounds=bounds, normalize=True)
                    xC, YC = collapse_transform(data, params, normalize=True)
                    Q = compute_quality(df, xC, YC)
                    records.append({
                        'b0': b0, 'c0': c0,
                        'Uc0': Uc0, 'a0': a0,
                        'Uc': params[0], 'a': params[1], 'b': params[2], 'c': params[3],
                        'Uc_err': errs[0], 'a_err': errs[1], 'b_err': errs[2], 'c_err': errs[3],
                        'Q': Q
                    })
                except Exception as e:
                    records.append({'b0': b0, 'c0': c0, 'Uc0': Uc0, 'a0': a0, 'Uc': np.nan, 'a': np.nan, 'b': np.nan, 'c': np.nan, 'Uc_err': np.nan, 'a_err': np.nan, 'b_err': np.nan, 'c_err': np.nan, 'Q': np.nan})
                    continue

    df_res = pd.DataFrame(records)
    df_res = df_res[np.isfinite(df_res['Uc']) & np.isfinite(df_res['a']) & np.isfinite(df_res['Q'])].copy()

    # Correlations
    rho_a = spearmanr(df_res['a0'], df_res['a']).statistic if len(df_res)>2 else np.nan
    rho_Uc = spearmanr(df_res['Uc0'], df_res['Uc']).statistic if len(df_res)>2 else np.nan

    print(f"Spearman corr: a_final vs a0 = {rho_a:.3f}; Uc_final vs Uc0 = {rho_Uc:.3f}")

    # Plots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Heatmap for a fixed (b0,c0) (take the first)
    subset = df_res[(df_res['b0']==bc_list[0][0]) & (df_res['c0']==bc_list[0][1])]
    ax = fig.add_subplot(gs[0,0])
    if len(subset) > 0:
        pivot_Q = subset.pivot_table(index='a0', columns='Uc0', values='Q')
        if pivot_Q.size > 0:
            x0, x1 = float(np.min(pivot_Q.columns.values)), float(np.max(pivot_Q.columns.values))
            y0, y1 = float(np.min(pivot_Q.index.values)), float(np.max(pivot_Q.index.values))
            im = ax.imshow(pivot_Q.values, origin='lower', aspect='auto',
                           extent=[x0, x1, y0, y1], cmap='viridis')
            fig.colorbar(im, ax=ax, label='Quality')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_xlabel('Initial U_c0'); ax.set_ylabel('Initial a0')
    ax.set_title('FSE: Quality heatmap (one b0,c0)')

    # Scatter final a vs initial a (colored by Q)
    ax = fig.add_subplot(gs[0,1])
    sc = ax.scatter(df_res['a0'], df_res['a'], c=df_res['Q'], cmap='viridis', s=25)
    fig.colorbar(sc, ax=ax, label='Quality')
    if len(df_res) > 0:
        ax.plot([df_res['a0'].min(), df_res['a0'].max()], [df_res['a0'].min(), df_res['a0'].max()], 'r--', alpha=0.6)
    ax.set_xlabel('Initial a0'); ax.set_ylabel('Final ν^{-1}')
    ax.set_title(f'Final a vs initial a (Spearman={rho_a:.2f})')

    # Scatter final Uc vs initial Uc (colored by Q)
    ax = fig.add_subplot(gs[0,2])
    sc = ax.scatter(df_res['Uc0'], df_res['Uc'], c=df_res['Q'], cmap='viridis', s=25)
    fig.colorbar(sc, ax=ax, label='Quality')
    if len(df_res) > 0:
        lb, ub = df_res['Uc0'].min(), df_res['Uc0'].max()
        ax.plot([lb,ub],[lb,ub],'r--', alpha=0.6)
    ax.set_xlabel('Initial U_c0'); ax.set_ylabel('Final U_c')
    ax.set_title(f'Final U_c vs initial U_c (Spearman={rho_Uc:.2f})')

    # Histogram of final a
    ax = fig.add_subplot(gs[1,0])
    ax.hist(df_res['a'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final ν^{-1}')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of final ν^{-1}')

    # Histogram of final Uc
    ax = fig.add_subplot(gs[1,1])
    ax.hist(df_res['Uc'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Final U_c')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of final U_c')

    # Quality vs final a
    ax = fig.add_subplot(gs[1,2])
    ax.scatter(df_res['a'], df_res['Q'], s=20, alpha=0.6)
    ax.set_xlabel('Final ν^{-1}')
    ax.set_ylabel('Quality')
    ax.set_title('Quality vs final ν^{-1}')

    out = os.path.join(os.path.dirname(__file__), 'sensitivity_fse_allL.png')
    plt.tight_layout(); plt.savefig(out, dpi=220); plt.close()
    print(f"Saved plot: {os.path.basename(out)}; n={len(df_res)}")

    if len(df_res) > 0:
        print("FSE summary:")
        print(df_res[['Uc','a','b','c','Q']].describe().to_string())

    return df_res


def main():
    nofse_df = analyze_nofse_drop_l7()
    fse_df = analyze_fse_all_l()
    # Print quick sensitivity judgment
    def quick_judge(df, init_col, final_col):
        if len(df) < 3:
            return 'insufficient'
        rho = spearmanr(df[init_col], df[final_col]).statistic
        return f"Spearman({final_col} vs {init_col})={rho:.3f}"

    print("\nQuick sensitivity summary:")
    print("No-FSE:", quick_judge(nofse_df, 'a0', 'a'), quick_judge(nofse_df, 'Uc0', 'Uc'))
    print("FSE:", quick_judge(fse_df, 'a0', 'a'), quick_judge(fse_df, 'Uc0', 'Uc'))

if __name__ == "__main__":
    main() 