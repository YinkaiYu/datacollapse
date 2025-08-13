import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def run_generalized_bootstrap(n_trials=160, n_boot_ext=3, seed=0,
                              uc_center=8.696, uc_span=0.08,
                              a_min=0.8, a_max=1.6,
                              bounds=((8.0, 9.0), (0.6, 2.0))):
    rng = np.random.default_rng(seed)
    df_full = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    df = df_full[df_full["L"] != 7].copy().reset_index(drop=True)
    data_base = df[["L","U","Y"]].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)

    records = []
    for t in tqdm(range(n_trials), desc="No-FSE generalized bootstrap"):
        Uc0 = float(rng.uniform(uc_center-uc_span, uc_center+uc_span))
        a0  = float(rng.uniform(a_min, a_max))
        for b in range(n_boot_ext):
            # external bootstrap by resampling noise
            Yb = data_base[:,2] + rng.normal(0.0, sigma)
            data = data_base.copy()
            data[:,2] = Yb
            try:
                (params, errs) = fit_data_collapse(
                    data, sigma, Uc0, a0,
                    n_knots=10, lam=1e-3, n_boot=0, bounds=bounds)
                xC, YC = collapse_transform(data, params)
                Q = compute_quality(df, xC, YC)
                records.append({
                    'trial': t, 'rep': b,
                    'Uc0': Uc0, 'a0': a0,
                    'Uc': params[0], 'a': params[1], 'Q': Q
                })
            except Exception:
                continue

    res = pd.DataFrame(records)
    out_csv = os.path.join(os.path.dirname(__file__), 'generalized_bootstrap_nofse_dropL7.csv')
    res.to_csv(out_csv, index=False)

    # Estimate equivalence band ΔQ using MAD or percentile width
    if len(res) > 0:
        Q_vals = res['Q'].to_numpy(float)
        Q_median = float(np.median(Q_vals))
        mad = float(np.median(np.abs(Q_vals - Q_median)))
        # Use ~1.4826*MAD as robust sigma
        Q_sigma_robust = 1.4826 * mad if mad > 0 else float(np.std(Q_vals))
        delta_Q = 2.0 * Q_sigma_robust  # ~95% band (approx)
        Q_p25, Q_p75 = np.percentile(Q_vals, [25, 75])
        Q_top = float(np.max(Q_vals))
        Uc_med = float(np.median(res['Uc'])); a_med = float(np.median(res['a']))
        Uc_p25, Uc_p75 = np.percentile(res['Uc'], [25, 75])
        a_p25, a_p75 = np.percentile(res['a'], [25, 75])
    else:
        Q_median, Q_sigma_robust, delta_Q = np.nan, np.nan, np.nan
        Q_p25 = Q_p75 = Q_top = np.nan
        Uc_med = a_med = np.nan
        Uc_p25 = Uc_p75 = a_p25 = a_p75 = np.nan

    # Plot distributions
    if len(res) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # Q 分布
        axes[0,0].hist(res['Q'], bins=40, edgecolor='black', alpha=0.7)
        axes[0,0].axvline(Q_median, color='green', lw=2, label=f'median Q={Q_median:.1f}')
        axes[0,0].axvspan(Q_median-delta_Q/2, Q_median+delta_Q/2, color='orange', alpha=0.2, label=f'ΔQ≈{delta_Q:.1f}')
        axes[0,0].axvspan(Q_p25, Q_p75, color='blue', alpha=0.10, label=f'IQR [{Q_p25:.1f}, {Q_p75:.1f}]')
        axes[0,0].axvline(Q_top, color='red', lw=1.5, ls='--', label=f'Top Q={Q_top:.1f}')
        axes[0,0].set_xlabel('Quality'); axes[0,0].set_ylabel('Count'); axes[0,0].legend()
        axes[0,0].set_title('Q distribution, median/IQR/ΔQ')

        # a 分布
        axes[0,1].hist(res['a'], bins=40, edgecolor='black', alpha=0.7)
        axes[0,1].axvline(1.0, color='orange', ls='--', alpha=0.7)
        axes[0,1].axvline(a_med, color='green', lw=2, label=f'median ν^(-1)={a_med:.3f}')
        axes[0,1].axvspan(a_p25, a_p75, color='blue', alpha=0.10, label=f'IQR [{a_p25:.3f}, {a_p75:.3f}]')
        axes[0,1].set_xlabel('ν^{-1}')
        axes[0,1].set_ylabel('Count'); axes[0,1].legend()

        # Uc 分布
        axes[1,0].hist(res['Uc'], bins=40, edgecolor='black', alpha=0.7)
        axes[1,0].axvline(Uc_med, color='green', lw=2, label=f'median U_c={Uc_med:.3f}')
        axes[1,0].axvspan(Uc_p25, Uc_p75, color='blue', alpha=0.10, label=f'IQR [{Uc_p25:.3f}, {Uc_p75:.3f}]')
        axes[1,0].set_xlabel('U_c'); axes[1,0].set_ylabel('Count'); axes[1,0].legend()

        # Q vs a
        sc = axes[1,1].scatter(res['a'], res['Q'], s=10, alpha=0.6)
        axes[1,1].axvline(1.0, color='orange', ls='--', alpha=0.7)
        axes[1,1].set_xlabel('ν^{-1}')
        axes[1,1].set_ylabel('Quality')
        axes[1,1].set_title('Quality vs ν^{-1}')

        out_png = os.path.join(os.path.dirname(__file__), 'generalized_bootstrap_nofse_dropL7.png')
        plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    # Quick textual summary
    print(f"No-FSE generalized bootstrap: n={len(res)}")
    print(f"Q_median={Q_median:.2f}, robust_sigma≈{Q_sigma_robust:.2f}, equivalence ΔQ≈{delta_Q:.2f}")
    if len(res) > 0:
        print(f"Uc median={Uc_med:.6f}, a median={a_med:.6f}")

    return res


if __name__ == '__main__':
    run_generalized_bootstrap() 