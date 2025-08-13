import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
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


def run_generalized_bootstrap_fse(n_trials=120, n_boot_ext=3, seed=0,
                                  Uc_min=8.30, Uc_max=8.80,
                                  a_min=0.9, a_max=1.5,
                                  b_span=0.25, c_span=0.25,
                                  n_grid=5):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "real_data_combined.csv"))
    data_base = df[["L","U","Y"]].to_numpy(float)
    sigma = df["sigma"].to_numpy(float)

    records = []
    for t in tqdm(range(n_trials), desc="FSE generalized bootstrap"):
        # random initial centers（更广）
        Uc0 = float(rng.uniform(Uc_min, Uc_max))
        a0  = float(rng.uniform(a_min, a_max))
        b0  = float(rng.uniform(0.3, 1.4))
        c0  = float(rng.uniform(-1.45, -0.35))

        # local grids around (b0,c0)
        b_grid = np.unique(np.clip(np.linspace(b0-b_span, b0+b_span, int(n_grid)), 0.0, 2.0))
        c_grid = np.unique(np.clip(np.linspace(c0-c_span, c0+c_span, int(n_grid)), -1.5, -0.05))

        for b in range(n_boot_ext):
            # external bootstrap on Y
            Yb = data_base[:,2] + rng.normal(0.0, sigma)
            data = data_base.copy()
            data[:,2] = Yb
            try:
                params, errs = fit_data_collapse_fse_robust(
                    data, sigma, Uc0, a0,
                    b_grid=b_grid, c_grid=c_grid,
                    n_knots=10, lam=1e-3, n_boot=0,
                    bounds_Ua=((8.0, 9.0), (0.8, 2.0)),
                    normalize=True
                )
                xC, YC = collapse_transform(data, params, normalize=True)
                Q = compute_quality(df, xC, YC)
                records.append({
                    'trial': t, 'rep': b,
                    'Uc': params[0], 'a': params[1], 'b': params[2], 'c': params[3],
                    'Q': Q
                })
            except Exception:
                continue

    res = pd.DataFrame(records)
    out_csv = os.path.join(os.path.dirname(__file__), 'generalized_bootstrap_fse_allL.csv')
    res.to_csv(out_csv, index=False)

    # Estimate ΔQ via robust sigma
    if len(res) > 0:
        Q_vals = res['Q'].to_numpy(float)
        Q_median = float(np.median(Q_vals))
        mad = float(np.median(np.abs(Q_vals - Q_median)))
        Q_sigma_robust = 1.4826 * mad if mad > 0 else float(np.std(Q_vals))
        delta_Q = 2.0 * Q_sigma_robust
        Q_p25, Q_p75 = np.percentile(Q_vals, [25, 75])
        Q_top = float(np.max(Q_vals))
        Uc_med = float(np.median(res['Uc'])); a_med = float(np.median(res['a']))
        Uc_p25, Uc_p75 = np.percentile(res['Uc'], [25, 75])
        a_p25, a_p75 = np.percentile(res['a'], [25, 75])
    else:
        Q_median = Q_sigma_robust = delta_Q = np.nan
        Q_p25 = Q_p75 = Q_top = np.nan
        Uc_med = a_med = np.nan
        Uc_p25 = Uc_p75 = a_p25 = a_p75 = np.nan

    # Plot distributions（带标注）
    if len(res) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # Q分布
        axes[0,0].hist(res['Q'], bins=40, edgecolor='black', alpha=0.7)
        axes[0,0].axvline(Q_median, color='green', lw=2, label=f'median Q={Q_median:.1f}')
        axes[0,0].axvspan(Q_median-delta_Q/2, Q_median+delta_Q/2, color='orange', alpha=0.2,
                          label=f'ΔQ≈{delta_Q:.1f}')
        axes[0,0].axvspan(Q_p25, Q_p75, color='blue', alpha=0.10, label=f'IQR [{Q_p25:.1f}, {Q_p75:.1f}]')
        axes[0,0].axvline(Q_top, color='red', lw=1.5, ls='--', label=f'Top Q={Q_top:.1f}')
        axes[0,0].set_xlabel('Quality'); axes[0,0].set_ylabel('Count'); axes[0,0].legend()
        axes[0,0].set_title('FSE Q distribution, median/IQR/ΔQ')
        # a分布
        axes[0,1].hist(res['a'], bins=40, edgecolor='black', alpha=0.7)
        axes[0,1].axvline(a_med, color='green', lw=2, label=f'median ν^(-1)={a_med:.3f}')
        axes[0,1].axvspan(a_p25, a_p75, color='blue', alpha=0.10, label=f'IQR [{a_p25:.3f}, {a_p75:.3f}]')
        axes[0,1].set_xlabel('ν^{-1}'); axes[0,1].set_ylabel('Count'); axes[0,1].legend()
        # Uc分布
        axes[1,0].hist(res['Uc'], bins=40, edgecolor='black', alpha=0.7)
        axes[1,0].axvline(Uc_med, color='green', lw=2, label=f'median U_c={Uc_med:.3f}')
        for x,lab,col in [(8.38,'8.38','red'),(8.46,'8.46','purple'),(8.67,'8.67','blue')]:
            axes[1,0].axvline(x, color=col, ls='--', alpha=0.6, label=lab)
        axes[1,0].axvspan(Uc_p25, Uc_p75, color='blue', alpha=0.10, label=f'IQR [{Uc_p25:.3f}, {Uc_p75:.3f}]')
        axes[1,0].set_xlabel('U_c'); axes[1,0].set_ylabel('Count'); axes[1,0].legend()
        # Q vs Uc
        axes[1,1].scatter(res['Uc'], res['Q'], s=10, alpha=0.6)
        axes[1,1].set_xlabel('U_c'); axes[1,1].set_ylabel('Quality')
        axes[1,1].set_title('Quality vs U_c')

        out_png = os.path.join(os.path.dirname(__file__), 'generalized_bootstrap_fse_allL.png')
        plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    # Quick summary
    print(f"FSE generalized bootstrap: n={len(res)}")
    print(f"Q_median={Q_median:.2f}, robust_sigma≈{Q_sigma_robust:.2f}, equivalence ΔQ≈{delta_Q:.2f}")
    if len(res) > 0:
        print(f"Uc median={Uc_med:.6f}, a median={a_med:.6f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=120)
    p.add_argument('--boots', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--Uc_min', type=float, default=8.30)
    p.add_argument('--Uc_max', type=float, default=8.80)
    p.add_argument('--a_min', type=float, default=0.9)
    p.add_argument('--a_max', type=float, default=1.5)
    p.add_argument('--b_span', type=float, default=0.25)
    p.add_argument('--c_span', type=float, default=0.25)
    p.add_argument('--n_grid', type=int, default=5)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_generalized_bootstrap_fse(n_trials=args.trials, n_boot_ext=args.boots, seed=args.seed,
                                  Uc_min=args.Uc_min, Uc_max=args.Uc_max,
                                  a_min=args.a_min, a_max=args.a_max,
                                  b_span=args.b_span, c_span=args.c_span,
                                  n_grid=args.n_grid) 