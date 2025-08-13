import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, minimize_scalar

# NOTE: Avoid Chinese in plot text per user preference [[memory:5669012]]
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_crossing_on_domain(U_dom, Y1_on_dom, Y2_on_dom, edge_tol_frac=0.02):
    Uo = np.asarray(U_dom, float)
    Y1o = np.asarray(Y1_on_dom, float)
    Y2o = np.asarray(Y2_on_dom, float)

    order = np.argsort(Uo)
    Uo = Uo[order]; Y1o = Y1o[order]; Y2o = Y2o[order]
    eps = 1e-12
    Uo = np.maximum.accumulate(Uo + np.arange(len(Uo))*eps)

    if len(np.unique(Uo)) < 4:
        return np.nan

    s1 = UnivariateSpline(Uo, Y1o, s=max(1.0, len(Uo)*0.01), k=3)
    s2 = UnivariateSpline(Uo, Y2o, s=max(1.0, len(Uo)*0.01), k=3)
    diff = lambda u: float(s1(u) - s2(u))

    umin, umax = float(Uo.min()), float(Uo.max())
    width = umax - umin
    if width <= 0:
        return np.nan

    xs = np.linspace(umin, umax, 300)
    vals = np.array([diff(x) for x in xs])
    sign = np.sign(vals)

    uc = np.nan
    for i in range(len(xs)-1):
        if sign[i] == 0:
            uc = xs[i]
            break
        if sign[i]*sign[i+1] < 0:
            try:
                uc = brentq(diff, xs[i], xs[i+1], maxiter=200)
                break
            except Exception:
                continue

    if not np.isfinite(uc):
        try:
            res = minimize_scalar(lambda u: abs(diff(u)), bounds=(umin, umax), method='bounded')
            if res.success:
                uc = float(res.x)
        except Exception:
            uc = np.nan

    if not np.isfinite(uc):
        return np.nan

    if (uc - umin) < edge_tol_frac*width or (umax - uc) < edge_tol_frac*width:
        return np.nan

    return uc


essential_cols = ["L","U","Y","sigma"]

def compute_common_window(df: pd.DataFrame, n_grid: int = 600, band_width: float = 0.06):
    L_values = sorted(df["L"].unique())
    U_min_common = max(df[df["L"]==L]["U"].min() for L in L_values)
    U_max_common = min(df[df["L"]==L]["U"].max() for L in L_values)
    if U_max_common <= U_min_common:
        raise RuntimeError("No common U range across all L.")
    U_grid = np.linspace(U_min_common, U_max_common, n_grid)

    Ys = []
    for L in L_values:
        sub = df[df["L"]==L]
        U = sub["U"].to_numpy(float)
        Y = sub["Y"].to_numpy(float)
        Ys.append(np.interp(U_grid, U, Y))
    Y_mat = np.vstack(Ys)
    spread = Y_mat.std(axis=0)  # across L

    k_min = int(np.argmin(spread))
    U_center = U_grid[k_min]
    U_lo = max(U_min_common, U_center - band_width/2)
    U_hi = min(U_max_common, U_center + band_width/2)
    if U_hi - U_lo < band_width/4:
        mid = 0.5*(U_min_common + U_max_common)
        U_lo, U_hi = mid - band_width/2, mid + band_width/2
        U_lo = max(U_lo, U_min_common); U_hi = min(U_hi, U_max_common)
    return float(U_lo), float(U_hi), U_grid, spread


def estimate_crossing_uc(csv_path: str, n_boot: int = 200, seed: int = 0,
                          min_overlap: float = 0.05, band_width: float = 0.06,
                          plateau_fraction: float = 0.5):
    """Return plateau-averaged Uc from largest-L_eff pairs.
    plateau_fraction: fraction of pairs (by largest L_eff) used for weighted average (0.3-0.6 recommended).
    """
    df = pd.read_csv(csv_path)
    for col in essential_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in data.")

    U_lo, U_hi, U_grid, spread = compute_common_window(df, n_grid=600, band_width=band_width)

    L_values = sorted(df["L"].unique())
    perL = {}
    for L in L_values:
        sub = df[df["L"] == L].copy()
        perL[L] = {
            'U': sub['U'].to_numpy(float),
            'Y': sub['Y'].to_numpy(float),
            'sigma': sub['sigma'].to_numpy(float)
        }

    rng = np.random.default_rng(seed)

    pairs = []
    for i in range(len(L_values)):
        for j in range(i+1, len(L_values)):
            L1 = int(L_values[i]); L2 = int(L_values[j])
            U1 = perL[L1]['U']; Y1 = perL[L1]['Y']; s1 = perL[L1]['sigma']
            U2 = perL[L2]['U']; Y2 = perL[L2]['Y']; s2 = perL[L2]['sigma']

            umin_pair = max(U1.min(), U2.min())
            umax_pair = min(U1.max(), U2.max())
            umin = max(umin_pair, U_lo)
            umax = min(umax_pair, U_hi)
            if not np.isfinite(umin) or not np.isfinite(umax) or umax - umin < min_overlap:
                continue

            U_dom = np.linspace(umin, umax, 240)
            boot_vals = []
            for _ in range(max(1, n_boot)):
                Y1b = Y1 + rng.normal(0.0, s1)
                Y2b = Y2 + rng.normal(0.0, s2)
                Y1b_dom = np.interp(U_dom, U1, Y1b)
                Y2b_dom = np.interp(U_dom, U2, Y2b)
                uc = find_crossing_on_domain(U_dom, Y1b_dom, Y2b_dom)
                if np.isfinite(uc):
                    boot_vals.append(uc)

            if len(boot_vals) == 0:
                continue

            uc_mean = float(np.mean(boot_vals))
            uc_std = float(np.std(boot_vals, ddof=1))
            L_eff = 0.5*(L1 + L2)
            pairs.append((L1, L2, L_eff, 1.0/L_eff, uc_mean, uc_std))

    pairs = [p for p in pairs if np.isfinite(p[4]) and np.isfinite(p[5]) and p[5] > 0]
    if not pairs:
        raise RuntimeError("No valid crossings after window/overlap filtering.")

    arr = np.array(pairs, float)
    L_eff = arr[:,2]
    inv_Leff = arr[:,3]
    uc_vals = arr[:,4]
    uc_errs = arr[:,5]

    # Select plateau by largest L_eff
    n = len(arr)
    k = max(3, int(np.ceil(plateau_fraction * n)))
    idx = np.argsort(-L_eff)[:k]
    uc_sel = uc_vals[idx]
    err_sel = uc_errs[idx]
    w = 1.0/np.maximum(1e-8, err_sel**2)
    Uc_plateau = float(np.sum(w*uc_sel) / np.sum(w))
    Uc_plateau_err = float(np.sqrt(1.0/np.sum(w)))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    ax = axes[0]
    ax.errorbar(inv_Leff, uc_vals, yerr=uc_errs, fmt='o', ms=6, capsize=3, label='Crossings')
    ax.errorbar(inv_Leff[idx], uc_vals[idx], yerr=uc_errs[idx], fmt='o', ms=7, capsize=3,
                color='tab:orange', label='Plateau set')
    ax.axhline(Uc_plateau, color='tab:green', lw=2, label=f'Plateau avg = {Uc_plateau:.4f}±{Uc_plateau_err:.4f}')
    ax.set_xlabel('1 / L_eff')
    ax.set_ylabel(r'$U_c$ from crossings')
    ax.set_title(r'Crossing-based $U_c$ (plateau averaging)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(U_grid, spread, '-', lw=1.8)
    ax2.axvspan(U_lo, U_hi, color='orange', alpha=0.2, label='window')
    ax2.set_xlabel('U'); ax2.set_ylabel('Across-L std of $R_{01}$')
    ax2.set_title('Window by minimal across-L spread')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    out_png = os.path.join(os.path.dirname(__file__), 'crossing_uc_extrapolation.png')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"Plateau-averaged U_c (largest L_eff set): {Uc_plateau:.6f} ± {Uc_plateau_err:.6f}")
    print(f"Used pairs total={n}, plateau_k={k}; window: [{U_lo:.3f}, {U_hi:.3f}]")
    print(f"Saved plot: {os.path.basename(out_png)}")
    return Uc_plateau, Uc_plateau_err, pairs


def main():
    data_csv = os.path.join(os.path.dirname(__file__), 'real_data_combined.csv')
    estimate_crossing_uc(data_csv, n_boot=120, min_overlap=0.03, band_width=0.10, plateau_fraction=0.5)

if __name__ == '__main__':
    main() 