#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from datacollapse import (
    fit_data_collapse,
    fit_data_collapse_fse_robust,
    collapse_transform,
)

rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.figsize'] = (10, 6)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMAGES_DIR = os.path.join(ROOT, 'docs', 'images')
DEFAULT_CSV_PATH = os.path.join(HERE, 'sample_data.csv')


def load_sample_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    if not set(['L','U','Y']).issubset(df.columns):
        raise ValueError('CSV must contain columns L,U,Y[,sigma]')
    L = df['L'].to_numpy(float)
    U = df['U'].to_numpy(float)
    Y = df['Y'].to_numpy(float)
    data = np.column_stack([L,U,Y])
    if 'sigma' in df.columns:
        err = df['sigma'].to_numpy(float)
    else:
        err = None
    return data, err, df


def compute_window(df: pd.DataFrame, width: float = 0.10):
    U_vals = df['U'].to_numpy(float)
    U_min, U_max = float(U_vals.min()), float(U_vals.max())
    if U_max - U_min <= width:
        return U_min, U_max
    centers = np.linspace(U_min + 0.5*width, U_max - 0.5*width, 60)
    best_c, best_score = centers[0], np.inf
    for c in centers:
        lo, hi = c - 0.5*width, c + 0.5*width
        sub = df[(df['U'] >= lo) & (df['U'] <= hi)]
        scores = []
        for _, g in sub.groupby('L'):
            if len(g) >= 3:
                scores.append(np.nanstd(g['Y'].to_numpy(float)))
        if len(scores) >= 2:
            s = float(np.nanmean(scores))
            if s < best_score:
                best_score, best_c = s, c
    return best_c - 0.5*width, best_c + 0.5*width


def plot_raw(df: pd.DataFrame, save_path: str):
    plt.figure()
    L_vals = sorted(df['L'].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(L_vals)))
    for i, Lv in enumerate(L_vals):
        sub = df[df['L'] == Lv]
        plt.plot(sub['U'].to_numpy(), sub['Y'].to_numpy(), 'o-',
                 color=cmap[i], label=f'L = {int(Lv)}', linewidth=2, markersize=6, alpha=0.9)
    plt.xlabel('U (Control Parameter)')
    plt.ylabel('R (dimensionless observable)')
    plt.title('Raw Data (before collapse)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='System Size', frameon=True, fancybox=True, shadow=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def collapse_quality(x: np.ndarray, Yc: np.ndarray, L: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xr = float(np.nanmax(x) - np.nanmin(x))
    rngs = []
    for Lv in np.unique(L):
        m = (L == Lv)
        if np.sum(m) >= 3:
            yr = float(np.nanmax(Yc[m]) - np.nanmin(Yc[m]))
            if np.isfinite(yr):
                rngs.append(yr)
    if not rngs:
        return 0.0
    return xr / (np.mean(rngs) + 1e-12)


def plot_collapse(data: np.ndarray, params, save_path: str, *, normalize=False, L_ref='geom'):
    plt.figure()
    x, Yc = collapse_transform(data, params, normalize=normalize, L_ref=L_ref)
    L = data[:,0]
    L_vals = sorted(np.unique(L))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(L_vals)))
    for i, Lv in enumerate(L_vals):
        mask = (L == Lv)
        plt.plot(x[mask], Yc[mask], 'o', color=cmap[i], label=f'L = {int(Lv)}', alpha=0.9)
    plt.xlabel(r'$(U - U_c)\, L^{1/\nu}$')
    ylabel = 'R' if not normalize else r'$R/(1+bL^c)$'
    plt.ylabel(ylabel)
    title = 'Collapse (without finite-size correction)' if not normalize else 'Collapse (with finite-size correction)'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title='System Size', frameon=True, fancybox=True, shadow=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def best_nofse_params(data_win: np.ndarray, err: np.ndarray | None):
    U = data_win[:,1]
    u_min, u_max = float(np.min(U)), float(np.max(U))
    Uc0_list = np.linspace(u_min - 0.05, u_max + 0.05, 9)
    a0_list = np.linspace(0.6, 1.8, 7)
    bounds = ((u_min - 0.1, u_max + 0.1), (0.3, 2.2))
    best = None
    for Uc0 in Uc0_list:
        for a0 in a0_list:
            try:
                params, _ = fit_data_collapse(
                    data_win, err, Uc0, a0,
                    n_knots=12, lam=1e-3, n_boot=0, random_state=0,
                    bounds=bounds,
                    optimizer='NM_then_Powell', random_restarts=12, maxiter=5000,
                )
                x, Yc = collapse_transform(data_win, params)
                q = collapse_quality(x, Yc, data_win[:,0])
                if (best is None) or (q > best[0]):
                    best = (q, params)
            except Exception:
                continue
    if best is None:
        params, _ = fit_data_collapse(
            data_win, err, (u_min+u_max)/2, 1.0,
            n_knots=12, lam=1e-3, n_boot=0, random_state=0,
            bounds=bounds,
            optimizer='NM_then_Powell', random_restarts=8, maxiter=4000,
        )
        return params
    return best[1]


def main():
    parser = argparse.ArgumentParser(description='Build README collapse images from CSV')
    parser.add_argument('--csv', type=str, default=DEFAULT_CSV_PATH, help='Path to CSV with columns L,U,Y[,sigma]')
    parser.add_argument('--width', type=float, default=0.10, help='U-window width for near-critical selection')
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    width = float(args.width)

    print('Loading data:', csv_path)
    data, err, df = load_sample_csv(csv_path)

    raw_png = os.path.join(IMAGES_DIR, 'raw_data.png')
    plot_raw(df, raw_png)
    print('Saved:', raw_png)

    lo, hi = compute_window(df, width=width)
    df_win = df[(df['U'] >= lo) & (df['U'] <= hi)].copy()
    data_win = np.column_stack([df_win['L'].to_numpy(float), df_win['U'].to_numpy(float), df_win['Y'].to_numpy(float)])
    err_win = df_win['sigma'].to_numpy(float) if 'sigma' in df_win.columns else None

    print(f'Fitting without finite-size correction on window [{lo:.4f}, {hi:.4f}] ...')
    params_nofse = best_nofse_params(data_win, err_win)
    nofse_png = os.path.join(IMAGES_DIR, 'nofse_collapse.png')
    plot_collapse(data_win, params_nofse, nofse_png, normalize=False)
    print('Saved:', nofse_png, 'params=', params_nofse)

    print('Fitting with finite-size correction (robust) ...')
    b_grid = np.linspace(0.2, 1.2, 6)
    c_grid = np.linspace(-1.5, -0.3, 7)
    u_min, u_max = float(np.min(data_win[:,1])), float(np.max(data_win[:,1]))
    bounds_Ua = ((u_min - 0.05, u_max + 0.05), (0.3, 2.2))
    (params_fse, _) = fit_data_collapse_fse_robust(
        data_win, err_win, U_c_0=(u_min+u_max)/2, a_0=1.0,
        b_grid=b_grid, c_grid=c_grid,
        n_knots=12, lam=1e-3, n_boot=0, random_state=0,
        bounds_Ua=bounds_Ua,
        normalize=True, L_ref='geom',
        optimizer='NM_then_Powell', random_restarts=5, maxiter=5000,
    )
    fse_png = os.path.join(IMAGES_DIR, 'fse_collapse.png')
    plot_collapse(data_win, params_fse, fse_png, normalize=True, L_ref='geom')
    print('Saved:', fse_png, 'params=', params_fse)

if __name__ == '__main__':
    main() 