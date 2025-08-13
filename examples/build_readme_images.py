#!/usr/bin/env python3
import os
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
CSV_PATH = os.path.join(HERE, 'sample_data.csv')


def load_sample_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    # expected columns: L, U, Y, sigma
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


def plot_collapse(data: np.ndarray, params, save_path: str, *, normalize=False, L_ref='geom'):
    plt.figure()
    x, Yc = collapse_transform(data, params, normalize=normalize, L_ref=L_ref)
    L = data[:,0]
    # sort by L for consistent color mapping
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


def main():
    print('Loading sample data:', CSV_PATH)
    data, err, df = load_sample_csv(CSV_PATH)

    # Plot raw
    raw_png = os.path.join(IMAGES_DIR, 'raw_data.png')
    plot_raw(df, raw_png)
    print('Saved:', raw_png)

    # Fit without finite-size correction
    print('Fitting without finite-size correction...')
    (params_nofse, errs_nofse) = fit_data_collapse(
        data, err, U_c_0=8.6, a_0=1.0,
        n_knots=12, lam=1e-3, n_boot=0, random_state=0,
        bounds=((np.min(data[:,1])-0.2, np.max(data[:,1])+0.2), (0.3, 2.5)),
        optimizer='NM_then_Powell', random_restarts=8, maxiter=4000,
    )
    nofse_png = os.path.join(IMAGES_DIR, 'nofse_collapse.png')
    plot_collapse(data, params_nofse, nofse_png, normalize=False)
    print('Saved:', nofse_png, 'params=', params_nofse)

    # Fit with finite-size correction (robust variant)
    print('Fitting with finite-size correction (robust)...')
    b_grid = np.linspace(0.2, 1.2, 6)
    c_grid = np.linspace(-1.5, -0.3, 7)
    u_min, u_max = float(np.min(data[:,1])), float(np.max(data[:,1]))
    bounds_Ua = ((u_min - 0.2, u_max + 0.2), (0.3, 2.5))
    (params_fse, errs_fse) = fit_data_collapse_fse_robust(
        data, err, U_c_0=8.6, a_0=1.0,
        b_grid=b_grid, c_grid=c_grid,
        n_knots=12, lam=1e-3, n_boot=0, random_state=0,
        bounds_Ua=bounds_Ua,
        normalize=True, L_ref='geom',
        optimizer='NM_then_Powell', random_restarts=3, maxiter=4000,
    )
    fse_png = os.path.join(IMAGES_DIR, 'fse_collapse.png')
    plot_collapse(data, params_fse, fse_png, normalize=True, L_ref='geom')
    print('Saved:', fse_png, 'params=', params_fse)

if __name__ == '__main__':
    main() 