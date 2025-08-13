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
rcParams['figure.figsize'] = (8, 5)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, 'docs', 'images')
DEFAULT_CSV = os.path.join(ROOT, 'real_data_test', 'real_data_combined.csv')


def load_csv(path: str):
    df = pd.read_csv(path)
    need = {'L','U','Y'}
    if not need.issubset(df.columns):
        raise ValueError('CSV must contain columns L,U,Y[,sigma]')
    if 'sigma' not in df.columns:
        df['sigma'] = np.nan
    df = df.sort_values(['L','U']).reset_index(drop=True)
    return df


def plot_raw(df: pd.DataFrame, save_path: str):
    plt.figure()
    Ls = sorted(df['L'].unique())
    cmap = plt.get_cmap('tab10', max(10, len(Ls)))
    for i, Lv in enumerate(Ls):
        g = df[df['L']==Lv]
        plt.errorbar(g['U'], g['Y'], yerr=g['sigma'], fmt='o-', ms=3, lw=1, alpha=0.9,
                     color=cmap(i%10), label=f'L={int(Lv)}')
    plt.xlabel('U'); plt.ylabel('R'); plt.title('Raw data'); plt.legend(ncol=2, fontsize=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=240); plt.close()


def plot_collapse(df: pd.DataFrame, x: np.ndarray, Yc: np.ndarray, save_path: str, title: str, normalize=False):
    plt.figure()
    Ls = sorted(df['L'].unique())
    cmap = plt.get_cmap('tab10', max(10, len(Ls)))
    for i, Lv in enumerate(Ls):
        m = (df['L'].to_numpy()==Lv)
        plt.errorbar(x[m], Yc[m], yerr=df['sigma'].to_numpy()[m], fmt='o', ms=3, alpha=0.9,
                     color=cmap(i%10), label=f'L={int(Lv)}')
    plt.xlabel(r'$(U-U_c)\,L^{1/\nu}$')
    plt.ylabel('R' if not normalize else r'$R/(1+bL^c)$')
    plt.title(title); plt.legend(ncol=2, fontsize=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=240); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=DEFAULT_CSV)
    args = ap.parse_args()

    df = load_csv(args.csv)
    print('Loaded:', args.csv, 'points:', len(df))

    # Raw plot
    plot_raw(df, os.path.join(IMAGES_DIR, 'raw_data.png'))

    # No finite-size correction (drop L=7)
    DF2 = df[df['L']!=7].copy().reset_index(drop=True)
    data2 = DF2[['L','U','Y']].to_numpy(float)
    sigma2 = DF2['sigma'].to_numpy(float)
    bounds = ((8.0, 9.0), (0.6, 2.0))
    Uc0, a0 = 8.67, 1.20
    print('Fitting without finite-size correction (drop L=7)...')
    (params2, errs2) = fit_data_collapse(data2, sigma2, Uc0, a0,
                                         n_knots=10, lam=1e-3, n_boot=10, random_state=0,
                                         bounds=bounds,
                                         optimizer='NM_then_Powell', maxiter=4000, random_restarts=8)
    x2, Yc2 = collapse_transform(data2, params2)
    plot_collapse(DF2, x2, Yc2, os.path.join(IMAGES_DIR, 'nofse_collapse.png'),
                  title='Collapse (drop L=7)', normalize=False)
    print('No-FSC params:', params2, 'errs:', errs2)

    # With finite-size correction (robust, all L, normalize)
    DATA = df[['L','U','Y']].to_numpy(float)
    SIG = df['sigma'].to_numpy(float)
    b0, c0 = 0.8, -1.0
    b_grid = np.linspace(b0-0.3, b0+0.3, 7)
    c_grid = np.linspace(c0-0.3, c0+0.3, 7)
    print('Fitting with finite-size correction (robust, all L, normalize)...')
    (params4, errs4) = fit_data_collapse_fse_robust(DATA, SIG, 8.45, 1.20,
                                                    b_grid=b_grid, c_grid=c_grid,
                                                    n_knots=10, lam=1e-3, n_boot=6, random_state=0,
                                                    bounds_Ua=((8.0, 9.0), (0.8, 2.0)),
                                                    normalize=True,
                                                    optimizer='NM_then_Powell', maxiter=4000, random_restarts=2)
    x4, Yc4 = collapse_transform(DATA, params4, normalize=True)
    plot_collapse(df, x4, Yc4, os.path.join(IMAGES_DIR, 'fse_collapse.png'),
                  title='Collapse (all L, finite-size correction)', normalize=True)
    print('FSC params:', params4, 'errs:', errs4)

if __name__ == '__main__':
    main() 