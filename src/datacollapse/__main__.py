import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .datacollapse import (
    fit_data_collapse,
    fit_data_collapse_fse,
    fit_data_collapse_fse_robust,
    collapse_transform,
)


def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    if not {'L','U','Y'}.issubset(df.columns):
        raise ValueError('CSV must contain columns L,U,Y[,sigma]')
    if 'sigma' not in df.columns:
        df['sigma'] = np.nan
    df = df.sort_values(['L','U']).reset_index(False)
    data = df[['L','U','Y']].to_numpy(float)
    err  = df['sigma'].to_numpy(float)
    return df, data, err


def plot_before(df: pd.DataFrame, out_png: str):
    plt.figure()
    for L in sorted(df['L'].unique()):
        g = df[df['L']==L]
        line, = plt.plot(g['U'], g['Y'], '-', lw=1.1, label=f'L={int(L)}')
        plt.errorbar(g['U'], g['Y'], yerr=g['sigma'], fmt='o', ms=3, capsize=2, elinewidth=1, color=line.get_color())
    plt.xlabel('U'); plt.ylabel('R'); plt.title('Raw curves')
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_after(df: pd.DataFrame, x: np.ndarray, Yc: np.ndarray, out_png: str, *, normalize=False):
    plt.figure()
    for L in sorted(df['L'].unique()):
        m = (df['L'].to_numpy()==L)
        plt.errorbar(x[m], Yc[m], yerr=df['sigma'].to_numpy()[m], fmt='o', ms=3, capsize=2, elinewidth=1, label=f'L={int(L)}')
    plt.xlabel(r'$(U-U_c)\,L^{1/\nu}$')
    plt.ylabel('R' if not normalize else r'$R/(1+bL^c)$')
    title = 'Collapse (without finite-size correction)' if not normalize else 'Collapse (with finite-size correction, normalized)'
    plt.title(title)
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser(description='datacollapse CLI')
    ap.add_argument('--csv', type=str, required=True, help='Path to CSV with columns L,U,Y[,sigma]')
    ap.add_argument('--mode', type=str, default='fse-robust', choices=['nofse','fse','fse-robust'])
    ap.add_argument('--outdir', type=str, default='.')
    ap.add_argument('--normalize', action='store_true', help='Use normalized finite-size correction')
    ap.add_argument('--Lref', type=str, default='geom', help="Reference L for normalization: 'geom' or number")
    ap.add_argument('--n_knots', type=int, default=10)
    ap.add_argument('--lam', type=float, default=1e-3)
    ap.add_argument('--n_boot', type=int, default=4)
    ap.add_argument('--optimizer', type=str, default='NM_then_Powell')
    ap.add_argument('--maxiter', type=int, default=4000)
    ap.add_argument('--random_restarts', type=int, default=2)
    args = ap.parse_args()

    df, data, err = load_csv(args.csv)

    # before plot
    plot_before(df, os.path.join(args.outdir, 'before.png'))

    params = None; errs = None; normalize = False

    if args.mode == 'nofse':
        params, errs = fit_data_collapse(
            data, err, 8.67, 1.20,
            n_knots=args.n_knots, lam=args.lam, n_boot=args.n_boot, random_state=0,
            bounds=((min(df['U'])-0.2, max(df['U'])+0.2), (0.3, 2.5)),
            optimizer=args.optimizer, maxiter=args.maxiter, random_restarts=args.random_restarts
        )
        x, Yc = collapse_transform(data, params)
        plot_after(df, x, Yc, os.path.join(args.outdir, 'after.png'), normalize=False)

    elif args.mode == 'fse':
        # example initial guess
        params, errs = fit_data_collapse_fse(
            data, err, 8.45, 1.20, 0.8, -1.0,
            n_knots=args.n_knots, lam=args.lam, n_boot=args.n_boot, random_state=0,
            bounds=((min(df['U'])-0.2, max(df['U'])+0.2), (0.3, 2.5), (0.0, 3.0), (-2.0, -0.05)),
            normalize=args.normalize, L_ref=args.Lref,
            optimizer=args.optimizer, maxiter=args.maxiter, random_restarts=args.random_restarts
        )
        x, Yc = collapse_transform(data, params, normalize=args.normalize, L_ref=args.Lref)
        plot_after(df, x, Yc, os.path.join(args.outdir, 'after.png'), normalize=args.normalize)

    else:  # fse-robust
        params, errs = fit_data_collapse_fse_robust(
            data, err, 8.45, 1.20,
            b_grid=np.linspace(0.5, 1.1, 7), c_grid=np.linspace(-1.3, -0.5, 7),
            n_knots=args.n_knots, lam=args.lam, n_boot=args.n_boot, random_state=0,
            bounds_Ua=((min(df['U'])-0.2, max(df['U'])+0.2), (0.3, 2.5)),
            normalize=True, L_ref=args.Lref,
            optimizer=args.optimizer, maxiter=args.maxiter, random_restarts=args.random_restarts
        )
        x, Yc = collapse_transform(data, params, normalize=True, L_ref=args.Lref)
        plot_after(df, x, Yc, os.path.join(args.outdir, 'after.png'), normalize=True)

    print('Params:', params, '+/-', errs)

if __name__ == '__main__':
    main() 