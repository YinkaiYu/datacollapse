
import os, sys, re, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# local datacollapse
sys.path.insert(0, '.')
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse_robust, collapse_transform

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def build_combined_from_datascale(data_scale_dir: str, out_csv: str) -> pd.DataFrame:
    rows = []
    # scan files
    patterns = ['**/*.csv', '**/*.txt']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_scale_dir, p), recursive=True))
    def infer_L(path):
        # try L=xx in filename or parent dirs
        m = re.search(r'L\s*=\s*(\d+)', path)
        if m: return int(m.group(1))
        parts = os.path.normpath(path).split(os.sep)
        for seg in parts[::-1]:
            m = re.search(r'L\s*=\s*(\d+)', seg)
            if m: return int(m.group(1))
        # fallback: try _Lxx
        m = re.search(r'_L(\d+)', path)
        if m: return int(m.group(1))
        return None
    def read_table(p):
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_csv(p, sep='\s+', engine='python')
        return df
    def map_columns(df):
        cols = {c.lower(): c for c in df.columns}
        u = cols.get('u')
        y = cols.get('r01') or cols.get('y') or cols.get('r_01')
        s = cols.get('sigma') or cols.get('err') or cols.get('error')
        if not (u and y):
            raise ValueError('Missing U or Y (R01) columns')
        if s is None:
            # approximate sigma if absent
            s = y
            df[s] = np.std(df[y]) * 0.05
        return df[[u, y, s]].rename(columns={u:'U', y:'Y', s:'sigma'})
    for fp in files:
        L = infer_L(fp)
        if L is None: 
            continue
        try:
            d = read_table(fp)
            d = map_columns(d)
            d['L'] = int(L)
            rows.append(d[['L','U','Y','sigma']])
        except Exception:
            continue
    if not rows:
        raise RuntimeError('No usable files found under Data_scale/. Expect files with columns U,Y(R01),sigma and L hinted by filenames/folders.')
    df = pd.concat(rows, ignore_index=True)
    # optional cleaning/sorting
    df = df.sort_values(['L','U']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def load_or_build_combined(data_scale_dir: str, out_csv: str) -> pd.DataFrame:
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    if not os.path.isdir(data_scale_dir):
        raise FileNotFoundError(f'Data_scale directory not found: {data_scale_dir}')
    return build_combined_from_datascale(data_scale_dir, out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-scale', type=str, default='Data_scale')
    args = ap.parse_args()

    # build combined CSV
    combined_csv = 'real_data_combined.csv'
    df = load_or_build_combined(args.data_scale, combined_csv)

    # unified colors
    Ls = sorted(df['L'].unique().tolist())
    cmap = plt.get_cmap('tab10', max(10, len(Ls)))
    L_colors = {int(Lv): cmap(i % 10) for i, Lv in enumerate(Ls)}

    # Raw
    plt.figure(figsize=(6,4))
    for Lv, d in df.groupby('L'):
        c = L_colors[int(Lv)]
        plt.errorbar(d['U'], d['Y'], yerr=d['sigma'], fmt='o-', ms=3, lw=1, alpha=0.9, color=c, label=f'L={int(Lv)}')
    plt.xlabel('U'); plt.ylabel(r'$R_{01}$'); plt.title('Raw data'); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig('collapse_raw.png', dpi=220); plt.close()

    # Data collapse (drop L=7): fit
    DF2 = df[df['L']!=7].copy().reset_index(drop=True)
    data2 = DF2[['L','U','Y']].to_numpy(float)
    sigma2 = DF2['sigma'].to_numpy(float)
    bounds = ((8.0, 9.0), (0.6, 2.0))
    Uc0, a0 = 8.67, 1.20
    (params2, errs2) = fit_data_collapse(data2, sigma2, Uc0, a0, n_knots=10, lam=1e-3,
                                         n_boot=50, random_state=0, bounds=bounds,
                                         optimizer='NM_then_Powell', maxiter=4000, random_restarts=8)
    xC2, YC2 = collapse_transform(data2, params2)
    plt.figure(figsize=(6,4))
    for Lv, g in DF2.assign(x=xC2, Yc=YC2).groupby('L'):
        c = L_colors[int(Lv)]
        plt.errorbar(g['x'], g['Yc'], yerr=g['sigma'], fmt='o', ms=3, alpha=0.9, color=c, label=f'L={int(Lv)}')
    plt.xlabel(r'(U-U_c)\,L^{\,1/\nu}$'); plt.ylabel(r'$R_{01}$')
    plt.title('Data collapse (drop L=7)')
    plt.legend(ncol=2, fontsize=8)
    _txt2 = "U_c="+str(round(params2[0],6))+"±"+str(round(errs2[0],6))+"\nν^(-1)="+str(round(params2[1],6))+"±"+str(round(errs2[1],6))
    plt.text(0.02, 0.98, _txt2, transform=plt.gca().transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    plt.tight_layout(); plt.savefig('collapse_nofse_dropL7.png', dpi=220); plt.close()

    # Data collapse (all L, with finite size correction): robust fit
    DATA = df[['L','U','Y']].to_numpy(float)
    sigma = df['sigma'].to_numpy(float)
    b0 = 0.8; c0 = -1.0
    b_grid = np.linspace(b0-0.3, b0+0.3, 7)
    c_grid = np.linspace(c0-0.3, c0+0.3, 7)
    (params4, errs4) = fit_data_collapse_fse_robust(DATA, sigma, 8.45, 1.20,
                                                    b_grid=b_grid, c_grid=c_grid,
                                                    n_knots=10, lam=1e-3, n_boot=8, random_state=0,
                                                    bounds_Ua=((8.0, 9.0), (0.8, 2.0)),
                                                    normalize=True,
                                                    optimizer='NM_then_Powell', maxiter=4000, random_restarts=2)
    xC4, YC4 = collapse_transform(DATA, params4, normalize=True)
    plt.figure(figsize=(6,4))
    for Lv, g in df.assign(x=xC4, Yc=YC4).groupby('L'):
        c = L_colors[int(Lv)]
        plt.errorbar(g['x'], g['Yc'], yerr=g['sigma'], fmt='o', ms=3, alpha=0.9, color=c, label=f'L={int(Lv)}')
    plt.xlabel(r'(U-U_c)\,L^{\,1/\nu}$'); plt.ylabel(r'$R_{01}$')
    plt.title('Data collapse (all L, with finite size correction)')
    plt.legend(ncol=2, fontsize=8)
    _txt4 = "U_c="+str(round(params4[0],6))+"±"+str(round(errs4[0],6))+"\nν^(-1)="+str(round(params4[1],6))+"±"+str(round(errs4[1],6))+"\n(b,c)=("+str(round(params4[2],6))+","+str(round(params4[3],6))+")"
    plt.text(0.02, 0.98, _txt4, transform=plt.gca().transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))
    plt.tight_layout(); plt.savefig('collapse_fse_allL.png', dpi=220); plt.close()

    print('Done. Figures and parameter annotations saved. Combined CSV at:', combined_csv)

if __name__ == '__main__':
    main()
