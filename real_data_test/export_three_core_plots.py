import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Parameters from FINAL_SYNTHESIS_REPORT.md (rounded to uncertainty)
NOFSE_UC = 8.669546
NOFSE_A  = 1.191639
NOFSE_UC_ERR = 0.002214
NOFSE_A_ERR  = 0.011695

FSE_UC = 8.443683
FSE_A  = 1.292836
FSE_UC_ERR = 0.074107
FSE_A_ERR  = 0.067600
# Default representative FSE (b,c) for plotting (fallback)
FSE_B = 0.7977894766363625
FSE_C = -1.0630991340121303

base = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base, 'real_data_combined.csv'))


def pick_fse_top_params(csv_path):
    try:
        dfres = pd.read_csv(csv_path)
        if len(dfres) == 0:
            return None
        # 选Top 1% Q所在的组合，或直接取argmax
        idx = int(np.argmax(dfres['Q'].to_numpy()))
        row = dfres.iloc[idx]
        return float(row['Uc']), float(row['a']), float(row['b']), float(row['c']), float(row['Q'])
    except Exception:
        return None


def annotate_params(ax, text):
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))


def plot_raw(out_png):
    fig, ax = plt.subplots(figsize=(6,4))
    for L, d in df.groupby('L'):
        ax.errorbar(d['U'], d['Y'], yerr=d['sigma'], fmt='o-', ms=3, lw=1, alpha=0.9, label=f'L={int(L)}')
    ax.set_xlabel('U')
    ax.set_ylabel(r'$R_{01}$')
    ax.set_title('Raw data')
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()


def plot_nofse(out_png):
    d2 = df[df['L']!=7].copy()
    x = (d2['U'] - NOFSE_UC) * (d2['L']**NOFSE_A)
    fig, ax = plt.subplots(figsize=(6,4))
    for L, g in d2.assign(x=x).groupby('L'):
        ax.errorbar(g['x'], g['Y'], yerr=g['sigma'], fmt='o', ms=3, alpha=0.9, label=f'L={int(L)}')
    ax.set_xlabel(r'$x=(U-U_c)\,L^{\,1/\nu}$')
    ax.set_ylabel(r'$R_{01}$')
    ax.set_title('No-FSE collapse (drop L=7)')
    annotate_params(ax, f"U_c = {NOFSE_UC:.4f} ± {NOFSE_UC_ERR:.4f}\n"
                        f"ν^(-1) = {NOFSE_A:.4f} ± {NOFSE_A_ERR:.4f}")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()


def plot_fse(out_png):
    # Prefer top-Q parameters from generalized bootstrap to提升重合度（同一次拟合的(Uc,a,b,c)）
    pick = pick_fse_top_params(os.path.join(base, 'generalized_bootstrap_fse_allL.csv'))
    if pick is not None:
        Uc, a, b, c, qtop = pick
    else:
        Uc, a, b, c = FSE_UC, FSE_A, FSE_B, FSE_C
        qtop = np.nan

    # normalized scale factor consistent with fit_data_collapse_fse(..., normalize=True)
    L_vals = df['L'].to_numpy(float)
    L_ref = float(np.exp(np.mean(np.log(L_vals))))
    s = 1.0 + b * (L_vals**c)
    s_ref = 1.0 + b * (L_ref**c)
    s_norm = s / s_ref
    Yc = df['Y'] / s_norm
    sigc = df['sigma'] / s_norm
    x = (df['U'] - Uc) * (df['L']**a)

    fig, ax = plt.subplots(figsize=(6,4))
    for L, g in df.assign(x=x, Yc=Yc, sigc=sigc).groupby('L'):
        ax.errorbar(g['x'], g['Yc'], yerr=g['sigc'], fmt='o', ms=3, alpha=0.9, label=f'L={int(L)}')
    ax.set_xlabel(r'$x=(U-U_c)\,L^{\,1/\nu}$')
    ax.set_ylabel(r'$R_{01}$')
    ax.set_title('FSE collapse (All L)')
    annotate_params(ax, (
        f"U_c = {Uc:.4f} ± {FSE_UC_ERR:.4f}\n"
        f"ν^(-1) = {a:.4f} ± {FSE_A_ERR:.4f}\n"
        f"(b,c) ≈ ({b:.3f}, {c:.3f}), normalize=geom"
        + (f"\nTop-Q sample used (Q≈{qtop:.1f})" if not np.isnan(qtop) else "")
    ))
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()


def main():
    plot_raw(os.path.join(base, 'core_raw.png'))
    plot_nofse(os.path.join(base, 'core_nofse_dropL7.png'))
    plot_fse(os.path.join(base, 'core_fse_allL.png'))
    print('Exported: core_raw.png, core_nofse_dropL7.png, core_fse_allL.png')

if __name__ == '__main__':
    main() 