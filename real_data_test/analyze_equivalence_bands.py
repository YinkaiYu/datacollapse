import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def robust_sigma(arr):
    arr = np.asarray(arr, float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad if mad > 0 else float(np.std(arr))


def main():
    base = os.path.dirname(__file__)
    f_nofse = os.path.join(base, 'generalized_bootstrap_nofse_dropL7.csv')
    f_fse   = os.path.join(base, 'generalized_bootstrap_fse_allL.csv')

    if not (os.path.exists(f_nofse) and os.path.exists(f_fse)):
        print('Missing bootstrap CSV files.'); return

    dn = pd.read_csv(f_nofse)
    df = pd.read_csv(f_fse)

    dn = dn.replace([np.inf, -np.inf], np.nan).dropna(subset=['Q','Uc','a'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Q','Uc','a'])

    # Per-method equivalence bands
    Qn_med = float(np.median(dn['Q'])); sig_n = robust_sigma(dn['Q']); dQ_n = 2.0*sig_n
    Qf_med = float(np.median(df['Q'])); sig_f = robust_sigma(df['Q']); dQ_f = 2.0*sig_f

    # Per-method Uc,a centers
    Ucn_med = float(np.median(dn['Uc'])); an_med = float(np.median(dn['a']))
    Ucf_med = float(np.median(df['Uc'])); af_med = float(np.median(df['a']))

    # Fractions in Uc bins
    bins = [(8.30,8.50),(8.50,8.60),(8.60,8.75)]
    frac_n = []; frac_f = []
    for lo,hi in bins:
        frac_n.append(float(((dn['Uc']>=lo)&(dn['Uc']<hi)).mean()))
        frac_f.append(float(((df['Uc']>=lo)&(df['Uc']<hi)).mean()))

    # Cross-method "equivalence" using narrower band (intersection)
    dQ_cross = min(dQ_n, dQ_f)
    # Fractions whose Q within their method median ± dQ_cross/2 (centered per-method)
    frac_eq_n = float(((dn['Q']>=Qn_med-dQ_cross/2)&(dn['Q']<=Qn_med+dQ_cross/2)).mean())
    frac_eq_f = float(((df['Q']>=Qf_med-dQ_cross/2)&(df['Q']<=Qf_med+dQ_cross/2)).mean())

    # Save summary CSV
    out_csv = os.path.join(base, 'equivalence_summary.csv')
    pd.DataFrame({
        'metric':['Q_median','robust_sigma','DeltaQ','Uc_median','a_median','frac_eq'],
        'NoFSE':[Qn_med, sig_n, dQ_n, Ucn_med, an_med, frac_eq_n],
        'FSE':[Qf_med, sig_f, dQ_f, Ucf_med, af_med, frac_eq_f]
    }).to_csv(out_csv, index=False)

    # Comparative plot
    fig, axes = plt.subplots(2,2, figsize=(12,9))
    # Uc hist
    axes[0,0].hist(dn['Uc'], bins=40, alpha=0.6, label='No-FSE', edgecolor='black')
    axes[0,0].hist(df['Uc'], bins=40, alpha=0.6, label='FSE', edgecolor='black')
    for x,lab,col in [(8.38,'8.38','red'),(8.46,'8.46','purple'),(8.67,'8.67','green')]:
        axes[0,0].axvline(x, color=col, ls='--', alpha=0.7, label=lab)
    axes[0,0].set_xlabel('U_c'); axes[0,0].set_ylabel('Count'); axes[0,0].legend()
    axes[0,0].set_title('Uc distributions (No-FSE vs FSE)')

    # Q hist with bands
    axes[0,1].hist(dn['Q'], bins=40, alpha=0.6, label=f'No-FSE (ΔQ≈{dQ_n:.0f})', edgecolor='black')
    axes[0,1].hist(df['Q'], bins=40, alpha=0.6, label=f'FSE (ΔQ≈{dQ_f:.0f})', edgecolor='black')
    axes[0,1].axvline(Qn_med, color='blue', lw=2, alpha=0.7, label='No-FSE median')
    axes[0,1].axvline(Qf_med, color='orange', lw=2, alpha=0.7, label='FSE median')
    axes[0,1].set_xlabel('Quality'); axes[0,1].set_ylabel('Count'); axes[0,1].legend()
    axes[0,1].set_title('Q distributions and medians')

    # Bin fractions bar
    x = np.arange(len(bins)); w=0.35
    axes[1,0].bar(x-w/2, frac_n, width=w, label='No-FSE')
    axes[1,0].bar(x+w/2, frac_f, width=w, label='FSE')
    axes[1,0].set_xticks(x, [f'[{lo:.2f},{hi:.2f})' for lo,hi in bins])
    axes[1,0].set_ylabel('Fraction'); axes[1,0].set_title('Uc bin fractions'); axes[1,0].legend()

    # Q vs Uc scatter
    axes[1,1].scatter(dn['Uc'], dn['Q'], s=10, alpha=0.5, label='No-FSE')
    axes[1,1].scatter(df['Uc'], df['Q'], s=10, alpha=0.5, label='FSE')
    axes[1,1].set_xlabel('U_c'); axes[1,1].set_ylabel('Quality'); axes[1,1].legend()
    axes[1,1].set_title('Q vs Uc')

    out_png = os.path.join(base, 'equivalence_comparison.png')
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

    print('Saved:', os.path.basename(out_csv), os.path.basename(out_png))

if __name__ == '__main__':
    main() 