import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# import from installed/editable package if available; fallback to src
try:
	from datacollapse import fit_data_collapse, fit_data_collapse_fse_robust, collapse_transform
except Exception:
	ROOT = os.path.dirname(os.path.dirname(__file__))
	sys.path.insert(0, os.path.join(ROOT, 'src'))
	from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse_robust, collapse_transform

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, 'sample_data.csv')


def main():
	# Load real sample data
	df = pd.read_csv(CSV)
	data = df[['L','U','Y']].to_numpy(float)
	err = df['sigma'].to_numpy(float) if 'sigma' in df.columns else None

	# BEFORE
	plt.figure(figsize=(6,4))
	for L in sorted(df['L'].unique()):
		sub = df[df['L']==L]
		line, = plt.plot(sub['U'], sub['Y'], '-', lw=1.1, label=f'L={int(L)}')
		plt.errorbar(sub['U'], sub['Y'], yerr=sub['sigma'], fmt='o', ms=3, capsize=2, elinewidth=1, color=line.get_color())
	plt.xlabel('U'); plt.ylabel('R'); plt.title('Raw curves')
	plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
	plt.tight_layout(); plt.savefig(os.path.join(HERE, 'minimal_before.png'), dpi=200); plt.close()

	# WITHOUT finite-size correction (e.g., drop L=7 if present)
	DF2 = df[df['L']!=7].copy().reset_index(drop=True) if (df['L']==7).any() else df.copy()
	data2 = DF2[['L','U','Y']].to_numpy(float)
	err2 = DF2['sigma'].to_numpy(float) if 'sigma' in DF2.columns else None
	(params2, errs2) = fit_data_collapse(
		data2, err2, 8.67, 1.20,
		n_knots=10, lam=1e-3, n_boot=6, random_state=0,
		bounds=((8.0, 9.0), (0.6, 2.0)), optimizer='NM_then_Powell', random_restarts=6
	)
	x2, Yc2 = collapse_transform(data2, params2)
	plt.figure(figsize=(6,4))
	for L in sorted(DF2['L'].unique()):
		m = (DF2['L'].to_numpy()==L)
		plt.errorbar(x2[m], Yc2[m], yerr=DF2['sigma'].to_numpy()[m], fmt='o', ms=3, alpha=0.9, label=f'L={int(L)}')
	plt.xlabel(r'$(U-U_c)\,L^{1/\nu}$'); plt.ylabel('R'); plt.title('Collapse (without finite-size correction)')
	plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
	plt.tight_layout(); plt.savefig(os.path.join(HERE, 'minimal_nofse.png'), dpi=200); plt.close()

	# WITH finite-size correction (robust, normalized)
	b0, c0 = 0.8, -1.0
	b_grid = np.linspace(b0-0.3, b0+0.3, 7)
	c_grid = np.linspace(c0-0.3, c0+0.3, 7)
	(params4, errs4) = fit_data_collapse_fse_robust(
		data, err, 8.45, 1.20,
		b_grid=b_grid, c_grid=c_grid,
		n_knots=10, lam=1e-3, n_boot=4, random_state=0,
		bounds_Ua=((8.0, 9.0), (0.8, 2.0)), normalize=True,
		optimizer='NM_then_Powell', maxiter=4000, random_restarts=2
	)
	x4, Yc4 = collapse_transform(data, params4, normalize=True)
	plt.figure(figsize=(6,4))
	for L in sorted(df['L'].unique()):
		m = (df['L'].to_numpy()==L)
		plt.errorbar(x4[m], Yc4[m], yerr=df['sigma'].to_numpy()[m], fmt='o', ms=3, alpha=0.9, label=f'L={int(L)}')
	plt.xlabel(r'$(U-U_c)\,L^{1/\nu}$'); plt.ylabel(r'$R/(1+bL^c)$')
	plt.title('Collapse (with finite-size correction, normalized)')
	plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
	plt.tight_layout(); plt.savefig(os.path.join(HERE, 'minimal_fse.png'), dpi=200); plt.close()

	print('No-FSC params:', params2, '+/-', errs2)
	print('FSC params:', params4, '+/-', errs4)

if __name__ == '__main__':
	main() 