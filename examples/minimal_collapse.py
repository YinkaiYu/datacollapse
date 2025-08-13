import os, sys, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 允许从项目src导入
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse_robust, collapse_transform


def build_combined_from_datascale(data_scale_dir: str, out_csv: str) -> pd.DataFrame:
	rows = []
	patterns = ['**/*.csv', '**/*.txt']
	files = []
	for p in patterns:
		files.extend(glob.glob(os.path.join(data_scale_dir, p), recursive=True))
	def infer_L(path):
		m = re.search(r'L\s*=\s*(\d+)', path)
		if m: return int(m.group(1))
		parts = os.path.normpath(path).split(os.sep)
		for seg in parts[::-1]:
			m = re.search(r'L\s*=\s*(\d+)', seg)
			if m: return int(m.group(1))
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
			s = y; df[s] = np.std(df[y]) * 0.05
		return df[[u,y,s]].rename(columns={u:'U', y:'Y', s:'sigma'})
	for fp in files:
		L = infer_L(fp)
		if L is None: continue
		try:
			d = read_table(fp)
			d = map_columns(d)
			d['L'] = int(L)
			rows.append(d[['L','U','Y','sigma']])
		except Exception:
			continue
	if not rows:
		raise RuntimeError('No usable files in Data_scale/')
	df = pd.concat(rows, ignore_index=True).sort_values(['L','U']).reset_index(drop=True)
	df.to_csv(out_csv, index=False)
	return df


def main():
	base = os.path.join(ROOT, 'real_data_test', 'collab_package')
	data_scale = os.path.join(base, 'Data_scale')
	out_csv = os.path.join(base, 'real_data_combined.csv')
	os.makedirs(base, exist_ok=True)

	# 1) 构建合并数据
	df = build_combined_from_datascale(data_scale, out_csv)
	print('combined saved:', out_csv, 'shape=', df.shape)

	# 统一颜色
	Ls = sorted(df['L'].unique().tolist())
	cmap = plt.get_cmap('tab10', max(10, len(Ls)))
	L_colors = {int(Lv): cmap(i % 10) for i, Lv in enumerate(Ls)}

	# 2) 原始图
	plt.figure(figsize=(6,4))
	for Lv, d in df.groupby('L'):
		c = L_colors[int(Lv)]
		plt.errorbar(d['U'], d['Y'], yerr=d['sigma'], fmt='o-', ms=3, lw=1, alpha=0.9, color=c, label=f'L={int(Lv)}')
	plt.xlabel('U'); plt.ylabel(r'$R_{01}$'); plt.title('Raw data'); plt.legend(ncol=2, fontsize=8)
	plt.tight_layout(); plt.savefig(os.path.join(base, 'example_raw.png'), dpi=220); plt.close()

	# 3) No-FSE（drop L=7）
	DF2 = df[df['L']!=7].copy().reset_index(drop=True)
	data2 = DF2[['L','U','Y']].to_numpy(float)
	sigma2 = DF2['sigma'].to_numpy(float)
	params2, errs2 = fit_data_collapse(
		data2, sigma2, 8.67, 1.20,
		n_knots=10, lam=1e-3, n_boot=30, random_state=0,
		bounds=((8.0,9.0),(0.6,2.0)), optimizer='NM_then_Powell', random_restarts=6
	)
	x2, Y2 = collapse_transform(data2, params2)
	plt.figure(figsize=(6,4))
	for Lv, g in DF2.assign(x=x2, Yc=Y2).groupby('L'):
		c = L_colors[int(Lv)]
		plt.errorbar(g['x'], g['Yc'], yerr=g['sigma'], fmt='o', ms=3, alpha=0.9, color=c, label=f'L={int(Lv)}')
	plt.xlabel(r'$x=(U-U_c)\,L^{\,1/\nu}$'); plt.ylabel(r'$R_{01}$'); plt.title('Data collapse (drop L=7)'); plt.legend(ncol=2, fontsize=8)
	plt.tight_layout(); plt.savefig(os.path.join(base, 'example_nofse_dropL7.png'), dpi=220); plt.close()
	print('No-FSE params:', params2, 'errs:', errs2)

	# 4) FSE（robust）
	DATA = df[['L','U','Y']].to_numpy(float)
	sigma = df['sigma'].to_numpy(float)
	b_grid = np.linspace(0.5, 1.1, 7)
	c_grid = np.linspace(-1.3, -0.5, 7)
	params4, errs4 = fit_data_collapse_fse_robust(
		DATA, sigma, 8.45, 1.20,
		b_grid=b_grid, c_grid=c_grid,
		n_knots=10, lam=1e-3, n_boot=8, random_state=0,
		bounds_Ua=((8.0,9.0),(0.8,2.0)), normalize=True,
		optimizer='NM_then_Powell', maxiter=4000, random_restarts=0
	)
	x4, Y4 = collapse_transform(DATA, params4, normalize=True)
	plt.figure(figsize=(6,4))
	for Lv, g in df.assign(x=x4, Yc=Y4).groupby('L'):
		c = L_colors[int(Lv)]
		plt.errorbar(g['x'], g['Yc'], yerr=g['sigma'], fmt='o', ms=3, alpha=0.9, color=c, label=f'L={int(Lv)}')
	plt.xlabel(r'$x=(U-U_c)\,L^{\,1/\nu}$'); plt.ylabel(r'$R_{01}$'); plt.title('Data collapse (all L, with finite size correction)'); plt.legend(ncol=2, fontsize=8)
	plt.tight_layout(); plt.savefig(os.path.join(base, 'example_fse_allL.png'), dpi=220); plt.close()
	print('FSE params:', params4, 'errs:', errs4)

if __name__ == '__main__':
	main() 