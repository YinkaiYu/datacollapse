import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from datacollapse.datacollapse import fit_data_collapse, collapse_transform

def test_fit_and_transform_runs():
	# synthetic small dataset
	L = np.array([7,9,11,13,15], float)
	U = np.linspace(8.5, 8.9, 50)
	Ls, Us = np.meshgrid(L, U, indexing='ij')
	x = (Us - 8.67) * (Ls**1.2)
	f = np.tanh(x)
	sigma = 0.02*np.ones_like(f)
	y = f + np.random.default_rng(0).normal(0, sigma)
	data = np.vstack([Ls.ravel(), Us.ravel(), y.ravel()]).T
	err = sigma.ravel()
	params, errs = fit_data_collapse(data, err, 8.66, 1.1, n_knots=8, lam=1e-3, n_boot=0,
		bounds=((8.0,9.0),(0.6,2.0)))
	xc, yc = collapse_transform(data, params)
	assert xc.shape == yc.shape == (data.shape[0],) 