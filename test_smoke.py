
import os, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse, fit_data_collapse_fse_robust

def test_fitters_smoke_normalized_and_robust():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data.csv"))
    data = df[["L","U","Y"]].to_numpy(float)
    err  = df["sigma"].to_numpy(float)

    # normalized FSE
    (Uc, a, b, c), errs = fit_data_collapse_fse(
        data, err, 8.64, 1.8, 0.8, -0.3,
        n_knots=8, n_boot=0, bounds=((8.5,8.8),(0.8,3.5),(0,3),(-2,-1e-3)),
        normalize=True
    )
    assert np.isfinite(Uc) and c < 0

    # robust grid
    b_grid = np.linspace(0.0, 3.0, 6)
    c_grid = np.linspace(-1.0, -0.05, 5)
    (Uc, a, b, c), errs = fit_data_collapse_fse_robust(
        data, err, 8.64, 1.8, b_grid, c_grid, n_knots=8, n_boot=0,
        bounds_Ua=((8.4,8.9),(0.2,4.0)), normalize=True
    )
    assert np.isfinite(Uc) and c < 0


def test_fse_normalized():
    import os, numpy as np, pandas as pd, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from datacollapse.datacollapse import fit_data_collapse_fse, collapse_transform
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data.csv"))
    data = df[["L","U","Y"]].to_numpy(float)
    err  = df["sigma"].to_numpy(float)
    (Uc, a, b, c), _ = fit_data_collapse_fse(data, err, 8.64, 1.8, 0.8, -0.3, n_knots=8, n_boot=0,
                                             bounds=((8.5,8.8), (0.8,3.5), (0.0, 3.0), (-2.0, -1e-3)),
                                             normalize=True)
    x, Ycorr = collapse_transform(data, (Uc, a, b, c), normalize=True)
    assert np.isfinite(Uc) and np.isfinite(a) and np.isfinite(b) and (c<0) and np.all(np.isfinite(x)) and np.all(np.isfinite(Ycorr))
