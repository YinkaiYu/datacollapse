# datacollapse Library Documentation

A Python library for data collapse analysis in critical phenomena, supporting both standard and finite-size-corrected forms. The core approach:
- Represents unknown universal function f(x) using "linear splines + second-difference smoothing penalty", avoiding pre-assumed analytical forms
- Performs outer nonlinear optimization over (U_c, a=ν^{-1}) (and b,c for finite-size corrections)
- Uses error bars σ for weighting and bootstrap for parameter uncertainty estimation
- Provides optimizer selection, multi-start random restarts, and robust grid+inner optimization to reduce initial value sensitivity

## Installation and Dependencies
- Pure Python implementation (`datacollapse/datacollapse.py`)
- Dependencies: numpy, scipy (optional; fallback to grid refinement if scipy unavailable, slower)

## Quick Start
```python
import numpy as np
import pandas as pd
from datacollapse import fit_data_collapse, fit_data_collapse_fse_robust, collapse_transform

# Prepare data: ndarray(N,3) -> [L, U, Y]
# err: same-length sigma vector, or None (will auto-estimate)

data = df[["L","U","Y"]].to_numpy(float)
sigma = df["sigma"].to_numpy(float)

# Without finite-size correction
params, errs = fit_data_collapse(
    data, sigma, U_c_0=8.67, a_0=1.20,
    n_knots=10, lam=1e-3, n_boot=50, random_state=0,
    bounds=((8.0,9.0),(0.6,2.0)),
    optimizer="NM_then_Powell", maxiter=4000, random_restarts=8
)
# Visualize collapse
x, Yc = collapse_transform(data, params)

# With finite-size correction (robust variant, recommended): grid over (b,c), inner optimize (U_c,a)
params_fse, errs_fse = fit_data_collapse_fse_robust(
    data, sigma, U_c_0=8.45, a_0=1.20,
    b_grid=np.linspace(0.5,1.1,7), c_grid=np.linspace(-1.3,-0.5,7),
    n_knots=10, lam=1e-3, n_boot=10, random_state=0,
    bounds_Ua=((8.0,9.0),(0.8,2.0)),
    normalize=True,
    optimizer="NM_then_Powell", maxiter=4000, random_restarts=0
)
xf, Yf = collapse_transform(data, params_fse, normalize=True)
```

## Main API

### `fit_data_collapse(data, err, U_c_0, a_0, *, n_knots=12, lam=1e-3, n_boot=20, random_state=0, bounds=None, optimizer="NM_then_Powell", maxiter=4000, random_restarts=0, progress=None)`
Fit without finite-size correction: Y ≈ f((U - U_c) L^a)

**Parameters:**
- `data`: (N,3) array [L, U, Y]
- `err`: σ (same length as Y), or None for auto-estimation
- `U_c_0, a_0`: Initial guesses for critical point and scaling exponent
- `n_knots`: Number of spline knots for f(x)
- `lam`: Smoothing penalty coefficient (larger = smoother)
- `n_boot`: Bootstrap iterations for uncertainty estimation
- `bounds`: ((Uc_lo,Uc_hi),(a_lo,a_hi)) parameter bounds
- `optimizer`: "NM" | "Powell" | "NM_then_Powell"
- `random_restarts`: Number of random restart attempts (multi-start)
- `progress`: Optional callback function receiving progress dict

**Returns:** `(U_c, a), (σ_Uc, σ_a)`

### `fit_data_collapse_fse(data, err, U_c_0, a_0, b_0, c_0, *, normalize=False, L_ref="geom", ...)`
Direct optimization of four parameters (U_c,a,b,c) with finite-size correction: Y ≈ f((U - U_c) L^a) · (1 + b L^c)

**Additional Parameters:**
- `b_0, c_0`: Initial guesses for finite-size correction parameters
- `normalize`: If True, use s_norm=(1+b L^c)/(1+b L_ref^c)
- `L_ref`: Reference size for normalization ("geom" for geometric mean, or float)

### `fit_data_collapse_fse_robust(data, err, U_c_0, a_0, b_grid, c_grid, *, normalize=False, L_ref="geom", ...)`
Robust variant for finite-size correction:
1. Grid over (b,c) combinations
2. For each grid cell, optimize only (U_c,a)
3. Select best grid cell; bootstrap repeats inner procedure

**Additional Parameters:**
- `b_grid, c_grid`: Arrays of b and c values to grid over
- `bounds_Ua`: Bounds for (U_c,a) only, since (b,c) are fixed on grid

**Recommended for:** Mitigating initial value sensitivity and (b,c) boundary attraction

### `collapse_transform(data, params, *, normalize=False, L_ref="geom")`
Transform data to collapse coordinates (x, Yc) for plotting

**Parameters:**
- `data`: (N,3) array [L, U, Y]
- `params`: Fitted parameters from fit functions
- `normalize, L_ref`: Must match fitting call for finite-size correction

**Returns:** `x, Yc` arrays for plotting

## Universal Function f(x) Representation
- Places K=n_knots uniform knots on x-axis, constructs linear spline basis A(x)
- Objective: min_coeffs Σ w_i (Y_i − f(x_i))^2 + λ‖D²·coeffs‖²
  - w_i=1/σ_i²; D² is second-difference matrix, λ controls smoothness
- This is a linear problem with closed-form solution; outer loop only optimizes θ=(U_c, a[, b, c])

## Parameter Selection Guidelines

### Spline Parameters
- `n_knots`: 10–14 commonly used; larger = more flexible but risk overfitting
- `lam`: 1e-4–1e-2 commonly used; larger = smoother f(x)

### Bounds and Initial Values
- `bounds`: Provide reasonable ranges for (U_c, a) to avoid anomalous local minima
- Initial values: `a_0≥1.1` often more stable; for finite-size correction, recommend robust variant with grid over (b,c)
- Uncertainty: Set `n_boot≥30–50` for reliable error estimates

### Robustness
- `random_restarts > 0`: Multi-start random restarts (recommended)
- For finite-size correction: Prefer `fit_data_collapse_fse_robust` over direct `fit_data_collapse_fse`

## Reproducibility
- `random_state`: Controls bootstrap and restart random numbers
- All random operations use numpy.random.Generator for reproducible results

## Examples
See `examples/minimal_collapse.py` which includes:
- Building `real_data_combined.csv` from Data_scale directory
- Running both standard and finite-size-corrected fits
- Generating and saving three comparison plots

## Performance Notes
- Linear spline fitting is fast (closed-form solution)
- Bootstrap and multi-start add computational cost proportional to iterations
- Robust finite-size correction is slower due to grid search but more reliable

## Error Handling
- Invalid parameters (e.g., c≥0 in finite-size correction) return infinite objective
- Missing scipy gracefully falls back to grid-based optimization
- Bootstrap failures are handled by returning NaN for that iteration 