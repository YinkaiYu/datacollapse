
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Sequence, Callable, List

# ----------------- helpers -----------------
def _build_linear_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    K = len(knots); N = len(x)
    A = np.zeros((N, K), float)
    xc = np.clip(x, knots[0], knots[-1])
    j = np.searchsorted(knots, xc, side="right") - 1
    j = np.clip(j, 0, K-2)
    x0 = knots[j]; x1 = knots[j+1]
    denom = (x1 - x0)
    denom[denom==0.0] = 1e-12
    t = (xc - x0)/denom
    A[np.arange(N), j]   += (1.0 - t)
    A[np.arange(N), j+1] += t
    return A

def _second_diff_penalty(K: int) -> np.ndarray:
    D = np.zeros((K-2, K), float)
    for i in range(K-2):
        D[i,i]   = 1.0
        D[i,i+1] = -2.0
        D[i,i+2] = 1.0
    return D.T @ D

def _prep_sigma(err: Optional[np.ndarray], Y: np.ndarray) -> np.ndarray:
    if err is None:
        sigma = np.full_like(Y, np.std(Y)*0.05 + 1e-6, float)
    else:
        e = np.asarray(err, float)
        sigma = e[:, -1] if e.ndim>1 else e
        sigma = np.asarray(sigma, float)
        sigma[sigma<=0] = np.min(sigma[sigma>0]) if np.any(sigma>0) else 1e-6
    return sigma

# 优化器选择与多起点最小化
def _minimize_once(objfun: Callable[[np.ndarray], float], x0, bounds=None,
                   optimizer: str = "NM_then_Powell", maxiter: int = 4000):
    """One-shot minimize with chosen optimizer(s)."""
    try:
        import numpy as _np
        import scipy.optimize as opt
        x0 = _np.array(x0, float)
        method = None
        if optimizer in ("NM", "Nelder-Mead"):
            method = "Nelder-Mead"
            res = opt.minimize(lambda th: objfun(th), x0=x0,
                               method=method,
                               options=dict(xatol=1e-6, fatol=1e-9, maxiter=maxiter))
            th = res.x; fval = float(res.fun)
            return th, fval
        elif optimizer in ("Powell",):
            # 如果提供bounds，Powell可以使用
            res = opt.minimize(lambda th: objfun(th), x0=x0,
                               method="Powell", bounds=bounds,
                               options=dict(xtol=1e-6, maxiter=maxiter))
            th = res.x; fval = float(res.fun)
            return th, fval
        else:
            # 默认：先NM，再Powell细化
            res = opt.minimize(lambda th: objfun(th), x0=x0,
                               method="Nelder-Mead",
                               options=dict(xatol=1e-6, fatol=1e-9, maxiter=maxiter))
            th = res.x; fval = float(res.fun)
            if bounds is not None:
                lo = _np.array([b[0] for b in bounds], float)
                hi = _np.array([b[1] for b in bounds], float)
                th = _np.clip(th, lo, hi)  # clip NM output before Powell
            res2 = opt.minimize(lambda th: objfun(th), x0=th, method="Powell",
                                bounds=bounds, options=dict(xtol=1e-6, maxiter=maxiter))
            if res2.fun < res.fun:
                th, fval = res2.x, float(res2.fun)
            return th, fval
    except Exception:
        # SciPy不可用或失败时的grid-refine回退
        th_best = np.array(x0, float); f_best = objfun(th_best)
        half = np.array([0.2]*len(th_best), float)
        for scale in [1.0, 0.5, 0.25, 0.125]:
            grids = [np.linspace(th_best[i]-half[i]*scale, th_best[i]+half[i]*scale, 17)
                     for i in range(len(th_best))]
            from itertools import product
            for th in product(*grids):
                th = np.array(th, float)
                if bounds is not None:
                    ok = all(bounds[i][0] <= th[i] <= bounds[i][1] for i in range(len(th)))
                    if not ok: continue
                f = objfun(th)
                if f < f_best: th_best, f_best = th, f
        return th_best, f_best


def _multi_start_minimize(objfun: Callable[[np.ndarray], float], x0: Sequence[float], *,
                          bounds=None, optimizer: str = "NM_then_Powell",
                          maxiter: int = 4000, random_restarts: int = 0,
                          rng: Optional[np.random.Generator] = None,
                          progress: Optional[Callable[[str, dict], None]] = None) -> Tuple[np.ndarray, float]:
    """Run multi-start minimization. Returns best (theta, fval)."""
    starts: List[np.ndarray] = [np.array(x0, float)]
    if random_restarts and random_restarts > 0:
        rng = rng or np.random.default_rng(0)
        if bounds is not None:
            lo = np.array([b[0] for b in bounds], float)
            hi = np.array([b[1] for b in bounds], float)
            for _ in range(int(random_restarts)):
                starts.append(rng.uniform(lo, hi))
        else:
            for _ in range(int(random_restarts)):
                starts.append(np.array(x0, float) + rng.normal(0.0, 0.2, size=len(x0)))
    best_th = None; best_val = np.inf
    for i, s in enumerate(starts):
        th, fval = _minimize_once(objfun, s, bounds=bounds, optimizer=optimizer, maxiter=maxiter)
        if progress:
            progress("restart_step", {"idx": i, "theta": th.tolist(), "fval": float(fval)})
        if fval < best_val:
            best_th, best_val = th, fval
    return np.array(best_th, float), float(best_val)


def _fit_params_with_spline(L, U, Y, w, Uc, a, knots, lam):
    x = (U - Uc) * (L**a)
    A = _build_linear_design(x, knots)
    sqrtw = np.sqrt(w)
    Aw = A * sqrtw[:,None]
    yw = Y * sqrtw
    P = _second_diff_penalty(A.shape[1]) * lam
    M = Aw.T @ Aw + P
    rhs = Aw.T @ yw
    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        M = M + np.eye(M.shape[0])*1e-10
        coeffs = np.linalg.solve(M, rhs)
    resid = Y - (A @ coeffs)
    chi2 = float(np.sum(w * resid**2)) / len(Y)
    return chi2

def _bootstrap_params(fit_once, Y, sigma, theta_hat, n_boot=20, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(0, int(n_boot))):
        Yb = Y + rng.normal(0.0, sigma)
        samples.append(fit_once(Yb, theta_hat))
    S = np.array(samples)
    if S.size == 0: return np.full(len(theta_hat), np.nan)
    return S.std(axis=0, ddof=1)

def _scale_factor(L, b, c, normalize: bool, L_ref):
    s = 1.0 + b*(L**c)
    if not normalize:
        return s
    if isinstance(L_ref, (int,float)):
        Lr = float(L_ref)
    elif L_ref == "geom":
        Lr = float(np.exp(np.mean(np.log(L))))
    else:
        raise ValueError("L_ref must be 'geom' or a positive number.")
    s_ref = 1.0 + b*(Lr**c)
    return s / s_ref

# ----------------- public API -----------------
def fit_data_collapse(data: np.ndarray,
                      err: Optional[np.ndarray],
                      U_c_0: float, a_0: float,
                      n_knots: int = 12, lam: float = 1e-3,
                      n_boot: int = 20, random_state: int = 0,
                      bounds=None,
                      *,
                      optimizer: str = "NM_then_Powell",
                      maxiter: int = 4000,
                      random_restarts: int = 0,
                      progress: Optional[Callable[[str, dict], None]] = None):
    data = np.asarray(data, float)
    L, U, Y = data[:,0], data[:,1], data[:,2]
    sigma = _prep_sigma(err, Y)
    w0 = 1.0/(sigma**2)

    x0 = (U - U_c_0) * (L**a_0)
    lo, hi = np.quantile(x0, 0.01), np.quantile(x0, 0.99)
    knots = np.linspace(lo, hi, max(3, int(n_knots)))

    def objective(th, Y_use):
        Uc, a = th
        return _fit_params_with_spline(L, U, Y_use, w0, Uc, a, knots, lam)

    rng = np.random.default_rng(random_state)
    th_hat, _ = _multi_start_minimize(lambda th: objective(th, Y), (U_c_0, a_0),
                                      bounds=bounds, optimizer=optimizer, maxiter=maxiter,
                                      random_restarts=random_restarts, rng=rng, progress=progress)
    def fit_once(Y_use, th_init):
        th, _ = _multi_start_minimize(lambda th: objective(th, Y_use), th_init,
                                      bounds=bounds, optimizer=optimizer, maxiter=maxiter,
                                      random_restarts=random_restarts, rng=rng, progress=None)
        return np.array(th, float)
    std = _bootstrap_params(fit_once, Y, sigma, np.array(th_hat, float), n_boot, random_state)
    return (float(th_hat[0]), float(th_hat[1])), (float(std[0]), float(std[1]))

def fit_data_collapse_fse(data: np.ndarray,
                          err: Optional[np.ndarray],
                          U_c_0: float, a_0: float, b_0: float, c_0: float,
                          n_knots: int = 12, lam: float = 1e-3,
                          n_boot: int = 20, random_state: int = 0,
                          bounds=None,
                          normalize: bool = False, L_ref: 'geom|float' = "geom",
                          *,
                          optimizer: str = "NM_then_Powell",
                          maxiter: int = 4000,
                          random_restarts: int = 0,
                          progress: Optional[Callable[[str, dict], None]] = None):
    """Fit (Uc,a,b,c) with model: Y ≈ f((U-Uc)L^a) * s(L),
       where s(L) = 1 + b L^c  (or normalized version if normalize=True)."""
    data = np.asarray(data, float)
    L, U, Y = data[:,0], data[:,1], data[:,2]
    sigma = _prep_sigma(err, Y)
    w_base = 1.0/(sigma**2)

    x0 = (U - U_c_0) * (L**a_0)
    lo, hi = np.quantile(x0, 0.01), np.quantile(x0, 0.99)
    knots = np.linspace(lo, hi, max(3, int(n_knots)))

    def objective(th, Y_use):
        Uc, a, b, c = th
        if c >= 0: return np.inf
        s = _scale_factor(L, b, c, normalize, L_ref)
        if np.any(s <= 0): return np.inf
        Yc = Y_use / s
        w  = w_base * (s**2)  # variance propagation
        return _fit_params_with_spline(L, U, Yc, w, Uc, a, knots, lam)

    rng = np.random.default_rng(random_state)
    th0 = (U_c_0, a_0, b_0, c_0)
    th_hat, _ = _multi_start_minimize(lambda th: objective(th, Y), th0,
                                      bounds=bounds, optimizer=optimizer, maxiter=maxiter,
                                      random_restarts=random_restarts, rng=rng, progress=progress)
    def fit_once(Y_use, th_init):
        th, _ = _multi_start_minimize(lambda th: objective(th, Y_use), th_init,
                                      bounds=bounds, optimizer=optimizer, maxiter=maxiter,
                                      random_restarts=random_restarts, rng=rng, progress=None)
        return np.array(th, float)
    std = _bootstrap_params(fit_once, Y, sigma, np.array(th_hat, float), n_boot, random_state)
    return tuple(float(v) for v in th_hat), tuple(float(v) for v in std)

def fit_data_collapse_fse_robust(data: np.ndarray,
                                 err: Optional[np.ndarray],
                                 U_c_0: float, a_0: float,
                                 b_grid: Sequence[float], c_grid: Sequence[float],
                                 n_knots: int = 12, lam: float = 1e-3,
                                 n_boot: int = 10, random_state: int = 0,
                                 bounds_Ua = ((-np.inf, np.inf), (-np.inf, np.inf)),
                                 normalize: bool = False, L_ref: 'geom|float' = "geom",
                                 *,
                                 optimizer: str = "NM_then_Powell",
                                 maxiter: int = 4000,
                                 random_restarts: int = 0,
                                 progress: Optional[Callable[[str, dict], None]] = None):
    """Robust variant: grid over (b,c); for each cell optimize only (Uc,a).
       Returns (Uc,a,b,c) at the best cell. Bootstrap repeats the inner loop."""
    data = np.asarray(data, float)
    L, U, Y = data[:,0], data[:,1], data[:,2]
    sigma = _prep_sigma(err, Y)
    w_base = 1.0/(sigma**2)

    x0 = (U - U_c_0) * (L**a_0)
    lo, hi = np.quantile(x0, 0.01), np.quantile(x0, 0.99)
    knots = np.linspace(lo, hi, max(3, int(n_knots)))

    def obj_Ua(Uc, a, Y_use, b, c):
        if c >= 0: return np.inf
        s = _scale_factor(L, b, c, normalize, L_ref)
        if np.any(s <= 0): return np.inf
        Yc = Y_use / s
        w  = w_base * (s**2)
        return _fit_params_with_spline(L, U, Yc, w, Uc, a, knots, lam)

    def inner(Y_use, b, c, Uc_init=U_c_0, a_init=a_0):
        f = lambda th: obj_Ua(th[0], th[1], Y_use, b, c)
        th, _ = _multi_start_minimize(f, (Uc_init, a_init), bounds=bounds_Ua,
                                      optimizer=optimizer, maxiter=maxiter,
                                      random_restarts=random_restarts,
                                      rng=np.random.default_rng(random_state), progress=None)
        return th

    # coarse search
    best_val = np.inf; best = None; best_th = None
    for b in b_grid:
        for c in c_grid:
            th = inner(Y, b, c)
            val = obj_Ua(th[0], th[1], Y, b, c)
            if progress:
                progress("grid_cell", {"b": float(b), "c": float(c), "val": float(val)})
            if val < best_val:
                best_val, best, best_th = val, (b,c), th
    Uc_hat, a_hat = best_th; b_hat, c_hat = best

    # bootstrap
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(max(0, int(n_boot))):
        Yb = Y + rng.normal(0.0, sigma)
        val_best = np.inf; par_best = None
        for b in b_grid:
            for c in c_grid:
                th = inner(Yb, b, c, Uc_init=Uc_hat, a_init=a_hat)
                val = obj_Ua(th[0], th[1], Yb, b, c)
                if val < val_best:
                    par_best = (th[0], th[1], b, c); val_best = val
        samples.append(par_best)
    S = np.array(samples)
    std = S.std(axis=0, ddof=1) if S.size else np.full(4, np.nan)

    return (float(Uc_hat), float(a_hat), float(b_hat), float(c_hat)), tuple(float(v) for v in std)

def collapse_transform(data: np.ndarray, params, *,
                       normalize: bool = False, L_ref: 'geom|float' = "geom"):
    data = np.asarray(data, float)
    L, U, Y = data[:,0], data[:,1], data[:,2]
    Uc, a = params[:2]
    x = (U - Uc) * (L**a)
    if len(params) >= 4:
        b, c = params[2], params[3]
        s = _scale_factor(L, b, c, normalize, L_ref)
        Yc = Y / s
    else:
        Yc = Y
    return x, Yc
