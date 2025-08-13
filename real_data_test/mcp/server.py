import os, sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from datacollapse.datacollapse import (
	fit_data_collapse,
	fit_data_collapse_fse,
	fit_data_collapse_fse_robust,
	collapse_transform as collapse_transform_fn,
)
import numpy as np

app = FastAPI(title="datacollapse MCP")

# ---------- helpers ----------
def compute_quality_np(df_L: np.ndarray, x: np.ndarray, Yc: np.ndarray) -> float:
	x_range = float(np.max(x) - np.min(x)) if x.size else np.nan
	y_ranges = []
	for L in sorted(np.unique(df_L)):
		m = (df_L == L)
		if not np.any(m):
			continue
		yL = Yc[m]
		if yL.size:
			y_ranges.append(float(np.max(yL) - np.min(yL)))
	return float(x_range / np.mean(y_ranges)) if y_ranges else float('nan')

# ---------- models ----------
class NoFSEFitReq(BaseModel):
	data: List[List[float]]  # [L,U,Y]
	err: Optional[List[float]] = None
	Uc0: float
	a0: float
	n_knots: int = 12
	lam: float = 1e-3
	bounds: Optional[List[Tuple[float,float]]] = None
	n_boot: int = 0
	random_state: int = 0
	optimizer: str = "NM_then_Powell"
	maxiter: int = 4000
	random_restarts: int = 0

class NoFSEFitResp(BaseModel):
	Uc: float
	a: float
	Uc_err: float
	a_err: float

class FSEFitReq(BaseModel):
	data: List[List[float]]
	err: Optional[List[float]] = None
	Uc0: float
	a0: float
	b0: float
	c0: float
	n_knots: int = 12
	lam: float = 1e-3
	bounds: Optional[List[Tuple[float,float]]] = None
	n_boot: int = 0
	random_state: int = 0
	normalize: bool = False
	L_ref: str = "geom"
	optimizer: str = "NM_then_Powell"
	maxiter: int = 4000
	random_restarts: int = 0

class FSEFitResp(BaseModel):
	Uc: float
	a: float
	b: float
	c: float
	u_err: float
	a_err: float
	b_err: float
	c_err: float

class FSEFitRobustReq(BaseModel):
	data: List[List[float]]
	err: Optional[List[float]] = None
	Uc0: float
	a0: float
	b_grid: List[float]
	c_grid: List[float]
	n_knots: int = 12
	lam: float = 1e-3
	n_boot: int = 10
	random_state: int = 0
	bounds_Ua: Optional[List[Tuple[float,float]]] = None
	normalize: bool = False
	L_ref: str = "geom"
	optimizer: str = "NM_then_Powell"
	maxiter: int = 4000
	random_restarts: int = 0

class CollapseReq(BaseModel):
	data: List[List[float]]
	params: List[float]  # [Uc,a] or [Uc,a,b,c]
	normalize: bool = False
	L_ref: str = "geom"

class CollapseResp(BaseModel):
	x: List[float]
	Yc: List[float]

class QualityReq(BaseModel):
	L: List[float]
	x: List[float]
	Yc: List[float]

class QualityResp(BaseModel):
	Q: float

# ---------- endpoints ----------
@app.post("/fit_nofse", response_model=NoFSEFitResp)
def fit_nofse(req: NoFSEFitReq):
	data = np.asarray(req.data, float)
	err = None if req.err is None else np.asarray(req.err, float)
	params, errs = fit_data_collapse(
		data, err, req.Uc0, req.a0,
		n_knots=req.n_knots, lam=req.lam, n_boot=req.n_boot, random_state=req.random_state,
		bounds=None if req.bounds is None else tuple(map(tuple, req.bounds)),
		optimizer=req.optimizer, maxiter=req.maxiter, random_restarts=req.random_restarts
	)
	return NoFSEFitResp(Uc=float(params[0]), a=float(params[1]), Uc_err=float(errs[0]), a_err=float(errs[1]))

@app.post("/fit_fse", response_model=FSEFitResp)
def fit_fse(req: FSEFitReq):
	data = np.asarray(req.data, float)
	err = None if req.err is None else np.asarray(req.err, float)
	params, errs = fit_data_collapse_fse(
		data, err, req.Uc0, req.a0, req.b0, req.c0,
		n_knots=req.n_knots, lam=req.lam, n_boot=req.n_boot, random_state=req.random_state,
		bounds=None if req.bounds is None else tuple(map(tuple, req.bounds)),
		normalize=req.normalize, L_ref=req.L_ref,
		optimizer=req.optimizer, maxiter=req.maxiter, random_restarts=req.random_restarts
	)
	Uc,a,b,c = params
	eU,eA,eB,eC = errs
	return FSEFitResp(Uc=Uc, a=a, b=b, c=c, u_err=eU, a_err=eA, b_err=eB, c_err=eC)

@app.post("/fit_fse_robust", response_model=FSEFitResp)
def fit_fse_robust(req: FSEFitRobustReq):
	data = np.asarray(req.data, float)
	err = None if req.err is None else np.asarray(req.err, float)
	params, errs = fit_data_collapse_fse_robust(
		data, err, req.Uc0, req.a0,
		b_grid=np.asarray(req.b_grid, float), c_grid=np.asarray(req.c_grid, float),
		n_knots=req.n_knots, lam=req.lam, n_boot=req.n_boot, random_state=req.random_state,
		bounds_Ua=None if req.bounds_Ua is None else tuple(map(tuple, req.bounds_Ua)),
		normalize=req.normalize, L_ref=req.L_ref,
		optimizer=req.optimizer, maxiter=req.maxiter, random_restarts=req.random_restarts
	)
	Uc,a,b,c = params
	eU,eA,eB,eC = errs if errs and len(errs)==4 else (np.nan, np.nan, np.nan, np.nan)
	return FSEFitResp(Uc=Uc, a=a, b=b, c=c, u_err=eU, a_err=eA, b_err=eB, c_err=eC)

@app.post("/collapse_transform", response_model=CollapseResp)
def collapse_transform(req: CollapseReq):
	data = np.asarray(req.data, float)
	x, Yc = collapse_transform_fn(data, req.params, normalize=req.normalize, L_ref=req.L_ref)
	return CollapseResp(x=x.tolist(), Yc=Yc.tolist())

@app.post("/compute_quality", response_model=QualityResp)
def compute_quality(req: QualityReq):
	L = np.asarray(req.L, float)
	x = np.asarray(req.x, float)
	Yc = np.asarray(req.Yc, float)
	Q = compute_quality_np(L, x, Yc)
	return QualityResp(Q=float(Q)) 