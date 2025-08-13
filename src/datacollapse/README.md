# datacollapse 库使用说明

本库用于临界现象中的数据坍缩（data collapse）分析，支持无有限尺寸修正（drop L）与含有限尺寸修正（FSE）两种形式。核心思路是：
- 将未知的普适函数 f(x) 用“线性样条 + 二阶差分平滑惩罚”表征，避免为 f 预先假设解析式；
- 对 (U_c, a=ν^{-1})（以及 FSE 的 b,c）做外层非线性优化；
- 用误差棒 σ 做加权与 bootstrap 估计参数不确定度；
- 提供优化器选择、多起点随机重启和稳健网格+内层优化（FSE）以降低初值敏感。

## 安装和依赖
- 纯Python单文件实现（`datacollapse/datacollapse.py`）
- 依赖：numpy、scipy（可选；无scipy时回退网格细化，较慢）

## 快速开始
```python
import numpy as np
import pandas as pd
from datacollapse.datacollapse import fit_data_collapse, fit_data_collapse_fse_robust, collapse_transform

# 准备 data: ndarray(N,3) -> [L, U, Y]
# err: 同长度 sigma 向量，或 None（将自动估计）

data = df[["L","U","Y"]].to_numpy(float)
sigma = df["sigma"].to_numpy(float)

# No-FSE（drop L=7示例）
params, errs = fit_data_collapse(
    data, sigma, U_c_0=8.67, a_0=1.20,
    n_knots=10, lam=1e-3, n_boot=50, random_state=0,
    bounds=((8.0,9.0),(0.6,2.0)),
    optimizer="NM_then_Powell", maxiter=4000, random_restarts=8
)
# 可视化坍缩
x, Yc = collapse_transform(data, params)

# FSE 稳健变体（推荐）：对(b,c)做网格，内层优化(U_c,a)
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

## 主要API
- `fit_data_collapse(data, err, U_c_0, a_0, *, n_knots=12, lam=1e-3, n_boot=20, random_state=0, bounds=None, optimizer="NM_then_Powell", maxiter=4000, random_restarts=0, progress=None)`
  - 返回：(U_c, a), (σ_Uc, σ_a)
  - data: (N,3) 数组 [L, U, Y]；err: σ（与Y同长）
  - n_knots: 样条结点数；lam: 平滑惩罚系数
  - bounds: ((Uc_lo,Uc_hi),(a_lo,a_hi))
  - optimizer: "NM" | "Powell" | "NM_then_Powell"
  - random_restarts: 随机重启次数（多起点）
  - progress: 可选回调，接收字典进度信息

- `fit_data_collapse_fse(data, err, U_c_0, a_0, b_0, c_0, *, normalize=False, L_ref="geom", ...)`
  - 直接优化四参 (U_c,a,b,c)
  - normalize=True 时采用 s_norm=(1+b L^c)/(1+b L_ref^c)

- `fit_data_collapse_fse_robust(data, err, U_c_0, a_0, b_grid, c_grid, *, normalize=False, L_ref="geom", ...)`
  - 稳健变体：
    1) 对(b,c)网格；
    2) 每个单元内仅优化(U_c,a)；
    3) 选最优单元；对bootstrap重复内层流程
  - 推荐用于缓解初值敏感与(b,c)边界吸附

- `collapse_transform(data, params, *, normalize=False, L_ref="geom")`
  - 将数据转换为坍缩坐标 (x, Yc)，用于绘图

## f(x) 的表征（为何无需假设解析式）
- 在 x 轴上放置 K=n_knots 个均匀结点，构造线性样条基 A(x)
- 目标：min_coeffs Σ w_i (Y_i − f(x_i))^2 + λ‖D²·coeffs‖²
  - w_i=1/σ_i²；D² 为二阶差分矩阵，λ 控制平滑度
- 这是线性问题，可闭式解；外层仅优化 θ=(U_c, a[, b, c])

## 参数选择建议
- n_knots: 10–14 常用；越大越灵活；
- lam: 1e-4–1e-2 常用；越大越平滑；
- bounds: 给(U_c, a)合理范围，避免陷入异常局部；
- 初值：a_0≥1.1 往往更稳定；FSE 建议用 robust 变体并网格(b,c)
- 不确定度：设置 n_boot≥30–50

## 稳健性与可重复性
- `random_state` 控制bootstrap与重启随机数
- `random_restarts` 多起点随机重启（建议>0）
- FSE推荐优先 `fit_data_collapse_fse_robust`

## 例子
参见 `examples/minimal_collapse.py`，包含：
- 从 Data_scale 构建 `real_data_combined.csv`
- 运行 No-FSE（drop L=7）与 FSE（robust）拟合
- 生成三张图并保存 