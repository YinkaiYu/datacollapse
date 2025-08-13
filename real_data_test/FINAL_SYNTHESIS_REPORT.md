# 最终综合结论与推荐参数

## 方法与选择原则

- 基于广义bootstrap结果，按质量上四分位（top 25%）筛选高质量子集进行统计；
- 给出 `U_c` 与 `ν^{-1}` 的中位数作为推荐值，robust σ（≈1.4826·MAD）作为不确定度；
- No-FSE Drop L=7 作为主结果，FSE All-L 作为对照；

## 统计结果（高质量子集）

### No-FSE (Drop L=7)

- 样本数 n=480，高质量 n_high=120（Q≥Q@75%=103.16）

- 推荐 U_c = 8.669546 ± 0.002214

- 推荐 ν^(-1) = 1.191639 ± 0.011695


### FSE (All L)

- 样本数 n=1280，高质量 n_high=320（Q≥Q@75%=123.66）

- 推荐 U_c = 8.448503 ± 0.074273

- 推荐 ν^(-1) = 1.285148 ± 0.065871


## 图表索引

- 交点基线：`crossing_uc_extrapolation.png`
- 初始敏感性：`sensitivity_nofse_dropL7.png`，`sensitivity_fse_allL.png`
- 宽扫描与稳健扫描：`wide_a_scan_nofse_dropL7.png`，`robust_nofse_dropL7_scan.png`
- FSE网格：`robust_fse_grid_heatmap.png`
- 广义bootstrap：`generalized_bootstrap_nofse_dropL7.png`，`generalized_bootstrap_fse_allL.png`
- 等效带对比：`equivalence_comparison.png`

## 核心三图（参数已标注）

- 原始数据：`core_raw.png`

![Raw](core_raw.png)

- No-FSE（Drop L=7）坍缩（标注 U_c 与 ν^(-1) 及不确定度）：`core_nofse_dropL7.png`

![No-FSE Drop L=7](core_nofse_dropL7.png)

- FSE（All L）坍缩（标注 U_c、ν^(-1)、(b,c) 及不确定度，normalize=geom）：`core_fse_allL.png`

![FSE All L](core_fse_allL.png)

## 质量等效带与占比（摘要）

- 结果来源：`equivalence_summary.csv`
- 关键数值：
  - No-FSE：Q_median≈82.54，robust σ≈33.08 → ΔQ≈66.15；U_c中位≈8.6717，ν^(-1)中位≈1.0907；等效带内占比≈0.796
  - FSE：  Q_median≈87.86，robust σ≈42.73 → ΔQ≈85.45；U_c中位≈8.4477，ν^(-1)中位≈1.0704；等效带内占比≈0.578
- 直观图：`equivalence_comparison.png`

![Equivalence](equivalence_comparison.png)

## 可复制命令与文件清单（附录）

- 生成三张核心图（含参数与不确定度标注）：
  - `cd real_data_test && python export_three_core_plots.py`
- 广义bootstrap（No-FSE / FSE）：
  - No-FSE：`cd real_data_test && python generalized_bootstrap_nofse.py`
  - FSE（扩规模示例）：`cd real_data_test && python generalized_bootstrap_fse.py --trials 360 --boots 4 --seed 0 --n_grid 7 --b_span 0.20 --c_span 0.20`
- 等效带对比与最终推荐：
  - `cd real_data_test && python analyze_equivalence_bands.py && python summarize_final_recommendation.py`
- 主要输出文件：
  - `core_raw.png`，`core_nofse_dropL7.png`，`core_fse_allL.png`
  - `generalized_bootstrap_nofse_dropL7.csv/.png`，`generalized_bootstrap_fse_allL.csv/.png`
  - `equivalence_summary.csv`，`equivalence_comparison.png`
  - `FINAL_SYNTHESIS_REPORT.md`
