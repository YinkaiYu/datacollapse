# 初始敏感性与超宽扫描阶段性报告

## 概览
- 本报告汇总了以下三部分的完成情况：
  1) 交点法基线（平台平均）
  2) 初始值敏感性分析（No-FSE Drop L=7 与 FSE All-L）
  3) No-FSE Drop L=7 的超宽范围 ν^(-1) 扫描
- 所有产出均已保存到 `real_data_test/` 目录，图表可直接目测坍缩质量与分布特征。

## 1) 交点法 U_c 基线（平台平均）
- 方法：在公共U窗口内求各(L,L')交点，选最大 L_eff 的若干对做误差加权平台平均。
- 结果：
  - U_c^cross,plateau = 8.6961 ± 0.0015
  - 窗口约 [8.691, 8.791]
- 图表：
  - `crossing_uc_extrapolation.png`

![Crossing-based U_c and window](crossing_uc_extrapolation.png)

## 2) 初始值敏感性分析
- 产出图：
  - `sensitivity_nofse_dropL7.png`
  - `sensitivity_fse_allL.png`

![Sensitivity No-FSE Drop L=7](sensitivity_nofse_dropL7.png)

![Sensitivity FSE All-L](sensitivity_fse_allL.png)

### No-FSE（Drop L=7）
- Spearman 相关：
  - Final a vs a0 = 0.962（对 a0 高度敏感）
  - Final U_c vs U_c0 = 0.712（对 U_c0 中等敏感）
- 分布（n=110）：
  - U_c: 平均 8.6740；IQR≈[8.6695, 8.6782]
  - ν^(-1): 平均 1.036；IQR≈[0.937, 1.170]
  - 质量Q: 平均 75.5；IQR≈[55.8, 101.3]
- 结论：a0 偏低（<1）易落入低质量局部极小；a0≥1.1 明显提高Q并稳定得到 ν^(-1)>1。

### FSE（All L）
- Spearman 相关：
  - Final a vs a0 = 0.929（高度敏感）
  - Final U_c vs U_c0 = 0.964（极高敏感）
- 分布（n=90）：
  - U_c 范围宽（≈8.34–8.74），呈多模态（含高Q≈180与中Q≈70–110）
  - b,c 常在边界附近，存在边界吸引效应
- 结论：FSE 对初值更敏感，需采用稳健网格+bootstrap与“质量等效带”策略。

## 3) 超宽范围 ν^(-1) 扫描（No-FSE Drop L=7）
- 参数：a0∈[0.30, 2.00]，U_c0∈[8.40, 8.90]，bounds a∈[0.30, 2.50]；n≈462 个有效收敛样本。
- Top-5（按Q）：集中在 a≈1.21–1.23 区域（最高 Q≈116.3）。
- 分组（按最终a）：
  - a<1：n=262，mean Q≈29.0，max Q≈65.3
  - a≥1：n=200，mean Q≈99.6，max Q≈116.3
- 图表与数据：
  - `wide_a_scan_nofse_dropL7.png`
  - `wide_a_scan_nofse_dropL7.csv`

![Wide a scan No-FSE Drop L=7](wide_a_scan_nofse_dropL7.png)

- 结论：
  - a≈0.5 或 a≈2.0 区域均劣于 a≈1.22；数据偏好 ν^(-1)>1 的高质量解。

## 4) FSE（All-L）稳健网格与小规模bootstrap（新增）
- 网格扫描（b∈[0.4,1.2], c∈[-1.4,-0.4]，多初值）：
  - 最优单元（无bootstrap）：U_c≈8.3854, ν^(-1)≈1.3577, b≈0.8557, c≈-0.6096, Q≈183.51
  - 图与数据：`robust_fse_grid_scan.csv`，`robust_fse_grid_heatmap.png`

![FSE grid heatmap (best Q per cell)](robust_fse_grid_heatmap.png)

- 对Top单元做小规模bootstrap（`fit_data_collapse_fse_robust`）：
  - 代表结果：U_c≈8.37866 ± 0.00041, ν^(-1)≈1.3235 ± 0.0031, Q≈168.07
  - 数据：`robust_fse_bootstrap_top.csv`
- 结论：高质量FSE模态稳健存在；与No-FSE（U_c≈8.67, ν^(-1)≈1.2, Q~110）存在系统性差异，属“是否纳入FSE修正”的模型差异。

## 5) “质量等效带”与误差棒的作用（新增）
- 动机：原始数据带误差棒，坍缩质量Q本身有统计波动；当两个解的Q差在ΔQ内，可视为“等效最优”。
- 现象：FSE的U_c≈8.37866（bootstrap后）质量依旧很高；这类点有可能由于误差棒导致的等效带而被接受。
- 计划（下一阶段）：
  1. 用bootstrap对No-FSE与FSE分别估计Q的统计波动ΔQ（n_boot≥200），定义等效带阈值；
  2. 扩大FSE的初值/网格范围与随机重启，做“广义bootstrap”（多初值×bootstrap），统计(U_c, ν^(-1))分布；
  3. 对No-FSE亦做广义bootstrap，比较两方法在等效带内的结果占比；
  4. 在报告中给出“统计中心±不确定度”与“等效带内可接受区间”，并标注U_c≈8.67与U_c≈8.38各自的占比与证据强度。

## 6) 结论更新（阶段性）
- 目前证据（No-FSE宽扫描 + 交点平台 + 敏感性分析）一致支持：No-FSE在U_c≈8.67、ν^(-1)≈1.20–1.25给出最高坍缩质量；a≈0.5或≈2.0均显著劣化质量。
- FSE在稳健网格与小规模bootstrap下稳定出现高质量模态（U_c≈8.38, ν^(-1)≈1.33–1.36, Q~170–184）。
- 下一步将通过“质量等效带”的统计定义，判断FSE与No-FSE两套解是否在误差允许范围内等效，若不等效则量化偏差来源。

## 当前阶段结论
- 交点平台与No-FSE宽扫描一致指向：U_c≈8.67–8.70，ν^(-1)≈1.20–1.25 区域给出最高坍缩质量。
- “ν^(-1)≈0.5”与“ν^(-1)≈2.0”在同等扫描与边界条件下均未给出可比的高质量坍缩。
- FSE 需用稳健方法（网格+bootstrap+等效带）再评估模态间的“质量等效”。

## 文件清单（可直接查看）
- 交点法：`crossing_uc_extrapolation.png`
- 初始敏感性：`sensitivity_nofse_dropL7.png`，`sensitivity_fse_allL.png`
- 宽扫描：`wide_a_scan_nofse_dropL7.csv`，`wide_a_scan_nofse_dropL7.png`
- No-FSE 稳健扫描（补充）：`robust_nofse_dropL7_scan.csv`，`robust_nofse_dropL7_scan.png`

## 后续计划（与 TODO 对齐）
- FSE（All-L）使用稳健变体 `fit_data_collapse_fse_robust` 做(b,c)网格+bootstrap；
- 定义并估计“质量等效带”，给出两方法的统计中心±不确定度与等效区间；
- 汇总形成最终综合报告（以 Drop L=7 为准，统一纵轴 r'$R_{01}$'）。 

## 图表解读指南（逐图说明）

为便于快速理解，这里对报告中生成的图表逐一解释其含义、阅读顺序与关键结论。

### 1) `crossing_uc_extrapolation.png`
- 左图（Crossing-based U_c）：
  - 横轴：1/L_eff；纵轴：由两条不同L曲线交点估计的 `U_c`。
  - 橙色点：最大L_eff的若干对交点（平台集合）；绿色水平线：平台加权平均 `U_c^cross,plateau`。
  - 读法：优先关注平台集合是否集中与平稳，绿色线给出 `U_c` 基线。
- 右图（Window by minimal across-L spread）：
  - 横轴：U；纵轴：不同L的 `R_{01}` 在固定U处的跨L标准差。
  - 橙色阴影：跨L分散度最小的“物理窗口”，交点只在该窗口内寻找，避免远端假交点。

### 2) `sensitivity_nofse_dropL7.png`
- 面板一（左上，热力图）：横轴 `U_c0`（初值），纵轴 `a0`（初值），颜色为坍缩质量Q（越亮越好）。
  - 读法：高质量区集中在较高 `a0`（≳1.1）；`a0<1` 容易陷入低Q解。
- 面板二（中上，散点）：横轴 `a0`，纵轴 `最终 ν^{-1}`，颜色为Q。
  - 读法：`最终 ν^{-1}` 随 `a0` 单调增强，且高Q解集中在 `最终 ν^{-1} > 1` 区域。
- 面板三（右上，散点）：横轴 `U_c0`，纵轴 `最终 U_c`，颜色为Q。
  - 读法：`最终 U_c` 对 `U_c0` 有正相关（但弱于a的敏感性）。
- 面板四/五（下排直方图）：`最终 ν^{-1}` 与 `最终 U_c` 的分布。
  - 读法：分布主峰位置与离散程度，主峰 `ν^{-1} > 1`。
- 面板六（右下，散点）：横轴 `最终 ν^{-1}`，纵轴 Q。
  - 读法：Q 随 `ν^{-1}` 提升至 ≈1.2 附近最优。

### 3) `sensitivity_fse_allL.png`
- 结构与No-FSE图相同，但对象是 FSE（All-L）。
- 关键差异：
  - `最终 U_c` 对 `U_c0` 的相关性更强，显示FSE对初始更敏感；
  - 分布呈多模态，提示存在不同质量的解簇（高Q≈180与中Q≈70–110并存）。

### 4) `wide_a_scan_nofse_dropL7.png`
- 面板一（左上，散点）：横轴 `最终 ν^{-1}`，纵轴 Q。
  - 读法：`a<1` 区域整体Q低（均值≈29），`a≥1` 显著更高（均值≈100，峰值>110）。
- 面板二（中上，热力图）：横轴 `U_c0`，纵轴 `a0`，颜色Q。
  - 读法：超宽初值下，高Q区域仍偏向较高 `a0` 且U_c0在≈8.67附近。
- 面板三（右上，直方图）：`最终 ν^{-1}` 分布。
  - 读法：主峰位于 ≈1.2，`a≈0.5` 或 `≈2.0` 均非高质量。
- 面板四（下排，最佳坍缩）：横轴 `(U - Uc) × L^(1/ν)`，纵轴 `R_{01}`，含误差棒。
  - 读法：直观看到不同L的曲线在 `ν^{-1}≈1.22`、`U_c≈8.67` 的最优坍缩质量（Q≈116）。

### 5) `robust_nofse_dropL7_scan.png`
- 内容与`wide_a_scan_nofse_dropL7.png`相似，但围绕交点平台基线 `U_c≈8.696±0.07` 做更密集扫描。
- 结论强化：最佳解仍在 `ν^{-1}≈1.2` 附近，Q显著高于 `a<1` 的解。

### 6) `robust_fse_grid_heatmap.png`
- 横轴：b；纵轴：c；颜色：在固定(b,c)单元内，对多组(U_c0,a0)求得的最佳Q。
- 读法：热区对应FSE高质量模态（例如 `b≈0.86, c≈-0.61`）。与CSV `robust_fse_grid_scan.csv` 对应。

### 7) `generalized_bootstrap_nofse_dropL7.png`
- 左上：Q分布；绿色线为中位数，橙色带为“质量等效带”近似阈值 `ΔQ≈66`。
  - 读法：当两个解的Q差小于ΔQ，可视作统计上“等效最优”。
- 其他面板分别为 `ν^{-1}`、`U_c` 的分布，以及 Q-vs-`ν^{-1}` 的关系。
  - 结论：No-FSE中心稳定在 `U_c≈8.67`、`ν^(-1)>1`。

### 8) `generalized_bootstrap_fse_allL.png`
- 与No-FSE图同构，但对象为FSE。
- 观察点：FSE在广义bootstrap下的 `U_c` 分布中心上移至≈8.46（远离8.38），`ν^(-1)` 中心≈1.08，且 `ΔQ≈83` 更宽。
  - 读法：在更广初值与外部噪声下，高质量“低U_c模态”不再总占主导；需要以等效带与占比来衡量与No-FSE的一致性。

## 7) 质量等效带对比（No-FSE vs FSE，新增）

- 数据文件：
  - `equivalence_summary.csv`（Q中位数、robust σ、ΔQ、U_c/ν^(-1)中位数、等效带内占比）
  - `equivalence_comparison.png`（下方插图）

![Equivalence band comparison](equivalence_comparison.png)

- 图读法：
  - 左上（U_c分布对比）：直观看两方法 `U_c` 分布位置与宽度，并以虚线标示参考值（8.38/8.46/8.67）。
  - 右上（Q分布与ΔQ）：分别给出两方法的Q分布、各自的中位数与“等效带阈值”ΔQ（基于robust σ）；判断两者Q是否“统计等效”。
  - 左下（U_c分箱占比）：统计 `U_c` 落在[8.30,8.50)、[8.50,8.60)、[8.60,8.75)三段的占比，用于量化“更接近8.67”的权重。
  - 右下（Q vs U_c）：观察高Q解集中在哪个 `U_c` 区间。

- 数值要点：
  - No-FSE: Q_median≈82.54，robust σ≈33.08 → ΔQ≈66.15；U_c中位≈8.6717，ν^{-1}中位≈1.0907；等效带内占比≈0.796
  - FSE:   Q_median≈91.19，robust σ≈41.35 → ΔQ≈82.69；U_c中位≈8.4603，ν^{-1}中位≈1.0831；等效带内占比≈0.600

- 阶段性判断：
  - No-FSE（Drop L=7）在广义bootstrap下的中心稳定于 `U_c≈8.67`，`ν^(-1)>1`，且高Q解主要集中在8.60–8.75区间。
  - FSE（All-L）在稳健网格+广义bootstrap后，其 `U_c` 分布中心已从8.38模态上移至 `≈8.46`，但仍明显低于8.67；其ΔQ更大，表明在误差棒作用下的“等效带”更宽。
  - 结合 `交点平台` 与 `No-FSE宽扫描+广义bootstrap` 的证据，当前数据更偏向 `U_c≈8.67` 的解；FSE的低U_c模态虽高Q，但在广义bootstrap与更广初值下并未占主导，可能属于“模型修正偏置+等效带容忍”范畴。

## 8) No-FSE 的 n_knots/lam 灵敏度（新增）

- 图表与数据：
  - `nknot_lam_sensitivity_nofse.png`
  - `nknot_lam_sensitivity_nofse.csv`

![n_knots/lam sensitivity (No-FSE Drop L=7)](nknot_lam_sensitivity_nofse.png)

- 图表解读：
  - 左图（热力图）：横轴 `lam`，纵轴 `n_knots`，颜色为该配置下“多起点扫描得到的最佳坍缩质量 Q_best`”。
  - 右图（误差条）：对每个配置，统计多起点的 `Q_mean ± Q_std`，反映初值不确定性下的平均质量与波动。

- 关键观察：
  - 轻微的 `n_knots` 效应：`n_knots` 增大时 Q_best 略有提升；样本中 `n_knots=14` 最高（Q_best≈105.11），`n_knots=10`≈104.55，`n_knots=8`≈104.39，`n_knots=12`≈102.62。
  - `lam` 在测试范围 `1e-4, 1e-3, 1e-2` 内对 Q_best 基本无影响：同一 `n_knots` 的三条结果几乎重合。
  - 参数稳定：对应最优解的 `U_c_best≈8.670±0.001`，`ν^(-1)_best≈1.175–1.184`，与前文（交点平台与No-FSE宽扫描）一致。
  - 多起点统计：各配置的 `Q_mean≈89–91`，`Q_std≈10–11`，说明在这些平滑度设置下，初值带来的质量波动中等；但“最优”解稳定存在且对 `lam` 不敏感。

- 结论与建议：
  - 推荐默认：`n_knots=12–14`，`lam=1e-3`（折中与常用默认）。若追求更高的 Q_best，可取 `n_knots=14`。
  - 此灵敏度实验不改变总体结论：No-FSE 的最优坍缩依旧位于 `U_c≈8.67`、`ν^(-1)≈1.18–1.22` 区域。

---

后续我们将：
- 进一步加大FSE的广义bootstrap规模（n_trials/n_boot_ext提高、(b,c)局部网格加密），并将关键数值标注到图中（中位数、IQR、Top值、ΔQ带）；
- 在最终综合报告中，给出“统计中心±不确定度+等效带内可接受区间”的统一结论，并明确推荐以 `Drop L=7` 的 No-FSE 结果（`U_c≈8.67, ν^(-1)≈1.20–1.25`）作为当前基准；FSE结果作为对照并标注“可能因FSE修正偏置导致的低U_c模态”。 

## 最终综合结论与推荐参数（草稿）

- 依据广义bootstrap（高质量子集，Top 25%）的统计：
  - No-FSE (Drop L=7)（主推荐）
    - U_c = 8.66955 ± 0.00221
    - ν^(-1) = 1.19164 ± 0.01170
  - FSE (All L)（对照）
    - U_c = 8.46616 ± 0.05714
    - ν^(-1) = 1.27767 ± 0.05298
- 说明：
  - Q阈值（高质量子集）：No-FSE Q≥103.16；FSE Q≥120.51
  - 样本数：No-FSE n=480（高质量120）；FSE n=360（高质量90）
- 详细版请见：`FINAL_SYNTHESIS_REPORT.md` 