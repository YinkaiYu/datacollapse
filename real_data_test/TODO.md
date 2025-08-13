# Data Collapse QCP 项目 To-Do（持续更新）

更新时间：自动随实现进度更新

## 1) 基线与可视化
- [x] 交点法 U_c 基线：仅在公共窗口与重叠区内找交点，排除边界交点（完成：crossing_uc_extrapolation.png，U_c^cross≈8.696±0.002）
- [x] 非线性/平台外推：平台平均完成
- [x] 生成图：交点散点 + 平台区标注

## 2) No-FSE（Drop L=7）稳健化
- [x] 多起点网格：U_c∈[U_c^cross±0.07], a∈[1.0,1.4]，宽 bounds（robust_nofse_dropL7_scan.csv/.png）
- [x] 收集(U_c, a, Q)，对高质量子集（Q≥阈值）做统计（已在广义bootstrap中体现）
- [x] n_knots∈{8,10,12}, lam∈{1e-4,1e-3,1e-2} 灵敏度
- [x] 图：参数分布、质量-参数关系、最佳坍缩图（y轴 r'$R_{01}$'）
- [x] 广义bootstrap（多起点×外部bootstrap）与ΔQ估计（generalized_bootstrap_nofse_dropL7.*）

## 3) FSE（All-L）稳健化
- [x] 使用 fit_data_collapse_fse_robust：网格 b∈[0.4,1.2], c∈[-1.4,-0.4]（robust_fse_grid_scan.*）
- [x] bootstrap（Top单元小规模）统计 (U_c,a,b,c) 与 Q（robust_fse_bootstrap_top.csv）
- [x] 小型优化器对比：默认(NM→Powell) vs 仅Powell（compare_fse_optimizers.csv/.png）
- [x] 模态识别：低U_c高Q模态与更高U_c模态并存（图已展示）
- [x] 广义bootstrap（多起点×外部bootstrap）与ΔQ估计（generalized_bootstrap_fse_allL.*，已扩规模并标注中位数/ΔQ）

## 4) 质量等效带与统计整合
- [x] 估计 collapse 质量的统计波动 ΔQ（bootstrap/重采样）
- [x] |Q1−Q2|<ΔQ 视为“等效最优”，给出等效带内占比（equivalence_summary.csv）
- [x] 对比 U_c^cross 与两方法统计区间的一致性/偏差来源（equivalence_comparison.png + 报告解读）

## 5) 库级改进（如需）
- [x] 在 fit 接口中增加 optimizer / maxiter / random_restarts 可选参数
- [x] 文档化默认策略与新选项差异；极端敏感场景启用 random_restarts

## 6) 报告与交付
- [x] 以 Drop L=7 为基准的阶段性报告：图文并茂、直接插图（INITIAL_SENSITIVITY_AND_WIDE_SCAN_REPORT.md）
- [ ] 最终综合报告：给出“统计中心±不确定度 + 等效带内可接受区间”的统一结论与推荐参数（No-FSE为基准，FSE为对照）

## 6.1) 合作者交付包（新增，库发布前先行）
- [x] 三张图：原始数据、No-FSE（Drop L=7）坍缩、FSE（All-L）坍缩（collab_package/*.png）
- [x] 简短笔记：Ansatz 与推荐参数（中位数±不确定度）（collab_package/collab_note.md）
- [x] 一键复现脚本与依赖（reproduce_three_plots.py, requirements.txt）
- [x] 打包ZIP（collab_package.zip）

## 新增下一步（Next)
- [x] No-FSE/FSE：在关键图上直接标注中位数、IQR、Top值、ΔQ带（部分已做，继续统一风格）
- [x] FSE：进一步扩大广义bootstrap（n_trials≥300, n_boot_ext≥4），并局部加密(b,c)网格，验证 U_c 分布是否进一步向8.55–8.67收敛
- [x] No-FSE：补做 n_knots/lam 灵敏度小实验并汇总结论
- [x] 形成最终综合报告小节（结论/建议/表格）：
  - [x] 推荐值与不确定度：U_c、ν^(-1)，No-FSE（主）与FSE（对照）
  - [x] 等效带判定：列出等效带内外的占比与差异来源说明
  - [x] 可复制的命令与文件清单附录
- [ ] 报告图片替换为统一标注风格版本（如需重渲染）

## 7) datacollapse 库完善与发布（新增）
- [x] 功能增强：
  - [x] fit_data_collapse/fit_data_collapse_fse 增加 optimizer/maxiter/random_restarts 参数
  - [x] 提供稳健接口文档与示例（包含 fse_robust 的推荐网格与bootstrap用法）
  - [x] 进度回调/日志输出（可选），便于长任务监控
  - [x] 随机种子与可重复性控制（统一random_state传递）
- [ ] 工程与发布：
  - [x] 完善README（快速开始、API、示例、最佳实践、常见问题）
  - [x] 增加单元测试与CI（GitHub Actions）
  - [x] 版本化与变更记录（CHANGELOG）
  - [x] 在GitHub创建开源仓库并push代码（含许可协议）
  - [ ] 预留PyPI发布脚手架（可选）

## 8) MCP 集成（LLM工具化，新增）
- [ ] 需求与设计：
  - [ ] 定义MCP工具接口：fit_data_collapse、fit_data_collapse_fse、fit_data_collapse_fse_robust、collapse_transform、compute_quality
  - [ ] 输入/输出JSON Schema与参数校验（bounds、n_knots、lam、normalize、L_ref等）
  - [ ] 任务管理：长任务异步执行、进度事件（percent/ETA）、结果缓存与下载
  - [ ] 安全与限流：最大点数、超时、并发与速率限制、沙箱路径
- [ ] 实现：
  - [ ] MCP服务（Python FastAPI/uvicorn），封装datacollapse API
  - [ ] 事件/流式进度（Server-Sent Events或WebSocket）
  - [ ] 日志与审计（任务参数、时间、耗时、版本）
- [ ] 适配与示例：
  - [ ] LLM调用示例（JSON示例、典型错误与修正）
  - [ ] 端到端样例：提交任务→查询进度→获取结果→绘图
- [ ] 发布：
  - [ ] README与OpenAPI规范/使用说明
  - [ ] GitHub仓库与CI构建
  - [ ] （可选）Docker镜像与部署脚本 