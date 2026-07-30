# 脚本职责索引

所有 Python、PowerShell 和 VBScript 可执行文件集中在本目录，以保持旧脚本之间
的直接导入兼容。以下按研究职责分类；文件在磁盘上保持平铺，避免改变历史调用链。

## 数据构建与候选生成

| 文件 | 作用 |
|---|---|
| `build_fullwave_metric_dataset_v2.py` | 扫描完整 full-wave 指标，去重并按场景构建残差 critic 数据集。 |
| `build_stage1_metric_dataset.py` | 构建第一阶段指标数据集及 train/val/test 场景分组。 |
| `generate_adaptive_candidate_pool_v2.py` | 生成自适应 ratio、结构化 mask 和多策略候选池。 |
| `generate_guarded_fixed_count_teacher.py` | 在固定激活数下生成带物理门控的 teacher。 |
| `generate_mixed_guarded_canonical_teacher.py` | 合并多种门控 teacher，形成统一 canonical 样本。 |
| `generate_multitask_hfss_dataset.py` | 生成 K=1/2/4/6、多 ratio 的目标方向、mask 和任务权值基础数据。 |
| `generate_optimized_teacher_labels.py` | 生成早期 PSLL/mask 优化 teacher 标签。 |
| `prepare_round6_fullwave_candidates.py` | 规划并物化 Round6 的 1000 个 full-wave 候选及 HFSS 分块任务。 |
| `prepare_task_eep_smoke_and_paired_scenes.py` | 准备任务级 EEP/HFSS smoke、ratio 配对和小间隔场景。 |
| `rerank_stage1_candidates.py` | 使用 Stage-1 critic 对场景组内候选重排序。 |

## 权值、mask 与匹配优化

| 文件 | 作用 |
|---|---|
| `build_grounded_patch_s256_proxy.py` | 从小阵列/局部耦合信息构建 grounded-patch S256 代理，仅供诊断。 |
| `design_eep_port_match.py` | 基于完整 S 参数分析端口类别并设计 EEP/端口匹配参数。 |
| `design_modal_subarray_network.py` | 设计 2x2/4x4 子阵偶模、奇模解耦与匹配网络。 |
| `design_v115_grounded_modal_network.py` | 在可信 S4 上综合有限 Q 双参考面横向偶奇模网络，并分开计算 transducer/insertion efficiency。 |
| `design_port_class_matching.py` | 对角点、边缘、内部等端口类别设计匹配网络。 |
| `experiment_fixed_count_swap_psll.py` | 在固定阵元数下执行局部 swap 搜索以改善 PSLL。 |
| `generate_iso_lcmv_teacher.py` | 生成带任务隔离约束的 LCMV/ZF teacher 权值。 |
| `project_active_return_weights.py` | 将权值投影到 active-RL、幅度和功率约束附近。 |
| `reconstruct_task_lcmv_socp_psll.py` | 重构 `w_1...w_K`，执行区域零陷 LCMV/SOCP 与 PSLL 二级优化。 |
| `run_pagan_lite_mvp.py` | 执行 physics-aware GAN 思路的轻量 MVP 候选实验。 |
| `run_stage_a_multiscenario_matching.py` | 在 2400 个场景上进行跨场景 minimax 匹配与 active-RL 评估。 |

## 模型训练

| 文件 | 作用 |
|---|---|
| `train_fullwave_residual_critic_v2.py` | 训练 HFSS 相对 AF/EEP 的 full-wave residual critic。 |
| `train_grounded_patch_active_return_proxy_critic.py` | 训练 grounded-patch active-return 代理 critic。 |
| `train_hfss_grid_mask.py` | 训练二维 16x16 网格 mask 模型。 |
| `train_hfss_steering_mask.py` | 训练条件于目标方向的稀疏 mask 模型。 |
| `train_hfss_surrogate.py` | 训练基础 HFSS/阵列指标 surrogate。 |
| `train_stage1_metric_critic.py` | 训练 Stage-1 指标 critic，并输出分类、校准和排序指标。 |

## HFSS 建模、导出与收敛

| 文件 | 作用 |
|---|---|
| `export_full_s256p.py` | 从 matched_v2 工程导出完整 256 端口 Touchstone。 |
| `export_grounded_patch_eep_operator.py` | 导出 grounded-patch 端口 Etheta/Ephi、S 参数并构建 EEP 算子。 |
| `hfss_task_fullwave_validate.py` | 回灌 combined/task 权值，导出 full-wave 方向图并计算任务级指标。 |
| `prepare_16x16_feed_neighborhood_mesh.py` | 为 256 个馈电邻域建立统一确定性局部网格。 |
| `prepare_16x16_feed_sheet_mesh.py` | 为馈电 sheet 建立统一网格和 DDM 恢复工程。 |
| `prepare_16x16_portmesh_stabilization.py` | 构建 16x16 端口区域网格稳定化任务。 |
| `prepare_eep_16port_smoke.py` | 选择代表端口，准备 16 端口 EEP 和 S 参数 smoke。 |
| `prepare_matched_ura16_model.py` | 从快速模型构建修复几何接触的 matched_v2 对照工程。 |
| `prescreen_xcoupling_retune_4x4.py` | 对固定 dx 的端加载和横向耦合候选进行 4x4 预筛。 |
| `run_balanced_feed_smoke.py` | 构建并运行显式平衡馈电 smoke。 |
| `run_cps_microstrip_balun_smoke.py` | 构建带参考地的 CPS/微带 balun 与匹配 smoke。 |
| `run_embedded8x8_geometry_smoke.py` | 将通过预筛的几何扩展到严格 8x8 嵌入式验证。 |
| `run_endload_8x8_followup.py` | 对端加载几何执行 8x8 后续求解和门控。 |
| `run_geometry_feed_smoke.py` | 生成并验证小阵列几何/馈电候选。 |
| `run_grounded_patch_array_rebuild.py` | 重建更易收敛的 grounded-patch 1x1/4x4/8x8/16x16 工程。 |
| `run_modal_fullarray_rebuild.py` | 将模态解耦/匹配候选扩展到全阵工程。 |
| `run_sanitized_endload_validation.py` | 验证消除切触和微小重叠后的单一实体端加载几何。 |
| `run_staged_16x16_convergence.py` | 分阶段续算 16x16，记录每轮 Delta S、资源和 RF 门限。 |
| `run_twoport_balanced_match_fixture.py` | 构建 1x1/2x2 真实两端口平衡馈电与匹配夹具。 |
| `run_v114_small_cell_broadband_feed.py` | 分阶段构建、求解并门控 v1.14 的 1x1/2x2/4x4 物理宽带馈电候选；硬停止禁止越过失败的小阵门。 |
| `run_v115_physical_modal_feed_fixture.py` | 构建四 PRE/四 POST 参考面的物理 S8 馈电夹具，回放可信 S4 和冻结代表激励。 |
| `validate_eep_superposition_smoke.py` | 对比 EEP 直接叠加与相同权值的 direct HFSS 复场。 |

## 实验性收敛恢复与资源审计

| 文件 | 作用与当前状态 |
|---|---|
| `audit_port_mesh_consistency.py` | 审计 256 端口定义、积分线、网格覆盖和 small-segment 分布。 |
| `prepare_clean_ddm4_restart.py` | 保留物理几何和 0.18 mm 面网格，删除旧 MeshLink 解并创建轻量 DDM4 分支。 |
| `run_volumetric_feedmesh_ddm.py` | 通用分阶段 DDM 控制器；包含内存、矩阵规模、拓扑和 Delta S 硬停止条件。 |
| `run_guarded_ddm_continuation.py` | 对旧 feedsheet 分支执行带反弹检测的受控续算。 |
| `prepare_volumetric_feedmesh_recovery.py` | 创建 256 个全深度体积馈电网格区域；本机内存不足时不应继续。 |
| `prepare_volumetric_feedmesh_pecsheet.py` | 在体积网格分支中将 patch/ground 转为零厚度 PEC sheet。 |
| `prepare_pecsheet_perfecte_recovery.py` | 为零厚度 PEC sheet 显式添加 Perfect E 边界。 |
| `prepare_ddm6_pecsheet_branch.py` | 创建六域并行资源 smoke；24 GB 主机上存在 OOM 风险。 |
| `prepare_layered_feedmesh_recovery.py` | 创建 core/halo 分层体积网格候选。 |
| `prepare_layered_mesh_relaxation.py` | 将分层网格放宽到 0.22/0.30 mm 以降低内存。 |
| `collect_layered_pass1_failure.py` | 保存分层网格首轮 late-OOM 证据并锁定标签。 |
| `summarize_volumetric_feedmesh_recovery.py` | 汇总体积、PEC sheet、DDM6 等恢复分支及失败原因。 |

体积和分层网格脚本属于受控实验记录，不是当前推荐主线。当前主线是
`prepare_clean_ddm4_restart.py` 加 `run_volumetric_feedmesh_ddm.py --ddm-tasks 4`；
只有连续两轮 `Delta S <= 0.05` 且 RL、互易性、无源性同时通过后才能开放标签。

## 分析、比较与阶段判定

| 文件 | 作用 |
|---|---|
| `analyze_ddm_recovery_stage.py` | 分析 16x16 DDM 恢复阶段的 S 参数、Delta S 和工程 gate。 |
| `audit_meshlink_baseline.py` | 对比旧 MeshLink、原 direct 与 clean DDM 快照，审计逐端口阻抗、耦合、匹配灵敏度和固定网格验证决策。 |
| `run_fixed_mesh_ddm_run02.py` | 建立独立 run02，移除目标侧二次局部细化，依次执行 MeshLink import smoke、资源受控 DDM、S256 导出及 old pass5 交叉验证。 |
| `run_fixed_mesh_direct_crosscheck.py` | 独立执行相同固定网格的 Direct Solver，管理 D 盘 off-core scratch，并完成 direct/DDM 交叉门控与版本化工程基准冻结。 |
| `run_fixed_mesh_eep_fieldsolve.py` | 从可信固定网格建立保存辐射场的资源受控 DDM 求解，并与 direct S256 交叉验证。 |
| `run_fixed_mesh_eep_ddm_fieldsolve_run05.py` | 以 80% 内存门限启动通过的 run05 场求解配置；run03/run04 包装器保留失败审计参数。 |
| `prepare_trusted_eep_residual_validation.py` | 构造 K=1/2/4/6、跨 ratio、大扫描角及 K=6 小间隔的 96 候选新标签集。 |
| `validate_trusted_eep_hfss_residuals.py` | 执行无比例校正的 EEP/HFSS 复场验证，生成方向图残差、S256 匹配指标、工程 gate 和场景级划分。 |
| `package_trusted_residual_dataset.py` | 在不覆盖原始结果的前提下生成带标准键、schema 和证据清单的 critic 数据包。 |
| `gate_trusted_residual_critic_training.py` | 检查残差尺度、正负样本支持和 active-RL 可行集，决定是否允许训练神经 residual critic。 |
| `audit_trusted_active_rl_semantics.py` | 在可信 matched S256 上比较 combined/task-level 及 -20/-30/-40 dB 有效驱动端口的 active-RL 口径。 |
| `optimize_trusted_eep_s256_joint_weights.py` | 联合优化任务级 `w1...wK`，保持 EEP 主响应和区域泄漏约束，同时投影 combined/task active RL。 |
| `evaluate_trusted_eep_s256_joint_smoke.py` | 汇总96候选联合优化结果，输出 K/ratio/扫描角统计、场景oracle和HFSS标签开放决定。 |
| `refine_trusted_dense_local_eep_joint.py` | 用稠密 local-5deg EEP 算子、合成目标点等式和 S256 active-RL 投影重新优化同一96候选。 |
| `prepare_dense_joint_hfss_smoke.py` | 仅将通过严格工程门控的稀疏多波束候选打包为50-100 case HFSS任务。 |
| `evaluate_dense_joint_hfss_smoke.py` | 审计稠密优化后的HFSS批次，按K/ratio/扫描角汇总并分别判定物理标签与residual critic是否开放。 |
| `generate_dense_boundary_hard_negatives.py` | 围绕可信正样本生成可复现的幅相量化、通道失效门限扰动及低ratio配对候选。 |
| `generate_expanded_independent_residual_scenes.py` | 生成独立目标方向场景、中等实现误差、定向主瓣失败及重新优化的低ratio配对样本。 |
| `generate_gate15_boundary_scenes.py` | 生成PSLL、nearest isolation与local isolation门内/门外/对照三元组，并强制新增样本保持主瓣通过。 |
| `prepare_frozen_critic_prospective_validation.py` | 在读取任何新HFSS结果前冻结checkpoint、pooled calibrator、输入替代、门限、候选概率、场景选择和SHA-256证据。 |
| `evaluate_frozen_critic_prospective_hfss.py` | 将事前冻结预测与未见方向集合的完整HFSS标签配对，核验哈希/门限/方向泄漏并输出前瞻AUROC、校准、准入和场景级bootstrap指标。 |
| `audit_gate15_boundary_results.py` | 严格审计三类full-wave门限跨越、主瓣保持、复场重构和分组裕量。 |
| `subset_hfss_candidate_dataset.py` | 非破坏性抽取候选轴子集，用于正式HFSS批次前的小规模映射smoke。 |
| `build_dense_implementation_residual_dataset.py` | 合并正样本、实现误差和三类gate15边界样本，按sample_index分组构建critic训练数据。 |
| `calibrate_pooled_residual_critic.py` | 仅用场景分组验证集拟合跨种子正则isotonic校准器并输出五种子校准指标。 |
| `evaluate_dense_implementation_residual_critic.py` | 汇总五随机种子critic指标，并在显式绑定pooled calibrator时执行AUROC、ECE和支持度晋级门控。 |
| `analyze_eep_16port_operator.py` | 解析 16 端口复数 EEP 并构建/检查线性算子。 |
| `analyze_full_s256p_active_return.py` | 计算 S256 下的 active impedance、逐端口/总反射 RL 分布。 |
| `analyze_paired_task_results.py` | 分析相同方向集合下不同 ratio/权值的配对任务结果。 |
| `analyze_staged_smatrix_stability.py` | 汇总多轮 S 矩阵变化、互易性、无源性和连续收敛状态。 |
| `assess_hfss_model_suitability.py` | 评估当前 HFSS 建模是否适合作为训练和工程基准。 |
| `compare_embedded8x8_hardware.py` | 配对比较 8x8 原始与优化硬件候选的匹配和耦合。 |
| `evaluate_embedded8x8_hierarchical_network.py` | 评估 8x8 类别匹配与跨 4x4 二级模态网络。 |
| `evaluate_projected_combined_af.py` | 分析权值投影前后的 combined AF 指标，仅作代理证据。 |
| `evaluate_stage1_acceptance.py` | 根据预设指标判断 Stage-1/Stage-2 是否验收。 |
| `evaluate_teacher_model_vs_original.py` | 比较 teacher/model 与原始 mask/权值的方向图和功耗指标。 |
| `make_eh_plane_outputs.py` | 清洗 HFSS E/H 面数据并输出 CSV/PNG。 |
| `merge_task_lcmv_psll_runs.py` | 合并分批任务级 LCMV/SOCP/PSLL 结果。 |
| `summarize_active_matching_stage.py` | 汇总有源匹配阶段按 K、ratio、扫描角的真实指标。 |
| `summarize_eep_hfss_joint_smoke_stage.py` | 汇总 EEP/HFSS 联合 smoke，并生成标签开放/锁定决定。 |
| `summarize_v114_small_cell_broadband_feed.py` | 汇总 v1.14 小阵馈电、模态耦合、冻结 active-RL 回放和阶段锁定决定。 |
| `calibrate_v115_physical_fixture.py` | 用物理 S8 校准分布式夹具 surrogate，并仅在同一拓扑内做物理感知重综合。 |
| `summarize_v115_grounded_modal_network.py` | 固化 v1.15 电路、物理 S8、双参考面效率和停止决定。 |
| `run_v116_physical_s8_hfss_optimization.py` | 用 HFSS LocalVariables 顺序执行物理 S8 粗筛、局部优化和独立复验，并以冻结激励做双参考面门控。 |
| `summarize_v116_physical_s8_hfss_optimization.py` | 固化 v1.16 的 14 个精确 HFSS S8、最佳元件、独立复验和阶段权限。 |
| `run_v117_integrated_2x2_smoke.py` | 构建并门控共地双层馈电网络、物理扇出、探针和四贴片一体化 2x2 HFSS，导出 S4、三频 EEP 和功率账本。 |
| `summarize_v117_integrated_2x2_smoke.py` | 固化 v1.17 一体化 2x2 的资源、S4、EEP、失败门控和后续锁定决定。 |

## AEDT 启动器与历史快速模型

| 文件 | 作用 |
|---|---|
| `build_ura16_quick.vbs` | 在 AEDT 中创建初始 16x16 PEC 偶极子快速工程。 |
| `inspect_sources.vbs` | 导出 full-array 工程的 HFSS source 名称。 |
| `run_eh_plane_reports.vbs` | 设置全阵等功率激励并导出 E/H 面报告。 |
| `run_fullarray_pattern.vbs` | 求解并导出全阵 broadside 二维/一维方向图。 |
| `smoke_test.vbs` | 检查 AEDT COM 自动化和工程保存功能。 |
| `run_build.ps1` | 调用 AEDT 执行 `build_ura16_quick.vbs` 并检查日志/工程。 |
| `run_eh_plane_reports.ps1` | 调用 E/H 面 VBS 并验证输出文件。 |
| `run_fullarray_pattern.ps1` | 调用全阵方向图 VBS，等待求解并检查导出。 |
| `run_generate_multitask_dataset.ps1` | 使用固定参数启动基础多任务数据生成。 |
| `make_eh_plane_plot.ps1` | 用 PowerShell/System.Drawing 绘制 E/H 面对比图。 |

## 空间整理与维护

| 文件 | 作用 |
|---|---|
| `archive_superseded_root_results_for_scratch.ps1` | 将确认过时的大型 `.aedtresults` 迁移到归档盘并建立 junction。 |
| `cleanup_failed_hfss_scratch_20260718.ps1` | 清理已确认失败的 HFSS scratch，保留审计信息。 |
| `cleanup_invalid_pass09_retry02.ps1` | 定点删除 pass09 retry02 的无效求解节点和锁文件。 |
| `cleanup_fieldsolve_scratch_20260723.ps1` | 仅清理已完成并通过验证的 fieldsolve 临时 scratch，保留求解结果。 |
| `cleanup_failed_fieldsolve_run03.ps1` | 清理失败 run03 的大体积 scratch，并生成文件哈希审计清单。 |
| `cleanup_superseded_pass19_cache.ps1` | 删除被替代的 pass19 cache，并保存 profile/manifest。 |
| `cleanup_verified_hfss_artifacts.ps1` | 对白名单 HFSS 缓存执行 dry-run 或显式 `-Execute` 清理。 |

## 调用约定

```powershell
# Python 脚本
python scripts\<name>.py --help

# PowerShell 启动器
powershell -ExecutionPolicy Bypass -File scripts\<name>.ps1
```

清理脚本具有破坏性，只能在 AEDT/HFSS 进程停止、目标路径审计通过且确认已有必要
快照后使用。`cleanup_verified_hfss_artifacts.ps1` 默认只准备审计信息，实际删除必须
显式传入 `-Execute`。
