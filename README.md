# DBF-Spare

## Current verified hardware status (v1.23)

The true balanced differential 1x1 launch passes its three-frequency HFSS S2
gate with 24.99 dB input RL and 95.27% insertion efficiency. A load-aware
single-block modal circuit passed its finite-Q and tolerance upper-bound gates,
but its 10 GHz physical S8 mapping reached only 3.15 dB active RL and 92.51%
actual-load insertion efficiency. It degraded 56 of 57 paired frozen stimuli,
so three-frequency HFSS, integrated 2x2, array expansion, training labels, and
critic retraining remain locked. See
`baselines/2026-08-04-v123-load-aware-modal-transformer-stop-gate/BASELINE.md`.

面向低功耗多任务波束的 16x16 稀疏相控阵研究项目。项目以 HFSS 全波模型和
嵌入单元方向图（EEP）为物理基准，联合优化阵元 mask、任务级复权值、旁瓣、
任务间隔离度、有源回波损耗与归一化功耗，并为每个任务场景寻找满足工程门限的
最小阵元开启比例。

## 项目状态

当前最新阶段基线为 `v1.18.0-equalized-post-transition`（2026-07-30）。v1.16
网络元件和门限全部冻结，四路 POST 扇出已等长为 `23.3928 mm`，并完成一体化
三频 HFSS S4 验证。等长化将最差 active RL 从 `1.601 dB` 提升到 `7.226 dB`，
但仍低于 `11 dB`；一体化与级联的 `max |Delta S|` 仅从 `0.307` 降到 `0.277`。
独立复验、4x4/16x16、标签和 critic 继续锁定，下一硬件分支必须采用一体化
多端口 POST 过渡/解耦，停止继续扫描公共长度和互不耦合的局部匹配段。

## 目录结构

```text
DBF-Spare/
|-- README.md                 项目入口和常用命令
|-- docs/                     项目框架、实验要求和结果索引
|-- scripts/                  数据、优化、训练、HFSS 与分析脚本
|-- models/hfss/              可版本化的 HFSS 工程文件
|-- baselines/<date>/         带 SHA-256 的紧凑证据快照
|-- tools/                    仓库级结果索引工具
`-- hfss_outputs/             本地大型仿真/训练输出，不进入 Git
```

详细说明：

- [项目框架与执行边界](docs/PROJECT_GUIDE.md)
- [脚本职责索引](scripts/README.md)
- [HFSS 模型说明](models/hfss/README.md)
- [当前结果索引](docs/RESULTS_INDEX.md)
- [实验要求与重建设计决策](docs/EXPERIMENT_REQUIREMENTS_AND_REBUILD_DECISION_20260717.md)
- [2026-07-21 基线](baselines/2026-07-21/BASELINE.md)

## 核心闭环

```text
任务场景编码
  -> 自适应 ratio 与结构化 mask 候选
  -> 区域鲁棒 LCMV/SOCP 任务权值
  -> 有源回波与功率约束投影
  -> EEP/full-wave 物理门控
  -> residual critic 候选排序
  -> 局部 mask 搜索
  -> HFSS 最终验证
```

稀疏搜索顺序为 `0.5 -> 0.6 -> 0.7 -> 0.8`，首次满足严格门限即停止；
`ratio=1.0` 只作全阵对照。

## 常用命令

从仓库根目录运行：

```powershell
# 验证当前基线快照没有被修改
python tools\build_result_index.py --tag 2026-07-25-prospective-hfss --verify-only

# 查看脚本参数，不启动大规模 HFSS
python scripts\run_staged_16x16_convergence.py --help

# 生成或重建紧凑结果索引（要求本地原始结果仍完整）
python tools\build_result_index.py --tag 2026-07-25-prospective-hfss

# 提交并同步后续代码修改
git add -A
git commit -m "describe the change"
git push
```

## 工程门限

- 连续两轮自适应求解 `Delta S <= 0.05`。
- S 矩阵互易性误差 `<= 1e-4`，最大奇异值 `<= 1.001`。
- 匹配后被动 RL、激活端口 active RL 和总反射 RL 均 `>= 10 dB`。
- PSLL 基础筛选 `<= 0 dB`，阶段目标 `-3 dB`，扩展目标 `-6 dB`。
- 最近任务隔离度 `>= 25 dB`，目标附近 +/-5 度隔离度 `>= 20 dB`。
- 最弱目标增益下降 `<= 0.5 dB`，多波束峰值幅度差 `<= 3 dB`。

原始 AEDT 结果树、生成数据集、checkpoint、场数据和临时文件体量超过 60 GiB，
由 `.gitignore` 排除。可复核的关键证据复制到 `baselines/<date>/snapshots/`，并在
manifest 中记录来源、证据等级和 SHA-256。
