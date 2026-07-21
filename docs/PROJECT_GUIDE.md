# 项目框架与执行指南

## 1. 实现目的

本项目研究 16x16、256 通道、10 GHz 多任务相控阵的自适应稀疏阵元选择。目标
不是在固定开启比例下单独压低阵因子旁瓣，而是在相同 weakest-target gain 或
相同 EIRP 的比较条件下，为每个目标方向集合找到同时满足以下约束的最小 ratio：

- 多波束主瓣方向和幅度保持；
- PSLL、最近目标隔离度和目标邻域泄漏满足门限；
- 逐激活端口及总反射的有源回波损耗满足门限；
- 单通道幅度、总权值范数、白噪声增益和功耗受控；
- HFSS 全波结果完整且数值收敛。

## 2. 系统框架

项目分为六层：

1. **物理基准层**：HFSS 阵元、馈电、互耦、S 参数和 EEP。
2. **数据层**：按 `sample_index` 组织场景、mask、任务级权值与 full-wave 指标。
3. **候选层**：结构化 seed、局部 swap/add/remove、历史 hard-positive 和生成模型。
4. **权值层**：区域零陷 LCMV/ZF、SOCP 或顺序凸投影，显式保留 `w_1...w_K`。
5. **学习层**：critic 学习 HFSS 相对 EEP 的残差、不确定性和候选组内排序。
6. **验证层**：EEP 门控、HFSS-in-loop 和按 K/ratio/扫描角/目标间隔的统计。

GAN 或注意力模型只负责提出候选，不负责证明物理可行。最终结论必须来自已收敛、
匹配合格且输出完整的 HFSS 全波结果。

## 3. 数据流

```text
目标方向集合 + K + 物理条件
  -> ratio=0.5 的 16-32 个结构化 mask
  -> EEP 流形上的区域 LCMV/SOCP
  -> active-RL / 功率投影
  -> EEP 严格 gate
  -> residual critic 的保守置信区间排序
  -> near-boundary / hard-negative / high-uncertainty 进入 HFSS
  -> 失败则局部搜索或提升 ratio
  -> 首个严格可行解与 ratio=1.0 对照
```

训练、验证和测试必须按独立 `sample_index` 分组，同一场景的所有 ratio、mask 和
权值变体必须位于同一个 split，防止场景泄漏。

## 4. 证据等级

| 等级 | 定义 | 允许用途 |
|---|---|---|
| A | HFSS 已收敛且通过物理/工程门限 | 工程结论与正式标签 |
| B | HFSS 输出有效，但收敛或 RF 门限未完全通过 | 诊断、hard negative |
| C | 基于受阻物理基准验证的 EEP 接口 | 管线验证 |
| D | AF、合成 S256 或学习代理 | 候选筛选、预训练 |
| X | 空结果、不完整、失效或过时结果 | 禁止进入训练 |

`combined_complete=1` 和 `isolation_complete=1` 是 full-wave 指标入库的必要条件，
但不能替代数值收敛、匹配和工程 gate。

## 5. 目录职责

| 路径 | 作用 | 是否进入 Git |
|---|---|---|
| `scripts/` | 全部可执行研究脚本与启动器 | 是 |
| `models/hfss/` | 小型、可复现的 HFSS 工程基准 | 是 |
| `docs/` | 项目规则、结果索引和设计决策 | 是 |
| `baselines/` | 精选结果快照、manifest 和校验和 | 是 |
| `tools/` | 仓库级索引和校验工具 | 是 |
| `hfss_outputs/` | 原始求解、数据集、图、checkpoint | 否 |
| `.python_deps/` | 本地 Python 依赖缓存 | 否 |

## 6. 当前可信结论

- 1x1、4x4、8x8 grounded-patch 基准通过各自的收敛和被动匹配门限。
- 256 端口 EEP 的线性叠加映射在 smoke 中达到约 `1e-12` 量级 NMSE。
- 16x16 DDM pass2 仍未达到 `Delta S <= 0.05`，且最差匹配后被动 RL 低于
  `10 dB`，所以不能开放新的工程训练标签。
- 现有 critic 可用于分析和预训练；在场景级严格 gate 上的 top-1 表现仍不足以
  替代物理门控。

## 7. 推荐执行顺序

1. 固定 0.18 mm 馈电邻域统一网格，继续验证连续两轮 Delta S。
2. 仅在 S256 收敛与匹配门限通过后重导可信 256 端口 EEP。
3. 用 50-100 个 direct HFSS case 检查 EEP 的主瓣、PSLL 和隔离度误差分布。
4. 重跑任务级 `w_1...w_K`、区域零陷和 active-RL 联合约束。
5. 检查 best-of-N oracle，只有候选空间确有可行解才训练或重训 critic。
6. 增量采集 near-boundary、hard-negative、hard-positive 和 ratio 配对样本。

任何大规模 HFSS 任务都必须先通过 smoke test、候选 oracle 检查和磁盘/内存检查。

## 8. 运行与版本管理

所有脚本默认从自身位置推导项目根目录，因此建议从仓库根目录运行：

```powershell
python scripts\<script_name>.py --help
powershell -ExecutionPolicy Bypass -File scripts\<task_name>.ps1
```

本地 `main` 已跟踪 `origin/main`。每次完成一个可验证的小阶段后，先更新结果索引
和文档，再执行 `git add -A`、`git commit`、`git push`。大型输出不应使用
`git add -f` 强行提交；需要保留的证据应通过 `tools/build_result_index.py` 生成
紧凑快照。
