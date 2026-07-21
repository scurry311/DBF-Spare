# HFSS 工程文件

| 文件 | 作用 | 当前定位 |
|---|---|---|
| `smoke_test.aedt` | AEDT 自动化接口和工程保存的最小冒烟测试 | 工具链检查 |
| `ura16_quick_10ghz.aedt` | 初始 16x16 简化 PEC 偶极子快速模型 | 历史/快速建模参考 |
| `ura16_quick_10ghz_fullarray_run.aedt` | 全阵激励和方向图导出工程 | 旧 full-array 数据接口 |
| `ura16_quick_10ghz_matched_v2.aedt` | 修复几何接触并进行匹配处理的快速模型 | 旧 matched_v2 对照 |

这些工程文件用于保留自动化接口和历史可复现实验，不代表当前 16x16 grounded-patch
DDM 已通过严格工程 gate。大型 `.aedtresults` 求解树保存在本地 `hfss_outputs/`
或归档位置，不进入 Git。

脚本通过相对路径访问此目录。直接调用旧 VBS/PowerShell 启动器时，应从仓库内的
`scripts/` 使用对应启动文件，不要把工程复制回项目根目录。
