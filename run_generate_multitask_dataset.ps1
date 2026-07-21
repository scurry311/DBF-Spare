$ErrorActionPreference = "Stop"

$root = "D:\codex_workspace\hfss_ura16_quick_model"
$script = Join-Path $root "generate_multitask_hfss_dataset.py"

python $script --samples-per-combo 100 --seed 20260625 --out-dir "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\multitask_dataset"
