$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "generate_multitask_hfss_dataset.py"

python $script --samples-per-combo 100 --seed 20260625 --out-dir (Join-Path $root "hfss_outputs\multitask_dataset")
