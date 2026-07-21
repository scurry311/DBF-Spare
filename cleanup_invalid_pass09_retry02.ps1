$ErrorActionPreference = "Stop"

$runRoot = [System.IO.Path]::GetFullPath(
    "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\grounded_patch_direct_16x16_portmesh_staged_convergence_20260718_run02"
).TrimEnd('\')
$invalid = [System.IO.Path]::GetFullPath(
    (Join-Path $runRoot "grounded_patch_16x16\grounded_patch_16x16.aedtresults\URA_GroundedPatch_10GHz.results\DV1332_S1289_V1775_F1382")
)
$lock = [System.IO.Path]::GetFullPath(
    (Join-Path $runRoot "grounded_patch_16x16\grounded_patch_16x16.aedt.lock")
)

if (-not $invalid.StartsWith($runRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Invalid result path is outside the audited run"
}
if ((Split-Path -Leaf $invalid) -ne "DV1332_S1289_V1775_F1382") {
    throw "Unexpected invalid result leaf"
}
$active = Get-Process ansysedt, hf3d, HFSSCOMENGINE -ErrorAction SilentlyContinue
if ($active) {
    throw "Refusing cleanup while AEDT/HFSS is active"
}

$bytes = 0
if (Test-Path -LiteralPath $invalid -PathType Container) {
    $bytes = (Get-ChildItem -LiteralPath $invalid -Recurse -File -Force |
        Measure-Object -Property Length -Sum).Sum
    Remove-Item -LiteralPath $invalid -Recurse -Force
}
if (Test-Path -LiteralPath $lock -PathType Leaf) {
    Remove-Item -LiteralPath $lock -Force
}

[pscustomobject]@{
    removed_invalid_result = $invalid
    removed_bytes = [int64]$bytes
    pass08_result_preserved = Test-Path -LiteralPath (
        Join-Path $runRoot "grounded_patch_16x16\grounded_patch_16x16.aedtresults\URA_GroundedPatch_10GHz.results\DV1332_S1289_V0_F1382"
    )
    d_free_bytes_after = [int64](Get-PSDrive D).Free
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json
