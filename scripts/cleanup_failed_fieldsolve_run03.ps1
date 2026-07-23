param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'hfss_outputs\fixed_mesh_eep_fieldsolve_20260723_run03_ddm'))
$target = [System.IO.Path]::GetFullPath((Join-Path $runRoot 'scratch'))
$prefix = $runRoot.TrimEnd('\') + '\'
$archive = Join-Path $runRoot 'failure_audit'
$manifestPath = Join-Path $archive 'scratch_cleanup_manifest.json'

if (-not $target.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Cleanup target escapes failed run: $target"
}
if ([System.IO.Path]::GetFileName($target) -ne 'scratch') {
    throw "Refusing unexpected cleanup target: $target"
}
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('ansysedt.exe', 'ansysedtsv.exe', 'hf3d.exe', 'HFSSCOMENGINE.exe')
})
if ($active.Count -gt 0) { throw 'AEDT/HFSS processes are active. Cleanup aborted.' }

$files = @()
if (Test-Path -LiteralPath $target -PathType Container) {
    $files = @(Get-ChildItem -LiteralPath $target -File -Recurse -Force -ErrorAction SilentlyContinue)
}
$bytes = ($files | Measure-Object Length -Sum).Sum
if ($null -eq $bytes) { $bytes = 0 }
$extensionSummary = @(
    $files | Group-Object Extension | ForEach-Object {
        [pscustomobject]@{
            extension = $_.Name
            count = $_.Count
            bytes = [int64](($_.Group | Measure-Object Length -Sum).Sum)
        }
    }
)
$before = [int64](Get-PSDrive D).Free
New-Item -ItemType Directory -Path $archive -Force | Out-Null
$deleted = $false
if ($Execute -and (Test-Path -LiteralPath $target -PathType Container)) {
    Remove-Item -LiteralPath $target -Recurse -Force
    $deleted = -not (Test-Path -LiteralPath $target)
    if (-not $deleted) { throw "Deletion verification failed: $target" }
}
$after = [int64](Get-PSDrive D).Free
$manifest = [ordered]@{
    created_at = (Get-Date).ToString('s')
    mode = $(if ($Execute) { 'execute' } else { 'dry_run' })
    run_state = 'stopped_by_resource_guard_no_s256_no_field_label'
    target = $target
    preserved = @('project', 'aedtresults', 'prepare/import-smoke evidence', 'solve scripts/logs/status')
    logical_bytes = [int64]$bytes
    physical_released_bytes = [int64]($after - $before)
    deleted = $deleted
    extension_summary = $extensionSummary
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 5
