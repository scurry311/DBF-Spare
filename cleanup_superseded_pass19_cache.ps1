$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath('D:\codex_workspace\hfss_ura16_quick_model')
$target = [System.IO.Path]::GetFullPath(
    'D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\grounded_patch_direct_16x16_staged_convergence_20260717_run01\grounded_patch_16x16\grounded_patch_16x16.aedtresults'
)
$targetRun = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetDirectoryName($target))
$rootPrefix = $projectRoot.TrimEnd('\') + '\'
if (-not $target.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Target escapes project root: $target"
}
if ([System.IO.Path]::GetFileName($target) -notlike '*.aedtresults') {
    throw "Refusing non-aedtresults target: $target"
}

$active = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -in @('ansysedt.exe', 'hf3d.exe', 'HFSSCOMENGINE.exe') }
)
foreach ($process in $active) {
    if ($process.CommandLine -and $process.CommandLine.Contains($targetRun)) {
        throw "Target is referenced by active process $($process.ProcessId)"
    }
}

$manifestDir = Join-Path $projectRoot 'hfss_outputs\cleanup_archive_20260718_pass19_superseded'
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
$manifestPath = Join-Path $manifestDir 'cleanup_manifest.json'
$files = @()
$bytes = 0
if (Test-Path -LiteralPath $target -PathType Container) {
    $files = @(Get-ChildItem -LiteralPath $target -Recurse -File -Force -ErrorAction SilentlyContinue)
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { $bytes = 0 }
}
$profiles = @($files | Where-Object { $_.Extension -eq '.profile' })
foreach ($profile in $profiles) {
    Copy-Item -LiteralPath $profile.FullName -Destination (
        Join-Path $manifestDir ("profile_{0}" -f $profile.Name)
    ) -Force
}
$beforeFree = (Get-PSDrive D).Free
if (Test-Path -LiteralPath $target -PathType Container) {
    Remove-Item -LiteralPath $target -Recurse -Force
}
$deleted = -not (Test-Path -LiteralPath $target)
if (-not $deleted) {
    throw "Deletion verification failed: $target"
}
$afterFree = (Get-PSDrive D).Free
$manifest = [ordered]@{
    created_at = (Get-Date).ToString('s')
    target = $target
    active_process_count = $active.Count
    target_referenced_by_active_process = $false
    archived_profile_count = $profiles.Count
    deleted = $deleted
    logical_bytes = [int64]$bytes
    physical_released_bytes = [int64]($afterFree - $beforeFree)
    preserved = @('project .aedt', 'per-pass S256 snapshots', 'per-pass profiles', 'CSV/JSON diagnostics')
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 4
