param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$target = [System.IO.Path]::GetFullPath('D:\HFSS_FeedSheet08_Stage').TrimEnd('\')
$expected = 'D:\HFSS_FeedSheet08_Stage'
$archive = Join-Path $projectRoot 'hfss_outputs\cleanup_archive_20260723_fieldsolve_scratch'
$manifestPath = Join-Path $archive 'cleanup_manifest.json'

if (-not $target.Equals($expected, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unexpected cleanup target: $target"
}

$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('ansysedt.exe', 'ansysedtsv.exe', 'hf3d.exe', 'HFSSCOMENGINE.exe')
})
if ($active.Count -gt 0) {
    throw 'AEDT/HFSS processes are active. Scratch cleanup aborted.'
}

$files = @()
if (Test-Path -LiteralPath $target -PathType Container) {
    $files = @(Get-ChildItem -LiteralPath $target -File -Recurse -Force -ErrorAction SilentlyContinue)
}
$bytes = ($files | Measure-Object -Property Length -Sum).Sum
if ($null -eq $bytes) { $bytes = 0 }
$beforeFree = [int64](Get-PSDrive D).Free

$records = @(
    Get-ChildItem -LiteralPath $target -Directory -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            $childBytes = (Get-ChildItem -LiteralPath $_.FullName -File -Recurse -Force -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
            if ($null -eq $childBytes) { $childBytes = 0 }
            [pscustomobject]@{
                name = $_.Name
                bytes = [int64]$childBytes
                last_write = $_.LastWriteTime.ToString('s')
            }
        }
)

New-Item -ItemType Directory -Path $archive -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $target 'ansys_lmgrd_20260721.log')) {
    Copy-Item -LiteralPath (Join-Path $target 'ansys_lmgrd_20260721.log') -Destination $archive -Force
}

$deleted = $false
if ($Execute -and (Test-Path -LiteralPath $target -PathType Container)) {
    Remove-Item -LiteralPath $target -Recurse -Force
    $deleted = -not (Test-Path -LiteralPath $target)
    if (-not $deleted) { throw "Deletion verification failed: $target" }
}

$afterFree = [int64](Get-PSDrive D).Free
$manifest = [ordered]@{
    created_at = (Get-Date).ToString('s')
    mode = $(if ($Execute) { 'execute' } else { 'dry_run' })
    target = $target
    policy = 'Delete inactive AEDT solver scratch only; preserve project files, accepted baseline, profiles, S256 and compact audit evidence.'
    active_solver_process_count = $active.Count
    logical_bytes = [int64]$bytes
    deleted = $deleted
    physical_released_bytes = [int64]($afterFree - $beforeFree)
    child_records = $records
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 5
