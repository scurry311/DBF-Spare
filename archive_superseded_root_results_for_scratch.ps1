param(
    [string]$ProjectRoot = "D:\codex_workspace\hfss_ura16_quick_model",
    [string]$ArchiveRoot = "C:\Users\aqwer\hfss_archive\hfss_ura16_quick_model_20260718"
)

$ErrorActionPreference = "Stop"

$project = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
$archive = [System.IO.Path]::GetFullPath($ArchiveRoot).TrimEnd('\')
$names = @(
    "ura16_quick_10ghz_matched_v2.aedtresults",
    "ura16_quick_10ghz_fullarray_run.aedtresults"
)

New-Item -ItemType Directory -Path $archive -Force | Out-Null
$records = @()

foreach ($name in $names) {
    $source = [System.IO.Path]::GetFullPath((Join-Path $project $name))
    $target = [System.IO.Path]::GetFullPath((Join-Path $archive $name))

    if (-not $source.StartsWith($project + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing source outside project root: $source"
    }
    if (-not $target.StartsWith($archive + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing target outside archive root: $target"
    }
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Missing source directory: $source"
    }
    if (Test-Path -LiteralPath $target) {
        throw "Archive target already exists: $target"
    }

    $item = Get-Item -LiteralPath $source -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Source is already a reparse point: $source"
    }

    $bytes = (Get-ChildItem -LiteralPath $source -Recurse -File -Force |
        Measure-Object -Property Length -Sum).Sum
    $cDrive = Get-PSDrive -Name C
    if ($cDrive.Free -lt ($bytes + 8GB)) {
        throw "C drive lacks archive headroom for $name. Need payload plus 8 GiB reserve."
    }

    Move-Item -LiteralPath $source -Destination $target
    New-Item -ItemType Junction -Path $source -Target $target | Out-Null

    $records += [pscustomobject]@{
        name = $name
        source_junction = $source
        archive_target = $target
        bytes = [int64]$bytes
        moved_utc = (Get-Date).ToUniversalTime().ToString('o')
    }
}

$manifest = [pscustomobject]@{
    project_root = $project
    archive_root = $archive
    records = $records
    d_free_bytes_after = [int64](Get-PSDrive -Name D).Free
    c_free_bytes_after = [int64](Get-PSDrive -Name C).Free
}
$manifestPath = Join-Path $archive "archive_manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 5
