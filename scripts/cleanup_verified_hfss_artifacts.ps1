param(
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$outputRoot = Join-Path $projectRoot 'hfss_outputs'
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$archiveRoot = Join-Path $outputRoot ("cleanup_archive_{0}" -f $runStamp)
$profileRoot = Join-Path $archiveRoot 'profiles'
$manifestPath = Join-Path $archiveRoot 'cleanup_manifest.json'

$relativeTargets = @(
    'hfss_outputs\modal_decoupling_20260714_run01\fullarray\ura16_tload3_l10p4_modal_candidate.aedtresults',
    'hfss_outputs\xcoupling_load_tune_8x8_20260716_run01\strict8x8_load_bar2p4_cap028_blend052\strict8x8_load_bar2p4_cap028_blend052.aedtresults',
    'hfss_outputs\hardware_xcoupling_20260716_run01\smooth_compact_l10p4_bar3p0_dx16p0\smooth_compact_l10p4_bar3p0_dx16p0.aedtresults',
    'hfss_outputs\embedded8x8_modal_smoke_20260716_run04\smooth_blended_l11p2_bar2p0\smooth_blended_l11p2_bar2p0.aedtresults',
    'hfss_outputs\grounded_patch_direct_16x16_resource_smoke_20260717_run01\grounded_patch_16x16\grounded_patch_16x16.aedtresults',
    'hfss_outputs\grounded_patch_direct_8x8_eep_gate_20260717_run01\grounded_patch_8x8\grounded_patch_8x8.aedtresults',
    'hfss_outputs\grounded_patch_direct_8x8_eep_gate_20260717_run02\grounded_patch_8x8\grounded_patch_8x8.aedtresults'
)

$active = Get-Process ansysedt, hf3d -ErrorAction SilentlyContinue
if ($active) {
    throw 'AEDT/HFSS processes are active. Cleanup aborted.'
}

New-Item -ItemType Directory -Path $profileRoot -Force | Out-Null
$rootPrefix = $projectRoot.TrimEnd('\') + '\'
$records = @()

foreach ($relativeTarget in $relativeTargets) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $relativeTarget))
    if (-not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes project root: $candidate"
    }
    if ([System.IO.Path]::GetFileName($candidate) -notlike '*.aedtresults') {
        throw "Refusing non-aedtresults target: $candidate"
    }

    if (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        $records += [pscustomobject]@{
            relative_path = $relativeTarget
            absolute_path = $candidate
            existed = $false
            bytes = 0
            profile_count = 0
            deleted = $false
        }
        continue
    }

    $files = @(Get-ChildItem -LiteralPath $candidate -Recurse -File -Force -ErrorAction SilentlyContinue)
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { $bytes = 0 }
    $profiles = @($files | Where-Object { $_.Extension -eq '.profile' })
    $targetTag = ($relativeTarget -replace '[^A-Za-z0-9._-]', '_')
    $targetProfileDir = Join-Path $profileRoot $targetTag

    if ($profiles.Count -gt 0) {
        New-Item -ItemType Directory -Path $targetProfileDir -Force | Out-Null
        foreach ($profile in $profiles) {
            $relativeProfile = $profile.FullName.Substring($candidate.Length).TrimStart('\')
            $profileTag = ($relativeProfile -replace '[^A-Za-z0-9._-]', '_')
            Copy-Item -LiteralPath $profile.FullName -Destination (Join-Path $targetProfileDir $profileTag) -Force
        }
    }

    $record = [pscustomobject]@{
        relative_path = $relativeTarget
        absolute_path = $candidate
        existed = $true
        bytes = [int64]$bytes
        profile_count = $profiles.Count
        deleted = $false
    }

    if ($Execute) {
        Remove-Item -LiteralPath $candidate -Recurse -Force
        $record.deleted = -not (Test-Path -LiteralPath $candidate)
        if (-not $record.deleted) {
            throw "Deletion verification failed: $candidate"
        }
    }
    $records += $record
}

$manifest = [ordered]@{
    created_at = (Get-Date).ToString('s')
    mode = $(if ($Execute) { 'execute' } else { 'dry_run' })
    project_root = $projectRoot
    policy = 'Delete verified HFSS mesh/field caches only; preserve projects, Touchstone, metrics, logs, scripts, and archived profiles.'
    total_candidate_bytes = [int64](($records | Measure-Object -Property bytes -Sum).Sum)
    deleted_bytes = [int64](($records | Where-Object deleted | Measure-Object -Property bytes -Sum).Sum)
    records = $records
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$manifest | ConvertTo-Json -Depth 5
