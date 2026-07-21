$ErrorActionPreference = "Stop"

$scratch = [System.IO.Path]::GetFullPath("D:\HFSS_Stage").TrimEnd('\')
if ($scratch -ne "D:\HFSS_Stage") {
    throw "Unexpected scratch path: $scratch"
}

$active = Get-Process ansysedt, hf3d, HFSSCOMENGINE -ErrorAction SilentlyContinue
if ($active) {
    throw "Refusing to clean scratch while AEDT/HFSS processes are active"
}

$bytes = 0
if (Test-Path -LiteralPath $scratch -PathType Container) {
    $bytes = (Get-ChildItem -LiteralPath $scratch -Recurse -File -Force -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    Remove-Item -LiteralPath $scratch -Recurse -Force
}

[pscustomobject]@{
    removed_path = $scratch
    removed_bytes = [int64]$bytes
    d_free_bytes_after = [int64](Get-PSDrive D).Free
    completed_utc = (Get-Date).ToUniversalTime().ToString('o')
} | ConvertTo-Json
