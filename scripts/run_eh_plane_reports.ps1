$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$ansys = "D:\v231\Win64\ansysedt.exe"
$script = Join-Path $PSScriptRoot "run_eh_plane_reports.vbs"
$log = [System.IO.Path]::ChangeExtension($script, ".log")

if (-not (Test-Path -LiteralPath $ansys)) {
    throw "ansysedt.exe not found: $ansys"
}
if (-not (Test-Path -LiteralPath $script)) {
    throw "Script not found: $script"
}

Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath $ansys -ArgumentList @("-RunScriptAndExit", $script) -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddMinutes(15)

while (-not $proc.HasExited) {
    if ((Get-Date) -gt $deadline) {
        throw "Timed out waiting for HFSS E/H-plane report export."
    }
    Start-Sleep -Seconds 3
    $proc.Refresh()
}

if ($proc.ExitCode -ne 0) {
    throw "HFSS report export exited with code $($proc.ExitCode). See $log"
}

$outDir = Join-Path $root "hfss_outputs\eh_planes"
$expected = @(
    "hfss_e_plane_phi0_raw.csv",
    "hfss_h_plane_phi90_raw.csv"
)
foreach ($name in $expected) {
    $path = Join-Path $outDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Expected output not found: $path"
    }
}

Write-Host "HFSS E/H-plane report export completed."
