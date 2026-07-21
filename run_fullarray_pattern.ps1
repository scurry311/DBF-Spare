$ErrorActionPreference = "Stop"

$AedtExe = "D:\v231\Win64\ansysedt.exe"
$ScriptPath = Join-Path $PSScriptRoot "run_fullarray_pattern.vbs"
$LogPath = [System.IO.Path]::ChangeExtension($ScriptPath, ".log")
$OutputDir = Join-Path $PSScriptRoot "hfss_outputs\fullarray_broadside"

if (-not (Test-Path -LiteralPath $AedtExe)) {
    throw "AEDT executable not found: $AedtExe"
}
if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Script not found: $ScriptPath"
}

if (Test-Path -LiteralPath $LogPath) {
    Remove-Item -LiteralPath $LogPath -Force
}

Write-Host "Running full-array broadside HFSS pattern export..."
Write-Host "  AEDT:   $AedtExe"
Write-Host "  Script: $ScriptPath"
Write-Host "  Output: $OutputDir"

Start-Process -FilePath $AedtExe -ArgumentList @("-RunScriptAndExit", $ScriptPath) | Out-Null

$deadline = (Get-Date).AddHours(3)
do {
    Start-Sleep -Seconds 15
    $running = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -ieq "ansysedt.exe" -and
            $_.CommandLine -like "*run_fullarray_pattern.vbs*"
        }
    if (Test-Path -LiteralPath $LogPath) {
        $last = Get-Content -LiteralPath $LogPath -Tail 6
        Write-Host ($last -join " | ")
    }
    if ((Get-Date) -gt $deadline) {
        throw "Timed out waiting for AEDT full-array pattern run."
    }
} while ($running)

if (Test-Path -LiteralPath $LogPath) {
    $logText = Get-Content -LiteralPath $LogPath -Raw
    if ($logText -match "\[error\]|Script Error|Script macro error") {
        throw "AEDT script reported an error. See $LogPath"
    }
}

$expected = @(
    "fullarray_gain_total_theta_phi.csv",
    "fullarray_gain_total_phi0.csv",
    "fullarray_gain_total_phi90.csv"
)
foreach ($name in $expected) {
    $path = Join-Path $OutputDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Expected output was not created: $path"
    }
}

Write-Host "Full-array pattern export completed."
