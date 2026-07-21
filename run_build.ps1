$ErrorActionPreference = "Stop"

$AedtExe = "D:\v231\Win64\ansysedt.exe"
$ScriptPath = Join-Path $PSScriptRoot "build_ura16_quick.vbs"
$ProjectPath = Join-Path $PSScriptRoot "ura16_quick_10ghz.aedt"

if (-not (Test-Path -LiteralPath $AedtExe)) {
    throw "AEDT executable not found: $AedtExe"
}

if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Build script not found: $ScriptPath"
}

Write-Host "Running AEDT script..."
Write-Host "  AEDT:    $AedtExe"
Write-Host "  Script:  $ScriptPath"
Write-Host "  Output:  $ProjectPath"

$logPath = [System.IO.Path]::ChangeExtension($ScriptPath, ".log")
if (Test-Path -LiteralPath $logPath) {
    Remove-Item -LiteralPath $logPath -Force
}

$process = Start-Process -FilePath $AedtExe -ArgumentList @("-RunScriptAndExit", $ScriptPath) -PassThru
$deadline = (Get-Date).AddMinutes(25)

do {
    Start-Sleep -Seconds 5
    $running = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -ieq "ansysedt.exe" -and
            $_.CommandLine -like "*build_ura16_quick.vbs*"
        }
    if ((Get-Date) -gt $deadline) {
        throw "Timed out waiting for AEDT script to finish."
    }
} while ($running)

if (Test-Path -LiteralPath $logPath) {
    $logText = Get-Content -LiteralPath $logPath -Raw
    if ($logText -match "\[error\]|Script Error|Script macro error") {
        throw "AEDT script reported an error. See $logPath"
    }
}

if (-not (Test-Path -LiteralPath $ProjectPath)) {
    throw "Expected project was not created: $ProjectPath"
}

Write-Host "Done: $ProjectPath"
