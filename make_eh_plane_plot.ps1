$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$root = "D:\codex_workspace\hfss_ura16_quick_model"
$outDir = Join-Path $root "hfss_outputs\eh_planes"
$eCsv = Join-Path $outDir "e_plane_phi0_gain_total.csv"
$hCsv = Join-Path $outDir "h_plane_phi90_gain_total.csv"
$outputPath = Join-Path $outDir "eh_plane_1d_gain_total_labeled.png"

$eRows = Import-Csv -LiteralPath $eCsv
$hRows = Import-Csv -LiteralPath $hCsv

$width = 1200
$height = 760
$left = 100
$right = 1090
$top = 95
$bottom = 610
$plotW = $right - $left
$plotH = $bottom - $top

$bitmap = New-Object System.Drawing.Bitmap($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.Clear([System.Drawing.Color]::White)

$fontTitle = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
$font = New-Object System.Drawing.Font("Segoe UI", 11)
$fontSmall = New-Object System.Drawing.Font("Segoe UI", 9)
$brushText = [System.Drawing.Brushes]::Black
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(45, 45, 45), 1.4)
$penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225, 225, 225), 1)
$penE = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(214, 55, 55), 3)
$penH = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(45, 105, 195), 3)
$brushE = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(214, 55, 55))
$brushH = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(45, 105, 195))

function XForTheta([double]$theta) {
    return [float]($left + ($theta / 90.0) * $plotW)
}

function YForRel([double]$relDb) {
    if ($relDb -lt -40.0) { $relDb = -40.0 }
    if ($relDb -gt 0.0) { $relDb = 0.0 }
    return [float]($top + ((0.0 - $relDb) / 40.0) * $plotH)
}

function BuildPoints($rows) {
    $points = New-Object "System.Collections.Generic.List[System.Drawing.PointF]"
    foreach ($row in $rows) {
        $theta = [double]$row.Theta_deg
        $rel = [double]$row.GainTotal_relative_dB
        $points.Add((New-Object System.Drawing.PointF((XForTheta $theta), (YForRel $rel))))
    }
    return $points.ToArray()
}

$graphics.DrawString("HFSS Full-Array E/H Plane 1D Pattern", $fontTitle, $brushText, 100, 28)
$graphics.DrawString("GainTotal normalized to peak, 10 GHz, all 256 ports active", $font, $brushText, 100, 61)

foreach ($db in -40, -30, -20, -10, 0) {
    $y = YForRel $db
    $graphics.DrawLine($penGrid, $left, $y, $right, $y)
    $graphics.DrawString([string]$db, $fontSmall, $brushText, 58, $y - 8)
}

foreach ($theta in 0, 15, 30, 45, 60, 75, 90) {
    $x = XForTheta $theta
    $graphics.DrawLine($penGrid, $x, $top, $x, $bottom)
    $graphics.DrawString([string]$theta, $fontSmall, $brushText, $x - 8, $bottom + 10)
}

$graphics.DrawRectangle($penAxis, $left, $top, $plotW, $plotH)
$graphics.DrawString("Theta (deg)", $font, $brushText, $left + $plotW / 2 - 38, $bottom + 42)

$graphics.TranslateTransform(25, $top + $plotH / 2 + 55)
$graphics.RotateTransform(-90)
$graphics.DrawString("Normalized dB(GainTotal)", $font, $brushText, 0, 0)
$graphics.ResetTransform()

$ePoints = BuildPoints $eRows
$hPoints = BuildPoints $hRows
$graphics.DrawLines($penE, $ePoints)
$graphics.DrawLines($penH, $hPoints)

$graphics.FillRectangle($brushE, $right - 235, 50, 26, 10)
$graphics.DrawString("E-plane, Phi=0 deg", $fontSmall, $brushText, $right - 200, 43)
$graphics.FillRectangle($brushH, $right - 235, 72, 26, 10)
$graphics.DrawString("H-plane, Phi=90 deg", $fontSmall, $brushText, $right - 200, 65)

$peak = ($eRows + $hRows | ForEach-Object { [double]$_.GainTotal_dBi } | Measure-Object -Maximum).Maximum
$graphics.DrawString(("Peak GainTotal = {0:N4} dBi at Theta=0 deg" -f $peak), $fontSmall, $brushText, 100, $height - 55)
$graphics.DrawString("E-plane: XZ plane; H-plane: YZ plane. Dipole polarization is along X.", $fontSmall, $brushText, 100, $height - 32)

$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)

$graphics.Dispose()
$bitmap.Dispose()
$fontTitle.Dispose()
$font.Dispose()
$fontSmall.Dispose()
$penAxis.Dispose()
$penGrid.Dispose()
$penE.Dispose()
$penH.Dispose()
$brushE.Dispose()
$brushH.Dispose()

Get-Item -LiteralPath $outputPath | Select-Object FullName, Length, LastWriteTime
