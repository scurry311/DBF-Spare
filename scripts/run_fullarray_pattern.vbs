Option Explicit

Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport
Dim projectPath, outDir, fso, file, sources, editArgs(), rootDir
Dim i, reportName, solutionName, sphereName

Set fso = CreateObject("Scripting.FileSystemObject")
rootDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
projectPath = rootDir & "\models\hfss\ura16_quick_10ghz_fullarray_run.aedt"
outDir = rootDir & "\hfss_outputs\fullarray_broadside"
sphereName = "InfiniteSphere_Theta0_90_Phi0_360"
solutionName = "Setup_10GHz : LastAdaptive"

EnsureFolder rootDir & "\hfss_outputs"
EnsureFolder outDir

Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject("ura16_quick_10ghz_fullarray_run")
Set oDesign = oProject.SetActiveDesign("URA16_Quick_10GHz")
Set oSol = oDesign.GetModule("Solutions")

sources = oSol.GetAllSources()
ReDim editArgs(UBound(sources) + 1)
editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
For i = LBound(sources) To UBound(sources)
    editArgs(i + 1) = Array("Name:=", CStr(sources(i)), "Magnitude:=", "0.00390625W", "Phase:=", "0deg")
Next
oSol.EditSources editArgs

Set file = fso.CreateTextFile(outDir & "\fullarray_source_summary.txt", True)
file.WriteLine "Project: " & projectPath
file.WriteLine "Design: URA16_Quick_10GHz"
file.WriteLine "Definition: all 256 ports active, equal magnitude, equal phase"
file.WriteLine "Magnitude: 0.00390625W incident power per Driven Modal port"
file.WriteLine "Total incident power: 1W"
file.WriteLine "Phase: 0deg"
file.WriteLine "Source count: " & CStr(UBound(sources) - LBound(sources) + 1)
file.Close

oProject.Save
oDesign.Analyze "Setup_10GHz"

Set oReport = oDesign.GetModule("ReportSetup")
DeleteIfExists oReport, "FullArray_GainTotal_ThetaPhi"
DeleteIfExists oReport, "FullArray_GainTotal_Phi0"
DeleteIfExists oReport, "FullArray_GainTotal_Phi90"

oReport.CreateReport "FullArray_GainTotal_ThetaPhi", "Far Fields", "Rectangular Contour Plot", solutionName, _
    Array("Context:=", sphereName), _
    Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("All")), _
    Array("X Component:=", "Theta", "Y Component:=", "Phi", "Z Component:=", Array("dB(GainTotal)")), _
    Array()
oReport.ExportToFile "FullArray_GainTotal_ThetaPhi", outDir & "\fullarray_gain_total_theta_phi.csv"

oReport.CreateReport "FullArray_GainTotal_Phi0", "Far Fields", "Rectangular Plot", solutionName, _
    Array("Context:=", sphereName), _
    Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("0deg")), _
    Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
    Array()
oReport.ExportToFile "FullArray_GainTotal_Phi0", outDir & "\fullarray_gain_total_phi0.csv"

oReport.CreateReport "FullArray_GainTotal_Phi90", "Far Fields", "Rectangular Plot", solutionName, _
    Array("Context:=", sphereName), _
    Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("90deg")), _
    Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
    Array()
oReport.ExportToFile "FullArray_GainTotal_Phi90", outDir & "\fullarray_gain_total_phi90.csv"

oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub EnsureFolder(path)
    If Not fso.FolderExists(path) Then
        fso.CreateFolder(path)
    End If
End Sub

Sub DeleteIfExists(reportModule, name)
    On Error Resume Next
    reportModule.DeleteReports Array(name)
    On Error GoTo 0
End Sub
