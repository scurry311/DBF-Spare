Option Explicit

Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol, oReport
Dim projectPath, outRoot, outDir, fso, sources, editArgs()
Dim i, sphereName, solutionName

projectPath = "D:\codex_workspace\hfss_ura16_quick_model\ura16_quick_10ghz_fullarray_run.aedt"
outRoot = "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs"
outDir = outRoot & "\eh_planes"
sphereName = "InfiniteSphere_Theta0_90_Phi0_360"
solutionName = "Setup_10GHz : LastAdaptive"

Set fso = CreateObject("Scripting.FileSystemObject")
EnsureFolder outRoot
EnsureFolder outDir

Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject("ura16_quick_10ghz_fullarray_run")
Set oDesign = oProject.SetActiveDesign("URA16_Quick_10GHz")
Set oSol = oDesign.GetModule("Solutions")

' Keep the report tied to the full-array broadside excitation:
' all 256 ports active, 1/256 W incident power per port, 1 W total, 0 deg.
sources = oSol.GetAllSources()
ReDim editArgs(UBound(sources) + 1)
editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
For i = LBound(sources) To UBound(sources)
    editArgs(i + 1) = Array("Name:=", CStr(sources(i)), "Magnitude:=", "0.00390625W", "Phase:=", "0deg")
Next
oSol.EditSources editArgs

Set oReport = oDesign.GetModule("ReportSetup")
DeleteIfExists oReport, "FullArray_EPlane_GainTotal_Phi0"
DeleteIfExists oReport, "FullArray_HPlane_GainTotal_Phi90"

' Dipoles are X-directed. With boresight along +Z:
' E-plane = XZ plane = Phi 0 deg; H-plane = YZ plane = Phi 90 deg.
oReport.CreateReport "FullArray_EPlane_GainTotal_Phi0", "Far Fields", "Rectangular Plot", solutionName, _
    Array("Context:=", sphereName), _
    Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("0deg")), _
    Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
    Array()
oReport.ExportToFile "FullArray_EPlane_GainTotal_Phi0", outDir & "\hfss_e_plane_phi0_raw.csv"

oReport.CreateReport "FullArray_HPlane_GainTotal_Phi90", "Far Fields", "Rectangular Plot", solutionName, _
    Array("Context:=", sphereName), _
    Array("Freq:=", Array("10GHz"), "Theta:=", Array("All"), "Phi:=", Array("90deg")), _
    Array("X Component:=", "Theta", "Y Component:=", Array("dB(GainTotal)")), _
    Array()
oReport.ExportToFile "FullArray_HPlane_GainTotal_Phi90", outDir & "\hfss_h_plane_phi90_raw.csv"

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
