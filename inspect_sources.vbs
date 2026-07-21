Option Explicit

Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol
Dim projectPath, outPath, fso, file, sources, i

projectPath = "D:\codex_workspace\hfss_ura16_quick_model\ura16_quick_10ghz_fullarray_run.aedt"
outPath = "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\sources.txt"

Set fso = CreateObject("Scripting.FileSystemObject")
If Not fso.FolderExists("D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs") Then
    fso.CreateFolder("D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs")
End If

Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject projectPath
Set oProject = oDesktop.SetActiveProject("ura16_quick_10ghz_fullarray_run")
Set oDesign = oProject.SetActiveDesign("URA16_Quick_10GHz")
Set oSol = oDesign.GetModule("Solutions")

sources = oSol.GetAllSources()
Set file = fso.CreateTextFile(outPath, True)
For i = LBound(sources) To UBound(sources)
    file.WriteLine CStr(sources(i))
Next
file.Close

oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
