Option Explicit

Dim oAnsoftApp, oDesktop, oProject, oDesign, oSol
Dim projectPath, outPath, fso, file, sources, i, rootDir

Set fso = CreateObject("Scripting.FileSystemObject")
rootDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
projectPath = rootDir & "\models\hfss\ura16_quick_10ghz_fullarray_run.aedt"
outPath = rootDir & "\hfss_outputs\sources.txt"
If Not fso.FolderExists(rootDir & "\hfss_outputs") Then
    fso.CreateFolder(rootDir & "\hfss_outputs")
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
