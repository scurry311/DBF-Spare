Option Explicit

Dim oAnsoftApp, oDesktop, oProject
Dim projectPath, fso, rootDir

Set fso = CreateObject("Scripting.FileSystemObject")
rootDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
projectPath = rootDir & "\models\hfss\smoke_test.aedt"

Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.SaveAs projectPath, True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
