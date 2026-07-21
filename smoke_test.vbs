Option Explicit

Dim oAnsoftApp, oDesktop, oProject
Dim projectPath

projectPath = "D:\codex_workspace\hfss_ura16_quick_model\smoke_test.aedt"

Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.SaveAs projectPath, True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
