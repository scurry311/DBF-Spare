Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\independent_10ghz\direct_repeat_a\v138_direct_repeat_a.aedt"
Set oProject = oDesktop.SetActiveProject("v138_direct_repeat_a")
Set oDesign = oProject.SetActiveDesign("V132_VerticalDifferential")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\independent_10ghz\direct_repeat_a\v138_direct_repeat_a.s1p", Array("All"), True, 50, "S", -1, 0, 15, True, False, False

oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
