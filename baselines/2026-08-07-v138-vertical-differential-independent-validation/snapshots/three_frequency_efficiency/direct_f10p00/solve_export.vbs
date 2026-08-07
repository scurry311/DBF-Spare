Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, reportName
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\three_frequency_efficiency\direct_f10p00\v138_direct_f10p00.aedt"
Set oProject = oDesktop.SetActiveProject("v138_direct_f10p00")
Set oDesign = oProject.SetActiveDesign("V132_VerticalDifferential")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\three_frequency_efficiency\direct_f10p00\v138_direct_f10p00.s1p", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
Set oReport = oDesign.GetModule("ReportSetup")
reportName = "V138_RadiationEfficiency"
On Error Resume Next
oReport.DeleteReports Array(reportName)
Err.Clear
oReport.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("10GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile reportName, "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\three_frequency_efficiency\direct_f10p00\radiation_efficiency.csv"
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
