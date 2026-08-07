Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oReport, fso, quantityFile, statusFile, quantities, item
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\three_frequency_efficiency\direct_f9p96\v138_direct_f9p96.aedt"
Set oProject = oDesktop.SetActiveProject("v138_direct_f9p96")
Set oDesign = oProject.SetActiveDesign("V132_VerticalDifferential")
Set oReport = oDesign.GetModule("ReportSetup")
Set fso = CreateObject("Scripting.FileSystemObject")
Set quantityFile = fso.CreateTextFile("D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\available_quantities.txt", True)
On Error Resume Next
quantities = oReport.GetAllQuantities("Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"))
For Each item In quantities
    quantityFile.WriteLine CStr(item)
Next
quantityFile.Close
Set statusFile = fso.CreateTextFile("D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\export_status.csv", True)
statusFile.WriteLine "quantity,error_number"
Err.Clear
oReport.DeleteReports Array("V138_PowerAudit_RadiationEfficiency")
Err.Clear
oReport.CreateReport "V138_PowerAudit_RadiationEfficiency", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("9.96GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile "V138_PowerAudit_RadiationEfficiency", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\RadiationEfficiency.csv"
statusFile.WriteLine "RadiationEfficiency," & CStr(Err.Number)
Err.Clear
Err.Clear
oReport.DeleteReports Array("V138_PowerAudit_SystemEfficiency")
Err.Clear
oReport.CreateReport "V138_PowerAudit_SystemEfficiency", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("9.96GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("SystemEfficiency"))
If Err.Number = 0 Then oReport.ExportToFile "V138_PowerAudit_SystemEfficiency", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\SystemEfficiency.csv"
statusFile.WriteLine "SystemEfficiency," & CStr(Err.Number)
Err.Clear
Err.Clear
oReport.DeleteReports Array("V138_PowerAudit_RadiatedPower")
Err.Clear
oReport.CreateReport "V138_PowerAudit_RadiatedPower", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("9.96GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("RadiatedPower"))
If Err.Number = 0 Then oReport.ExportToFile "V138_PowerAudit_RadiatedPower", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\RadiatedPower.csv"
statusFile.WriteLine "RadiatedPower," & CStr(Err.Number)
Err.Clear
Err.Clear
oReport.DeleteReports Array("V138_PowerAudit_AcceptedPower")
Err.Clear
oReport.CreateReport "V138_PowerAudit_AcceptedPower", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("9.96GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("AcceptedPower"))
If Err.Number = 0 Then oReport.ExportToFile "V138_PowerAudit_AcceptedPower", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\AcceptedPower.csv"
statusFile.WriteLine "AcceptedPower," & CStr(Err.Number)
Err.Clear
Err.Clear
oReport.DeleteReports Array("V138_PowerAudit_IncidentPower")
Err.Clear
oReport.CreateReport "V138_PowerAudit_IncidentPower", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("9.96GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("IncidentPower"))
If Err.Number = 0 Then oReport.ExportToFile "V138_PowerAudit_IncidentPower", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\IncidentPower.csv"
statusFile.WriteLine "IncidentPower," & CStr(Err.Number)
Err.Clear
Err.Clear
oReport.DeleteReports Array("V138_PowerAudit_ReflectedPower")
Err.Clear
oReport.CreateReport "V138_PowerAudit_ReflectedPower", "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V132"), Array("Freq:=", Array("9.96GHz")), Array("X Component:=", "Freq", "Y Component:=", Array("ReflectedPower"))
If Err.Number = 0 Then oReport.ExportToFile "V138_PowerAudit_ReflectedPower", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\power_report_audit\direct_f9p96\ReflectedPower.csv"
statusFile.WriteLine "ReflectedPower," & CStr(Err.Number)
Err.Clear
statusFile.Close
On Error GoTo 0
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication
