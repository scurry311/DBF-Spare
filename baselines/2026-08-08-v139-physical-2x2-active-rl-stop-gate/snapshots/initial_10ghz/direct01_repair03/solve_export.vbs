Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oSolutions, oReport, vars, variation, ports
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.OpenProject "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\v139_direct01_repair03.aedt"
Set oProject = oDesktop.SetActiveProject("v139_direct01_repair03")
Set oDesign = oProject.SetActiveDesign("V139_Physical2x2Differential")
oDesign.Analyze "Setup_10GHz"
oProject.Save
Set oSolutions = oDesign.GetModule("Solutions")
Set oReport = oDesign.GetModule("ReportSetup")
vars = oSolutions.ListVariations("Setup_10GHz:LastAdaptive")
variation = CStr(vars(LBound(vars)))
oSolutions.ExportNetworkData variation, Array("Setup_10GHz:LastAdaptive"), 3, "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\v139_direct01_repair03.s4p", Array("All"), True, 50, "S", -1, 0, 15, True, False, False
ports = Array("P00", "P10", "P01", "P11")
ApplyCalibration oSolutions, ports, Array("1W","0W","0W","0W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "basis_0", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_basis_0.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","1W","0W","0W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "basis_1", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_basis_1.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","0W","1W","0W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "basis_2", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_basis_2.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","0W","0W","1W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "basis_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_basis_3.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("1W","1W","0W","0W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "pair_re_0_1", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_re_0_1.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("1W","1W","0W","0W"), Array("0deg","90deg","0deg","0deg")
ExportEfficiency oReport, "pair_im_0_1", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_im_0_1.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("1W","0W","1W","0W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "pair_re_0_2", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_re_0_2.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("1W","0W","1W","0W"), Array("0deg","0deg","90deg","0deg")
ExportEfficiency oReport, "pair_im_0_2", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_im_0_2.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("1W","0W","0W","1W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "pair_re_0_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_re_0_3.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("1W","0W","0W","1W"), Array("0deg","0deg","0deg","90deg")
ExportEfficiency oReport, "pair_im_0_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_im_0_3.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","1W","1W","0W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "pair_re_1_2", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_re_1_2.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","1W","1W","0W"), Array("0deg","0deg","90deg","0deg")
ExportEfficiency oReport, "pair_im_1_2", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_im_1_2.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","1W","0W","1W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "pair_re_1_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_re_1_3.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","1W","0W","1W"), Array("0deg","0deg","0deg","90deg")
ExportEfficiency oReport, "pair_im_1_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_im_1_3.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","0W","1W","1W"), Array("0deg","0deg","0deg","0deg")
ExportEfficiency oReport, "pair_re_2_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_re_2_3.csv", "10GHz"
ApplyCalibration oSolutions, ports, Array("0W","0W","1W","1W"), Array("0deg","0deg","0deg","90deg")
ExportEfficiency oReport, "pair_im_2_3", "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\efficiency_pair_im_2_3.csv", "10GHz"
oProject.Save
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub ApplyCalibration(solModule, portArray, magnitudeArray, phaseArray)
    Dim sources, editArgs(), i, j, sourceName, portName, magnitude, phase
    sources = solModule.GetAllSources()
    ReDim editArgs(UBound(sources) + 1)
    editArgs(0) = Array("IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False)
    For i = LBound(sources) To UBound(sources)
        sourceName = CStr(sources(i))
        magnitude = "0W"
        phase = "0deg"
        For j = LBound(portArray) To UBound(portArray)
            portName = CStr(portArray(j))
            If LCase(Split(sourceName, ":")(0)) = LCase(portName) Then
                magnitude = CStr(magnitudeArray(j))
                phase = CStr(phaseArray(j))
            End If
        Next
        editArgs(i + 1) = Array("Name:=", sourceName, "Magnitude:=", magnitude, "Phase:=", phase)
    Next
    solModule.EditSources editArgs
End Sub
Sub ExportEfficiency(reportModule, stateName, outputPath, frequencyValue)
    Dim reportName
    reportName = "V139_Eff_" & stateName
    On Error Resume Next
    reportModule.DeleteReports Array(reportName)
    Err.Clear
    reportModule.CreateReport reportName, "Antenna Parameters", "Data Table", "Setup_10GHz : LastAdaptive", Array("Context:=", "InfiniteSphere_V139"), Array("Freq:=", Array(frequencyValue)), Array("X Component:=", "Freq", "Y Component:=", Array("RadiationEfficiency"))
    If Err.Number = 0 Then reportModule.ExportToFile reportName, outputPath
    On Error GoTo 0
End Sub
