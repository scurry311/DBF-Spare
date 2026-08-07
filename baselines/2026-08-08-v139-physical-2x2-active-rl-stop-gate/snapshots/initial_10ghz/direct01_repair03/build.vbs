Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V139", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "2.2", "dielectric_loss_tangent:=", "0.0009")
oProject.InsertDesign "HFSS", "V139_Physical2x2Differential", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("V139_Physical2x2Differential")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "SubstratePart_00_00", -15.0000000, -15.0000000, -0.7870000, 6.7700000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_00_01", -15.0000000, -7.6800000, -0.7870000, 6.7700000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_00_02", -15.0000000, -7.3200000, -0.7870000, 6.7700000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_00_03", -15.0000000, 7.3200000, -0.7870000, 6.7700000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_00_04", -15.0000000, 7.6800000, -0.7870000, 6.7700000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_01_00", -8.2300000, -15.0000000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_01_02", -8.2300000, -7.3200000, -0.7870000, 0.3600000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_01_04", -8.2300000, 7.6800000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_02_00", -7.8700000, -15.0000000, -0.7870000, 0.7400000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_02_01", -7.8700000, -7.6800000, -0.7870000, 0.7400000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_02_02", -7.8700000, -7.3200000, -0.7870000, 0.7400000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_02_03", -7.8700000, 7.3200000, -0.7870000, 0.7400000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_02_04", -7.8700000, 7.6800000, -0.7870000, 0.7400000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_03_00", -7.1300000, -15.0000000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_03_02", -7.1300000, -7.3200000, -0.7870000, 0.3600000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_03_04", -7.1300000, 7.6800000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_04_00", -6.7700000, -15.0000000, -0.7870000, 13.5400000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_04_01", -6.7700000, -7.6800000, -0.7870000, 13.5400000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_04_02", -6.7700000, -7.3200000, -0.7870000, 13.5400000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_04_03", -6.7700000, 7.3200000, -0.7870000, 13.5400000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_04_04", -6.7700000, 7.6800000, -0.7870000, 13.5400000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_05_00", 6.7700000, -15.0000000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_05_02", 6.7700000, -7.3200000, -0.7870000, 0.3600000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_05_04", 6.7700000, 7.6800000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_06_00", 7.1300000, -15.0000000, -0.7870000, 0.7400000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_06_01", 7.1300000, -7.6800000, -0.7870000, 0.7400000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_06_02", 7.1300000, -7.3200000, -0.7870000, 0.7400000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_06_03", 7.1300000, 7.3200000, -0.7870000, 0.7400000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_06_04", 7.1300000, 7.6800000, -0.7870000, 0.7400000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_07_00", 7.8700000, -15.0000000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_07_02", 7.8700000, -7.3200000, -0.7870000, 0.3600000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_07_04", 7.8700000, 7.6800000, -0.7870000, 0.3600000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_08_00", 8.2300000, -15.0000000, -0.7870000, 6.7700000, 7.3200000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_08_01", 8.2300000, -7.6800000, -0.7870000, 6.7700000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_08_02", 8.2300000, -7.3200000, -0.7870000, 6.7700000, 14.6400000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_08_03", 8.2300000, 7.3200000, -0.7870000, 6.7700000, 0.3600000, 0.7870000, "RO5880_V139", True
CreateBox oEditor, "SubstratePart_08_04", 8.2300000, 7.6800000, -0.7870000, 6.7700000, 7.3200000, 0.7870000, "RO5880_V139", True

' Frozen v1.38 radiator P00 at (-7.500,-7.500) mm.
CreateMetalSheetZ oEditor, "PrimaryN_P00", -13.0000000, -7.9000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryN_P00", -12.3000000, -6.7750000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckN_P00", -8.7000000, -7.9000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryN_P00,SecondaryN_P00,NeckN_P00"
CreateMetalSheetZ oEditor, "PrimaryP_P00", -7.2000000, -7.9000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryP_P00", -7.2000000, -6.7750000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckP_P00", -7.2000000, -7.9000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryP_P00,SecondaryP_P00,NeckP_P00"
CreateBox oEditor, "ViaN_P00", -8.2300000, -7.6800000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateBox oEditor, "ViaP_P00", -7.1300000, -7.6800000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateMetalSheetZ oEditor, "PadN_P00", -8.3000000, -7.9000000, -0.7870000, 0.5000000, 0.8000000
CreateMetalSheetZ oEditor, "PadP_P00", -7.2000000, -7.9000000, -0.7870000, 0.5000000, 0.8000000
CreateSheetZ oEditor, "PortSheet_P00", -7.8400000, -7.9000000, -0.7870000, 0.6800000, 0.8000000
AssignDifferentialPortZ oBoundary, "P00", "PortSheet_P00", -7.8000000, -7.2000000, -7.5000000, -0.7870000

' Frozen v1.38 radiator P10 at (7.500,-7.500) mm.
CreateMetalSheetZ oEditor, "PrimaryN_P10", 2.0000000, -7.9000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryN_P10", 2.7000000, -6.7750000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckN_P10", 6.3000000, -7.9000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryN_P10,SecondaryN_P10,NeckN_P10"
CreateMetalSheetZ oEditor, "PrimaryP_P10", 7.8000000, -7.9000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryP_P10", 7.8000000, -6.7750000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckP_P10", 7.8000000, -7.9000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryP_P10,SecondaryP_P10,NeckP_P10"
CreateBox oEditor, "ViaN_P10", 6.7700000, -7.6800000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateBox oEditor, "ViaP_P10", 7.8700000, -7.6800000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateMetalSheetZ oEditor, "PadN_P10", 6.7000000, -7.9000000, -0.7870000, 0.5000000, 0.8000000
CreateMetalSheetZ oEditor, "PadP_P10", 7.8000000, -7.9000000, -0.7870000, 0.5000000, 0.8000000
CreateSheetZ oEditor, "PortSheet_P10", 7.1600000, -7.9000000, -0.7870000, 0.6800000, 0.8000000
AssignDifferentialPortZ oBoundary, "P10", "PortSheet_P10", 7.2000000, 7.8000000, -7.5000000, -0.7870000

' Frozen v1.38 radiator P01 at (-7.500,7.500) mm.
CreateMetalSheetZ oEditor, "PrimaryN_P01", -13.0000000, 7.1000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryN_P01", -12.3000000, 8.2250000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckN_P01", -8.7000000, 7.1000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryN_P01,SecondaryN_P01,NeckN_P01"
CreateMetalSheetZ oEditor, "PrimaryP_P01", -7.2000000, 7.1000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryP_P01", -7.2000000, 8.2250000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckP_P01", -7.2000000, 7.1000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryP_P01,SecondaryP_P01,NeckP_P01"
CreateBox oEditor, "ViaN_P01", -8.2300000, 7.3200000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateBox oEditor, "ViaP_P01", -7.1300000, 7.3200000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateMetalSheetZ oEditor, "PadN_P01", -8.3000000, 7.1000000, -0.7870000, 0.5000000, 0.8000000
CreateMetalSheetZ oEditor, "PadP_P01", -7.2000000, 7.1000000, -0.7870000, 0.5000000, 0.8000000
CreateSheetZ oEditor, "PortSheet_P01", -7.8400000, 7.1000000, -0.7870000, 0.6800000, 0.8000000
AssignDifferentialPortZ oBoundary, "P01", "PortSheet_P01", -7.8000000, -7.2000000, 7.5000000, -0.7870000

' Frozen v1.38 radiator P11 at (7.500,7.500) mm.
CreateMetalSheetZ oEditor, "PrimaryN_P11", 2.0000000, 7.1000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryN_P11", 2.7000000, 8.2250000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckN_P11", 6.3000000, 7.1000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryN_P11,SecondaryN_P11,NeckN_P11"
CreateMetalSheetZ oEditor, "PrimaryP_P11", 7.8000000, 7.1000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryP_P11", 7.8000000, 8.2250000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckP_P11", 7.8000000, 7.1000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryP_P11,SecondaryP_P11,NeckP_P11"
CreateBox oEditor, "ViaN_P11", 6.7700000, 7.3200000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateBox oEditor, "ViaP_P11", 7.8700000, 7.3200000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateMetalSheetZ oEditor, "PadN_P11", 6.7000000, 7.1000000, -0.7870000, 0.5000000, 0.8000000
CreateMetalSheetZ oEditor, "PadP_P11", 7.8000000, 7.1000000, -0.7870000, 0.5000000, 0.8000000
CreateSheetZ oEditor, "PortSheet_P11", 7.1600000, 7.1000000, -0.7870000, 0.6800000, 0.8000000
AssignDifferentialPortZ oBoundary, "P11", "PortSheet_P11", 7.2000000, 7.8000000, 7.5000000, -0.7870000

oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array("PrimaryN_P00", "PrimaryP_P00", "PadN_P00", "PadP_P00", "PrimaryN_P10", "PrimaryP_P10", "PadN_P10", "PadP_P10", "PrimaryN_P01", "PrimaryP_P01", "PadN_P01", "PadP_P01", "PrimaryN_P11", "PrimaryP_P11", "PadN_P11", "PadP_P11"), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "0.0350000mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)
CreateBox oEditor, "AirRegion", -27, -27, -12.7870000, 54, 54, 24.8220000, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:UnifiedFeedRadiatorMesh_0p100mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("PrimaryN_P00", "PrimaryP_P00", "PrimaryN_P10", "PrimaryP_P10", "PrimaryN_P01", "PrimaryP_P01", "PrimaryN_P11", "PrimaryP_P11"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "0.1800000mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", 16, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 5.0000000, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, "DrivenSolverType:=", "Direct Solver")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V139", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v139_physical_2x2_differential_array_20260808_run01\initial_10ghz\direct01_repair03\v139_direct01_repair03.aedt", True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Sub CreateBox(editor, objName, x, y, z, dx, dy, dz, material, solveInside)
    editor.CreateBox Array("NAME:BoxParameters", "XPosition:=", Mm(x), "YPosition:=", Mm(y), "ZPosition:=", Mm(z), "XSize:=", Mm(dx), "YSize:=", Mm(dy), "ZSize:=", Mm(dz)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(220 150 55)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """" & material & """", "SolveInside:=", solveInside)
End Sub
Sub CreateVia(editor, objName, x, y, z, radius, height, numSides)
    editor.CreateCylinder Array("NAME:CylinderParameters", "XCenter:=", Mm(x), "YCenter:=", Mm(y), "ZCenter:=", Mm(z), "Radius:=", Mm(radius), "Height:=", Mm(height), "WhichAxis:=", "Z", "NumSides:=", CStr(numSides)), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """copper""", "SolveInside:=", False)
End Sub
Sub CreateSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(80 120 255)", "Transparency:=", 0.15, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub CreateMetalSheetZ(editor, objName, x, y, z, width, height)
    editor.CreateRectangle Array("NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", Mm(x), "YStart:=", Mm(y), "ZStart:=", Mm(z), "Width:=", Mm(width), "Height:=", Mm(height), "WhichAxis:=", "Z"), Array("NAME:Attributes", "Name:=", objName, "Flags:=", "", "Color:=", "(230 160 60)", "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "MaterialValue:=", """vacuum""", "SolveInside:=", True)
End Sub
Sub UniteSelection(editor, names)
    editor.Unite Array("NAME:Selections", "Selections:=", names), Array("NAME:UniteParameters", "KeepOriginals:=", False)
End Sub
Sub SubtractKeepObject(editor, blankName, toolName)
    editor.Subtract Array("NAME:Selections", "Blank Parts:=", blankName, "Tool Parts:=", toolName), Array("NAME:SubtractParameters", "KeepOriginals:=", True)
End Sub
Sub AssignDifferentialPortZ(boundary, portName, sheetName, xNegative, xPositive, y, z)
    boundary.AssignLumpedPort Array("NAME:" & portName, "Objects:=", Array(sheetName), "RenormalizeAllTerminals:=", True, "DoDeembed:=", False, Array("NAME:Modes", Array("NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True, Array("NAME:IntLine", "Start:=", Array(Mm(xNegative), Mm(y), Mm(z)), "End:=", Array(Mm(xPositive), Mm(y), Mm(z))), "CharImp:=", "Zpi")), "ShowReporterFilter:=", False, "ReporterFilter:=", Array(True), "FullResistance:=", "50ohm", "FullReactance:=", "0ohm")
End Sub
Function Mm(value)
    Mm = CStr(Round(CDbl(value), 7)) & "mm"
End Function
