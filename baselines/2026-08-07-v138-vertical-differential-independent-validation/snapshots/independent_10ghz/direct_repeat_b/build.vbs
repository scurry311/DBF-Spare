Option Explicit
Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad, oMesh
Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.GetDefinitionManager().AddMaterial Array("NAME:RO5880_V132", "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1, "permittivity:=", "2.2", "dielectric_loss_tangent:=", "0.0009")
oProject.InsertDesign "HFSS", "V132_VerticalDifferential", "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign("V132_VerticalDifferential")
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)
Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")
CreateBox oEditor, "Substrate", -7.5000000, -7.5000000, -0.7870000, 15.0000000, 7.3200000, 0.7870000, "RO5880_V132", True
CreateBox oEditor, "SubstrateTop", -7.5000000, 0.1800000, -0.7870000, 15.0000000, 7.3200000, 0.7870000, "RO5880_V132", True
CreateBox oEditor, "SubstrateRowLeft", -7.5000000, -0.1800000, -0.7870000, 6.7700000, 0.3600000, 0.7870000, "RO5880_V132", True
CreateBox oEditor, "SubstrateRowCenter", -0.3700000, -0.1800000, -0.7870000, 0.7400000, 0.3600000, 0.7870000, "RO5880_V132", True
CreateBox oEditor, "SubstrateRowRight", 0.7300000, -0.1800000, -0.7870000, 6.7700000, 0.3600000, 0.7870000, "RO5880_V132", True
UniteSelection oEditor, "Substrate,SubstrateTop,SubstrateRowLeft,SubstrateRowCenter,SubstrateRowRight"

' Negative conductor: frozen fork, one short via, and one bottom capacitive launch pad.
CreateMetalSheetZ oEditor, "PrimaryN", -5.5000000, -0.4000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryN", -4.8000000, 0.7250000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckN", -1.2000000, -0.4000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryN,SecondaryN,NeckN"
CreateBox oEditor, "ViaN", -0.7300000, -0.1800000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateMetalSheetZ oEditor, "PadN", -0.8000000, -0.4000000, -0.7870000, 0.5000000, 0.8000000


' Positive conductor is the exact x mirror; no common reference ground exists.
CreateMetalSheetZ oEditor, "PrimaryP", 0.3000000, -0.4000000, 0, 5.2000000, 0.8000000
CreateMetalSheetZ oEditor, "SecondaryP", 0.3000000, 0.7250000, 0, 4.5000000, 0.5500000
CreateMetalSheetZ oEditor, "NeckP", 0.3000000, -0.4000000, 0, 0.9000000, 1.3250000
UniteSelection oEditor, "PrimaryP,SecondaryP,NeckP"
CreateBox oEditor, "ViaP", 0.3700000, -0.1800000, -0.7870000, 0.3600000, 0.3600000, 0.7870000, "copper", False
CreateMetalSheetZ oEditor, "PadP", 0.3000000, -0.4000000, -0.7870000, 0.5000000, 0.8000000

oBoundary.AssignFiniteCond Array("NAME:CopperSheetFiniteConductivity", "Objects:=", Array("PrimaryN", "PrimaryP", "PadN", "PadP"), "UseMaterial:=", True, "Material:=", "copper", "UseThickness:=", True, "Thickness:=", "0.0350000mm", "Roughness:=", "0um", "InfGroundPlane:=", False, "IsTwoSided:=", True, "IsShellElement:=", False)

' The bottom reference plane is fixed; pad gap/area and via radius are independent matching controls.
CreateSheetZ oEditor, "PortSheet_DIFF", -0.3400000, -0.4000000, -0.7870000, 0.6800000, 0.8000000
AssignDifferentialPortZ oBoundary, "P_DIFF", "PortSheet_DIFF", -0.3000000, 0.3000000, 0, -0.7870000

CreateBox oEditor, "AirRegion", -19.5000000, -19.5000000, -12.7870000, 39.0000000, 39.0000000, 24.8570000, "air", True
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))
Set oMesh = oDesign.GetModule("MeshSetup")
oMesh.AssignLengthOp Array("NAME:ViaPadMesh_0p100mm", "RefineInside:=", False, "Enabled:=", True, "Objects:=", Array("PrimaryN", "PrimaryP"), "RestrictElem:=", False, "NumMaxElem:=", "1000", "RestrictLength:=", True, "MaxLength:=", "0.1000000mm", "UseAdvSizing:=", False)
oAnalysis.InsertSetup "HfssDriven", Array("NAME:Setup_10GHz", "SolveType:=", "Single", "Frequency:=", "10GHz", "MaxDeltaS:=", 0.05, "MaximumPasses:=", 16, "MinimumPasses:=", 2, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 8.0000000, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True, "SetLambdaTarget:=", False, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, "DrivenSolverType:=", "Direct Solver")
Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array("NAME:InfiniteSphere_V132", "UseCustomRadiationSurface:=", False, "ThetaStart:=", "0deg", "ThetaStop:=", "180deg", "ThetaStep:=", "5deg", "PhiStart:=", "0deg", "PhiStop:=", "360deg", "PhiStep:=", "5deg", "UseLocalCS:=", False)
oProject.SaveAs "D:\codex_workspace\hfss_ura16_quick_model\hfss_outputs\v138_vertical_differential_independent_validation_20260807_run01\independent_10ghz\direct_repeat_b\v138_direct_repeat_b.aedt", True
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
