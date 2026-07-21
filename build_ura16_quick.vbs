Option Explicit

' Quick HFSS URA16 model for the beam-multitask project.
' Geometry: 16 x 16 centered planar array, dx = dy = lambda/2 at 10 GHz.
' Element: simplified PEC dipole with a lumped-port sheet at the center gap.

Dim oAnsoftApp, oDesktop, oProject, oDesign, oEditor, oBoundary, oAnalysis, oRad
Dim projectPath, designName
Dim f0GHz, lambdaMm, spacingMm, dipoleLengthMm, gapMm, armLenMm, radiusMm
Dim portHeightMm, airPadMm, airX, airY, airZ
Dim nx, ny, ix, iy, idx
Dim xc, yc, xStartLeft, xStartRight
Dim nameBase, leftName, rightName, sheetName, portName

projectPath = "D:\codex_workspace\hfss_ura16_quick_model\ura16_quick_10ghz.aedt"
designName = "URA16_Quick_10GHz"

f0GHz = 10.0
lambdaMm = 30.0
spacingMm = 15.0
dipoleLengthMm = 15.0
gapMm = 0.5
armLenMm = (dipoleLengthMm - gapMm) / 2.0
radiusMm = 0.25
portHeightMm = 1.0
airPadMm = 15.0
nx = 16
ny = 16

Set oAnsoftApp = CreateObject("Ansoft.ElectronicsDesktop")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.RestoreWindow

oDesktop.NewProject
Set oProject = oDesktop.GetActiveProject()
oProject.InsertDesign "HFSS", designName, "DrivenModal", ""
Set oDesign = oProject.SetActiveDesign(designName)
Set oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.SetModelUnits Array("NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False)

Set oBoundary = oDesign.GetModule("BoundarySetup")
Set oAnalysis = oDesign.GetModule("AnalysisSetup")

' Parameters kept as project variables for quick inspection.
oDesign.ChangeProperty Array( _
    "NAME:AllTabs", _
    Array( _
        "NAME:LocalVariableTab", _
        Array("NAME:PropServers", "LocalVariables"), _
        Array( _
            "NAME:NewProps", _
            Array("NAME:f0", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "10GHz"), _
            Array("NAME:lambda0", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "30mm"), _
            Array("NAME:spacing", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "15mm"), _
            Array("NAME:array_nx", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "16"), _
            Array("NAME:array_ny", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "16"), _
            Array("NAME:max_active_ratio", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", "0.5") _
        ) _
    ) _
)

' Air region around the whole array.
airX = (nx - 1) * spacingMm + dipoleLengthMm + 2 * airPadMm
airY = (ny - 1) * spacingMm + dipoleLengthMm + 2 * airPadMm
airZ = 2 * airPadMm
oEditor.CreateBox Array( _
        "NAME:BoxParameters", _
        "XPosition:=", Mm(-airX / 2.0), _
        "YPosition:=", Mm(-airY / 2.0), _
        "ZPosition:=", Mm(-airZ / 2.0), _
        "XSize:=", Mm(airX), _
        "YSize:=", Mm(airY), _
        "ZSize:=", Mm(airZ) _
    ), _
    Array( _
        "NAME:Attributes", _
        "Name:=", "AirRegion", _
        "Flags:=", "Wireframe#", _
        "Color:=", "(143 175 143)", _
        "Transparency:=", 0.85, _
        "PartCoordinateSystem:=", "Global", _
        "MaterialValue:=", """air""", _
        "SolveInside:=", True _
    )
oBoundary.AssignRadiation Array("NAME:Rad_AirRegion", "Objects:=", Array("AirRegion"))

' 256 simplified dipoles. Element index follows ix * ny + iy from the codebase.
idx = 0
For ix = 0 To nx - 1
    For iy = 0 To ny - 1
        xc = (ix - 0.5 * (nx - 1)) * spacingMm
        yc = (iy - 0.5 * (ny - 1)) * spacingMm
        nameBase = Pad3(idx)
        leftName = "Dipole_" & nameBase & "_L"
        rightName = "Dipole_" & nameBase & "_R"
        sheetName = "PortSheet_" & nameBase
        portName = "P" & nameBase

        xStartLeft = xc - gapMm / 2.0 - armLenMm
        xStartRight = xc + gapMm / 2.0

        CreateArm oEditor, leftName, xStartLeft, yc, 0.0, radiusMm, armLenMm
        CreateArm oEditor, rightName, xStartRight, yc, 0.0, radiusMm, armLenMm
        CreatePortSheet oEditor, sheetName, xc - gapMm / 2.0, yc, -portHeightMm / 2.0, gapMm, portHeightMm
        AssignLumpedPort oBoundary, portName, sheetName, xc - gapMm / 2.0, yc, 0.0, xc + gapMm / 2.0, yc, 0.0

        idx = idx + 1
    Next
Next

' Single-frequency solution setup for quick validation.
oAnalysis.InsertSetup "HfssDriven", Array( _
    "NAME:Setup_10GHz", _
    "SolveType:=", "Single", _
    "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, _
    "MaximumPasses:=", 4, _
    "MinimumPasses:=", 1, _
    "MinimumConvergedPasses:=", 1, _
    "PercentRefinement:=", 20, _
    "BasisOrder:=", 1, _
    "DoLambdaRefine:=", True, _
    "DoMaterialLambda:=", True, _
    "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, _
    "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, _
    "SetPortMinMaxTri:=", False _
)

Set oRad = oDesign.GetModule("RadField")
oRad.InsertFarFieldSphereSetup Array( _
    "NAME:InfiniteSphere_Theta0_90_Phi0_360", _
    "UseCustomRadiationSurface:=", False, _
    "ThetaStart:=", "0deg", _
    "ThetaStop:=", "90deg", _
    "ThetaStep:=", "1deg", _
    "PhiStart:=", "0deg", _
    "PhiStop:=", "360deg", _
    "PhiStep:=", "2deg", _
    "UseLocalCS:=", False _
)

oProject.SaveAs projectPath, True
oDesktop.CloseProject oProject.GetName()
oDesktop.QuitApplication

Function Mm(value)
    Mm = CStr(Round(CDbl(value), 6)) & "mm"
End Function

Function Pad3(value)
    Pad3 = Right("000" & CStr(value), 3)
End Function

Sub CreateArm(editor, objName, xStart, yCenter, zCenter, radius, length)
    editor.CreateCylinder Array( _
            "NAME:CylinderParameters", _
            "XCenter:=", Mm(xStart), _
            "YCenter:=", Mm(yCenter), _
            "ZCenter:=", Mm(zCenter), _
            "Radius:=", Mm(radius), _
            "Height:=", Mm(length), _
            "WhichAxis:=", "X", _
            "NumSides:=", "12" _
        ), _
        Array( _
            "NAME:Attributes", _
            "Name:=", objName, _
            "Flags:=", "", _
            "Color:=", "(230 160 60)", _
            "Transparency:=", 0, _
            "PartCoordinateSystem:=", "Global", _
            "MaterialValue:=", """pec""", _
            "SolveInside:=", False _
        )
End Sub

Sub CreatePortSheet(editor, objName, xStart, yCenter, zStart, width, height)
    editor.CreateRectangle Array( _
            "NAME:RectangleParameters", _
            "IsCovered:=", True, _
            "XStart:=", Mm(xStart), _
            "YStart:=", Mm(yCenter), _
            "ZStart:=", Mm(zStart), _
            "Width:=", Mm(width), _
            "Height:=", Mm(height), _
            "WhichAxis:=", "Y" _
        ), _
        Array( _
            "NAME:Attributes", _
            "Name:=", objName, _
            "Flags:=", "", _
            "Color:=", "(80 120 255)", _
            "Transparency:=", 0.15, _
            "PartCoordinateSystem:=", "Global", _
            "MaterialValue:=", """vacuum""", _
            "SolveInside:=", True _
        )
End Sub

Sub AssignLumpedPort(boundary, portName, sheetName, x1, y1, z1, x2, y2, z2)
    boundary.AssignLumpedPort Array( _
        "NAME:" & portName, _
        "Objects:=", Array(sheetName), _
        "RenormalizeAllTerminals:=", True, _
        "DoDeembed:=", False, _
        Array( _
            "NAME:Modes", _
            Array( _
                "NAME:Mode1", _
                "ModeNum:=", 1, _
                "UseIntLine:=", True, _
                Array("NAME:IntLine", "Start:=", Array(Mm(x1), Mm(y1), Mm(z1)), "End:=", Array(Mm(x2), Mm(y2), Mm(z2))), _
                "CharImp:=", "Zpi" _
            ) _
        ), _
        "ShowReporterFilter:=", False, _
        "ReporterFilter:=", Array(True), _
        "FullResistance:=", "50ohm", _
        "FullReactance:=", "0ohm" _
    )
End Sub
