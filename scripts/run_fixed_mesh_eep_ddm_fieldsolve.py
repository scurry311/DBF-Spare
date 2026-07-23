#!/usr/bin/env python3
"""Run the field-enabled EEP solve with DDM to avoid direct-solver disk limits."""

from __future__ import annotations

import json

import run_fixed_mesh_eep_fieldsolve as field


field.OUT_DIR = (
    field.ROOT / "hfss_outputs" / "fixed_mesh_eep_fieldsolve_20260723_run02_ddm"
)
field.TARGET_NAME = "fixed_mesh_eep_ddm_fieldsolve_run02"
field.TARGET_PROJECT = (
    field.OUT_DIR / "project" / f"{field.TARGET_NAME}.aedt"
)
field.SETUP_NAME = "Setup_Frozen_DDM_EEP_Run02"
field.TOUCHSTONE = field.OUT_DIR / "solve" / "fixed_mesh_eep_ddm_fieldsolve.s256p"


def ddm_setup_array() -> str:
    return f'''Array( _
    "NAME:{field.SETUP_NAME}", _
    "SolveType:=", "Single", _
    "Frequency:=", "10GHz", _
    "MaxDeltaS:=", 0.05, _
    "UseMatrixConv:=", False, _
    "MaximumPasses:=", 1, _
    "MinimumPasses:=", 1, _
    "MinimumConvergedPasses:=", 1, _
    "PercentRefinement:=", 1, _
    "IsEnabled:=", True, _
    {field.mesh_link()}, _
    "BasisOrder:=", 1, _
    "DoLambdaRefine:=", False, _
    "DoMaterialLambda:=", False, _
    "SetLambdaTarget:=", False, _
    "UseMaxTetIncrease:=", False, _
    "PortAccuracy:=", 2, _
    "UseABCOnPort:=", False, _
    "SetPortMinMaxTri:=", False, _
    "DrivenSolverType:=", "Domain Decomposition", _
    "IterativeResidual:=", 0.000001, _
    "DDMSolverResidual:=", 0.000001, _
    "SaveRadFieldsOnly:=", False, _
    "SaveAnyFields:=", True)'''


field.setup_array = ddm_setup_array
field.direct.direct_profile_metrics = field.fixed.profile_metrics


if __name__ == "__main__":
    field.main()
