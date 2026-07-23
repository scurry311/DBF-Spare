#!/usr/bin/env python3
"""Retry the field-enabled fixed-mesh EEP solve with DDM and disk headroom."""

from __future__ import annotations

import run_fixed_mesh_eep_fieldsolve as field


field.OUT_DIR = (
    field.ROOT / "hfss_outputs" / "fixed_mesh_eep_fieldsolve_20260723_run03_ddm"
)
field.TARGET_NAME = "fixed_mesh_eep_ddm_fieldsolve_run03"
field.TARGET_PROJECT = field.OUT_DIR / "project" / f"{field.TARGET_NAME}.aedt"
field.SETUP_NAME = "Setup_Frozen_DDM_EEP_Run03"
field.TOUCHSTONE = field.OUT_DIR / "solve" / "fixed_mesh_eep_ddm_fieldsolve.s256p"
field.MIN_D_FREE_GB = 85.0


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
