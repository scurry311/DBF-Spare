#!/usr/bin/env python3
"""Solve the frozen-mesh 16x16 operator at the E2 10.04 GHz corner."""

from __future__ import annotations

import run_fixed_mesh_eep_fieldsolve as field


FREQUENCY_GHZ = 10.04

field.OUT_DIR = (
    field.ROOT / "hfss_outputs" / "v19_perturbed_operator_frequency_high_20260729_run01"
)
field.TARGET_NAME = "v19_perturbed_operator_frequency_high_run01"
field.TARGET_PROJECT = field.OUT_DIR / "project" / f"{field.TARGET_NAME}.aedt"
field.SETUP_NAME = "Setup_V19_Perturbed_FrequencyHigh_10p04GHz"
field.TOUCHSTONE = field.OUT_DIR / "solve" / "v19_frequency_high_10p04ghz.s256p"

# The low-frequency corner came within 46 MB of its hard physical-RAM stop.
# Require one additional GB before launch while keeping the validated 80% DDM path.
field.MIN_D_FREE_GB = 70.0
field.MIN_COMMIT_HEADROOM_GB = 27.0
field.MIN_PHYSICAL_AVAILABLE_GB = 16.0
field.AEDT_DISTRIBUTED_MACHINE_LIST = "list=scurry:1:4:80%:0"
field.CRITICAL_PHYSICAL_GB = 0.75
field.CRITICAL_COMMIT_GB = 3.0
field.CRITICAL_D_FREE_GB = 25.0
field.MAX_CRITICAL_POLLS = 2
field.RESOURCE_POLL_SECONDS = 20.0


def perturbed_setup_array() -> str:
    return f'''Array( _
    "NAME:{field.SETUP_NAME}", _
    "SolveType:=", "Single", _
    "Frequency:=", "{FREQUENCY_GHZ:.9g}GHz", _
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
    "SaveRadFieldsOnly:=", True, _
    "SaveAnyFields:=", True)'''


field.setup_array = perturbed_setup_array
field.direct.direct_profile_metrics = field.fixed.profile_metrics


if __name__ == "__main__":
    field.main()
