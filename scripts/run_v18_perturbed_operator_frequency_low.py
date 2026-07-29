#!/usr/bin/env python3
"""Solve the frozen-mesh 16x16 operator at the E2 9.96 GHz corner."""

from __future__ import annotations

import run_fixed_mesh_eep_fieldsolve as field


FREQUENCY_GHZ = 9.96

field.OUT_DIR = (
    field.ROOT / "hfss_outputs" / "v18_perturbed_operator_frequency_low_20260729_run01"
)
field.TARGET_NAME = "v18_perturbed_operator_frequency_low_run01"
field.TARGET_PROJECT = field.OUT_DIR / "project" / f"{field.TARGET_NAME}.aedt"
field.SETUP_NAME = "Setup_V18_Perturbed_FrequencyLow_9p96GHz"
field.TOUCHSTONE = field.OUT_DIR / "solve" / "v18_frequency_low_9p96ghz.s256p"

# The successful nominal field solve required the 80% mixed-precision path.
# Keep one local DDM task and stop if the host approaches resource exhaustion.
field.MIN_D_FREE_GB = 70.0
field.MIN_COMMIT_HEADROOM_GB = 27.0
field.MIN_PHYSICAL_AVAILABLE_GB = 15.0
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
