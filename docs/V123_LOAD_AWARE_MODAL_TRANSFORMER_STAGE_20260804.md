# v1.23 Load-Aware Single-Block Modal Transformer

## Frozen Boundary

The v1.22 run06 differential launch, PRE/POST planes, substrate, trace
dimensions, and finite-conductivity representation are frozen. The trusted
three-frequency S4 and all 285 side-2 stimuli are hash-locked. No HFSS solve is
authorized during the v1.23 modal-audit and circuit-upper-bound stage.

## Topology

The new correction is one physical block, not another cascade stage. Each
local x-neighbor pair uses a coupled-line even/odd section. A symmetric ground
branch changes both modal admittances, while one pair bridge changes only the
odd-mode admittance. The two x pairs share one manufacturable parameter set.
No nonlocal connection is allowed. The even-mode impedance must remain at
least 5 ohms above the odd-mode impedance in both nominal and tolerance
evaluations. The PRE/POST loading split is optimized inside the same block; an
endpoint is the output-only loading limit, not an additional cascade stage.

Two upper bounds are evaluated:

1. an ideal pair-local modal network with independent parameters for four
   modes, used to test whether the locality restriction is fundamentally
   sufficient;
2. a shared-geometry finite-Q network with coupled even/odd impedances,
   electrical lengths, ground loading, and bridge loading.

The full impedance target, local modal projection, unsupported residual, worst
v1.22 active-RL events, and target resistance-whitening transform are exported
before optimization.

## Power Gates

The 97% gate applies to the new correction block alone. The frozen launch
already has about 95.27% insertion efficiency, so imposing 97% on the combined
passive chain would be physically inconsistent. The combined launch plus
correction gate remains 95%, and actual-load insertion efficiency must also
remain at least 95%.

Nominal circuit design requires active RL and total RL of at least 12 dB. The
1000-sample manufacturing/Q tolerance gate uses the unchanged 11 dB engineering
threshold and requires at least a 95% joint pass rate. Only a complete pass may
authorize one 10 GHz network-only HFSS S8 smoke; three-frequency and integrated
HFSS remain locked until that smoke passes.

The circuit-upper-bound pass rate applies to active RL, total RL, passive RL,
and correction-block efficiency. Combined-launch and actual-load efficiency
are reported as a separate full-chain physical gate. This preserves the stated
97% correction-network requirement while preventing a successful matching
upper bound from concealing loss in the frozen launch.

## Completed Results

The three-frequency finite-Q circuit upper bound reached 12.368 dB active RL,
14.963 dB total RL, and 98.553% correction-block efficiency. The requested
1000-trial circuit tolerance gate passed in 997 trials. The same model predicted
only 93.320% minimum frozen-launch-plus-block efficiency, so it authorized one
diagnostic 10 GHz physical S8 and no later HFSS stage.

That physical S8 converged at Delta S = 0.017436 and passed reciprocity and
passivity checks. It failed the engineering gate: active RL was 3.149 dB,
total RL was 6.537 dB, matched-load network efficiency was 94.259%, and actual
load insertion efficiency was 92.508%. The physical-to-target maximum absolute
S difference was 0.5329.

Under identical trusted S4 termination and the same 57 frozen 10 GHz stimuli,
the physical v1.23 block degraded active RL in 56 cases and improved it in one.
The median active-RL change was -4.170 dB. The topology is therefore stopped
before 9.96/10.04 GHz, independent repeat, integrated 2x2, or label generation.
