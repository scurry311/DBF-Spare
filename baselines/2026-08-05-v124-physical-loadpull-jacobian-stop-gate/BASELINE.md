# v1.24 Physical Load-Pull/Modal Jacobian Stop-Gate Baseline

## Evidence Level

This checkpoint contains ten independently meshed 10 GHz network-only HFSS
S8 center-difference cases. Component values and Q are frozen. The trusted
antenna S4 is applied only in post-processing, so this is physical network
sensitivity evidence rather than integrated antenna or array evidence.

## Results

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Complete physical S8 cases | 10/10 | 10/10 | Passed |
| Lowest free memory | 3.81 GiB | >= 3 GiB | Passed |
| Effective geometry variables | 5/5 | >= 3 | Passed |
| Active-response Jacobian condition | 15.41 | <= 10000 | Passed |
| Maximum center nonlinearity | 1.371 | <= 0.50 | Failed |
| Strict numerical cases | 3/10 | 10/10 | Failed |
| Supported reflection explained | 9.55% | >= 60% | Failed |
| Trusted-S4 operator explained | 5.04% | >= 60% | Failed |
| Active response explained | 6.64% | >= 60% | Failed |
| Predicted best active RL | 3.52 dB | >= 10.5 dB | Failed |
| Predicted best total RL | 6.64 dB | >= 10.5 dB | Failed |

The Jacobian is full column rank, so failure is not caused by duplicate
variables. The five physical geometry directions point mostly outside the
required S8/load-pull correction subspace. A bounded search pushes every
variable to a manufacturing step limit without approaching the engineering
gate.

## Gate State

The local correction block is stopped. A predicted-geometry confirmation,
three-frequency extension, integrated 2x2, 4x4/16x16, labels, and critic
training are prohibited. The authorized next branch modifies the antenna feed
point/input impedance and uses 1x1 passive screening before a physical 2x2 S4
active-load comparison.
