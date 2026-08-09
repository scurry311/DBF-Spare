# v1.44 Grounded-Cavity Element Stop Gate

## Evidence Level

This baseline contains physical 10 GHz HFSS 1x1 results only.  No v1.44 2x2,
array, EEP, dataset, or learning result exists.

## Result

- New physical 1x1 candidates evaluated across grounded cavity, flush-wall,
  retuned patch, ceramic coax, inset tongue, and finite-Q element matching.
- Best new passive RL: `8.0976 dB`.
- Best converged numerical Delta S among the retained key points: `0.000568`.
- Best retained radiation efficiency: `98.38%`.
- No candidate reaches the `10 dB` stop line or the `15 dB` 1x1 gate.
- The 2x2 coupling and frozen active-RL gate therefore remains locked.

## Decision

Stop all v1.44 rectangular direct-probe and single-section matching variants.
Do not generate EEP, S256, HFSS labels, or critic data.  The next authorized
element must use an aperture-coupled or true balanced input with independent
matching variables.  A grounded neutralization layer may be evaluated only
after the new 1x1 input passes 15 dB.
