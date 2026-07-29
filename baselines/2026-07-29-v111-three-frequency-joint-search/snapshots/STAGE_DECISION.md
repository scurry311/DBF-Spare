# v20 Three-Frequency Stage Decision

## Confirmed results

- Strict oracle improved from 2/20 to 7/20.
- The 11 dB active-RL reserve oracle remains 0/20.
- Minimum feasible ratio counts are {'0.6': 3, '0.7': 2, '0.8': 2}; ratio 1.0 was not optimized.
- Failed-scene root constraints are {'mainlobe': 4, 'active_rl': 9}.
- Same-mask pattern/RL bridge counts by K are {2: 14, 4: 12, 6: 2}.
- Below-11-dB worst-port classes in the audit are {'interior': 198, 'edge': 58, 'corner': 3}.

## Decision

- The 18/20 acceptance gate failed, so no HFSS smoke list is frozen and critic training remains locked.
- The preregistered matching-redesign stop condition did not trigger: K=2 and K=4 each reached a 100% high-corner best-of-N rate at or above 10.5 dB.
- The next step is a targeted same-mask joint-feasibility rescue, not more random masks, HFSS labels, or critic training.
- If that rescue cannot create 11 dB reserve candidates, the 10.04 GHz matching bandwidth must then be broadened before HFSS.
