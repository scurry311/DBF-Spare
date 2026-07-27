# v1.6 Stage-B Robust Oracle

Evidence remains a 4x4-HFSS-calibrated 16x16 EEP/S256 proxy, not perturbed 16x16 HFSS.

| Stage | E2 overall | K2 | K4 | K6 |
|---|---:|---:|---:|---:|
| Initial | 1.33% | 4.00% | 0.00% | 0.00% |
| Common-weight projection | 69.33% | 88.00% | 76.00% | 44.00% |
| Mask rescue | 82.67% | 100.00% | 96.00% | 52.00% |

E1 new-scene oracle is 86.67%; E3 stress oracle is 0.00%.
Stage B does not pass. Remaining best-candidate causes: {'mainlobe': 8, 'active_rl': 4, 'nearest_iso': 1}.
Critic retraining and 16x16 HFSS smoke remain disabled.
