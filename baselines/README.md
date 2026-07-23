# Baselines

Each dated directory is an immutable, compact checkpoint of the research
state. It contains a human-readable decision, selected result snapshots, and a
SHA-256 manifest pointing back to the original local artifacts.

Large raw solver trees and generated datasets remain outside Git. A snapshot
is evidence of the recorded result, not permission to treat a blocked result
as an engineering label.

Current checkpoints:

- `2026-07-21`: blocked 16x16 adaptive-mesh physics gate.
- `2026-07-24`: trusted fixed-mesh S256 and EEP reconstruction; active-RL and
  strict engineering critic labels remain blocked.
