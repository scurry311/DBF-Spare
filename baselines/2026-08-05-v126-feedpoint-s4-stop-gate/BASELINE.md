# v1.26 Physical Feed-Point S4 Stop-Gate Baseline

## Evidence Level

This checkpoint contains one true physical 2x2 antenna evaluated by three
independent single-frequency HFSS adaptive solves. It includes a 285-excitation
active-RL replay and an optimistic diagonal-only load-pull bound. It contains
no 4x4/16x16, EEP, label, or critic evidence.

## Result

The physical model passes numerical, passive-RL, total-RL, and radiation
efficiency gates, but its worst active RL is 5.08 dB. Only 113/285 frozen
excitations reach 11 dB active RL. Ideal frequency-specific per-port Sii values
still reach only a 9.49 dB worst-frequency upper bound when measured
off-diagonal coupling is fixed.

## Gate State

Bridge and feed-point-only tuning are stopped. The next hardware model must
modify the radiator/input structure so diagonal and mutual-coupling terms move
together. Array expansion, EEP export, labels, and critic training remain
locked.
