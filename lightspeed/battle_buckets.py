"""Bucket scheme for battle-outcome (ΔHP) prediction. Shared by data generation
(gen_battle_outcomes.py) and the SL/RL heads. See EXPERIMENT_LOG.md (2026-06-12, battle-outcome aux task).

20 classes over hp_delta as a fraction of MAX HP:
    0           DEATH (battle lost)
    1           <= -50%
    2 .. 11     5%-wide damage bins: [-50,-45) ... [-5,0)   (indexed most-negative first)
    12          EXACT 0 (hp unchanged)
    13 .. 19    gains: (0,+5] (+5,+10] (+10,+15] (+15,+20] (+20,+35] (+35,+50] >+50

Gain bins stay 5%-fine through +20% because Burning Blood (Ironclad starter relic,
+6 HP applied in exitBattle) puts the modal easy-win outcome at ~6-12% of max HP.
"""
import numpy as np

NUM_BUCKETS = 20
DEATH = 0
EXACT_0 = 12

_GAIN_EDGES = np.array([0.05, 0.10, 0.15, 0.20, 0.35, 0.50])  # upper edges of buckets 13..18

BUCKET_LABELS = (
    ['DEATH', '<=-50%']
    + [f'({-50 + 5 * k},{-45 + 5 * k}]%' for k in range(10)]   # b=2..11: (-50,-45] ... (-5,0)
    + ['0']
    + ['(0,+5]%', '(+5,+10]%', '(+10,+15]%', '(+15,+20]%', '(+20,+35]%', '(+35,+50]%', '>+50%']
)
assert len(BUCKET_LABELS) == NUM_BUCKETS


def to_bucket(hp_delta: int, max_hp: int, died: bool) -> int:
    if died:
        return DEATH
    if hp_delta == 0:
        return EXACT_0
    f = hp_delta / max_hp
    if f < 0:
        k = int(-f / 0.05)        # damage bins are [5k%, 5(k+1)%) of max HP lost
        return 1 if k >= 10 else 11 - k
    return 13 + int(np.searchsorted(_GAIN_EDGES, f, side='left'))


def bucket_midpoint_frac(b: int) -> float:
    """Representative hp_frac_delta per bucket (for expected-value decoding).
    DEATH has no natural midpoint; use -1.0 (lose all of max HP) as a convention."""
    if b == DEATH:
        return -1.0
    if b == EXACT_0:
        return 0.0
    if b == 1:
        return -0.60                                 # <=-50% tail: call it -60%
    if 2 <= b <= 11:
        return -0.05 * (11 - b) - 0.025              # midpoint of (-5(12-b), -5(11-b)]
    edges = [0.0] + list(_GAIN_EDGES) + [0.60]      # >+50 tail: call it +55%
    return (edges[b - 13] + edges[b - 12]) / 2
