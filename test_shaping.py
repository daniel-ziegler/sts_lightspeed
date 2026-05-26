"""Verify potential-based reward shaping properties of compute_shaped_rewards.

Checks (the user's requirements):
  1. coefs=0 -> rewards identical to the base reward deltas (no behavior change by default).
  2. total-reward invariance: sum(shaped) - sum(base) is a per-game CONSTANT = -(shape_raw(s0) - off),
     independent of the interior trajectory.
  3. the offset cancels in every interior delta and only shifts the terminal step (by +off).
  4. terminal clawback == base_delta_T - shape_raw(s_last) + off  (bounded; "un-credited at end").
  5. interior steps credit HP / upgrade changes as expected.
"""
import sys
from ppo_train import compute_shaped_rewards, GameMetrics, compute_progress_reward
import slaythespire as sts

UND = sts.GameOutcome.UNDECIDED

def M(floor, hp, mx, nup):
    return GameMetrics(floor_num=floor, cur_hp=hp, max_hp=mx,
                       perfected_strike_count=0, num_upgraded=nup, outcome=UND)

# simple, outcome-independent base reward for the math checks
base = lambda m: 0.01 * m.floor_num
shape_raw = lambda m, hc, uc: hc * (m.cur_hp / m.max_hp if m.max_hp else 0.0) + uc * m.num_upgraded

def approx(a, b, t=1e-9): return abs(a - b) < t
def fail(msg): print("FAIL:", msg); sys.exit(1)

HC, UC, OFF = 0.18, 0.007, 0.176
# two trajectories sharing s0 (full HP, 0 upgrades) but different interiors/endings
traj_a = [M(0, 80, 80, 0), M(3, 60, 80, 2), M(8, 70, 80, 5), M(14, 30, 80, 9)]
traj_b = [M(0, 80, 80, 0), M(2, 75, 80, 1), M(20, 40, 90, 12)]
fin_a, fin_b = M(16, 0, 80, 9), M(34, 55, 90, 12)

for name, steps, fin in [("A", traj_a, fin_a), ("B", traj_b, fin_b)]:
    r_base, fb_base = compute_shaped_rewards(steps, fin, base, 0, 0, 0)
    # base deltas reference
    bv = [base(m) for m in steps] + [base(fin)]
    ref = [bv[i+1]-bv[i] for i in range(len(steps))]
    if not all(approx(x, y) for x, y in zip(r_base, ref)): fail(f"{name}: coefs=0 != base deltas")
    if not approx(fb_base, base(fin)): fail(f"{name}: terminal base value wrong")

    r_shaped, _ = compute_shaped_rewards(steps, fin, base, HC, UC, OFF)

    # (2) invariance: total shifts by exactly -(shape_raw(s0) - off), constant across trajectories
    delta_total = sum(r_shaped) - sum(r_base)
    expected_const = -(shape_raw(steps[0], HC, UC) - OFF)
    if not approx(delta_total, expected_const):
        fail(f"{name}: total shift {delta_total:.6f} != const {expected_const:.6f}")

    # (4) terminal clawback formula
    base_delta_T = base(fin) - base(steps[-1])
    expected_term = base_delta_T - shape_raw(steps[-1], HC, UC) + OFF
    if not approx(r_shaped[-1], expected_term):
        fail(f"{name}: terminal {r_shaped[-1]:.6f} != {expected_term:.6f}")

    # (3) offset only shifts the terminal step (interior deltas identical with/without offset)
    r_no_off, _ = compute_shaped_rewards(steps, fin, base, HC, UC, 0.0)
    for i in range(len(steps) - 1):
        if not approx(r_shaped[i], r_no_off[i]):
            fail(f"{name}: interior step {i} changed by offset")
    if not approx(r_shaped[-1] - r_no_off[-1], OFF):
        fail(f"{name}: terminal step not shifted by exactly +off")

# s0 is always full-HP/0-upg -> the per-game constant is the SAME for A and B
const_a = -(shape_raw(traj_a[0], HC, UC) - OFF)
const_b = -(shape_raw(traj_b[0], HC, UC) - OFF)
if not approx(const_a, const_b): fail("per-game constant differs between trajectories (s0 not canonical)")

# (5) interior credit: HP drop 80->60 (one step) is rewarded by HC*(60/80 - 80/80) = -0.045
r, _ = compute_shaped_rewards([M(1, 80, 80, 0), M(1, 60, 80, 0)], M(1, 60, 80, 0), lambda m: 0.0, HC, 0, 0)
if not approx(r[0], HC * (60/80 - 1.0)): fail("HP-drop interior credit wrong")
# upgrade +3 rewarded by UC*3
r, _ = compute_shaped_rewards([M(1, 80, 80, 0), M(1, 80, 80, 3)], M(1, 80, 80, 3), lambda m: 0.0, 0, UC, 0)
if not approx(r[0], UC * 3): fail("upgrade interior credit wrong")

# sanity: with the real progress reward + a victory, default path still works
win = GameMetrics(50, 70, 80, 0, 9, sts.GameOutcome.PLAYER_VICTORY)
r, fb = compute_shaped_rewards(traj_a, win, compute_progress_reward, HC, UC, OFF)
print(f"real-reward smoke: terminal_base={fb:.3f} sum={sum(r):.4f} term_step={r[-1]:.4f}")

print(f"per-game constant (HP+upg combo): {const_a:+.4f}  (offset reduces terminal clawback)")
print("ALL SHAPING TESTS PASSED")
