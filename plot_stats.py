# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PPO Training Stats Analysis

# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Current/relevant runs only (older experiments -- the shape_* upg-coefficient sweep, the hero
# lr/entropy forks, the GRPO runs -- are concluded; their stats remain in lambda_results/ if
# ever needed again). NOTE: all runs except honest1 trained on the CLAIRVOYANT engine (see
# EXPERIMENT_LOG.md) -- their win rates are inflated and not comparable to honest-era curves.
#   hero      PPO from-scratch baseline (v1: iters 1-99) + continuation (v2: 100-177)
#   heroe2    PPO num_epochs=2 fork -- cheat-era runner-up (0.744 @ 10k-sim paired eval)
#   ppo_hient PPO epochs=2, entropy_coef 0.05 + decay -- cheat-era champion (0.768 @ 10k)
#   ppo_ent10 PPO epochs=2, entropy_coef 0.10 ┐ entropy bracket
#   ppo_ent25 PPO epochs=2, entropy_coef 0.25 ┘ (0.25 over-flattened; killed)
#   honest1   honest-era from-scratch hero: honest CardPile engine, R5b encoding, dest-room aux
#   honest1asc  ascension 0-5 uniform mixture, warm start honest1.pt.iter_155 (its headline
#               win_rate is over the mixture -- NOT comparable to the A0-only runs; see the
#               per-ascension figure below)
stats_files = []
for _p in ('hero.pt.stats.jsonl', 'heroe2.pt.stats.jsonl',
           'ppo_hient.pt.stats.jsonl', 'ppo_ent10.pt.stats.jsonl', 'ppo_ent25.pt.stats.jsonl',
           'honest1.pt.stats.jsonl', 'honest1asc.pt.stats.jsonl'):
    _fp = f'lambda_results/{_p}'
    if os.path.exists(_fp):
        stats_files.append(_fp)

# Color palette for different runs
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# Per-group color overrides (takes precedence over the palette). Use to highlight headline runs.
GROUP_COLORS = {'hero': 'black', 'heroe2': 'darkorange',
                'ppo_hient': 'royalblue', 'ppo_ent10': 'navy', 'ppo_ent25': 'darkviolet',
                'honest1': 'crimson', 'honest1asc': 'darkgreen'}
GROUP_LW = {k: 2.8 for k in GROUP_COLORS}   # all headline runs

def load_run_data(filename):
    """Load data from a JSONL stats file"""
    data = []
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping")
        return None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print(f"Warning: {filename} is empty, skipping")
        return None

    # Resumes re-log some iteration numbers; keep the latest row per iteration and sort
    # by iteration so curves stay monotonic instead of jumping back at each resume.
    by_iter = {d.get('iteration'): d for d in data}
    data = [by_iter[k] for k in sorted(by_iter, key=lambda x: (x is None, x))]

    return {
        'filename': filename,
        'iterations': np.array([d.get('iteration', np.nan) for d in data]),
        'win_rates': np.array([d.get('win_rate', np.nan) for d in data]),
        'avg_floors': np.array([d.get('avg_floor', np.nan) for d in data]),
        'avg_rewards': np.array([d.get('avg_reward', np.nan) for d in data]),
        'collect_times': np.array([d.get('collect_time', np.nan) for d in data]),
        'train_times': np.array([d.get('train_time', np.nan) for d in data]),
        'policy_losses': np.array([d.get('policy_loss', np.nan) for d in data]),
        'value_losses': np.array([d.get('value_loss', np.nan) for d in data]),
        'entropies': np.array([d.get('entropy', np.nan) for d in data]),
        'kl_divs': np.array([d.get('kl_div', np.nan) for d in data]),
        'grad_norms': np.array([d.get('grad_norm', np.nan) for d in data]),
        'policy_grad_norms': np.array([d.get('policy_grad_norm', np.nan) for d in data]),
        'value_grad_norms': np.array([d.get('value_grad_norm', np.nan) for d in data]),
        'clipfracs': np.array([d.get('clipfrac', np.nan) for d in data]),
        'explained_variances': np.array([d.get('explained_variance', np.nan) for d in data]),
        'adv_norm_stds': np.array([d.get('adv_norm_std', np.nan) for d in data]),
        'num_experiences': np.array([d.get('num_experiences', np.nan) for d in data]),
        'num_trajectories': np.array([d.get('num_trajectories', np.nan) for d in data]),
        # Per-ascension-level breakdown (present only for ascension-mixture runs).
        **{f'win_rates_asc{a}': np.array([d.get(f'win_rate_asc{a}', np.nan) for d in data])
           for a in range(21)},
    }

# Load all runs
runs = []
for i, filename in enumerate(stats_files):
    run_data = load_run_data(filename)
    if run_data is not None:
        run_data['color'] = colors[i % len(colors)]
        run_data['label'] = os.path.basename(filename).replace('.pt.stats.jsonl', '').replace('shape_', '')
        runs.append(run_data)

print(f"Loaded {len(runs)} runs:")
for run in runs:
    print(f"  {run['label']}: {len(run['iterations'])} iterations")

# %%
SMOOTH_ALPHA = 0.25  # EMA factor; lower = smoother

def _smooth(y, alpha=SMOOTH_ALPHA):
    """EMA smoothing, NaN-aware (carries the last EMA value across gaps)."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    s = None
    for i, v in enumerate(y):
        if np.isnan(v):
            out[i] = s if s is not None else np.nan
        else:
            s = v if s is None else alpha * v + (1 - alpha) * s
            out[i] = s
    return out

# Replicate runs (same config, different init) share one color and are AVERAGED into a single
# bold smoothed line; each replicate's raw line is still drawn faint. Map replicate -> base.
REPLICATE_OF = {}  # (was used by the shape_* sweep replicates; none of the current runs replicate)

from collections import OrderedDict, defaultdict
groups = OrderedDict()
for run in runs:
    groups.setdefault(REPLICATE_OF.get(run['label'], run['label']), []).append(run)
group_list = list(groups.items())  # [(label, [member runs]), ...] -- one color per group

def _avg_over_iters(members, field):
    """Per-iteration mean across replicate members (averaged only where data exists)."""
    acc = defaultdict(list)
    for m in members:
        for x, y in zip(m['iterations'], m[field]):
            if not np.isnan(y):
                acc[x].append(y)
    xs = np.array(sorted(acc))
    ys = np.array([np.mean(acc[x]) for x in xs])
    return xs, ys

from matplotlib.lines import Line2D

def _gcolor(gi, gkey):
    return GROUP_COLORS.get(gkey, colors[gi % len(colors)])

def plot_group(ax, field, linestyle='-', lw=2, label_suffix='', add_label=True):
    """Each member's raw faint line + ONE bold EMA-smoothed line on the group average."""
    for gi, (gkey, members) in enumerate(group_list):
        color = _gcolor(gi, gkey)
        glw = GROUP_LW.get(gkey, lw)
        for m in members:
            ax.plot(m['iterations'], m[field], color=color, alpha=0.18, linewidth=1, linestyle=linestyle)
        gx, gy = _avg_over_iters(members, field)
        if len(gx):
            ax.plot(gx, _smooth(gy), color=color, alpha=1.0, linewidth=glw, linestyle=linestyle,
                    label=(gkey + label_suffix) if add_label else None)

def _run_handles():
    """Proxy legend handles: one per group, color = run."""
    return [Line2D([0], [0], color=_gcolor(gi, gkey), lw=GROUP_LW.get(gkey, 2), label=gkey)
            for gi, (gkey, _) in enumerate(group_list)]

def split_legend(ax, style_handles, run_loc='upper right', style_loc='upper left'):
    """Two legends: run/color (proxy per group) + subcomponent/linestyle."""
    runs_leg = ax.legend(handles=_run_handles(), fontsize=7, ncol=2, loc=run_loc, title='run')
    ax.add_artist(runs_leg)
    ax.legend(handles=style_handles, fontsize=8, loc=style_loc, title='component')

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Win rate over time
plot_group(ax1, 'win_rates')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Win Rate')
ax1.set_title('Win Rate Over Training')
ax1.grid(True, alpha=0.3)
if len(runs) > 1:
    ax1.legend()

# Average floor reached
plot_group(ax2, 'avg_floors')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Average Floor')
ax2.set_title('Average Floor Reached')
ax2.grid(True, alpha=0.3)
if len(runs) > 1:
    ax2.legend()

# Average reward
plot_group(ax3, 'avg_rewards')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Average Reward')
ax3.set_title('Average Reward Over Training')
ax3.grid(True, alpha=0.3)
if len(runs) > 1:
    ax3.legend()

# Collection and training times
plot_group(ax4, 'collect_times', linestyle='-', add_label=False)
plot_group(ax4, 'train_times', linestyle='--', add_label=False)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Execution Times')
ax4.grid(True, alpha=0.3)
split_legend(ax4, [Line2D([0], [0], color='gray', lw=2, linestyle='-', label='collect'),
                   Line2D([0], [0], color='gray', lw=2, linestyle='--', label='train')])

plt.tight_layout()
plt.show()

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Policy loss
plot_group(ax1, 'policy_losses')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Policy Loss')
ax1.set_title('Policy Loss Over Training')
ax1.grid(True, alpha=0.3)
if len(runs) > 1:
    ax1.legend()

# Value loss
plot_group(ax2, 'value_losses')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Value Loss')
ax2.set_title('Value Loss Over Training')
ax2.grid(True, alpha=0.3)
if len(runs) > 1:
    ax2.legend()

# Entropy
plot_group(ax3, 'entropies')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Entropy')
ax3.set_title('Policy Entropy Over Training')
ax3.grid(True, alpha=0.3)
if len(runs) > 1:
    ax3.legend()

# KL divergence
plot_group(ax4, 'kl_divs')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('KL Divergence')
ax4.set_title('KL Divergence Over Training')
ax4.grid(True, alpha=0.3)
if len(runs) > 1:
    ax4.legend()

plt.tight_layout()
plt.show()

# %%
fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, figsize=(15, 10))

# Gradient norms: total (solid), policy group (dashed), value group (dotted)
plot_group(ax1, 'grad_norms', add_label=False)
plot_group(ax1, 'policy_grad_norms', linestyle='--', lw=1.5, add_label=False)
plot_group(ax1, 'value_grad_norms', linestyle=':', lw=1.5, add_label=False)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Gradient Norm')
ax1.set_title('Gradient Norm (total / policy / value)')
ax1.grid(True, alpha=0.3)
split_legend(ax1, [Line2D([0], [0], color='gray', lw=2, linestyle='-', label='total'),
                   Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label='policy'),
                   Line2D([0], [0], color='gray', lw=1.5, linestyle=':', label='value')])

# Clip fraction
plot_group(ax2, 'clipfracs')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Clip Fraction')
ax2.set_title('PPO Clip Fraction Over Training')
ax2.grid(True, alpha=0.3)
if len(runs) > 1:
    ax2.legend()

# Value explained variance (1.0 = perfect, 0 = no better than predicting the mean)
plot_group(ax3, 'explained_variances')
ax3.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Explained Variance')
ax3.set_title('Value Explained Variance')
ax3.set_ylim(top=1.0)
ax3.grid(True, alpha=0.3)
if len(runs) > 1:
    ax3.legend()

plt.tight_layout()
plt.show()

# %%
# honest1 vs cheat-era from-scratch runs at matched WALL TIME. Win rates are NOT directly
# comparable across eras (the cheat-era engine was draw-order clairvoyant, worth ~tens of pp);
# floor progress is closer to comparable. This view shows learning-curve SHAPE and whether the
# R5b encoding + aux scaffold buys faster early learning despite the harder honest battles.
_LR = 'lambda_results'
hero_file = f'{_LR}/hero.pt.stats.jsonl'
CHEAT_BASELINES = [
    ('hero v1 (cheat era, from scratch)', f'{_LR}/hero.pt.stats.jsonl', 'black', 99),
    ('ppo_hient (cheat era, from scratch)', f'{_LR}/ppo_hient.pt.stats.jsonl', 'royalblue', 195),
]
HONEST_RUNS = [
    ('honest1 (honest engine, R5b+aux)', f'{_LR}/honest1.pt.stats.jsonl', 'crimson'),
]

def _wall_hours(run, max_iter=None):
    """Cumulative (collect+train) wall hours per logged iteration, optionally truncated."""
    keep = run['iterations'] <= max_iter if max_iter is not None else slice(None)
    times = (run['collect_times'][keep] + run['train_times'][keep])
    return np.cumsum(times) / 3600, keep

honest_runs = [(lbl, load_run_data(f), c) for lbl, f, c in HONEST_RUNS]
if any(r is not None for _, r, _ in honest_runs):
    fig, (ax_win, ax_floor) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, field, ylabel in ((ax_win, 'win_rates', 'Win rate'), (ax_floor, 'avg_floors', 'Avg floor')):
        for lbl, f, c, mx in CHEAT_BASELINES:
            run = load_run_data(f)
            if run is None:
                continue
            px, pkeep = _wall_hours(run, max_iter=mx)  # from-scratch portion only
            py = run[field][pkeep]
            ax.plot(px, py, color=c, alpha=0.14, linewidth=1)
            ax.plot(px, _smooth(py), color=c, linewidth=1.8, linestyle='--', label=lbl)
        for lbl, run, c in honest_runs:
            if run is None:
                continue
            gx, gkeep = _wall_hours(run)
            gy = run[field][gkeep]
            ax.plot(gx, gy, color=c, alpha=0.18, linewidth=1)
            ax.plot(gx, _smooth(gy), color=c, linewidth=2.6, label=lbl)
        ax.set_xlabel('Wall time (hours)'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs wall time — honest1 vs cheat-era from-scratch (win rates not era-comparable)')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("honest wall-time figure skipped (missing stats files)")

# %%
# honest1asc per-ascension win rates: one line per level + the bold mixture rate. honest1's
# A0 curve is overlaid shifted by its fork point (honest1asc iter 1 continues honest1 iter 155),
# so the crimson line shows what staying at A0 (with annealing) did from the same checkpoint --
# the green A0 line holding near it means the mixture isn't costing A0 competence.
ASC_RUN, ASC_FORK_ITER = 'honest1asc', 155
_asc_run = next((r for r in runs if r['label'] == ASC_RUN), None)
if _asc_run is not None:
    fig, ax = plt.subplots(figsize=(13, 6))
    asc_colors = plt.cm.viridis(np.linspace(0.0, 0.92, 21))
    for a in range(21):
        y = _asc_run[f'win_rates_asc{a}']
        if np.all(np.isnan(y)):
            continue
        ax.plot(_asc_run['iterations'], y, color=asc_colors[a], alpha=0.15, linewidth=0.8)
        ax.plot(_asc_run['iterations'], _smooth(y), color=asc_colors[a], linewidth=1.6,
                label=f'A{a}')
    ax.plot(_asc_run['iterations'], _smooth(_asc_run['win_rates']), color='darkgreen',
            linewidth=3.0, label='mixture (256 games/iter)')
    _h1 = next((r for r in runs if r['label'] == 'honest1'), None)
    if _h1 is not None:
        keep = _h1['iterations'] >= ASC_FORK_ITER
        ax.plot(_h1['iterations'][keep] - ASC_FORK_ITER, _smooth(_h1['win_rates'][keep]),
                color='crimson', linewidth=1.6, linestyle='--', alpha=0.8,
                label='honest1 A0 from same fork (annealed)')
    ax.set_xlabel(f'Iteration (0 = fork from honest1 iter {ASC_FORK_ITER})')
    ax.set_ylabel('Win rate')
    ax.set_title('honest1asc: per-ascension win rates (uniform mixture; dial-ups: A15 @40, A20 @~105)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("honest1asc per-ascension figure skipped (stats file missing)")

# %%
# Entropy phase plots: each run's training trajectory through (x, entropy) space as a faint
# EMA-smoothed line, ending at a strong dot for its current/final state, for x = avg floor and
# x = win rate.
#
# Dashed segments through the dots are objective-indifference lines: points along a line score
# equally under that run's training objective E[return] + coef_eff*H. Each run's empirical
# d(reward)/dx (fit over the run; includes the win bonus and shaping) converts the exchange rate
# to the x-axis. Segments are sized by a fixed *vertical* extent and axis limits are pinned to
# the data. PPO normalizes advantages by the (logged) adv_norm_std, so the effective per-return
# coefficient is entropy_coef * adv_norm_std (times the decisions-per-trajectory factor applied
# in the loop below). Runs with a decaying entropy_coef use the logged per-iteration value.
ENTROPY_COEF = {'ppo_hient': 0.05, 'ppo_ent10': 0.10, 'ppo_ent25': 0.25, 'honest1': 0.05,
                'honest1asc': 0.05}
PPO_NORMALIZED = {'ppo_hient', 'ppo_ent10', 'ppo_ent25', 'honest1', 'honest1asc'}  # adv / (logged) adv_norm_std

def _phase_panel(ax, xfield, xlabel, unit_name, unit_scale=1.0):
    """unit_scale: indifference slope is annotated per (unit_scale of x), e.g. 0.01 win rate."""
    for gi, (gkey, members) in enumerate(group_list):
        color = _gcolor(gi, gkey)
        xit, xv = _avg_over_iters(members, xfield)
        eit, en = _avg_over_iters(members, 'entropies')
        common = np.intersect1d(xit, eit)
        if common.size == 0 or np.all(np.isnan(en)):
            continue
        x_s = _smooth(xv[np.isin(xit, common)])
        ent_s = _smooth(en[np.isin(eit, common)])
        emphasized = gkey in GROUP_COLORS
        ax.plot(x_s, ent_s, color=color, alpha=0.45 if emphasized else 0.25,
                linewidth=1.6 if emphasized else 1.0)
        ax.scatter([x_s[-1]], [ent_s[-1]], color=color, s=110 if emphasized else 60,
                   zorder=5, edgecolors='white', linewidths=1.2, label=gkey)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()  # pin axes to the trajectory/dot data
    for gi, (gkey, members) in enumerate(group_list):
        if gkey not in ENTROPY_COEF:
            continue
        coef = ENTROPY_COEF[gkey]
        if gkey in PPO_NORMALIZED:  # PPO divides advantages by the (logged) EWMA std
            _, stds = _avg_over_iters(members, 'adv_norm_stds')
            stds = stds[~np.isnan(stds)]
            if stds.size == 0:
                continue
            coef = coef * stds[-1]
        # Per-trajectory decision count N: the loss is a per-DECISION mean, but the return
        # gradient aggregates over a trajectory's ~N decisions (policy gradient theorem:
        # grad E[R] = E[sum_t A_t grad log pi_t]), while the entropy term is genuinely
        # per-decision. So the effective entropy weight in trajectory-return units is
        # coef * N (times adv_norm_std for PPO). Forgetting N makes the lines ~30-40x too steep.
        _, nexp = _avg_over_iters(members, 'num_experiences')
        _, ntraj = _avg_over_iters(members, 'num_trajectories')
        okn = ~(np.isnan(nexp) | np.isnan(ntraj)) & (ntraj > 0)
        if okn.sum() == 0:
            continue
        n_per_traj = float(np.mean(nexp[okn] / ntraj[okn]))
        coef = coef * n_per_traj
        color = _gcolor(gi, gkey)
        _, xv = _avg_over_iters(members, xfield)
        _, rw = _avg_over_iters(members, 'avg_rewards')
        _, en = _avg_over_iters(members, 'entropies')
        ok = ~(np.isnan(xv) | np.isnan(rw))
        if ok.sum() < 5:
            continue
        dr_dx = np.polyfit(xv[ok], rw[ok], 1)[0]   # empirical reward per unit of x
        slope = -dr_dx / coef                      # nats of entropy per unit of x, at indifference
        x0, h0 = _smooth(xv)[-1], _smooth(en)[-1]
        dy = 0.15 * (ylim[1] - ylim[0])            # segment half-height: 15% of the y-range
        dx = dy / abs(slope)
        xs = np.array([x0 - dx, x0 + dx])
        ax.plot(xs, h0 + slope * (xs - x0), color=color, linestyle='--', linewidth=1.4, alpha=0.85)
        ax.annotate(f'{slope * unit_scale:.2f} nats/{unit_name}', xy=(x0, h0), xytext=(8, -14),
                    textcoords='offset points', fontsize=7, color=color)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Policy entropy')
    ax.set_title(f'Entropy vs {xlabel.lower()}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='best')

fig, (ax_floor, ax_win) = plt.subplots(2, 1, figsize=(16, 22), dpi=200)
_phase_panel(ax_floor, 'avg_floors', 'Avg floor reached', 'floor')
_phase_panel(ax_win, 'win_rates', 'Win rate', '+1% win', unit_scale=0.01)
fig.suptitle('Trajectories (faint), current state (dots), GRPO objective-indifference lines (dashed)',
             fontsize=13)
plt.tight_layout()
plt.show()

# %%
