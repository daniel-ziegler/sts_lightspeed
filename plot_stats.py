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

# All lambda reward-shaping runs (rounds 1-4); glob auto-includes new ones as they finish.
stats_files = sorted(glob.glob(
    '/home/dmz/osrc/sts_lightspeed.rl-ppo-fixes/lambda_results/shape_*.stats.jsonl'))
# Append the in-flight hero run + its forks + the from-scratch GRPO runs so they appear in the
# comparison panels alongside the shape_ runs. Forks branch off the hero: lr75 = lr halved
# @ iter 140; e2 = num_epochs 4->2 @ iter 170. GRPO runs (critic-free RLOO) start from scratch,
# so their iteration axis lines up with the from-scratch PPO runs; their value_loss /
# explained_variance panels read 0 (no critic).
# ppo_ent10/ppo_ent25: fresh from-scratch PPO (epochs=2) with a strong entropy bonus (coef 0.10
# / 0.25), testing whether PPO can hold GRPO-like high entropy while still learning faster.
for _p in ('hero.pt.stats.jsonl', 'herofork_lr75.pt.stats.jsonl', 'heroe2.pt.stats.jsonl',
           'heroent.pt.stats.jsonl', 'grpo_a.pt.stats.jsonl', 'grpo_b.pt.stats.jsonl',
           'ppo_ent10.pt.stats.jsonl', 'ppo_ent25.pt.stats.jsonl'):
    _fp = f'/home/dmz/osrc/sts_lightspeed.rl-ppo-fixes/lambda_results/{_p}'
    if os.path.exists(_fp):
        stats_files.append(_fp)
# (prior local PPO runs, if you want them instead:)
# stats_files = ['/home/dmz/osrc/sts_lightspeed.rl-ppo-fixes/runs/ppo_4ep.pt.stats.jsonl']

# Color palette for different runs
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# Per-group color overrides (takes precedence over the palette). Use to highlight headline runs.
GROUP_COLORS = {'hero': 'black', 'herofork_lr75': 'crimson', 'heroe2': 'darkorange',
                'heroent': 'green', 'grpo_a': 'magenta', 'grpo_b': 'teal',
                'ppo_ent10': 'navy', 'ppo_ent25': 'darkviolet'}
GROUP_LW = {'hero': 2.8, 'herofork_lr75': 2.8, 'heroe2': 2.8, 'heroent': 2.8,
            'grpo_a': 2.8, 'grpo_b': 2.8, 'ppo_ent10': 2.8, 'ppo_ent25': 2.8}   # headline runs

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
REPLICATE_OF = {'baseline2': 'baseline', 'upg3b': 'upg3', 'upg5b': 'upg5'}

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

# %% [markdown]
# ## Dose-response: upgrade-shaping coefficient

# %%
# Upgrade-shaping coefficient per run label. baseline = 0; upgN denotes the coef applied.
UPG_DOSE = {
    'baseline': 0.0,
    'upg3':     0.021,
    'upg5':     0.035,
    'upg10':    0.07,
    'upg20':    0.14,
    'upg40':    0.28,
}
# Runs that didn't complete the full 100 iters (e.g. upg20 crashed at 60).
PARTIAL_RUNS = {'upg20'}

# Build {dose: [member runs]} by merging replicates (REPLICATE_OF) and keeping only labels in UPG_DOSE.
dose_groups = defaultdict(list)
for run in runs:
    base = REPLICATE_OF.get(run['label'], run['label'])
    if base in UPG_DOSE:
        dose_groups[base].append(run)

def _last10_stats(members, field):
    """For each member, take its last 10 iter values for `field`; average mean/std across members."""
    means, stds, ns = [], [], []
    for m in members:
        ys = np.asarray(m[field], dtype=float)
        ys = ys[~np.isnan(ys)]
        if len(ys) == 0:
            continue
        tail = ys[-10:]
        means.append(np.mean(tail))
        stds.append(np.std(tail))
        ns.append(len(tail))
    if not means:
        return np.nan, np.nan, 0
    # Mean-of-means across replicates; pool std as mean of replicate stds (good enough for error bars).
    return float(np.mean(means)), float(np.mean(stds)), int(np.mean(ns))

rows = []  # (label, dose, partial, win_mean, win_std, floor_mean, floor_std, n_iters)
for label, dose in sorted(UPG_DOSE.items(), key=lambda kv: kv[1]):
    members = dose_groups.get(label, [])
    if not members:
        continue
    wm, ws, n = _last10_stats(members, 'win_rates')
    fm, fs, _ = _last10_stats(members, 'avg_floors')
    rows.append((label, dose, label in PARTIAL_RUNS, wm, ws, fm, fs, n))

# Print a numeric table so the values are at hand even without the figure.
print('\nDose-response (last-10-iter means, averaged over replicates):')
print(f"{'label':<10}{'coef':>8}{'partial':>9}{'win_mean':>10}{'win_std':>9}"
      f"{'floor_mean':>12}{'floor_std':>10}{'n':>4}")
for label, dose, partial, wm, ws, fm, fs, n in rows:
    print(f"{label:<10}{dose:>8.3f}{str(partial):>9}{wm:>10.4f}{ws:>9.4f}"
          f"{fm:>12.4f}{fs:>10.4f}{n:>4}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

def _plot_dose(ax, which, ylabel, title):
    """which: 'w' for win-rate, 'f' for avg-floor."""
    full_x, full_y, full_e = [], [], []
    part_x, part_y, part_e, part_lbl = [], [], [], []
    for label, dose, partial, wm, ws, fm, fs, n in rows:
        ym = wm if which == 'w' else fm
        ye = ws if which == 'w' else fs
        if np.isnan(ym):
            continue
        if partial:
            part_x.append(dose); part_y.append(ym); part_e.append(ye); part_lbl.append((label, n))
        else:
            full_x.append(dose); full_y.append(ym); full_e.append(ye)
    # Connect all points (full + partial) sorted by dose to show the curve shape.
    all_pts = sorted(list(zip(full_x, full_y)) + list(zip(part_x, part_y)))
    if all_pts:
        ax.plot([p[0] for p in all_pts], [p[1] for p in all_pts],
                color='steelblue', alpha=0.5, linewidth=1.2, zorder=1)
    ax.errorbar(full_x, full_y, yerr=full_e, fmt='o', color='steelblue',
                markersize=8, capsize=4, label='completed (100 iters)', zorder=2)
    if part_x:
        ax.errorbar(part_x, part_y, yerr=part_e, fmt='o', mfc='none', mec='steelblue',
                    color='steelblue', markersize=10, capsize=4,
                    label='partial', zorder=3)
        for x, y, (lbl, n) in zip(part_x, part_y, part_lbl):
            ax.annotate(f'{lbl} (partial, {n} iters)', xy=(x, y),
                        xytext=(6, 6), textcoords='offset points', fontsize=8)
    # Label each completed point with its run name for quick reference.
    for label, dose, partial, wm, ws, fm, fs, n in rows:
        ym = wm if which == 'w' else fm
        if np.isnan(ym) or partial:
            continue
        ax.annotate(label, xy=(dose, ym), xytext=(6, -10),
                    textcoords='offset points', fontsize=8, color='gray')
    ax.set_xlabel('Upgrade-shaping coefficient')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

_plot_dose(ax1, 'w', 'Win rate (last 10 iters)', 'Dose-response: win rate')
_plot_dose(ax2, 'f', 'Avg floor (last 10 iters)', 'Dose-response: avg floor')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Hero run (24h, full lambda box, upg5 shaping)

# %%
# Single-curve view of the headline run; vertical lines mark hyperparam interventions.
# Each entry: (iter_at_change, label_for_annotation). Append new ones here when adjustments land.
HERO_INTERVENTIONS = [
    (100, 'lr/2, workers 20->30, games 192->512, compile on'),
    (140, 'fork lr75: lr 1.5e-5->7.5e-6 (crimson)'),
    (170, 'forks @170: e2 num_epochs 4->2 (orange), ent coef 0.01->0.003 (green)'),
]
hero_file = '/home/dmz/osrc/sts_lightspeed.rl-ppo-fixes/lambda_results/hero.pt.stats.jsonl'
hero = load_run_data(hero_file)
# Forks branched off the hero (lr/2 @ iter 140, num_epochs->2 @ iter 170); overlay each on
# every hero panel. (label, file, color)
_FORKS = [
    ('fork lr/2', 'herofork_lr75.pt.stats.jsonl', 'crimson'),
    ('fork e2',   'heroe2.pt.stats.jsonl',        'darkorange'),
    ('fork ent',  'heroent.pt.stats.jsonl',       'green'),
]
forks = [(lbl, load_run_data(f'/home/dmz/osrc/sts_lightspeed.rl-ppo-fixes/lambda_results/{fn}'), c)
         for lbl, fn, c in _FORKS]
if hero is not None:
    print(f"hero: {len(hero['iterations'])} iters, last win={hero['win_rates'][-1]:.3f} "
          f"floor={hero['avg_floors'][-1]:.2f}")
    for lbl, fk, _ in forks:
        if fk is not None:
            print(f"{lbl}: {len(fk['iterations'])} iters, last win={fk['win_rates'][-1]:.3f} "
                  f"floor={fk['avg_floors'][-1]:.2f}")

    def _hero_plot(ax, field, ylabel, title, ylim=None):
        x, y = hero['iterations'], hero[field]
        ax.plot(x, y, color='steelblue', alpha=0.18, linewidth=1, label='raw')
        ax.plot(x, _smooth(y), color='steelblue', linewidth=2, label=f'hero EMA α={SMOOTH_ALPHA}')
        for lbl, fk, c in forks:
            if fk is not None and field in fk:
                fx, fy = fk['iterations'], fk[field]
                ax.plot(fx, fy, color=c, alpha=0.18, linewidth=1)
                ax.plot(fx, _smooth(fy), color=c, linewidth=2, label=f'{lbl} EMA')
        for it, lbl in HERO_INTERVENTIONS:
            ax.axvline(it, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
            ax.annotate(lbl, xy=(it, ax.get_ylim()[1]), xytext=(3, -10),
                        textcoords='offset points', fontsize=7, color='black')
        ax.set_xlabel('Iteration'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7, loc='best')

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    _hero_plot(axes[0][0], 'win_rates',          'Win rate', 'Hero — Win rate')
    _hero_plot(axes[0][1], 'avg_floors',         'Avg floor', 'Hero — Avg floor')
    _hero_plot(axes[0][2], 'avg_rewards',        'Avg reward', 'Hero — Avg reward')
    _hero_plot(axes[1][0], 'kl_divs',            'KL', 'Hero — KL divergence')
    _hero_plot(axes[1][1], 'grad_norms',         'Grad norm', 'Hero — Grad norm')
    _hero_plot(axes[1][2], 'explained_variances', 'Explained variance',
               'Hero — Value explained variance', ylim=(None, 1.0))
    plt.tight_layout()
    plt.show()
else:
    print(f"hero stats file not found yet: {hero_file}")

# %%
# From-scratch GRPO vs from-scratch PPO at matched WALL TIME (the fair learning-speed view —
# the iteration axis hides that the algorithms do different amounts of work per iteration).
# PPO baseline = hero v1 (iters 1-99, before the iter-100 hyperparameter change). Caveat: the
# GRPO runs share the box with each other (16 of 30 workers each) while hero v1 had it alone
# (20 workers), so GRPO wall times are inflated by contention.
_LR = '/home/dmz/osrc/sts_lightspeed.rl-ppo-fixes/lambda_results'
GRPO_RUNS = [
    ('grpo_a (RLOO, lr 1e-4)', f'{_LR}/grpo_a.pt.stats.jsonl', 'magenta'),
    ('grpo_b (RLOO, lr 3e-4)', f'{_LR}/grpo_b.pt.stats.jsonl', 'teal'),
]

def _wall_hours(run, max_iter=None):
    """Cumulative (collect+train) wall hours per logged iteration, optionally truncated."""
    keep = run['iterations'] <= max_iter if max_iter is not None else slice(None)
    times = (run['collect_times'][keep] + run['train_times'][keep])
    return np.cumsum(times) / 3600, keep

ppo_scratch = load_run_data(hero_file)
grpo_runs = [(lbl, load_run_data(f), c) for lbl, f, c in GRPO_RUNS]
if ppo_scratch is not None and any(r is not None for _, r, _ in grpo_runs):
    fig, (ax_win, ax_floor) = plt.subplots(1, 2, figsize=(14, 5))
    px, pkeep = _wall_hours(ppo_scratch, max_iter=99)  # hero v1 = the from-scratch portion
    for ax, field, ylabel in ((ax_win, 'win_rates', 'Win rate'), (ax_floor, 'avg_floors', 'Avg floor')):
        py = ppo_scratch[field][pkeep]
        ax.plot(px, py, color='black', alpha=0.18, linewidth=1)
        ax.plot(px, _smooth(py), color='black', linewidth=2.4, label='PPO from scratch (hero v1)')
        for lbl, run, c in grpo_runs:
            if run is None:
                continue
            gx, gkeep = _wall_hours(run)
            gy = run[field][gkeep]
            ax.plot(gx, gy, color=c, alpha=0.18, linewidth=1)
            ax.plot(gx, _smooth(gy), color=c, linewidth=2.4, label=lbl)
        ax.set_xlabel('Wall time (hours)'); ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs wall time — from-scratch GRPO vs PPO')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.show()
else:
    print("GRPO wall-time figure skipped (missing stats files)")

# %%
# Entropy phase plots: each run's training trajectory through (x, entropy) space as a faint
# EMA-smoothed line, ending at a strong dot for its current/final state, for x = avg floor and
# x = win rate. PPO runs sharpen (entropy falls) as they improve; the critic-free GRPO runs
# improve while staying far more stochastic.
#
# Dashed segments through the GRPO dots are objective-indifference lines. GRPO's advantages are
# raw return differences (no normalization), so its objective is exactly E[return] +
# entropy_coef*H: the optimizer trades entropy_coef nats per unit of return. Each run's empirical
# d(reward)/dx (fit over the run; includes the win bonus and shaping) converts that to the x-axis:
# points along the dashed line score equally under the training objective. The slopes are steep
# (near-vertical at this aspect ratio), so segments are sized by a fixed *vertical* extent and
# axis limits are pinned to the data. PPO lines need adv_norm_std (now logged by rl_train) since
# its advantages are EWMA-normalized: effective exchange rate = entropy_coef * adv_norm_std.
# Effective entropy/return exchange rate per run, in RAW-return units (the axis the plot uses):
#   GRPO uses raw advantages          -> effective coef = entropy_coef
#   PPO normalizes advantages by std  -> effective coef = entropy_coef * adv_norm_std
ENTROPY_COEF = {'grpo_a': 0.01, 'grpo_b': 0.01, 'ppo_ent10': 0.10, 'ppo_ent25': 0.25}
PPO_NORMALIZED = {'ppo_ent10', 'ppo_ent25'}  # divide advantages by the (logged) adv_norm_std

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
        if gkey in PPO_NORMALIZED:  # raw-unit coef = entropy_coef * latest adv_norm_std
            _, stds = _avg_over_iters(members, 'adv_norm_stds')
            stds = stds[~np.isnan(stds)]
            if stds.size == 0:
                continue
            coef = coef * stds[-1]
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
