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

# Define files to load - add your stats files here
stats_files = [
    'bigger.pt.stats.jsonl',
    'lr3e-5.pt.stats.jsonl',
]

# Auto-discover stats files if they exist
#stats_files = glob.glob('*.stats.jsonl')

# Color palette for different runs
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

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
        'clipfracs': np.array([d.get('clipfrac', np.nan) for d in data]),
    }

# Load all runs
runs = []
for i, filename in enumerate(stats_files):
    run_data = load_run_data(filename)
    if run_data is not None:
        run_data['color'] = colors[i % len(colors)]
        run_data['label'] = os.path.basename(filename).replace('.stats.jsonl', '')
        runs.append(run_data)

print(f"Loaded {len(runs)} runs:")
for run in runs:
    print(f"  {run['label']}: {len(run['iterations'])} iterations")

# %% [markdown]
# ## Performance Metrics

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Win rate over time
for run in runs:
    ax1.plot(run['iterations'], run['win_rates'], color=run['color'], 
             label=run['label'], linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Win Rate')
ax1.set_title('Win Rate Over Training')
ax1.grid(True, alpha=0.3)
if len(runs) > 1:
    ax1.legend()

# Average floor reached
for run in runs:
    ax2.plot(run['iterations'], run['avg_floors'], color=run['color'], 
             label=run['label'], linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Average Floor')
ax2.set_title('Average Floor Reached')
ax2.grid(True, alpha=0.3)
if len(runs) > 1:
    ax2.legend()

# Average reward
for run in runs:
    ax3.plot(run['iterations'], run['avg_rewards'], color=run['color'], 
             label=run['label'], linewidth=2)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Average Reward')
ax3.set_title('Average Reward Over Training')
ax3.grid(True, alpha=0.3)
if len(runs) > 1:
    ax3.legend()

# Collection and training times
for run in runs:
    ax4.plot(run['iterations'], run['collect_times'], color=run['color'], 
             linestyle='-', label=f'{run["label"]} Collection', linewidth=2)
    ax4.plot(run['iterations'], run['train_times'], color=run['color'], 
             linestyle='--', label=f'{run["label"]} Training', linewidth=2)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Time (seconds)')
ax4.set_title('Execution Times')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Training Losses and Optimization

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Policy loss
for run in runs:
    ax1.plot(run['iterations'], run['policy_losses'], color=run['color'], 
             label=run['label'], linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Policy Loss')
ax1.set_title('Policy Loss Over Training')
ax1.grid(True, alpha=0.3)
if len(runs) > 1:
    ax1.legend()

# Value loss
for run in runs:
    ax2.plot(run['iterations'], run['value_losses'], color=run['color'], 
             label=run['label'], linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Value Loss')
ax2.set_title('Value Loss Over Training')
ax2.grid(True, alpha=0.3)
if len(runs) > 1:
    ax2.legend()

# Entropy
for run in runs:
    ax3.plot(run['iterations'], run['entropies'], color=run['color'], 
             label=run['label'], linewidth=2)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Entropy')
ax3.set_title('Policy Entropy Over Training')
ax3.grid(True, alpha=0.3)
if len(runs) > 1:
    ax3.legend()

# KL divergence
for run in runs:
    ax4.plot(run['iterations'], run['kl_divs'], color=run['color'], 
             label=run['label'], linewidth=2)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('KL Divergence')
ax4.set_title('KL Divergence Over Training')
ax4.grid(True, alpha=0.3)
if len(runs) > 1:
    ax4.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Optimization Details

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Gradient norm
for run in runs:
    ax1.plot(run['iterations'], run['grad_norms'], color=run['color'], 
             label=run['label'], linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Gradient Norm')
ax1.set_title('Gradient Norm Over Training')
ax1.grid(True, alpha=0.3)
if len(runs) > 1:
    ax1.legend()

# Clip fraction
for run in runs:
    ax2.plot(run['iterations'], run['clipfracs'], color=run['color'], 
             label=run['label'], linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Clip Fraction')
ax2.set_title('PPO Clip Fraction Over Training')
ax2.grid(True, alpha=0.3)
if len(runs) > 1:
    ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary Statistics

# %%
print("Training Summary:")
print("=" * 50)

for run in runs:
    print(f"\n{run['label']}:")
    print(f"  Final win rate: {run['win_rates'][-1]:.1%}")
    print(f"  Final average floor: {run['avg_floors'][-1]:.1f}")
    print(f"  Final average reward: {run['avg_rewards'][-1]:.3f}")
    print(f"  Peak win rate: {max(run['win_rates']):.1%} at iteration {run['iterations'][run['win_rates'].index(max(run['win_rates']))]}")
    print(f"  Peak average floor: {max(run['avg_floors']):.1f} at iteration {run['iterations'][run['avg_floors'].index(max(run['avg_floors']))]}")
    print(f"  Average collection time: {np.mean(run['collect_times']):.1f}s")
    print(f"  Average training time: {np.mean(run['train_times']):.1f}s")
    print(f"  Total iterations: {len(run['iterations'])}")

if len(runs) > 1:
    print(f"\nComparison:")
    best_final_wr = max(runs, key=lambda r: r['win_rates'][-1])
    best_peak_wr = max(runs, key=lambda r: max(r['win_rates']))
    print(f"  Best final win rate: {best_final_wr['label']} ({best_final_wr['win_rates'][-1]:.1%})")
    print(f"  Best peak win rate: {best_peak_wr['label']} ({max(best_peak_wr['win_rates']):.1%})")

# %%