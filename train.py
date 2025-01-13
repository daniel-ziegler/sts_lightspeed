# %%
import dataclasses
import sys
import random
from dataclasses import dataclass, fields
from itertools import product
import json
from datetime import datetime
from typing import List
import argparse

from network import MAX_CHOICES, MAX_DECK_SIZE, NN, ActionType, FixedAction, ModelHP, SlayDataset, collate_fn, process_batch, collate_fn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import slaythespire as sts

# %%
torch.set_float32_matmul_precision('high')

# %%
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %%
@dataclass
class TrainingHP:
    batch_size: int = 128
    initial_lr: float = 1e-4
    final_lr: float = 1e-6
    weight_decay: float = 1e-4
    num_epochs: int = 1
    validation_fraction: float = 0.1
    validate_every_n_steps: int = 2000
    log_every_n_steps: int = 20

def parse_args() -> tuple[List[str], TrainingHP]:
    """Parse command line arguments and return data paths and training hyperparameters"""
    parser = argparse.ArgumentParser(description='Train the Slay the Spire AI model')
    
    # Add positional argument for data paths
    parser.add_argument('data_paths', nargs='*', help='Paths to parquet files containing training data')
    
    # Add arguments for each TrainingHP field
    defaults = TrainingHP()
    for field in fields(TrainingHP):
        parser.add_argument(
            f'--{field.name}', 
            type=field.type,
            default=getattr(defaults, field.name),
            help=f'Training hyperparameter {field.name} (default: {getattr(defaults, field.name)})'
        )
    
    args = parser.parse_args()
    
    # Create TrainingHP from parsed args
    hp_dict = {field.name: getattr(args, field.name) for field in fields(TrainingHP)}
    training_hp = TrainingHP(**hp_dict)
    
    return args.data_paths, training_hp

# %%
def is_validation_seed(seed: int, valid_fraction: float = 0.1) -> bool:
    """Deterministically decide if a seed should be in validation set"""
    # Use a simple hash function to get a value between 0 and 1
    hash_val = ((seed * 1327217885) & 0xFFFFFFFF) / 0xFFFFFFFF
    return hash_val < valid_fraction

class SlayDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        return {
            'deck': np.array(row['obs.deck.cards'], dtype=np.int32),
            'deck_upgrades': np.array(row['obs.deck.upgrades'], dtype=np.int32),
            'choices': np.array(row['cards_offered.cards'], dtype=np.int32),
            'choice_upgrades': np.array(row['cards_offered.upgrades'], dtype=np.int32),
            'fixed_obs': np.array(row['obs.fixed_observation'], dtype=np.int32),
            'fixed_actions': np.array(row['fixed_actions'], dtype=np.int32),
            'chosen_idx': row['chosen_idx'],
            'choice_type': row['choice_type'],
            'outcome': row['outcome'],
        }

def load_and_preprocess_data(paths: list[str], validation_fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess data from a parquet file, splitting into train and validation sets.
    
    Args:
        path: Path to the parquet file
        validation_fraction: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_df, valid_df)
    """
    df = pd.concat([pd.read_parquet(path) for path in paths])
    
    # Filter data to handle fixed actions
    data_df = df[
        df.apply(lambda r: (
            # Allow fixed actions (like SKIP)
            (r["choice_type"] == ActionType.FIXED) or
            # Or normal card choices within bounds
            (r["choice_type"] == ActionType.CARD and 
             r["chosen_idx"] >= 0 and 
             r["chosen_idx"] < len(r["cards_offered.cards"]) and
             r["chosen_idx"] < MAX_CHOICES)
        ), axis=1)
    ]

    # Split data
    valid_df = data_df[data_df['seed'].apply(lambda s: is_validation_seed(s, validation_fraction))]
    train_df = data_df[~data_df['seed'].apply(lambda s: is_validation_seed(s, validation_fraction))]

    # Balance validation set
    valid_positives = valid_df[valid_df['outcome'] == 1]
    valid_negatives = valid_df[valid_df['outcome'] == 0]
    n_samples = min(len(valid_positives), len(valid_negatives))
    balanced_valid_df = pd.concat([
        valid_positives.sample(n=n_samples, random_state=42),
        valid_negatives.sample(n=n_samples, random_state=42)
    ])
    
    return train_df, balanced_valid_df

data_paths, base_T = parse_args()
train_df, valid_df = load_and_preprocess_data(data_paths, base_T.validation_fraction)

# %%
np.random.seed(3)


# %%
# Test padding with longer batch element
 #long_batch = batch.copy()
 #long_batch.at[long_batch.index[0], "obs.deck.cards"] #  = np.array([7] * 100)  # Big deck
 ## long_batch.at[long_batch.index[0], "cards_offered.cards"] = np.array([1, 2, 3, 4, 5])  # More card choices
 ## %%
 #long_output = feed_to_net(long_batch)
 #
 ## Verify that outputs for unchanged elements are the same
 #assert torch.allclose(output['card_choice_logits'][1:], long_output['card_choice_logits'][1:,:output['card_choice_logits'].size(1)], rtol=1e-4, atol=1e-6)
 #print("Padding test passed: outputs for unchanged elements are the same.")
 ## %%
 #batch['outcome']

# %%
def train_step(net, opt, batch):
    """
    Perform one training step.
    Returns (loss, accuracy) tuple
    """
    device = net.device
    batch = {k: v.to(device) for k, v in batch.items()}
    output = process_batch(batch, net)
    
    # Get logits for chosen actions based on choice type
    batch_indices = torch.arange(len(batch['chosen_idx']), device=device)
    
    # Initialize chosen logits
    chosen_logits = torch.zeros(len(batch['chosen_idx']), device=device)
    
    # Handle card choices
    card_mask = batch['choice_type'] == ActionType.CARD
    if card_mask.any():
        chosen_logits[card_mask] = output['card_logits'][
            batch_indices[card_mask],
            batch['chosen_idx'][card_mask]
        ]
    
    # Handle fixed actions
    fixed_mask = batch['choice_type'] == ActionType.FIXED
    if fixed_mask.any():
        chosen_logits[fixed_mask] = output['fixed_logits'][
            batch_indices[fixed_mask],
            batch['chosen_idx'][fixed_mask]
        ]
    
    # Clip logits to avoid numerical issues
    chosen_logits = torch.clamp(chosen_logits, min=-20, max=20)
    
    # Check for NaN before loss
    if torch.isnan(chosen_logits).any():
        print("Warning: NaN in logits")
        print("Card logits:", output['card_logits'])
        print("Fixed logits:", output['fixed_logits'])
        print("Chosen logits:", chosen_logits)
        raise ValueError("NaN in logits")
    
    loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'])
    
    # Check for NaN loss
    if torch.isnan(loss):
        print("Warning: NaN loss")
        print("Chosen logits:", chosen_logits)
        print("Outcomes:", batch['outcome'])
        raise ValueError("NaN loss")
    
    opt.zero_grad()
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    opt.step()

    with torch.no_grad():
        accuracy = ((chosen_logits >= 0) == batch['outcome']).float().mean()
    
    return loss.item(), accuracy.item()

def train(net, train_df, valid_df, T: TrainingHP, device: torch.device):
    """
    Train the network using the provided parameters and data.
    
    Args:
        net: The neural network to train
        train_df: Training data DataFrame
        valid_df: Validation data DataFrame
        T: Training hyperparameters
        device: Device to train on
        
    Returns:
        Tuple of (save_path, final_validation_accuracy)
    """
    save_path = f"net.outcome.lr{T.initial_lr:.1e}.wd{T.weight_decay:.1e}.e{T.num_epochs}.pt"
    
    valid_loader = torch.utils.data.DataLoader(
        SlayDataset(valid_df),
        batch_size=T.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        SlayDataset(train_df),
        batch_size=T.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(net.parameters(), lr=T.initial_lr, weight_decay=T.weight_decay)
    total_steps = len(train_loader) * T.num_epochs

    valid_acc = np.nan
    
    for epoch in range(T.num_epochs):
        print(f"Epoch {epoch}")
        for i, batch in enumerate(train_loader):
            # Update learning rate
            current_step = epoch * len(train_loader) + i
            current_lr = T.initial_lr - (T.initial_lr - T.final_lr) * (current_step / total_steps)
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
            
            loss, acc = train_step(net, opt, batch)

            if i % T.log_every_n_steps == 0:
                print(f"{i}: loss={loss:.4f}, acc={acc:.3f}, lr={current_lr:.2e}")
            if i != 0 and i % T.validate_every_n_steps == 0 or i == len(train_loader) - 1:
                print(f"{i}: Validating")
                valid_losses, valid_acc = validate(valid_loader, net, device)
                print(f"Valid loss: {np.mean(valid_losses)}")
                print(f"Valid acc: {valid_acc}")

    torch.save(net.state_dict(), save_path)
    return save_path, valid_acc

def validate(valid_loader, net, device):
    """Run validation and return losses and accuracy."""
    valid_losses = []
    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for batch in valid_loader:
            output = process_batch(batch, net)
            
            # Get logits for chosen actions based on choice type
            batch_indices = torch.arange(len(batch['chosen_idx']), device=device)
            
            # Initialize chosen logits
            chosen_logits = torch.zeros(len(batch['chosen_idx']), device=device)
            
            # Handle card choices
            card_mask = batch['choice_type'] == ActionType.CARD
            if card_mask.any():
                chosen_logits[card_mask] = output['card_logits'][
                    batch_indices[card_mask],
                    batch['chosen_idx'][card_mask]
                ]
            
            # Handle fixed actions
            fixed_mask = batch['choice_type'] == ActionType.FIXED
            if fixed_mask.any():
                chosen_logits[fixed_mask] = output['fixed_logits'][
                    batch_indices[fixed_mask],
                    batch['chosen_idx'][fixed_mask]
                ]
            
            # Clip logits to avoid numerical issues
            chosen_logits = torch.clamp(chosen_logits, min=-20, max=20)
            
            loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'].to(device))
            pred = chosen_logits >= 0
            valid_losses.append(loss.item())
            valid_targets.append(batch['outcome'].cpu().numpy())
            valid_preds.append(pred.cpu().numpy())
        
        acc = np.mean(np.concatenate(valid_preds) == np.concatenate(valid_targets))
    return valid_losses, acc

def hyperparameter_sweep(train_df, valid_df):
    """Run hyperparameter sweep and save results."""
    learning_rates = np.geomspace(1e-4, 2e-5, 5)
    weight_decays = np.geomspace(1e-5, 1e-5, 1)
    
    results = []
    
    
    # Run sweep
    for lr, wd in product(learning_rates, weight_decays):
        print(f"\nTraining with lr={lr:.1e}, wd={wd:.1e}")
        
        # Create fresh model for each run
        H = ModelHP()
        net = NN(H)
        net = net.to(device)
        net = torch.compile(net, mode="reduce-overhead")
        
        # Update hyperparameters
        T = dataclasses.replace(base_T,
            initial_lr=lr,
            weight_decay=wd,
        )
        
        # Train model
        save_path, valid_acc = train(net, train_df, valid_df, T, device)

        del net
        
        # Store results
        result = {
            'learning_rate': lr,
            'weight_decay': wd,
            'valid_accuracy': valid_acc,
            'model_path': save_path,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)
        
        # Save intermediate results
        with open('sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    # Find best model
    best_result = max(results, key=lambda x: x['valid_accuracy'])
    print("\nBest model:")
    print(f"Learning rate: {best_result['learning_rate']:.1e}")
    print(f"Weight decay: {best_result['weight_decay']:.1e}")
    print(f"Validation accuracy: {best_result['valid_accuracy']:.4f}")
    print(f"Model path: {best_result['model_path']}")
    
    return best_result['model_path'], results

save_path, results = hyperparameter_sweep(train_df, valid_df)

# %%

# %%
H = ModelHP()
net = NN(H)
net = net.to(device)
net = torch.compile(net, mode="reduce-overhead")

# %%
state = torch.load(save_path, weights_only=True, map_location=device)
net.load_state_dict(state)

# %%
batch = valid_df.sample(20)
# %%
len(batch.iloc[0]['obs.deck.cards'])

# %%
with torch.no_grad():
    batch_data = collate_fn([SlayDataset(batch).__getitem__(i) for i in range(len(batch))])
    output = process_batch(batch_data, net)

    # Get win probabilities for all cards and mark chosen ones
    card_probs = torch.sigmoid(output['card_logits'])
    fixed_probs = torch.sigmoid(output['fixed_logits'])
    chosen_indices = batch_data['chosen_idx']
    batch_indices = torch.arange(len(batch), device=device)

# %%

# Print probabilities for each example in batch
for i in range(len(batch)):
    choice_type = batch_data['choice_type'][i].item()
    chosen_idx = chosen_indices[i].item()
    
    # Print card choices
    card_prob_strs = []
    probs = card_probs[i].cpu().numpy()
    cards_offered = batch.iloc[i]['cards_offered.cards']
    upgrades = batch.iloc[i]['cards_offered.upgrades']
    
    for j, (card_id, upgrade) in enumerate(zip(cards_offered, upgrades)):
        if probs[j] == float('-inf'):  # Skip masked values
            continue
        card = sts.Card(sts.CardId(card_id), upgrade)
        prob_str = f"{card}({probs[j]:.3f})"
        if choice_type == ActionType.CARD and j == chosen_idx:
            prob_str = f"[{prob_str}]"  # Mark chosen card with brackets
        card_prob_strs.append(prob_str)
    
    # Print fixed actions
    fixed_prob_strs = []
    probs = fixed_probs[i].cpu().numpy()
    fixed_actions = batch.iloc[i]['fixed_actions']
    
    for j, action in enumerate(fixed_actions):
        if probs[j] == float('-inf'):  # Skip masked values
            continue
        action_name = FixedAction(action).name
        prob_str = f"{action_name}({probs[j]:.3f})"
        if choice_type == ActionType.FIXED and j == chosen_idx:
            prob_str = f"[{prob_str}]"  # Mark chosen action with brackets
        fixed_prob_strs.append(prob_str)
    
    print(f"Example {i} ({ActionType(choice_type).name}):")
    if card_prob_strs:
        print(f"  Cards: {', '.join(card_prob_strs)}")
    if fixed_prob_strs:
        print(f"  Fixed: {', '.join(fixed_prob_strs)}")
    print(batch.iloc[i])
    print()
    

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Create fresh validation loader
valid_loader = torch.utils.data.DataLoader(
    SlayDataset(valid_df),
    batch_size=base_T.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)

valid_preds = []
valid_targets = []
card_predictions = {}  # Dictionary to store predictions for each card
fixed_predictions = {}  # Dictionary to store predictions for each fixed action

# Get predictions for validation set
with torch.no_grad():
    for batch in valid_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = process_batch(batch, net)
        
        # Get chosen action predictions for ROC curve
        batch_indices = torch.arange(len(batch['chosen_idx']), device=device)
        
        # Initialize chosen logits
        chosen_logits = torch.zeros(len(batch['chosen_idx']), device=device)
        
        # Handle card choices
        card_mask = batch['choice_type'] == ActionType.CARD
        if card_mask.any():
            chosen_logits[card_mask] = output['card_logits'][
                batch_indices[card_mask],
                batch['chosen_idx'][card_mask]
            ]
        
        # Handle fixed actions
        fixed_mask = batch['choice_type'] == ActionType.FIXED
        if fixed_mask.any():
            chosen_logits[fixed_mask] = output['fixed_logits'][
                batch_indices[fixed_mask],
                batch['chosen_idx'][fixed_mask]
            ]
        
        probs = torch.sigmoid(chosen_logits).cpu().numpy()
        valid_preds.extend(probs)
        valid_targets.extend(batch['outcome'].cpu().numpy())
        
        # Get predictions for all cards
        card_probs = torch.sigmoid(output['card_logits'])
        fixed_probs = torch.sigmoid(output['fixed_logits'])
        
        # For each example in batch
        for i in range(len(batch['choices'])):
            # Get all valid probabilities for this choice
            all_valid_probs = []
            
            # Get valid card probabilities
            choices = batch['choices'][i]
            upgrades = batch['choice_upgrades'][i]
            valid_mask = choices != sts.CardId.INVALID.value
            if valid_mask.any():
                all_valid_probs.extend(card_probs[i, valid_mask].cpu().numpy())
            
            # Get valid fixed action probabilities
            fixed_actions = batch['fixed_actions'][i]
            valid_mask = fixed_actions != FixedAction.INVALID.value
            if valid_mask.any():
                all_valid_probs.extend(fixed_probs[i, valid_mask].cpu().numpy())
            
            all_valid_probs = np.array(all_valid_probs)
            
            # Process card choices
            card_idx = 0  # Keep track of position in all_valid_probs
            choices = batch['choices'][i]
            upgrades = batch['choice_upgrades'][i]
            valid_mask = choices != sts.CardId.INVALID.value
            
            for j, (card_id, upgrade) in enumerate(zip(choices[valid_mask], upgrades[valid_mask])):
                card = sts.Card(sts.CardId(card_id), upgrade)
                card_key = str(card)
                
                # Calculate relative probability compared to all other options
                other_probs = np.concatenate([all_valid_probs[:card_idx], all_valid_probs[card_idx+1:]])
                relative_prob = all_valid_probs[card_idx] - np.mean(other_probs) if len(other_probs) > 0 else 0.0
                
                if card_key not in card_predictions:
                    card_predictions[card_key] = []
                card_predictions[card_key].append(relative_prob)
                card_idx += 1
            
            # Process fixed actions
            fixed_actions = batch['fixed_actions'][i]
            valid_mask = fixed_actions != FixedAction.INVALID.value
            
            for j, action in enumerate(fixed_actions[valid_mask]):
                action_name = FixedAction(action.item()).name
                
                # Calculate relative probability compared to all other options
                other_probs = np.concatenate([all_valid_probs[:card_idx], all_valid_probs[card_idx+1:]])
                relative_prob = all_valid_probs[card_idx] - np.mean(other_probs) if len(other_probs) > 0 else 0.0
                
                if action_name not in fixed_predictions:
                    fixed_predictions[action_name] = []
                fixed_predictions[action_name].append(relative_prob)
                card_idx += 1

# Calculate and plot ROC curve
fpr, tpr, _ = roc_curve(valid_targets, valid_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Calculate and print card statistics
print("\nCard Win Probability Statistics (relative to alternatives):")
card_stats = {}
for card_key, preds in card_predictions.items():
    preds = np.array(preds)
    card_stats[card_key] = {
        'mean': np.mean(preds),
        'std': np.std(preds),
        'count': len(preds)
    }

# Sort and print card statistics
sorted_cards = sorted(card_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
for card, stats in sorted_cards:
    if stats['count'] >= 10:  # Only show cards with enough samples
        print(f"{card:25} {stats['mean']:+.3f} ±{stats['std']:.3f} (n={stats['count']})")

# Calculate and print fixed action statistics
print("\nFixed Action Win Probability Statistics (relative to alternatives):")
fixed_stats = {}
for action_name, preds in fixed_predictions.items():
    preds = np.array(preds)
    fixed_stats[action_name] = {
        'mean': np.mean(preds),
        'std': np.std(preds),
        'count': len(preds)
    }

# Sort and print fixed action statistics
sorted_actions = sorted(fixed_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
for action, stats in sorted_actions:
    if stats['count'] >= 10:  # Only show actions with enough samples
        print(f"{action:25} {stats['mean']:+.3f} ±{stats['std']:.3f} (n={stats['count']})")

# %%
print(f"Saved to {save_path}")

# %%
