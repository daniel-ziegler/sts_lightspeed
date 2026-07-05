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

from silverbot.network import MAX_CHOICES, MAX_DECK_SIZE, NN, ActionType, FixedAction, ModelHP, SlayDataset, collate_fn, output_to_cpu, process_batch, choice_space, move_to_device
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

def parse_args() -> tuple[List[str], TrainingHP, str, object]:
    """Parse command line arguments and return data paths, training hyperparameters, prediction mode, and args object"""
    parser = argparse.ArgumentParser(description='Train the Slay the Spire AI model')
    
    # Add positional argument for data paths
    parser.add_argument('data_paths', nargs='*', help='Paths to parquet files containing training data')
    
    # Add prediction mode argument
    parser.add_argument('--prediction-mode', type=str, default='outcome', 
                        choices=['outcome', 'pstrike'], 
                        help='What to predict: outcome (win/loss) or pstrike (number of perfected strikes)')
    parser.add_argument('--no-torch-compile', action='store_true',
                        help='Disable torch.compile for the neural network')
    
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
    
    return args.data_paths, training_hp, args.prediction_mode, args

# %%
def is_validation_seed(seed: int, valid_fraction: float = 0.1) -> bool:
    """Deterministically decide if a seed should be in validation set"""
    # Use a simple hash function to get a value between 0 and 1
    hash_val = ((seed * 1327217885) & 0xFFFFFFFF) / 0xFFFFFFFF
    return hash_val < valid_fraction

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
    
    data_df = df
    # Filter data to handle fixed actions
    # data_df = df[
    #     df.apply(lambda r: (
    #         r["choice_type"] == ActionType.FIXED or
    #         r["choice_type"] == ActionType.RELIC or
    #         r["choice_type"] == ActionType.POTION or
    #         (r["choice_type"] == ActionType.CARD and 
    #          r["chosen_idx"] >= 0 and 
    #          r["chosen_idx"] < len(r["cards_offered.cards"]) and
    #          r["chosen_idx"] < MAX_CHOICES)
    #     ), axis=1)
    # ]
    print(data_df.columns)

    # Split data
    valid_df = data_df[data_df['seed'].apply(lambda s: is_validation_seed(s, validation_fraction))]
    train_df = data_df[~data_df['seed'].apply(lambda s: is_validation_seed(s, validation_fraction))]
    print(valid_df.columns)

    # Balance validation set
    valid_positives = valid_df[valid_df['outcome'] == 1]
    valid_negatives = valid_df[valid_df['outcome'] == 0]
    n_samples = min(len(valid_positives), len(valid_negatives))
    balanced_valid_df = pd.concat([
        valid_positives.sample(n=n_samples, random_state=42),
        valid_negatives.sample(n=n_samples, random_state=42)
    ])
    
    return train_df, balanced_valid_df

data_paths, base_T, prediction_mode, args = parse_args()
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
def train_step(net, opt, batch, prediction_mode='outcome'):
    """
    Perform one training step.
    Returns (loss, accuracy) tuple
    """
    device = net.device
    batch = move_to_device(batch, device)
    output = process_batch(batch, net)  # output is [batch_size, max_choices] flat logits
    
    # Get logits for chosen actions using flat indices
    batch_indices = torch.arange(len(batch['chosen_idx']), device=device)
    chosen_logits = output[batch_indices, batch['chosen_idx']]
    
    # Clip logits to avoid numerical issues
    chosen_logits = torch.clamp(chosen_logits, min=-20, max=20)
    
    # Check for NaN before loss
    if torch.isnan(chosen_logits).any():
        print("Warning: NaN in logits")
        print("Output shape:", output.shape)
        print("Chosen indices:", batch['chosen_idx'])
        print("Chosen logits:", chosen_logits)
        raise ValueError("NaN in logits")
    
    if prediction_mode == 'outcome':
        # Binary classification for win/loss
        loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'])
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss")
            print("Chosen logits:", chosen_logits)
            print("Outcomes:", batch['outcome'])
            raise ValueError("NaN loss")
        
        with torch.no_grad():
            accuracy = ((chosen_logits >= 0) == batch['outcome']).float().mean()
            
    elif prediction_mode == 'pstrike':
        # Regression for pstrike count
        pstrike_targets = batch['pstrike_count'].float()
        loss = F.mse_loss(chosen_logits, pstrike_targets)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss")
            print("Chosen logits:", chosen_logits)
            print("Pstrike targets:", pstrike_targets)
            raise ValueError("NaN loss")
        
        with torch.no_grad():
            # For regression, use mean absolute error as accuracy metric
            accuracy = F.l1_loss(chosen_logits, pstrike_targets).item()
    
    else:
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")
    
    opt.zero_grad()
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    opt.step()
    
    return loss.item(), accuracy.item()

def train(net, train_df, valid_df, T: TrainingHP, device: torch.device, prediction_mode='outcome'):
    """
    Train the network using the provided parameters and data.
    
    Args:
        net: The neural network to train
        train_df: Training data DataFrame
        valid_df: Validation data DataFrame
        T: Training hyperparameters
        device: Device to train on
        prediction_mode: What to predict ('outcome' or 'pstrike')
        
    Returns:
        Tuple of (save_path, final_validation_accuracy)
    """
    save_path = f"net.{prediction_mode}.lr{T.initial_lr:.1e}.wd{T.weight_decay:.1e}.e{T.num_epochs}.pt"
    
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
            
            loss, acc = train_step(net, opt, batch, prediction_mode)

            if i % T.log_every_n_steps == 0:
                if prediction_mode == 'outcome':
                    print(f"{i}: loss={loss:.4f}, acc={acc:.3f}, lr={current_lr:.2e}")
                else:
                    print(f"{i}: loss={loss:.4f}, mae={acc:.3f}, lr={current_lr:.2e}")
            if i != 0 and i % T.validate_every_n_steps == 0 or i == len(train_loader) - 1:
                print(f"{i}: Validating")
                valid_losses, valid_acc = validate(valid_loader, net, device, prediction_mode)
                print(f"Valid loss: {np.mean(valid_losses)}")
                if prediction_mode == 'outcome':
                    print(f"Valid acc: {valid_acc}")
                else:
                    print(f"Valid mae: {valid_acc}")

    torch.save(net.state_dict(), save_path)
    return save_path, valid_acc

def validate(valid_loader, net, device, prediction_mode='outcome'):
    """Run validation and return losses and accuracy."""
    valid_losses = []
    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = move_to_device(batch, device)
            output = process_batch(batch, net)  # output is [batch_size, max_choices] flat logits
            
            # Get logits for chosen actions using flat indices
            batch_indices = torch.arange(len(batch['chosen_idx']), device=device)
            chosen_logits = output[batch_indices, batch['chosen_idx']]
            
            # Clip logits to avoid numerical issues
            chosen_logits = torch.clamp(chosen_logits, min=-20, max=20)
            
            if prediction_mode == 'outcome':
                loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'])
                pred = chosen_logits >= 0
                valid_losses.append(loss.item())
                valid_targets.append(batch['outcome'].cpu().numpy())
                valid_preds.append(pred.cpu().numpy())
            elif prediction_mode == 'pstrike':
                pstrike_targets = batch['pstrike_count'].float()
                loss = F.mse_loss(chosen_logits, pstrike_targets)
                valid_losses.append(loss.item())
                valid_targets.append(pstrike_targets.cpu().numpy())
                valid_preds.append(chosen_logits.cpu().numpy())
        
        if prediction_mode == 'outcome':
            acc = np.mean(np.concatenate(valid_preds) == np.concatenate(valid_targets))
        else:
            # For regression, use mean absolute error
            acc = np.mean(np.abs(np.concatenate(valid_preds) - np.concatenate(valid_targets)))
    return valid_losses, acc

def hyperparameter_sweep(train_df, valid_df, prediction_mode='outcome'):
    """Run hyperparameter sweep and save results."""
    learning_rates = np.geomspace(5e-5, 1e-5, 5)
    weight_decays = np.geomspace(1e-5, 1e-5, 1)
    
    results = []
    
    
    # Run sweep
    for lr, wd in product(learning_rates, weight_decays):
        print(f"\nTraining with lr={lr:.1e}, wd={wd:.1e}")
        
        # Create fresh model for each run
        H = ModelHP(use_value_head=False)
        net = NN(H)
        net = net.to(device)
        if not args.no_torch_compile:
            net = torch.compile(net, mode="default")
        
        # Update hyperparameters
        T = dataclasses.replace(base_T,
            initial_lr=lr,
            weight_decay=wd,
        )
        
        # Train model
        save_path, valid_acc = train(net, train_df, valid_df, T, device, prediction_mode)

        del net
        
        # Store results
        result = {
            'learning_rate': lr,
            'weight_decay': wd,
            'valid_accuracy': valid_acc,
            'model_path': save_path,
            'timestamp': datetime.now().isoformat(),
            'prediction_mode': prediction_mode
        }
        results.append(result)
        
        # Save intermediate results
        with open(f'sweep_results_{prediction_mode}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
    # Find best model
    if prediction_mode == 'outcome':
        best_result = max(results, key=lambda x: x['valid_accuracy'])
    else:
        # For regression, lower MAE is better
        best_result = min(results, key=lambda x: x['valid_accuracy'])
    
    print("\nBest model:")
    print(f"Learning rate: {best_result['learning_rate']:.1e}")
    print(f"Weight decay: {best_result['weight_decay']:.1e}")
    if prediction_mode == 'outcome':
        print(f"Validation accuracy: {best_result['valid_accuracy']:.4f}")
    else:
        print(f"Validation MAE: {best_result['valid_accuracy']:.4f}")
    print(f"Model path: {best_result['model_path']}")
    
    return best_result['model_path'], results

save_path, results = hyperparameter_sweep(train_df, valid_df, prediction_mode)

# %%
H = ModelHP(use_value_head=False)
net = NN(H)
net = net.to(device)
if not args.no_torch_compile:
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
    output = process_batch(batch_data, net)  # output is [batch_size, max_choices] flat logits

    # Get win probabilities for all choices
    all_probs = torch.sigmoid(output)
    chosen_indices = batch_data['chosen_idx']
    batch_indices = torch.arange(len(batch), device=device)

# %%

# Print probabilities for each example in batch
for i in range(len(batch)):
    chosen_idx = chosen_indices[i].item()
    
    # Get probabilities for this batch item 
    probs = all_probs[i].cpu().numpy()
    
    # Reconstruct the choices structure for this batch item
    choices_dict = {
        'deck': [(card_id, upgrade) for card_id, upgrade in zip(
            batch.iloc[i]['cards_offered.cards'], 
            batch.iloc[i]['cards_offered.upgrades']
        )],
        'relics': list(batch.iloc[i]['relics_offered']),
        'potions': list(batch.iloc[i]['potions_offered']),
        'fixed': list(batch.iloc[i]['fixed_actions'])
    }
    
    # Build probability strings by category
    card_prob_strs = []
    relic_prob_strs = []
    potion_prob_strs = []
    fixed_prob_strs = []
    
    # Go through each valid logit and convert back to semantic choice
    for logit_idx in range(len(probs)):
        if logit_idx >= choice_space.length(choices_dict):
            break
            
        try:
            path = choice_space.ix_to_path(choices_dict, logit_idx)
            prob = probs[logit_idx]
            
            if path[0] == 'deck':
                # Card choice
                card_idx = path[1]
                if card_idx < len(choices_dict['deck']):
                    card_id, upgrade = choices_dict['deck'][card_idx]
                    card = sts.Card(sts.CardId(card_id), upgrade)
                    prob_str = f"{card}({prob:.3f})"
                    if logit_idx == chosen_idx:
                        prob_str = f"[{prob_str}]"
                    card_prob_strs.append(prob_str)
            
            elif path[0] == 'relics':
                # Relic choice
                relic_idx = path[1]
                if relic_idx < len(choices_dict['relics']):
                    relic_id = choices_dict['relics'][relic_idx]
                    relic_name = sts.RelicId(relic_id).name
                    prob_str = f"{relic_name}({prob:.3f})"
                    if logit_idx == chosen_idx:
                        prob_str = f"[{prob_str}]"
                    relic_prob_strs.append(prob_str)
            
            elif path[0] == 'potions':
                # Potion choice
                potion_idx = path[1]
                if potion_idx < len(choices_dict['potions']):
                    potion_id = choices_dict['potions'][potion_idx]
                    potion_name = sts.Potion(potion_id).name
                    prob_str = f"{potion_name}({prob:.3f})"
                    if logit_idx == chosen_idx:
                        prob_str = f"[{prob_str}]"
                    potion_prob_strs.append(prob_str)
            
            elif path[0] == 'fixed':
                # Fixed action choice
                action_idx = path[1]
                if action_idx < len(choices_dict['fixed']):
                    action = choices_dict['fixed'][action_idx]
                    action_name = FixedAction(action).name
                    prob_str = f"{action_name}({prob:.3f})"
                    if logit_idx == chosen_idx:
                        prob_str = f"[{prob_str}]"
                    fixed_prob_strs.append(prob_str)
                    
        except (IndexError, KeyError):
            # Skip invalid indices
            continue
    
    print(f"Example {i}:")
    if card_prob_strs:
        print(f"  Cards: {', '.join(card_prob_strs)}")
    if relic_prob_strs:
        print(f"  Relics: {', '.join(relic_prob_strs)}")
    if potion_prob_strs:
        print(f"  Potions: {', '.join(potion_prob_strs)}")
    if fixed_prob_strs:
        print(f"  Fixed: {', '.join(fixed_prob_strs)}")
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
card_predictions = {}
fixed_predictions = {}
relic_predictions = {}
potion_predictions = {}

# Get predictions for validation set
with torch.no_grad():
    for batch in valid_loader:
        batch = move_to_device(batch, device)
        output = process_batch(batch, net)  # output is [batch_size, max_choices] flat logits
        
        # Get chosen action predictions for ROC curve
        batch_indices = torch.arange(len(batch['chosen_idx']), device=device)
        chosen_logits = output[batch_indices, batch['chosen_idx']]
        
        probs = torch.sigmoid(chosen_logits).cpu().numpy()
        valid_preds.extend(probs)
        valid_targets.extend(batch['outcome'].cpu().numpy())
        
        responses = output_to_cpu(output, batch)
        
        for i, response in enumerate(responses):
            all_valid_probs = 1 / (1 + np.exp(-response))  # Sigmoid activation
            
            # Reconstruct the choices structure for this batch item
            # We need to get the original data to reconstruct choices
            batch_size = len(batch['chosen_idx'])
            
            # Count valid choices by category for this batch item
            deck_valid = batch['choices']['deck']['mask'][i].sum().item()
            relics_valid = batch['choices']['relics']['mask'][i].sum().item()
            potions_valid = batch['choices']['potions']['mask'][i].sum().item()
            fixed_valid = batch['choices']['fixed']['mask'][i].sum().item()
            
            # Extract choices from batch tensors
            deck_choices = []
            for j in range(deck_valid):
                card_id, upgrade = batch['choices']['deck']['value'][i, j].tolist()
                deck_choices.append((card_id, upgrade))
            
            relic_choices = []
            for j in range(relics_valid):
                relic_id = batch['choices']['relics']['value'][i, j].item()
                relic_choices.append(relic_id)
            
            potion_choices = []
            for j in range(potions_valid):
                potion_id = batch['choices']['potions']['value'][i, j].item()
                potion_choices.append(potion_id)
            
            fixed_choices = []
            for j in range(fixed_valid):
                action = batch['choices']['fixed']['value'][i, j].item()
                fixed_choices.append(action)
            
            choices_dict = {
                'deck': deck_choices,
                'relics': relic_choices,
                'potions': potion_choices,
                'fixed': fixed_choices
            }
            
            # Process each valid logit
            for logit_idx in range(len(all_valid_probs)):
                if logit_idx >= choice_space.length(choices_dict):
                    break
                    
                try:
                    path = choice_space.ix_to_path(choices_dict, logit_idx)
                    
                    # Calculate relative probability compared to all other options
                    other_probs = np.concatenate([all_valid_probs[:logit_idx], all_valid_probs[logit_idx+1:]])
                    relative_prob = all_valid_probs[logit_idx] - np.mean(other_probs) if len(other_probs) > 0 else 0.0
                    
                    if path[0] == 'deck':
                        # Card choice
                        card_idx = path[1]
                        if card_idx < len(choices_dict['deck']):
                            card_id, upgrade = choices_dict['deck'][card_idx]
                            card = sts.Card(sts.CardId(card_id), upgrade)
                            card_key = str(card)
                            
                            if card_key not in card_predictions:
                                card_predictions[card_key] = []
                            card_predictions[card_key].append(relative_prob)
                    
                    elif path[0] == 'relics':
                        # Relic choice
                        relic_idx = path[1]
                        if relic_idx < len(choices_dict['relics']):
                            relic_id = choices_dict['relics'][relic_idx]
                            relic_name = sts.RelicId(relic_id).name
                            
                            if relic_name not in relic_predictions:
                                relic_predictions[relic_name] = []
                            relic_predictions[relic_name].append(relative_prob)
                    
                    elif path[0] == 'potions':
                        # Potion choice
                        potion_idx = path[1]
                        if potion_idx < len(choices_dict['potions']):
                            potion_id = choices_dict['potions'][potion_idx]
                            potion_name = sts.Potion(potion_id).name
                            
                            if potion_name not in potion_predictions:
                                potion_predictions[potion_name] = []
                            potion_predictions[potion_name].append(relative_prob)
                    
                    elif path[0] == 'fixed':
                        # Fixed action choice
                        action_idx = path[1]
                        if action_idx < len(choices_dict['fixed']):
                            action = choices_dict['fixed'][action_idx]
                            action_name = FixedAction(action).name
                            
                            if action_name not in fixed_predictions:
                                fixed_predictions[action_name] = []
                            fixed_predictions[action_name].append(relative_prob)
                            
                except (IndexError, KeyError):
                    # Skip invalid indices
                    continue

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

# Calculate and print relic statistics
print("\nRelic Win Probability Statistics (relative to alternatives):")
relic_stats = {}
for relic_name, preds in relic_predictions.items():
    preds = np.array(preds)
    relic_stats[relic_name] = {
        'mean': np.mean(preds),
        'std': np.std(preds),
        'count': len(preds)
    }

# Sort and print relic statistics
sorted_relics = sorted(relic_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
for relic, stats in sorted_relics:
    if stats['count'] >= 10:  # Only show relics with enough samples
        print(f"{relic:25} {stats['mean']:+.3f} ±{stats['std']:.3f} (n={stats['count']})")

# Calculate and print potion statistics
print("\nPotion Win Probability Statistics (relative to alternatives):")
potion_stats = {}
for potion_name, preds in potion_predictions.items():
    preds = np.array(preds)
    potion_stats[potion_name] = {
        'mean': np.mean(preds),
        'std': np.std(preds),
        'count': len(preds)
    }

# Sort and print potion statistics
sorted_potions = sorted(potion_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
for potion, stats in sorted_potions:
    if stats['count'] >= 10:  # Only show potions with enough samples
        print(f"{potion:25} {stats['mean']:+.3f} ±{stats['std']:.3f} (n={stats['count']})")

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
