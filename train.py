# %%
import sys
import random
from dataclasses import dataclass

from network import MAX_CHOICES, NN, ModelHP, SlayDataset, collate_fn, process_batch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import slaythespire as sts
from randomplayouts import ActionType

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
    num_epochs: int = 2
    validation_fraction: float = 0.1
    validate_every_n_steps: int = 2000
    log_every_n_steps: int = 20

# %%
H = ModelHP()
T = TrainingHP()

net = NN(H)
net = net.to(device)
net = torch.compile(net, mode="reduce-overhead")

# %%
df = pd.read_parquet("rollouts0_100000.net.parquet")
# df = pd.read_parquet("rollouts0_6000.parquet")

# %%
data_df: pd.DataFrame = df[
    df.apply(lambda r: (
        r["chosen_idx"] < len(r["cards_offered.cards"]) and  # Original check
        r["chosen_idx"] < MAX_CHOICES  # Use constant
    ), axis=1)
]
assert (data_df["choice_type"] == ActionType.CARD).all()
print(f"Filtered out {len(df) - len(data_df)} rows with chosen_idx >= {MAX_CHOICES}")
print(f"Remaining rows: {len(data_df)}")

# %%
# Split into train and validation sets based on seed value
def is_validation_seed(seed: int, valid_fraction: float = 0.1) -> bool:
    """Deterministically decide if a seed should be in validation set"""
    # Use a simple hash function to get a value between 0 and 1
    hash_val = ((seed * 1327217885) & 0xFFFFFFFF) / 0xFFFFFFFF
    return hash_val < valid_fraction

# Split data
valid_df = data_df[data_df['seed'].apply(is_validation_seed)]
train_df = data_df[~data_df['seed'].apply(is_validation_seed)]
print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}")

# Create balanced validation set
valid_positives = valid_df[valid_df['outcome'] == 1]
valid_negatives = valid_df[valid_df['outcome'] == 0]
n_samples = min(len(valid_positives), len(valid_negatives))
balanced_valid_df = pd.concat([
    valid_positives.sample(n=n_samples, random_state=42),
    valid_negatives.sample(n=n_samples, random_state=42)
])
print(f"Balanced validation set length: {len(balanced_valid_df)}")
valid_df = balanced_valid_df
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
save_path = f"net.outcome.lr{T.initial_lr:.1e}-{T.final_lr:.1e}.wd{T.weight_decay:.1e}.e{T.num_epochs}.pt"

# %%
do_training = False # True

def train(batch, opt, device):
    """
    Perform one training step.
    Returns (loss, accuracy) tuple
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    output = process_batch(batch, net)
    
    chosen_logits = output['card_choice_winprob_logits'][
        torch.arange(len(batch['chosen_idx']), device=device),
        batch['chosen_idx']
    ]
    
    loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'])
    
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        accuracy = ((chosen_logits >= 0) == batch['outcome']).float().mean()
    
    return loss.item(), accuracy.item()

# %%
valid_loader = torch.utils.data.DataLoader(
    SlayDataset(valid_df),
    batch_size=T.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
)
    
if do_training:
    opt = torch.optim.AdamW(net.parameters(), lr=T.initial_lr, weight_decay=T.weight_decay)
    
    train_loader = torch.utils.data.DataLoader(
        SlayDataset(train_df),
        batch_size=T.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    total_steps = len(train_loader) * T.num_epochs
    
    for epoch in range(T.num_epochs):
        print(f"Epoch {epoch}")
        for i, batch in enumerate(train_loader):
            # Update learning rate
            current_step = epoch * len(train_loader) + i
            current_lr = T.initial_lr - (T.initial_lr - T.final_lr) * (current_step / total_steps)
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
            
            loss, acc = train(batch, opt, device)

            if i % T.log_every_n_steps == 0:
                print(f"{i}: loss={loss:.4f}, acc={acc:.3f}, lr={current_lr:.2e}")
            if i != 0 and i % T.validate_every_n_steps == 0 or i == len(train_loader) - 1:
                print(f"{i}: Validating")
                valid_losses = []
                valid_preds = []
                valid_targets = []
                with torch.no_grad():
                    for batch in valid_loader:
                        output = process_batch(batch, net)
                        chosen_logits = output['card_choice_winprob_logits'][
                            torch.arange(len(batch['chosen_idx']), device=device),
                            batch['chosen_idx']
                        ]
                        
                        loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'].to(device))
                        pred = chosen_logits >= 0
                        valid_losses.append(loss.item())
                        valid_targets.append(batch['outcome'].cpu().numpy())
                        valid_preds.append(pred.cpu().numpy())
                    print(f"Valid loss: {np.mean(valid_losses)}")
                    acc = np.mean(np.concatenate(valid_preds) == np.concatenate(valid_targets))
                    print(f"Valid acc: {acc}")

    torch.save(net.state_dict(), save_path)

# %%
state = torch.load(save_path, weights_only=True, map_location=device)
# %%
net.load_state_dict(state)

# %%
batch = valid_df.sample(128)
# %%
len(batch.iloc[0]['obs.deck.cards'])

# %%
batch_data = collate_fn([SlayDataset(batch).__getitem__(i) for i in range(len(batch))])
output = process_batch(batch_data, net)

# Get win probabilities for all cards and mark chosen ones
all_probs = torch.sigmoid(output['card_choice_winprob_logits'])
chosen_indices = batch_data['chosen_idx']
batch_indices = torch.arange(len(batch), device=device)

# Print probabilities for each example in batch
for i in range(min(20, len(batch))):
    probs = all_probs[i]
    chosen_idx = chosen_indices[i]
    cards_offered = batch.iloc[i]['cards_offered.cards']
    upgrades = batch.iloc[i]['cards_offered.upgrades']
    
    prob_strs = []
    for j, (card_id, upgrade) in enumerate(zip(cards_offered, upgrades)):
        if probs[j] == float('-inf'):  # Skip masked values
            continue
        card = sts.Card(sts.CardId(card_id), upgrade)
        prob_str = f"{card}({probs[j]:.3f})"
        if j == chosen_idx:
            prob_str = f"[{prob_str}]"  # Mark chosen probability with brackets
        prob_strs.append(prob_str)
    print(f"Example {i}: {', '.join(prob_strs)}")
    print(batch.iloc[i])
    print()
    

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot([len(d) for d in batch['obs.deck.cards']], torch.sigmoid(output['card_choice_winprob_logits']).mean(dim=1).numpy(force=True), 'o')
plt.xlabel('Deck size')
plt.ylabel('Win probability')
plt.title('Deck size vs predicted win probability')
plt.grid(True)
plt.show()
# %%


# %%
from sklearn.metrics import roc_curve, auc

batch_size = 512

valid_preds = []
valid_targets = []
card_predictions = {}  # Dictionary to store predictions for each card

# Get predictions for validation set
with torch.no_grad():
    for batch in valid_loader:
        output = process_batch(batch, net)
        
        # Get chosen card predictions for ROC curve
        chosen_logits = output['card_choice_winprob_logits'][
            torch.arange(len(batch['chosen_idx']), device=net.device),
            batch['chosen_idx'].to(net.device)
        ]
        probs = torch.sigmoid(chosen_logits).cpu().numpy()
        valid_preds.extend(probs)
        valid_targets.extend(batch['outcome'].numpy())
        
        # Get predictions for all cards
        all_probs = torch.sigmoid(output['card_choice_winprob_logits'])
        
        # For each example in batch
        for i in range(len(batch['choices'])):
            choices = batch['choices'][i]
            upgrades = batch['choice_upgrades'][i]
            
            # Get valid probabilities for this choice
            valid_mask = choices != sts.CardId.INVALID.value
            choice_probs = all_probs[i, valid_mask].cpu().numpy()
            
            # For each valid card choice
            for j, (card_id, upgrade) in enumerate(zip(choices[valid_mask], upgrades[valid_mask])):
                card = sts.Card(sts.CardId(card_id), upgrade)
                card_key = str(card)
                
                # Calculate relative win probability (compared to alternatives)
                other_probs = np.concatenate([choice_probs[:j], choice_probs[j+1:]])
                relative_prob = choice_probs[j] - np.mean(other_probs) if len(other_probs) > 0 else 0.0
                
                if card_key not in card_predictions:
                    card_predictions[card_key] = []
                card_predictions[card_key].append(relative_prob)

# Calculate card statistics
card_stats = {}
for card_key, preds in card_predictions.items():
    preds = np.array(preds)
    card_stats[card_key] = {
        'mean': np.mean(preds),  # Now represents average advantage over alternatives
        'std': np.std(preds),
        'count': len(preds)
    }

# Sort and print card statistics
sorted_cards = sorted(card_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\nCard Win Probability Statistics (relative to alternatives):")
print(f"{'Card':<30} {'Advantage':>8} {'Std':>8} {'Count':>8}")
print("-" * 56)
for card_key, stats in sorted_cards:
    print(f"{card_key:<30} {stats['mean']:8.3f} {stats['std']:8.3f} {stats['count']:8d}")

# Plot top cards
N = 20
plt.figure(figsize=(15, 8))
means = [stats['mean'] for _, stats in sorted_cards[:N]]
cards = [card for card, _ in sorted_cards[:N]]
plt.bar(range(N), means)
plt.xticks(range(N), cards, rotation=45, ha='right')
plt.ylabel('Mean Win Probability')
plt.title('Top Cards by Predicted Win Probability')
plt.tight_layout()
plt.show()

# Find optimal threshold and print confusion matrix
thresholds = np.arange(0, 1, 0.01)
accuracies = []
for threshold in thresholds:
    predictions = (valid_preds >= threshold).astype(int)
    accuracy = np.mean(predictions == valid_targets)
    accuracies.append(accuracy)

optimal_idx = np.argmax(accuracies)
optimal_threshold = thresholds[optimal_idx]
best_accuracy = accuracies[optimal_idx]

print(f'\nOptimal threshold: {optimal_threshold:.2f}')
print(f'Best accuracy: {best_accuracy:.3f}')

# Plot accuracies vs thresholds
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies)
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal threshold = {optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Classification Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Print confusion matrix
predictions = (valid_preds >= optimal_threshold).astype(int)
tp = np.sum((predictions == 1) & (valid_targets == 1))
tn = np.sum((predictions == 0) & (valid_targets == 0))
fp = np.sum((predictions == 1) & (valid_targets == 0))
fn = np.sum((predictions == 0) & (valid_targets == 1))

print("\nConfusion Matrix:")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Accuracy: {(tp + tn)/(tp + tn + fp + fn):.3f}")

# %%
print(f"Saved to {save_path}")

# %%
