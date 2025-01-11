# %%
import sys
import random
from enum import IntEnum, auto
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

import slaythespire as sts
from randomplayouts import ActionType
from slaythespire import getFixedObservationMaximums

# %%
torch.set_float32_matmul_precision('high')

# %%

@dataclass
class ModelHP:
    dim: int = 256
    ffn_dim_mult: int = 4
    n_layers: int = 4
    n_heads: int = 8
    norm_eps: float = 1e-5
    n_fixed_obs: int = len(getFixedObservationMaximums())

class InputType(IntEnum):
    Card = 0
    Relic = auto()
    Potion = auto()
    Choice = auto()

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, n_features: int):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(1000.0)) / (half_dim - 1)
        self.register_buffer('inv_freq', torch.exp(torch.arange(half_dim) * -emb))
        self.out_dim = dim * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for numerical values.
        x: tensor of integers to embed [batch_size, n_features]
        Returns: [batch_size, n_features * dim] tensor
        """
        # [batch_size, n_features, half_dim]
        emb = x.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        
        # [batch_size, n_features, dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # [batch_size, n_features * dim]
        return emb.reshape(x.shape[0], -1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float):
        super().__init__()
        self.eps = eps
        self.w = nn.parameter.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        normed = self._norm(x.float()).type_as(x)
        return normed * self.w

class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x2 = F.silu(self.w1(x)) * self.v(x)
        return self.w2(x2)

class TransformerBlock(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H
        self.norm1 = RMSNorm(H.dim, eps=H.norm_eps)
        self.attn = nn.MultiheadAttention(H.dim, H.n_heads, batch_first=True)
        self.norm2 = RMSNorm(H.dim, eps=H.norm_eps)
        self.ffn = FFN(H.dim, H.dim * H.ffn_dim_mult)

    def forward(self, x, mask):
        xn = self.norm1(x)
        xatt, _ = self.attn(xn, xn, xn, key_padding_mask=mask)
        x1 = x + xatt
        x2 = x1 + self.ffn(self.norm2(x1))
        return x2

class NN(nn.Module):
    def __init__(self, H: ModelHP):
        super().__init__()
        self.H = H

        self.input_type_embed = nn.Embedding(len(InputType), H.dim)
        self.card_embed = nn.Embedding(len(sts.CardId), H.dim, padding_idx=sts.CardId.INVALID.value)
        
        # Add sinusoidal embedding and projection
        n_fixed_obs = len(getFixedObservationMaximums())
        self.fixed_obs_embed = SinusoidalEmbedding(H.dim, n_fixed_obs)
        self.fixed_obs_proj = nn.Linear(self.fixed_obs_embed.out_dim, H.dim)

        self.layers = nn.ModuleList([TransformerBlock(H=H) for _ in range(H.n_layers)])

        self.norm = RMSNorm(H.dim, H.norm_eps)
        self.card_winprob = nn.Linear(H.dim, 1, bias=True)
        nn.init.uniform_(self.card_winprob.weight, -0.01, 0.01)
        nn.init.zeros_(self.card_winprob.bias)

    def forward(self, deck, card_choices, fixed_obs):
        max_deck_len = deck.size(1)
        max_choices_len = card_choices.size(1)

        # Create sinusoidal embeddings for all fixed observations at once
        fixed_obs_x = self.fixed_obs_proj(self.fixed_obs_embed(fixed_obs))
        
        cards = torch.cat((deck, card_choices), dim=1)
        mask = cards == sts.CardId.INVALID.value
        card_x = self.card_embed(cards) + self.input_type_embed(torch.tensor([int(InputType.Card)], device=device))
        card_x[:, max_deck_len:, :] += self.input_type_embed(torch.tensor([int(InputType.Choice)], device=device))
        
        x = torch.cat([
            fixed_obs_x.unsqueeze(1),
            card_x
        ], dim=1)
        
        full_mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=mask.dtype),
            mask
        ], dim=1)

        for l in self.layers:
            x = l(x, full_mask)
        xn = self.norm(x)
        
        card_x = xn[:, 1+max_deck_len:, :]
        card_choice_winprob_logits = self.card_winprob(card_x).squeeze(-1).float()
        card_choice_winprob_logits = card_choice_winprob_logits.masked_fill(mask[:, max_deck_len:], float('-inf'))
        
        return dict(
            card_choice_winprob_logits=card_choice_winprob_logits,
        )

def pad_sequence(sequences, max_len, device):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s, device=device) for s in sequences],
        batch_first=True,
        padding_value=sts.CardId.INVALID.value
    )[:, :max_len]

# %%
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %%
H = ModelHP()

net = NN(H)
net = net.to(device)
net.fixed_obs_embed = torch.compile(net.fixed_obs_embed)
net.fixed_obs_proj = torch.compile(net.fixed_obs_proj)
for i, l in enumerate(net.layers):
    net.layers[i] = torch.compile(l)

# %%
print(net.fixed_obs_embed.inv_freq)
# %%
# df = pd.read_parquet("rollouts0_100000.parquet")
df = pd.read_parquet("rollouts0_6000.parquet")

# %%
data_df: pd.DataFrame = df[df.apply(lambda r: r["chosen_idx"] < len(r["cards_offered.cards"]), axis=1)]
assert (data_df["chosen_type"] == ActionType.CARD).all()
len(data_df)

# %%
# Split into train and validation sets, ensuring rows with the same seed stay together
valid_seeds = data_df['seed'].drop_duplicates().sample(frac=0.1, random_state=42)
valid_df = data_df[data_df['seed'].isin(valid_seeds)]
train_df = data_df[~data_df['seed'].isin(valid_seeds)]
print(len(train_df), len(valid_df))

# %%
# Create balanced validation set
valid_positives = valid_df[valid_df['outcome'] == 1]
valid_negatives = valid_df[valid_df['outcome'] == 0]
n_samples = min(len(valid_positives), len(valid_negatives))
balanced_valid_df = pd.concat([
    valid_positives.sample(n=n_samples, random_state=42),
    valid_negatives.sample(n=n_samples, random_state=42)
])
print(f"Balanced validation set length: {len(balanced_valid_df)}")
print(f"Balanced validation set win rate: {balanced_valid_df['outcome'].mean():.2f}")
valid_df = balanced_valid_df
# %%
np.random.seed(3)


# %%
def feed_to_net(df):
    max_deck_len = max(len(d) for d in batch["obs.deck.cards"])
    max_choices_len = max(len(c) for c in batch["cards_offered.cards"])

    deck = pad_sequence(batch["obs.deck.cards"].apply(lambda x: x.astype(np.int32)), max_deck_len, device=device)
    choices = pad_sequence(batch["cards_offered.cards"].apply(lambda x: x.astype(np.int32)), max_choices_len, device=device)
    fixed_obs = torch.tensor(np.stack(batch["obs.fixed_observation"].to_numpy()), device=device)

    return net(deck, choices, fixed_obs)

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
batch_size = 512
# %%
lr = 3e-5
weight_decay = 5e-4

# %%
save_path = f"net.outcome.lr{lr:.1e}.wd{weight_decay:.1e}.pt"

# %%
do_training = True
# %%
if do_training:
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(2):
        train_df_shuffled = train_df.sample(frac=1)
        print(f"Epoch {epoch}")
        for i in range(len(train_df_shuffled) // batch_size):
            batch = train_df_shuffled.iloc[i*batch_size:(i+1)*batch_size]
            output = feed_to_net(batch)
            
            chosen_indices = torch.tensor(batch['chosen_idx'].to_numpy(), device=device)
            batch_indices = torch.arange(len(batch), device=device)
            chosen_logits = output['card_choice_winprob_logits'][batch_indices, chosen_indices]
            
            targets = torch.tensor(batch['outcome'].to_numpy(), device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(chosen_logits, targets)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 20 == 0:
                print(f"{i}: {loss.item():.4f}")
            if i % 1000 == 0 or i == len(train_df_shuffled) // batch_size - 1:
                torch.save(net.state_dict(), f"net.{i}.pt")
                print(f"{i}: Validating")
                valid_losses = []
                valid_preds = []
                valid_targets = []
                with torch.no_grad():
                    for j in range(len(valid_df) // batch_size):
                        batch = valid_df.iloc[j*batch_size:(j+1)*batch_size]
                        output = feed_to_net(batch)
                        chosen_indices = torch.tensor(batch['chosen_idx'].to_numpy(), device=device)
                        batch_indices = torch.arange(len(batch), device=device)
                        chosen_logits = output['card_choice_winprob_logits'][batch_indices, chosen_indices]
                        
                        targets = torch.tensor(batch['outcome'].to_numpy(), device=device)
                        loss = F.binary_cross_entropy_with_logits(chosen_logits, targets.float())
                        pred = chosen_logits >= 0
                        valid_losses.append(loss.item())
                        valid_targets.append(targets.numpy(force=True))
                        valid_preds.append(pred.numpy(force=True))
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
output = feed_to_net(batch)
# Get win probabilities for all cards and mark chosen ones
all_probs = torch.sigmoid(output['card_choice_winprob_logits'])
chosen_indices = torch.tensor(batch['chosen_idx'].to_numpy(), device=device)
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

batch_size = 256

valid_preds = []
valid_targets = []

# Get predictions for validation set
with torch.no_grad():
    for i in range(0, len(valid_df), batch_size):
        batch = valid_df.iloc[i:i+batch_size]
        output = feed_to_net(batch)
        
        # Get probabilities for chosen cards
        chosen_indices = torch.tensor(batch['chosen_idx'].to_numpy(), device=device)
        batch_indices = torch.arange(len(batch), device=device)
        probs = torch.sigmoid(output['card_choice_winprob_logits'][batch_indices, chosen_indices]).cpu().numpy()
        
        valid_preds.extend(probs)
        valid_targets.extend(batch['outcome'].to_numpy())

valid_preds = np.array(valid_preds)
valid_targets = np.array(valid_targets)

# Compute ROC curve
fpr, tpr, _ = roc_curve(valid_targets, valid_preds)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%
# Find optimal threshold based on accuracy
thresholds = np.arange(0, 1, 0.01)
accuracies = []
for threshold in thresholds:
    predictions = (valid_preds >= threshold).astype(int)
    accuracy = np.mean(predictions == valid_targets)
    accuracies.append(accuracy)

optimal_idx = np.argmax(accuracies)
optimal_threshold = thresholds[optimal_idx]
best_accuracy = accuracies[optimal_idx]

print(f'Optimal threshold: {optimal_threshold:.2f}')
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

# Print confusion matrix at optimal threshold
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

# Simple linear regression baseline using fixed observations
# print("\nLogistic Regression Baseline:")
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
# 
# # Prepare data
# train_X = np.stack(train_df["obs.fixed_observation"].to_numpy())
# train_y = train_df["outcome"].to_numpy()
# valid_X = np.stack(valid_df["obs.fixed_observation"].to_numpy())
# valid_y = valid_df["outcome"].to_numpy()
# 
# # Train logistic regression
# lr_model = LogisticRegression(random_state=42)
# lr_model.fit(train_X, train_y)
# 
# # Make predictions
# lr_preds = lr_model.predict_proba(valid_X)[:, 1]
# lr_binary_preds = lr_preds >= 0.5
# 
# # Calculate metrics
# lr_accuracy = accuracy_score(valid_y, lr_binary_preds)
# lr_conf_matrix = confusion_matrix(valid_y, lr_binary_preds)
# 
# print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
# print("\nConfusion Matrix:")
# print(f"True Positives: {lr_conf_matrix[1,1]}")
# print(f"True Negatives: {lr_conf_matrix[0,0]}")
# print(f"False Positives: {lr_conf_matrix[0,1]}")
# print(f"False Negatives: {lr_conf_matrix[1,0]}")
# 
# # Print feature importance
# fixed_obs_maxvals = getFixedObservationMaximums()
# importance = list(zip(range(len(fixed_obs_maxvals)), lr_model.coef_[0]))
# importance.sort(key=lambda x: abs(x[1]), reverse=True)
# print("\nFeature Importance (absolute coefficient value):")
# for idx, coef in importance[:10]:  # Top 10 most important features
#     print(f"Fixed observation {idx} (max {fixed_obs_maxvals[idx]}): {coef:.3f}")
# 
# %%
