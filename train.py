#!/usr/bin/env python3
import sys
import random
import argparse
from dataclasses import dataclass
from pathlib import Path

from network import MAX_CHOICES, NN, ModelHP, SlayDataset, collate_fn, process_batch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import slaythespire as sts
from randomplayouts import ActionType

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

def is_validation_seed(seed: int, valid_fraction: float = 0.1) -> bool:
    """Deterministically decide if a seed should be in validation set"""
    # Use a simple hash function to get a value between 0 and 1
    hash_val = ((seed * 1327217885) & 0xFFFFFFFF) / 0xFFFFFFFF
    return hash_val < valid_fraction

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and prepare data for training"""
    data_df: pd.DataFrame = df[
        df.apply(lambda r: (
            r["chosen_idx"] < len(r["cards_offered.cards"]) and  # Original check
            r["chosen_idx"] < MAX_CHOICES  # Use constant
        ), axis=1)
    ]
    assert (data_df["choice_type"] == ActionType.CARD).all()
    print(f"Filtered out {len(df) - len(data_df)} rows with chosen_idx >= {MAX_CHOICES}")
    print(f"Remaining rows: {len(data_df)}")
    return data_df

def split_data(data_df: pd.DataFrame, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets"""
    # Split data
    valid_df = data_df[data_df['seed'].apply(lambda s: is_validation_seed(s, valid_fraction))]
    train_df = data_df[~data_df['seed'].apply(lambda s: is_validation_seed(s, valid_fraction))]
    print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}")

    # Create balanced validation set if possible
    valid_positives = valid_df[valid_df['outcome'] == 1]
    valid_negatives = valid_df[valid_df['outcome'] == 0]
    
    if len(valid_positives) == 0 or len(valid_negatives) == 0:
        print("Warning: No positive or negative examples in validation set, using full validation set")
        return train_df, valid_df
    
    n_samples = min(len(valid_positives), len(valid_negatives))
    balanced_valid_df = pd.concat([
        valid_positives.sample(n=n_samples, random_state=42),
        valid_negatives.sample(n=n_samples, random_state=42)
    ])
    print(f"Balanced validation set length: {len(balanced_valid_df)}")
    return train_df, balanced_valid_df

def train_step(batch, net, opt, device):
    """Perform one training step"""
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

def validate(net, valid_loader, device):
    """Run validation and return loss and accuracy"""
    valid_losses = []
    valid_preds = []
    valid_targets = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = process_batch(batch, net)
            chosen_logits = output['card_choice_winprob_logits'][
                torch.arange(len(batch['chosen_idx']), device=device),
                batch['chosen_idx']
            ]
            
            loss = F.binary_cross_entropy_with_logits(chosen_logits, batch['outcome'])
            pred = chosen_logits >= 0
            valid_losses.append(loss.item())
            valid_targets.append(batch['outcome'].cpu().numpy())
            valid_preds.append(pred.cpu().numpy())
    
    avg_loss = np.mean(valid_losses)
    accuracy = np.mean(np.concatenate(valid_preds) == np.concatenate(valid_targets))
    return avg_loss, accuracy

def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, hp: TrainingHP, device: torch.device, output_path: str):
    """Train the network"""
    net = NN(ModelHP())
    net = net.to(device)
    net = torch.compile(net, mode="reduce-overhead")

    opt = torch.optim.AdamW(net.parameters(), lr=hp.initial_lr, weight_decay=hp.weight_decay)
    
    train_loader = torch.utils.data.DataLoader(
        SlayDataset(train_df),
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        SlayDataset(valid_df),
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    total_steps = len(train_loader) * hp.num_epochs
    
    for epoch in range(hp.num_epochs):
        print(f"Epoch {epoch}")
        for i, batch in enumerate(train_loader):
            # Update learning rate
            current_step = epoch * len(train_loader) + i
            current_lr = hp.initial_lr - (hp.initial_lr - hp.final_lr) * (current_step / total_steps)
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
            
            loss, acc = train_step(batch, net, opt, device)

            if i % hp.log_every_n_steps == 0:
                print(f"{i}: loss={loss:.4f}, acc={acc:.3f}, lr={current_lr:.2e}")
            if i != 0 and i % hp.validate_every_n_steps == 0 or i == len(train_loader) - 1:
                print(f"{i}: Validating")
                valid_loss, valid_acc = validate(net, valid_loader, device)
                print(f"Valid loss: {valid_loss:.4f}")
                print(f"Valid acc: {valid_acc:.3f}")

    torch.save(net.state_dict(), output_path)
    print(f"Saved model to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on Slay the Spire gameplay data')
    parser.add_argument('training_files', nargs='+', type=str,
                      help='Parquet files containing training data')
    parser.add_argument('--output', type=str, default=None,
                      help='Output path for trained model (default: net.outcome.pt)')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--initial-lr', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--final-lr', type=float, default=1e-6,
                      help='Final learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay for AdamW optimizer')
    parser.add_argument('--epochs', type=int, default=2,
                      help='Number of epochs to train')
    parser.add_argument('--validation-fraction', type=float, default=0.1,
                      help='Fraction of data to use for validation')
    parser.add_argument('--validate-every', type=int, default=2000,
                      help='Validate every N steps')
    parser.add_argument('--log-every', type=int, default=20,
                      help='Log every N steps')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up hyperparameters
    hp = TrainingHP(
        batch_size=args.batch_size,
        initial_lr=args.initial_lr,
        final_lr=args.final_lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        validation_fraction=args.validation_fraction,
        validate_every_n_steps=args.validate_every,
        log_every_n_steps=args.log_every,
    )

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    # Load and combine data
    dfs = []
    for file in args.training_files:
        print(f"Loading {file}...")
        df = pd.read_parquet(file)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Prepare data
    data_df = prepare_data(df)
    train_df, valid_df = split_data(data_df, hp.validation_fraction)

    # Set up output path
    if args.output is None:
        args.output = f"net.outcome.lr{hp.initial_lr:.1e}-{hp.final_lr:.1e}.wd{hp.weight_decay:.1e}.e{hp.num_epochs}.pt"

    # Train
    train(train_df, valid_df, hp, device, args.output)

if __name__ == "__main__":
    main()