"""Golden no-op check for the ascension-input change.

Run BEFORE the change to dump reference logits/values (--dump golden.pt), and AFTER
to compare (--check golden.pt). The warm-started net (zero-init ascension embedding)
must produce identical outputs at ascension 0.
"""
import argparse

import numpy as np
import torch

import slaythespire as sts
from network import NN, ModelHP, collate_fn, process_batch, load_network_backward_compatible
from playouts import construct_choice, flatten_dict

CKPT = 'runs/honest1.pt.iter_155'


def first_decisions(seed: int, n_decisions: int = 3):
    """Yield Choice objects for the first multi-option non-battle decisions of a game."""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    agent = sts.Agent()
    agent.verbosity_level = 0
    out = []
    while gc.outcome == sts.GameOutcome.UNDECIDED and len(out) < n_decisions:
        if gc.screen_state == sts.ScreenState.BATTLE:
            break
        actions = sts.GameAction.getAllActionsInState(gc)
        obs = sts.getNNRepresentation(gc)
        choice = construct_choice(gc, obs, actions)
        if choice is not None:
            total = (len(choice.cards_offered) + len(choice.relics_offered)
                     + len(choice.potions_offered) + len(choice.fixed_actions)
                     + len(choice.paths_offered))
            if total > 1:
                out.append(choice)
                # deterministic walk: always take the first listed action
                acts = (choice.card_actions + choice.path_actions + choice.relic_actions
                        + choice.potion_actions + choice.fixed_actions_list)
                acts[0].execute(gc)
                continue
        action = agent.pick_gameaction(gc)
        action.execute(gc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=str)
    ap.add_argument('--check', type=str)
    args = ap.parse_args()

    torch.manual_seed(0)
    hp = ModelHP(use_value_head=True)
    net = NN(hp)
    state = torch.load(CKPT, map_location='cpu', weights_only=True)
    net = load_network_backward_compatible(net, state)
    net.eval()

    results = {}
    for seed in (3, 17, 42, 1234, 99999):
        for di, choice in enumerate(first_decisions(seed)):
            batch = [{**flatten_dict(choice.as_dict()), 'chosen_idx': 0, 'outcome': 0.0}]
            tensors = collate_fn(batch)
            with torch.no_grad():
                out = process_batch(tensors, net)
            logits, value = out if isinstance(out, tuple) else (out, None)
            key = f'{seed}/{di}'
            results[f'{key}/logits'] = torch.as_tensor(logits).flatten().clone()
            if value is not None:
                results[f'{key}/value'] = torch.as_tensor(value).flatten().clone()

    if args.dump:
        torch.save(results, args.dump)
        print(f"dumped {len(results)} tensors to {args.dump}")
    if args.check:
        ref = torch.load(args.check, weights_only=True)
        assert set(ref) == set(results), f"key mismatch: {set(ref) ^ set(results)}"
        worst = 0.0
        for k in ref:
            d = (ref[k] - results[k]).abs().max().item()
            worst = max(worst, d)
            if d > 1e-5:
                print(f"MISMATCH {k}: max abs diff {d}")
        print(f"checked {len(ref)} tensors, worst abs diff {worst:.3e}")
        assert worst <= 1e-5, "golden check FAILED"
        print("golden check PASSED")


if __name__ == '__main__':
    main()
