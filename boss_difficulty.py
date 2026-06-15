"""Boss difficulty for the heart1 policy+MCTS, read off the learned battle-outcome head.

For every boss-encounter row in the battle dataset (states the policy reached at an act end,
plus alt-encounter bosses forced on same-act states), run the pretrained battle head to predict
the ΔHP-bucket distribution, and derive P(death) and E[ΔHP | survive]. Report per boss, with the
EMPIRICAL outcomes (the actual 32-reroll playout results in the same rows) alongside as ground
truth. Difficulty = predicted/empirical death rate; HP loss among survivors is the secondary axis.
"""
import argparse, glob
import numpy as np, pandas as pd, torch
import slaythespire as sts
from battle_buckets import NUM_BUCKETS, DEATH, bucket_midpoint_frac
from network import NN, ModelHP, BattleOutcomeHead, collate_fn, move_to_device, load_network_backward_compatible

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_E = sts.MonsterEncounter
BOSS_ACT = {_E.THE_GUARDIAN: 1, _E.HEXAGHOST: 1, _E.SLIME_BOSS: 1,
            _E.AUTOMATON: 2, _E.COLLECTOR: 2, _E.CHAMP: 2,
            _E.AWAKENED_ONE: 3, _E.TIME_EATER: 3, _E.DONU_AND_DECA: 3,
            _E.SHIELD_AND_SPEAR: 4, _E.THE_HEART: 4}
DUMMY = {'cards_offered.cards': [], 'cards_offered.upgrades': [], 'relics_offered': [],
         'potions_offered': [], 'paths_offered': [], 'choice_type': 0, 'chosen_idx': 0,
         'outcome': 0, 'return': 0.0,
         'fixed_actions': [{'action': 0, 'gold': 0, 'card': 0, 'relic': 0, 'info': 0}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--battle-dir', default='battle_data/val')
    ap.add_argument('--ckpt', default='pre_b20.pt')
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{args.battle_dir}/*.parquet')],
                   ignore_index=True)
    boss = df[df.encounter_kind == 'boss'].reset_index(drop=True)
    for k, v in DUMMY.items():
        boss[k] = [v] * len(boss) if isinstance(v, (list, dict)) else v
    print(f"{len(boss):,} boss-encounter rows ({boss.state_id.nunique()} states) over A0-20\n")

    net = NN(ModelHP(use_value_head=True)).to(DEVICE)
    head = BattleOutcomeHead(256, NUM_BUCKETS).to(DEVICE)
    st = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    net = load_network_backward_compatible(net, st['net']); head.load_state_dict(st['head'])
    net.eval(); head.eval()

    midpts = np.array([bucket_midpoint_frac(b) for b in range(NUM_BUCKETS)])
    rows = boss.to_dict('records')
    pdeath, psurv = [], []
    for i in range(0, len(rows), 256):
        chunk = rows[i:i + 256]
        b = move_to_device(collate_fn(chunk), DEVICE)
        enc = torch.tensor([int(r['encounter']) for r in chunk], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            pooled = net(b, return_pooled=True)[2]
            probs = torch.softmax(head(pooled, enc), -1).cpu().numpy()
        pdeath.extend(probs[:, DEATH])
        surv = probs.copy(); surv[:, DEATH] = 0; surv /= surv.sum(1, keepdims=True)
        psurv.extend((surv * midpts).sum(1))
    boss['pred_death'] = pdeath
    boss['pred_surv_frac'] = psurv

    print(f"{'boss':16s} {'act':>3} {'n':>5} | {'death% pred/emp':>16} | "
          f"{'A0-9':>9} {'A10-20':>9} (emp death) | {'ΔHP|surv pred/emp':>18}")
    print('-' * 92)
    recs = []
    for e, sub in boss.groupby('encounter'):
        name = str(_E(e)).replace('MonsterEncounter.', '')
        lo = sub[sub.ascension <= 9]; hi = sub[sub.ascension >= 10]
        surv = sub[~sub.died]
        recs.append(dict(
            name=name, act=BOSS_ACT.get(_E(e), 0), n=len(sub),
            pd=sub.pred_death.mean(), ed=sub.died.mean(),
            lo=lo.died.mean() if len(lo) else np.nan, hi=hi.died.mean() if len(hi) else np.nan,
            ps=sub.pred_surv_frac.mean(),
            es=surv.hp_frac_delta.mean() if len(surv) else np.nan))
    for r in sorted(recs, key=lambda r: -r['ed']):
        print(f"{r['name']:16s} {r['act']:>3} {r['n']:>5} | "
              f"{r['pd']*100:6.1f} /{r['ed']*100:6.1f} | "
              f"{r['lo']*100:8.1f}% {r['hi']*100:8.1f}% | "
              f"{r['ps']*100:+7.1f}% /{r['es']*100:+7.1f}%")


if __name__ == '__main__':
    main()
