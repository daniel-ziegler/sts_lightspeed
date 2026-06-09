"""Backfill per-ascension heart-kill rates into a run's stats jsonl from its episode parquets.

Each episode parquet (iter_N.parquet) holds per-decision rows; one game = one seed. Ascension
(fixed_observation[6]) and outcome are constant within a seed; key count
(fixed_observation[7..9]) is monotone, so its per-game max is the keys-at-end. A heart kill is
a WIN holding all 3 keys: 3 keys force the act-4 transition, where the only victory is the
Corrupt Heart -- so won AND max-keys==3 is exact and ascension-proof. (A floor threshold is
NOT: A20's double act-3 boss bumps an act-3-only win to floor 52, which a floor>51 test
miscounts as a heart kill.) We add heart_win_rate_asc{a} (and, for older stats lacking them,
win_rate_asc{a}/num_games_asc{a}) to each matching iteration row. iter_N.parquet corresponds to
stats `iteration` N.
"""
import argparse
import glob
import json
import os

import pandas as pd

ASC_OBS_IDX = 6
KEY_OBS_IDXS = (7, 8, 9)


def per_iter_asc_stats(ep_path: str) -> dict:
    df = pd.read_parquet(ep_path, columns=['seed', 'outcome', 'obs.fixed_observation'])
    fo = df['obs.fixed_observation']
    df = df.assign(
        asc=fo.apply(lambda x: int(x[ASC_OBS_IDX])),
        keys=fo.apply(lambda x: sum(int(x[i]) for i in KEY_OBS_IDXS)),
    )
    asc = df.groupby('seed')['asc'].first()
    won = df.groupby('seed')['outcome'].first() == 1
    max_keys = df.groupby('seed')['keys'].max()
    heart = won & (max_keys == 3)
    out = {}
    for a, sub in asc.groupby(asc):
        idx = sub.index
        n = len(idx)
        out[f'num_games_asc{a}'] = n
        out[f'win_rate_asc{a}'] = float(won.loc[idx].mean())
        out[f'heart_win_rate_asc{a}'] = float(heart.loc[idx].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes-dir', required=True, help='e.g. runs/heart1.pt.episodes')
    ap.add_argument('--stats', required=True, help='e.g. runs/heart1.pt.stats.jsonl')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    by_iter = {}
    for ep in glob.glob(os.path.join(args.episodes_dir, 'iter_*.parquet')):
        it = int(os.path.basename(ep)[len('iter_'):-len('.parquet')])
        try:
            by_iter[it] = per_iter_asc_stats(ep)
        except Exception as e:
            print(f"  skip iter {it}: {e}")
    print(f"computed per-ascension stats for {len(by_iter)} iterations")

    rows = []
    patched = 0
    with open(args.stats) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            extra = by_iter.get(d.get('iteration'))
            if extra:
                # Overwrite the per-ascension fields from the episodes (keys==3 is the exact,
                # ascension-proof heart-kill signal -- supersedes any earlier floor-based
                # backfill; matches the trainer's native act>=4 logging on rows that have it).
                d.update(extra)
                patched += 1
            rows.append(d)
    print(f"patched {patched} stats rows")

    if args.dry_run:
        sample = next((r for r in rows if any(k.startswith('heart_win_rate_asc') for k in r)), None)
        if sample:
            hk = {k: round(v, 3) for k, v in sample.items() if k.startswith('heart_win_rate_asc')}
            print(f"sample iter {sample['iteration']}: {hk}")
        return

    # Append-safe against a live trainer: re-read now and carry over any iteration rows that
    # appeared since we first read (they'd otherwise be lost in the atomic replace). Newer rows
    # already carry the field natively, so we keep them verbatim.
    seen = {r.get('iteration') for r in rows}
    with open(args.stats) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get('iteration') not in seen:
                rows.append(d)
                seen.add(d.get('iteration'))
    rows.sort(key=lambda r: (r.get('iteration') is None, r.get('iteration')))

    tmp = args.stats + '.tmp'
    with open(tmp, 'w') as f:
        for d in rows:
            f.write(json.dumps(d) + '\n')
    os.replace(tmp, args.stats)
    print(f"wrote {args.stats} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
