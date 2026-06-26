import json, sys, math
from collections import Counter
rows=[json.loads(l) for l in open('runs/eval_a20h_2575_greedy.jsonl') if l.strip()]
rows=[r for r in rows if r['won']!=-1]
n=len(rows)
hk=[r for r in rows if r['won']==1 and r['keys']==3]
def ci(p,n): return 1.96*math.sqrt(p*(1-p)/max(1,n))
hkr=len(hk)/max(1,n)
print(f"=== A20H GREEDY (temp 0), iter_2575, 10K sims ===")
print(f"games: {n} | heart-kill: {len(hk)}/{n} = {hkr:.3f} ± {ci(hkr,n):.3f}")
won=sum(r['won'] for r in rows); print(f"win rate: {won}/{n} = {won/n:.3f} ± {ci(won/n,n):.3f}")
print(f"avg floor: {sum(r['floor'] for r in rows)/n:.1f} | avg keys: {sum(r['keys'] for r in rows)/n:.2f}")
print("\nKey-count distribution:")
kc=Counter(r['keys'] for r in rows)
for k in range(4): print(f"  {k} keys: {kc[k]:4d} ({kc[k]/n:.1%})")
print("\nPER-KEY hold rate (which keys it gets):")
for name,fld in [('ruby (rest)','red_key'),('emerald (elite)','green_key'),('sapphire (chest)','blue_key')]:
    h=sum(1 for r in rows if r[fld]); print(f"  {name:18s}: {h}/{n} = {h/n:.1%}")
print("\nAmong games that MISSED >=1 key (keys<3): which specific key is missing")
miss=[r for r in rows if r['keys']<3]
for name,fld in [('ruby','red_key'),('emerald','green_key'),('sapphire','blue_key')]:
    m=sum(1 for r in miss if not r[fld]); print(f"  missing {name:10s}: {m}/{len(miss)} = {m/max(1,len(miss)):.1%}")
print("\nFailure modes (non-heart-kill), by key count + act reached:")
nf=[r for r in rows if not (r['won']==1 and r['keys']==3)]
def act(f): return 1+(f>=17)+(f>=34)+(f>=51)
for k in range(4):
    g=[r for r in nf if r['keys']==k]; c=Counter(act(r['floor']) for r in g)
    if g: print(f"  {k} keys ({len(g)}): act1={c[1]} act2={c[2]} act3={c[3]} act4={c[4]}")
