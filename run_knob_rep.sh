#!/bin/bash
# Replication block (fresh seeds): control vs knob-B (expl 18.5) vs expl-25 probe.
set -u
cd ~/osrc/sts_lightspeed.boss-eval
CKPT=runs/heroe2.pt.iter_270; N=1000; SS=6001000; SIMS=1000; W=8
LINK=slaythespire.cpython-310-x86_64-linux-gnu.so
ln -sf engines/sts_new3.so $LINK
echo "=== REP CONTROL $(date) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m lightspeed.eval_hero --ckpt $CKPT --n-games $N --seed-start $SS \
  --mcts-sims $SIMS --num-workers $W --battle-timeout 120 --out-csv runs/rep_ctl.csv
echo "=== REP KNOB B $(date) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m lightspeed.eval_hero --ckpt $CKPT --n-games $N --seed-start $SS \
  --mcts-sims $SIMS --num-workers $W --battle-timeout 120 \
  --exploration 18.5468 --widening-c 3.7028 --widening-alpha 0.52389 --out-csv runs/rep_knobB.csv
echo "=== REP EXPL25 $(date) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m lightspeed.eval_hero --ckpt $CKPT --n-games $N --seed-start $SS \
  --mcts-sims $SIMS --num-workers $W --battle-timeout 120 \
  --exploration 25.0 --widening-c 3.7028 --widening-alpha 0.52389 --out-csv runs/rep_expl25.csv
echo "=== REP ALL DONE $(date) ==="
ln -sf build/slaythespire.cpython-310-x86_64-linux-gnu.so $LINK
