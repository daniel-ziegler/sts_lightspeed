#!/bin/bash
# Deployment gate for honest-tune knob candidates vs the 57.2% CardPile baseline (eab_new2).
# Engine sts_new2_9e650c8 + queue-overflow fix... rebuilt below as sts_new3.
set -u
cd ~/osrc/sts_lightspeed.boss-eval
CKPT=runs/heroe2.pt.iter_270; N=1000; SS=6000000; SIMS=1000; W=8
LINK=slaythespire.cpython-310-x86_64-linux-gnu.so
ln -sf engines/sts_new3.so $LINK
echo "=== CONTROL (engine defaults on the fixed binary) $(date) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m silverbot.eval_hero --ckpt $CKPT --n-games $N --seed-start $SS \
  --mcts-sims $SIMS --num-workers $W --battle-timeout 120 \
  --out-csv runs/eab_new3ctl.csv
echo "=== KNOB A (expl 12.2, w 5.17/0.31) $(date) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m silverbot.eval_hero --ckpt $CKPT --n-games $N --seed-start $SS \
  --mcts-sims $SIMS --num-workers $W --battle-timeout 120 \
  --exploration 12.2008 --widening-c 5.1730 --widening-alpha 0.30896 \
  --out-csv runs/eab_knobA.csv
echo "=== KNOB B (expl 18.5, w 3.70/0.52) $(date) ==="
CUDA_VISIBLE_DEVICES=0 python3 -m silverbot.eval_hero --ckpt $CKPT --n-games $N --seed-start $SS \
  --mcts-sims $SIMS --num-workers $W --battle-timeout 120 \
  --exploration 18.5468 --widening-c 3.7028 --widening-alpha 0.52389 \
  --out-csv runs/eab_knobB.csv
echo "=== KNOB GATE ALL DONE $(date) ==="
ln -sf build/slaythespire.cpython-310-x86_64-linux-gnu.so $LINK
