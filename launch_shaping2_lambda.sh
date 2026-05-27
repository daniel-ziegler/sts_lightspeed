#!/bin/bash
# Follow-up shaping runs on lambda (box idle, 30 cores -> 10 workers each). iter 100 target.
# Matched hyperparams + tuned MCTS, only shaping differs. Offsets = coef * E[end-feature]
# (E[hp_end]=0.636, E[nup_end]=8.76).
set -u
cd /home/ubuntu/sts_rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--torch-compile no --num-games-per-step 96 --num-workers 10 \
--num-iterations 100 --num-epochs 4 --batch-size 192 --policy-lr 3e-5 \
--mcts-simulations 1000 --mcts-exploration 6.57 --mcts-widening-c 3.14 --mcts-widening-alpha 0.97 \
--save-every 5 --save-episodes"

# baseline: NO shaping, same tuned MCTS -> isolates the shaping effect against the shaped runs
nohup timeout 86400 python ppo_train.py $COMMON \
  --save-path runs/shape_baseline.pt > runs/shape_baseline.log 2>&1 &
echo "baseline PID $!"; sleep 3

# upg3 replicate (fresh init) -> confirm shape_upg3 (win 0.334) wasn't a lucky seed
nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-upg-coef 0.021 --shaping-offset 0.184 \
  --save-path runs/shape_upg3b.pt > runs/shape_upg3b.log 2>&1 &
echo "upg3b PID $!"; sleep 3

# upg5: push the upgrade dose further (5x = 0.035), offset 0.035*8.76
nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-upg-coef 0.035 --shaping-offset 0.307 \
  --save-path runs/shape_upg5.pt > runs/shape_upg5.log 2>&1 &
echo "upg5 PID $!"; sleep 3

# hp3: 3x HP shaping (0.18->0.54), offset 0.54*0.636 -> mirror the dose test on HP
nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-hp-coef 0.54 --shaping-offset 0.343 \
  --save-path runs/shape_hp3.pt > runs/shape_hp3.log 2>&1 &
echo "hp3 PID $!"; sleep 2
echo "launched 4 follow-up runs (baseline / upg3b / upg5 / hp3), num_iterations=100"
