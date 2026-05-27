#!/bin/bash
# Launch 3 PBRS reward-shaping experiments in parallel on the lambda box (30 cores -> 10 workers each).
# Identical hyperparams across all three; only the shaping potential differs. Stops at iter 100.
# Coeffs from analyze_shaping_coeffs.py (floor-controlled OLS of return-to-go):
#   c_hp=0.18/bar, c_upg=0.007/card; offsets = c * mean end-state feature (E[hp_end]=0.636, E[nup_end]=8.76).
set -u
cd /home/ubuntu/sts_rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON="--torch-compile no --num-games-per-step 96 --num-workers 10 \
--num-iterations 100 --num-epochs 4 --batch-size 192 --policy-lr 3e-5 \
--mcts-simulations 1000 --mcts-exploration 6.57 --mcts-widening-c 3.14 --mcts-widening-alpha 0.97 \
--save-every 5 --save-episodes"

nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-hp-coef 0.18 --shaping-offset 0.115 \
  --save-path runs/shape_hp.pt > runs/shape_hp.log 2>&1 &
echo "hp     PID $!"
sleep 3

nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-upg-coef 0.007 --shaping-offset 0.061 \
  --save-path runs/shape_upg.pt > runs/shape_upg.log 2>&1 &
echo "upg    PID $!"
sleep 3

nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-hp-coef 0.18 --shaping-upg-coef 0.007 --shaping-offset 0.176 \
  --save-path runs/shape_combo.pt > runs/shape_combo.log 2>&1 &
echo "combo  PID $!"
sleep 2
echo "launched 3 shaping runs (hp / upg / combo), num_iterations=100 each"
