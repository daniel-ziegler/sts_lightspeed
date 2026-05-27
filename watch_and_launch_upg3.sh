#!/bin/bash
# Deferred 4th shaping run: upgrades-only at 3x the data-derived coefficient (0.007 -> 0.021),
# offset = 0.021 * E[nup_end=8.76] = 0.184. Waits until one of the 3 current shaping runs finishes
# (frees ~10 cores), then launches. Run me under nohup so I persist across ssh disconnect.
set -u
cd /home/ubuntu/sts_rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Count live shaping runs by their timeout wrappers ([t] trick avoids self-match).
while [ "$(ps -eo cmd | grep -c '[t]imeout 86400 python ppo_train.py.*shape_')" -ge 3 ]; do
  sleep 60
done

echo "$(date -u) capacity freed -> launching shape_upg3"
COMMON="--torch-compile no --num-games-per-step 96 --num-workers 10 \
--num-iterations 100 --num-epochs 4 --batch-size 192 --policy-lr 3e-5 \
--mcts-simulations 1000 --mcts-exploration 6.57 --mcts-widening-c 3.14 --mcts-widening-alpha 0.97 \
--save-every 5 --save-episodes"

nohup timeout 86400 python ppo_train.py $COMMON \
  --shaping-upg-coef 0.021 --shaping-offset 0.184 \
  --save-path runs/shape_upg3.pt > runs/shape_upg3.log 2>&1 &
echo "$(date -u) launched shape_upg3 PID $!"
