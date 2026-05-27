#!/bin/bash
# Round 4 queue: launches configs keeping <=3 shape_ runs concurrent (chains onto round 3 as
# slots free; no oversubscription -> round-1 pace). Replicates (baseline2/upg5b) settle the
# dose-response controls; upg10/20/40 hunt for degenerate ALWAYS-SMITH (mirror of always-rest).
# Offsets = coef * E[nup_end]=8.76. Run under nohup.
set -u
cd /home/ubuntu/sts_rl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--torch-compile no --num-games-per-step 96 --num-workers 10 \
--num-iterations 100 --num-epochs 4 --batch-size 192 --policy-lr 3e-5 \
--mcts-simulations 1000 --mcts-exploration 6.57 --mcts-widening-c 3.14 --mcts-widening-alpha 0.97 \
--save-every 5 --save-episodes"
MAXC=5   # 5 in parallel: straggler duty-cycle leaves headroom on 30 cores

QUEUE=(
  "baseline2|"
  "upg5b|--shaping-upg-coef 0.035 --shaping-offset 0.307"
  "upg10|--shaping-upg-coef 0.07 --shaping-offset 0.613"
  "upg20|--shaping-upg-coef 0.14 --shaping-offset 1.226"
  "upg40|--shaping-upg-coef 0.28 --shaping-offset 2.453"
)

for entry in "${QUEUE[@]}"; do
  name="${entry%%|*}"; flags="${entry#*|}"
  while [ "$(ps -eo cmd | grep -c '[t]imeout 86400 python ppo_train.py.*shape_')" -ge $MAXC ]; do sleep 60; done
  echo "$(date -u) launching shape_${name}  flags=[${flags}]"
  nohup timeout 86400 python ppo_train.py $COMMON $flags \
    --save-path runs/shape_${name}.pt > runs/shape_${name}.log 2>&1 &
  echo "  PID $!"
  sleep 8
done
echo "$(date -u) round-4 queue fully launched"
