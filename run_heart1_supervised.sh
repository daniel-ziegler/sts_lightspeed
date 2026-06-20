#!/bin/bash
# Live heart1 RL training supervisor (snapshot of the box config: ubuntu@192.9.243.58:~/sts-ca/sts/).
# Auto-resumes from the latest checkpoint and restarts on exit. This is a record of the deployed
# launch args -- the box runs its own copy; sync changes there and re-snapshot here.
#
# Current schedule (as of iter ~1862, 2026-06-19):
#   - reward: heart (floor/190 + 0.05/key + 0.6 heart kill), PBRS shaping (upg 0.035 / starter
#     0.02 / offset 0.307), gamma 1.0
#   - lr: base re-anchored to the post-0.5x floor (policy 1e-5 / value 3.33335e-5), then a further
#     3x geometric decay (lr-final-frac 0.33333) over iters 1875-2175
#   - entropy coef: re-anchored to 0.002, then a further 3x geometric decay to 0.000667 over
#     iters 1875-2175 (prior 0.0083333 -> 0.002 decay done)
#   - engine: multi-select (GAMBLE/EXHAUST_MANY) + Colosseum/Match-and-Keep events now enabled
#   - --pipeline True (overlap collection N+1 with training N, ~1.17x; see COORD.md)
#   - --record-battle-states True: dumps replayable pre-battle action prefixes for every battle
#     start to runs/heart1.pt.battle_states/iter_N.txt (real-policy states; eval_states/A-B input).
#     ~MBs/iter -- pull + rotate. neow-miniBlessing games (10%) are skipped (unreplayable).
cd ~/sts-ca/sts
while true; do
  it=$(ls runs/heart1.pt.iter_* 2>/dev/null | grep -v optimizer | sed 's/.*iter_//' | sort -n | tail -1)
  RESUME=""
  [ -n "$it" ] && RESUME="--resume-from-step $it"
  python -u rl_train.py --algo ppo --seed 1 \
    --num-games-per-step 512 --num-epochs 2 --batch-size 192 \
    --num-iterations 9999 --mcts-simulations 1000 --battle-timeout 60 \
    --reward-function heart \
    --shaping-upg-coef 0.035 --shaping-offset 0.307 --shaping-starter-coef 0.02 \
    --policy-lr 1e-5 --value-lr 3.33335e-5 --entropy-coef 0.002 --entropy-coef-final 0.000667 --entropy-coef-decay-steps 300 --entropy-coef-decay-start 1875 \
    --lr-final-frac 0.33333 --lr-decay-steps 300 --lr-decay-start 1875 \
    --max-ascension 20 \
    --save-every 5 --save-episodes --record-battle-states True --num-workers 30 --torch-compile default --pipeline True \
    $RESUME --save-path runs/heart1.pt >> runs/heart1.log 2>&1
  echo "$(date -u +%FT%TZ) rl_train exited ($?); restarting from latest checkpoint in 30s" >> runs/heart1_supervisor.log
  sleep 30
done
