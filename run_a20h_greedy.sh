#!/bin/bash
# Robust A20H greedy eval: eval_hero is crash-resumable by seed (JSONL), so on any C++ abort
# (e.g. a rare engine assert) we restart and continue. Stops on clean completion, or if a restart
# makes ZERO new progress (stuck on a deterministically-crashing seed -> needs a look, not a loop).
cd /home/dmz/osrc/sts_lightspeed
JSONL=runs/eval_a20h_2575_greedy.jsonl
LOG=runs/eval_a20h_2575_greedy.log
prev=-1
while true; do
  done=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  if [ "$done" -ge 1000 ]; then echo "$(date -u +%T) complete ($done games)"; break; fi
  if [ "$done" -eq "$prev" ]; then
    echo "$(date -u +%T) NO PROGRESS after restart at $done games -- likely a deterministic crasher; stopping" >> "$LOG"
    break
  fi
  prev=$done
  echo "$(date -u +%T) (re)starting eval at $done/1000 done" >> "$LOG"
  nice -n 10 python3 -m lightspeed.eval_hero \
    --ckpt nets/heart1.pt.iter_2575 \
    --ascension 20 --mcts-sims 10000 --battle-timeout 900 \
    --num-workers 8 --n-games 1000 --temperature 0 \
    --out "$JSONL" >> "$LOG" 2>&1
  sleep 5
done
