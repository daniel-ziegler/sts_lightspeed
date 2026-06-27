#!/bin/bash
# Play N isolated single-game runs (fresh JVM each) and tally the heart/act3/loss/crash split.
#
# Usage: ./run_batch.sh <run_name> [games] [per_game_timeout_min]
#   Each game gets its OWN process via run_live.sh <run>_g<i> 1, so a fidelity assert / hang in one
#   game ends only that game -- the batch moves on to the next random seed. Settings are fixed to the
#   measurement config (A20, 10k sims, temp 0, random seed per game). Results accumulate in
#   runs/batch_<run>_results.txt so they survive each launch's errlog truncation.
set +e
RUN="${1:?usage: ./run_batch.sh <run_name> [games] [per_game_timeout_min]}"
N="${2:-30}"
TMO_MIN="${3:-75}"
REPO=/home/dmz/osrc/sts_lightspeed
ERRLOG="/mnt/c/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/communication_mod_errors.log"
RESULTS="$REPO/runs/batch_${RUN}_results.txt"
: > "$RESULTS"
TICKS=$(( TMO_MIN * 4 ))   # 15s ticks

echo "batch ${RUN}: ${N} games, A20 / 10k sims / temp 0 / random seed, ${TMO_MIN}min/game cap" | tee -a "$RESULTS"

for i in $(seq 1 "$N"); do
  # Launch one game (no SEED => random). run_live handles kill/config/autosave/errlog/launch.
  ASC=20 SIMS=10000 TEMP=0 "$REPO/run_live.sh" "${RUN}_g${i}" 1 >/dev/null 2>&1

  # Wait for THIS game's comm.py to exit (completion or crash), up to the per-game cap.
  exited=0
  for t in $(seq 1 "$TICKS"); do
    sleep 15
    if ! pgrep -f '[c]omm.py --character' >/dev/null; then exited=1; break; fi
  done
  pkill -9 -f comm.py 2>/dev/null

  # Parse this game's result from the (still-untruncated) errlog before the next launch clears it.
  seed=$(grep -aoE "= '[0-9A-Z]+'" "$ERRLOG" | head -1 | tr -d "= '")
  line=$(grep -a 'completed with result' "$ERRLOG" | tail -1)
  if [ -n "$line" ]; then
    kind=$(echo "$line" | grep -oE 'kind=[a-z]+' | cut -d= -f2)
    floor=$(echo "$line" | grep -oE 'max_floor=[0-9]+' | cut -d= -f2)
    echo "game $i seed=$seed kind=$kind max_floor=$floor" | tee -a "$RESULTS"
  elif [ "$exited" -eq 1 ]; then
    cr=$(grep -aoE "intent-damage mismatch for [A-Za-z]+|Unmapped [a-z]+ (power|move)[^\"]*|unmapped current move for [A-Za-z]+|Wrong number of cards|Too many cards" "$ERRLOG" | tail -1)
    echo "game $i seed=$seed kind=CRASH detail=[${cr:-unknown}]" | tee -a "$RESULTS"
  else
    echo "game $i seed=$seed kind=TIMEOUT (>${TMO_MIN}min, killed)" | tee -a "$RESULTS"
  fi
done

# Tally.
H=$(grep -c 'kind=heart' "$RESULTS"); A=$(grep -c 'kind=act3' "$RESULTS")
L=$(grep -c 'kind=loss' "$RESULTS"); C=$(grep -c 'kind=CRASH' "$RESULTS"); T=$(grep -c 'kind=TIMEOUT' "$RESULTS")
DONE=$((H+A+L))
echo "=== batch ${RUN} FINAL ===" | tee -a "$RESULTS"
echo "games=$N  completed=$DONE  crash=$C  timeout=$T" | tee -a "$RESULTS"
echo "heart=$H  act3=$A  loss=$L" | tee -a "$RESULTS"
[ "$DONE" -gt 0 ] && echo "heart_win_rate(of completed)=$H/$DONE   incl_act3=$((H+A))/$DONE" | tee -a "$RESULTS"
echo "heart_win_rate(of all $N)=$H/$N" | tee -a "$RESULTS"
