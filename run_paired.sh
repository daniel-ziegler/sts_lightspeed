#!/bin/bash
# Matched-seed A/B: for each of N fixed seeds, play the SAME seed twice at ascension 0 -- once driving
# the live combat search on the persistent bc (PBC_DRIVE=1, "M5") and once on master (both gates off) --
# and tally win/floor per arm + the matched per-seed delta. Each game is an isolated run_live launch
# (fresh JVM), so a crash/hang in one ends only that game. The 2h per-game cap is a hang safety net, not
# an expected duration. Results accumulate in runs/paired_<run>_results.txt (survives errlog truncation).
#
# Usage: ./run_paired.sh <run_name> [num_seeds]
set +e
RUN="${1:?usage: ./run_paired.sh <run_name> [num_seeds]}"
N="${2:-30}"
SIMS="${SIMS:-2000}"
ASCLVL="${ASC:-0}"                   # ascension for both arms (set ASC=10 for more signal)
TMO_MIN="${TMO_MIN:-120}"            # 2h per-game hang cap
REPO=/home/dmz/osrc/sts_lightspeed
ERRLOG="/mnt/c/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/communication_mod_errors.log"
RESULTS="$REPO/runs/paired_${RUN}_results.txt"
DESYNCS="$REPO/runs/paired_${RUN}_desync.txt"
: > "$RESULTS"; : > "$DESYNCS"
TICKS=$(( TMO_MIN * 4 ))             # 15s ticks

# Deterministic set of N valid base-35 StS seeds (so the matched run is reproducible).
mapfile -t SEEDS < <(python3 -c "
import sys; sys.path.insert(0,'$REPO'); import comm, random
r=random.Random(20260629)
print('\n'.join(comm.seed_long_to_string(r.getrandbits(63)) for _ in range($N)))
")

echo "paired ${RUN}: ${N} seeds x {DRIVE,master} at A${ASCLVL} / ${SIMS} sims / temp 0 / ${TMO_MIN}min cap" | tee -a "$RESULTS"

run_one () {  # $1=seed $2=tag(drive|master) $3=extra-env-prefix
  local seed="$1" tag="$2"
  env $3 SEED="$seed" ASC="$ASCLVL" SIMS="$SIMS" TEMP=0 "$REPO/run_live.sh" "${RUN}_${tag}" 1 >/dev/null 2>&1
  local timedout=1
  for t in $(seq 1 "$TICKS"); do
    sleep 15
    pgrep -f '[c]omm.py --character' >/dev/null || { timedout=0; break; }
  done
  pkill -9 -f comm.py 2>/dev/null
  # Harvest the DRIVE arm's fidelity/crash signal before the next launch truncates the errlog.
  if [ "$tag" = "drive" ]; then
    grep -aE "\[pbc DESYNC|\[pbc\] (carry|build|chosen)|Assertion|cannot be played with the selected target" "$ERRLOG" \
      | sed "s/^/$seed: /" >> "$DESYNCS" 2>/dev/null
  fi
  local line kind floor
  line=$(grep -a 'completed with result' "$ERRLOG" | tail -1)
  if [ -n "$line" ]; then
    kind=$(echo "$line" | grep -oE 'kind=[a-z0-9]+' | cut -d= -f2)
    floor=$(echo "$line" | grep -oE 'max_floor=[0-9]+' | cut -d= -f2)
    echo "${kind}:${floor}"
  elif [ "$timedout" -eq 1 ]; then echo "TIMEOUT:?"
  else echo "CRASH:?"; fi
}

i=0
for seed in "${SEEDS[@]}"; do
  i=$((i+1))
  d=$(run_one "$seed" drive  "PBC_DRIVE=1")
  m=$(run_one "$seed" master "")
  echo "seed=$seed drive=$d master=$m" | tee -a "$RESULTS"
done

# Tally.
echo "=== paired ${RUN} FINAL ===" | tee -a "$RESULTS"
for arm in drive master; do
  H=$(grep -oE "$arm=heart" "$RESULTS" | wc -l)
  A=$(grep -oE "$arm=act3" "$RESULTS" | wc -l)
  done_=$(grep -oE "$arm=(heart|act3|loss)" "$RESULTS" | wc -l)
  mf=$(grep -oE "$arm=[a-z]+:[0-9]+" "$RESULTS" | grep -oE ':[0-9]+' | tr -d : | awk '{s+=$1;n++} END{if(n)printf "%.1f",s/n}')
  echo "$arm: heart=$H act3=$A completed=$done_ mean_floor=$mf" | tee -a "$RESULTS"
done
echo "(matched per-seed deltas in $RESULTS)" | tee -a "$RESULTS"
