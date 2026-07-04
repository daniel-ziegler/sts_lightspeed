#!/bin/bash
# PBC_DRIVE robustness grind on FRESH seeds (single arm, no master): stress the persistent-bc drive
# across many games and surface any remaining card-select FALLBACK, engine ASSERT/crash, or coordinator
# HANG. Per-game isolation (fresh JVM via run_live) so a crash/hang ends only that game. Each game's
# signals are harvested to runs/grind_<run>_results.txt before the next launch truncates the errlog;
# run_live also archives the full errlog per game to runs/errlog_archive/. Offending lines go to
# runs/grind_<run>_issues.txt for triage.
#
# Usage: ./run_grind.sh <run_name> [num_games]
set +e
RUN="${1:?usage: ./run_grind.sh <run_name> [num_games]}"
N="${2:-50}"
SIMS="${SIMS:-2000}"
ASCLVL="${ASC:-0}"       # a fixed level, or ASC=rand for a random 0-20 per game (recorded per line)
TMO_MIN="${TMO_MIN:-90}"
REPO=/home/dmz/osrc/sts_lightspeed
ERRLOG="/mnt/c/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/communication_mod_errors.log"
RESULTS="$REPO/runs/grind_${RUN}_results.txt"
ISSUES="$REPO/runs/grind_${RUN}_issues.txt"
# RESUME=1 continues an interrupted grind: keep RESULTS/ISSUES and (below, once the seed list is
# built) skip the seeds already played. Requires invoking with the SAME seed list as the
# interrupted run (same run_name/N or SEEDS_FILE).
if [ -z "${RESUME:-}" ]; then : > "$RESULTS"; : > "$ISSUES"; fi
TICKS=$(( TMO_MIN * 4 ))

# Seed list: SEEDS_FILE=<path> plays an explicit list of base-35 seeds (one per line -- redo
# runs replaying specific seeds under current code); otherwise fresh deterministic seeds
# (RNG seed distinct from the paired set so these are genuinely new games).
if [ -n "${SEEDS_FILE:-}" ]; then
  mapfile -t SEEDS < <(grep -v '^\s*$' "$SEEDS_FILE")
  N=${#SEEDS[@]}
else
  mapfile -t SEEDS < <(python3 -c "
import sys; sys.path.insert(0,'$REPO'); import comm, random
r=random.Random(70150131)
print('\n'.join(comm.seed_long_to_string(r.getrandbits(63)) for _ in range($N)))
")
fi

# Resume: every played game left one '^g<i>' line, in seed order, so the line count IS the
# number of seeds to skip; numbering continues from there.
START=0
if [ -n "${RESUME:-}" ]; then
  START=$(grep -c '^g[0-9]' "$RESULTS")
  SEEDS=("${SEEDS[@]:$START}")
  N=${#SEEDS[@]}
fi

echo "grind ${RUN}: ${N} PBC_DRIVE games at A${ASCLVL} / ${SIMS} sims / temp 0 / ${TMO_MIN}min cap${SEEDS_FILE:+ (seeds from $SEEDS_FILE)}${RESUME:+ (resumed at g$((START+1)))}" | tee -a "$RESULTS"

i=$START
for seed in "${SEEDS[@]}"; do
  i=$((i+1))
  # Per-game ascension: a fixed level, or a random 0-20 when ASC=rand (mirrors run_batch.sh).
  if [ "$ASCLVL" = "rand" ]; then GAME_ASC=$(( RANDOM % 21 )); else GAME_ASC="$ASCLVL"; fi
  env PBC_DRIVE=1 SEED="$seed" ASC="$GAME_ASC" SIMS="$SIMS" TEMP=0 "$REPO/run_live.sh" "${RUN}" 1 >/dev/null 2>&1
  # Wait for the game to finish. comm.py's absence only means "game over" once it has been SEEN
  # at least once: a slow JVM/Steam launch can outlast run_live's 45s settle + the first tick,
  # and declaring the game dead then pkills comm.py right as the mod spawns it (a banner-only
  # errlog, driven=0, kind=CRASH -- a burned seed). Grace cap: if comm.py never appears within
  # 20 ticks (5 min), the launch genuinely failed and the errlog classification proceeds.
  timedout=1
  seen=0
  for t in $(seq 1 "$TICKS"); do
    sleep 15
    if pgrep -f '[c]omm.py --character' >/dev/null; then
      seen=1
    elif [ "$seen" -eq 1 ] || [ "$t" -ge 20 ]; then
      timedout=0; break
    fi
  done
  pkill -9 -f comm.py 2>/dev/null
  # Fallbacks are gone (divergence now crashes): a game that can't be resolved raises, so run_agent_cli
  # prints "Game error:" + a traceback and the game ends with no "completed with result" line. Count the
  # crash markers (pbc/shop/unmapped-select RuntimeErrors + the catch-all) rather than the retired
  # fallback strings.
  crash=$(grep -acE "Game error:|pbc/live select divergence|driven persistent bc|not parked at expected|shop choice unresolved|unmapped" "$ERRLOG" 2>/dev/null)
  asrt=$(grep -acE "Assertion|BATTLE SEARCH CRASH" "$ERRLOG" 2>/dev/null)
  hang=$(grep -acE "appears hung" "$ERRLOG" 2>/dev/null)
  driven=$(grep -acE "pbc-driven" "$ERRLOG" 2>/dev/null)
  line=$(grep -a 'completed with result' "$ERRLOG" | tail -1)
  kind=$(echo "$line" | grep -oE 'kind=[a-z0-9]+' | cut -d= -f2)
  floor=$(echo "$line" | grep -oE 'max_floor=[0-9]+' | cut -d= -f2)
  if [ -z "$kind" ]; then
    if [ "$hang" -gt 0 ]; then kind="HANG"
    elif [ "$timedout" -eq 1 ]; then kind="TIMEOUT"
    else kind="CRASH"; fi
  fi
  if [ "$crash" -gt 0 ] || [ "$asrt" -gt 0 ] || [ "$hang" -gt 0 ] || [ "$kind" = "CRASH" ]; then
    { echo "=== seed $seed (g$i, kind=$kind) ==="
      grep -aE "Game error:|Traceback|pbc/live select divergence|driven persistent bc|not parked at expected|shop choice unresolved|unmapped|Assertion|BATTLE SEARCH CRASH|appears hung|left pbc unclean|did not converge" "$ERRLOG"
    } >> "$ISSUES" 2>/dev/null
  fi
  echo "g$i seed=$seed asc=$GAME_ASC kind=${kind}:${floor} driven=$driven crashes=$crash asserts=$asrt hangs=$hang" | tee -a "$RESULTS"
done

echo "=== grind ${RUN} FINAL ===" | tee -a "$RESULTS"
sum() { grep -oE "$1=[0-9]+" "$RESULTS" | grep -oE '[0-9]+' | paste -sd+ | bc; }
echo "games=$(grep -c '^g[0-9]' "$RESULTS")  driven=$(sum driven)  crashes=$(sum crashes)  asserts=$(sum asserts)  hangs=$(sum hangs)" | tee -a "$RESULTS"
for k in heart act3 loss HANG CRASH TIMEOUT; do
  echo "$k: $(grep -oE "kind=${k}:" "$RESULTS" | wc -l)" | tee -a "$RESULTS"
done
