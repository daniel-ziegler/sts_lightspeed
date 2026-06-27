#!/bin/bash
# Launch a live Slay the Spire run driven by comm.py + heart1.
#
# Usage: ./run_live.sh <run_name> [games]
#   run_name : capture/run label, e.g. "v23" -> runs/comm_capture_v23.jsonl
#   games    : number of games (default 20)
#
# Kills any running game/comm.py, points the CommunicationMod config at the capture name and game
# count, clears the autosave, ARCHIVES the prior errlog (so its per-game "completed ... kind=" lines
# survive the truncate), launches ModTheSpire via mts_launch.bat, and verifies it came up.
# Then monitor with: ./monitor_live.sh <run_name>
set +e
RUN="${1:?usage: ./run_live.sh <run_name> [games]}"
GAMES="${2:-20}"
# Optional run knobs via env vars (default to comm.py's own defaults when unset):
#   SEED=<base35>  fixed StS seed (replays the same game each iteration)
#   ASC=<0-20>     ascension level
#   SIMS=<n>       combat MCTS simulations per decision
#   TEMP=<f>       net action-sampling temperature
SEED="${SEED:-}"; ASC="${ASC:-}"; SIMS="${SIMS:-}"; TEMP="${TEMP:-}"
REPO=/home/dmz/osrc/sts_lightspeed
CAP="comm_capture_${RUN}"
CFG="/mnt/c/Users/zieDa/AppData/Local/ModTheSpire/CommunicationMod/config.properties"
ERRLOG="/mnt/c/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/communication_mod_errors.log"
SAVE="/mnt/c/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/saves/IRONCLAD.autosave"
MTS_BAT='C:\Users\zieDa\mts_launch.bat'

taskkill.exe /F /IM java.exe 2>/dev/null || true
pkill -9 -f comm.py 2>/dev/null || true
sleep 3
echo "procs after kill (want java=0 comm.py=0): java=$(tasklist.exe 2>/dev/null | grep -ic java) comm.py=$(ps aux | grep -c '[c]omm.py')"

# Point the mod config at this run's capture name + game count. Run knobs go in as ENV vars in the
# command's `/usr/bin/env ...` prefix, NOT as appended CLI flags: ModTheSpire re-normalizes
# config.properties at startup and strips trailing CLI flags, but preserves these env assignments
# (the same way STS_COMM_CAPTURE survives). comm.py reads STS_START_SEED/STS_ASCENSION/STS_SIMS/
# STS_TEMPERATURE as the defaults for its matching flags.
sed -i "s/comm_capture_[A-Za-z0-9_]*/${CAP}/" "$CFG"
sed -i 's/ STS_START_SEED\\=[0-9A-Za-z]*//g; s/ STS_ASCENSION\\=[0-9]*//g; s/ STS_SIMS\\=[0-9]*//g; s/ STS_TEMPERATURE\\=[0-9.]*//g' "$CFG"
sed -i "s/--games [0-9]*/--games ${GAMES}/" "$CFG"
ENVV=""
[ -n "$SEED" ] && ENVV="$ENVV STS_START_SEED\\=$SEED"
[ -n "$ASC" ]  && ENVV="$ENVV STS_ASCENSION\\=$ASC"
[ -n "$SIMS" ] && ENVV="$ENVV STS_SIMS\\=$SIMS"
[ -n "$TEMP" ] && ENVV="$ENVV STS_TEMPERATURE\\=$TEMP"
# Insert the env assignments right after the existing STS_COMM_CAPTURE\=... token.
[ -n "$ENVV" ] && sed -i "s#\(STS_COMM_CAPTURE\\\\=[^ ]*\)#\1${ENVV}#" "$CFG"
echo "config: $(grep -o "${CAP}[^ ]*\|--games [0-9]*\|STS_START_SEED..[0-9A-Za-z]*\|STS_ASCENSION..[0-9]*\|STS_SIMS..[0-9]*\|STS_TEMPERATURE..[0-9.]*\|iter_[0-9]*" "$CFG" | tr '\n' ' ')"

: > "$SAVE" 2>/dev/null
rm -f "$SAVE" "$SAVE.backUp" 2>/dev/null
echo "autosave files remaining (want 0): $(ls "$SAVE" "$SAVE.backUp" 2>/dev/null | wc -l)"

# Archive the prior run's errlog before truncating -- it is the only place the per-game
# "completed ... kind=heart|act3|loss" lines live, and would otherwise be lost on relaunch.
mkdir -p "$REPO/runs/errlog_archive"
if [ -s "$ERRLOG" ]; then
  cp "$ERRLOG" "$REPO/runs/errlog_archive/errlog_$(date +%Y%m%d_%H%M%S).log"
  echo "archived prior errlog ($(wc -l < "$ERRLOG") lines)"
fi
: > "$ERRLOG"
echo "errlog truncated; whole file is this run ($RUN, $GAMES games)"

nohup cmd.exe /c "$MTS_BAT" >"/tmp/run_${RUN}_launch.out" 2>&1 &
echo "launch issued pid $!"
sleep 45
echo "java up (want >=1): $(tasklist.exe 2>/dev/null | grep -ic java)"
echo "comm.py up (want >=1): $(ps aux | grep -c '[c]omm.py')"
echo "capture bytes: $(stat -c %s "$REPO/runs/${CAP}.jsonl" 2>/dev/null)"
echo "monitor with: ./monitor_live.sh ${RUN}"
true
