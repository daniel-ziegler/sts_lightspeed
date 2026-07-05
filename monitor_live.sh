#!/bin/bash
# Monitor a live run started by run_live.sh and report the heart/act3/loss split.
#
# Usage: ./monitor_live.sh <run_name>
# Polls the errlog every 15s; exits on a crash/abort signature, a 15-min capture stall (HUNG),
# or once <games> games have completed. A "win" (PLAYER_VICTORY) is split into heart kills
# (reached act 4) vs act-3-only wins -- mirrors the offline eval's heart_win_rate / act3_win_rate.
set +e
RUN="${1:?usage: ./monitor_live.sh <run_name>}"
REPO=/home/dmz/osrc/sts_lightspeed
ERRLOG="/mnt/c/Program Files (x86)/Steam/steamapps/common/SlayTheSpire/communication_mod_errors.log"
CAP="$REPO/runs/comm_capture_${RUN}.jsonl"
CFG="/mnt/c/Users/zieDa/AppData/Local/ModTheSpire/CommunicationMod/config.properties"
GAMES="${2:-$(grep -oE '\-\-games [0-9]+' "$CFG" 2>/dev/null | grep -oE '[0-9]+' | head -1)}"
GAMES="${GAMES:-20}"
CRASH_RE='Unmapped (monster|player) power|ValueError|NotImplementedError|RuntimeError|BATTLE SEARCH CRASH|Assertion|could not drive|despite full belt|Wrong number of cards|Too many cards|Game error'
LASTSZ=-1; LASTSZ_CHG=0

report() {
  local W H A L
  W=$(grep -ac 'result: True' "$ERRLOG"); L=$(grep -ac 'result: False' "$ERRLOG")
  H=$(grep -ac 'kind=heart' "$ERRLOG"); A=$(grep -ac 'kind=act3' "$ERRLOG")
  echo "WINS=$W (heart=$H act3=$A)  LOSSES=$L"
  echo "win_rate(incl act3)=$W/$((W+L))   heart_win_rate=$H/$((W+L))   act3_only=$A/$((W+L))"
}

for i in $(seq 1 2400); do
  sleep 15
  CL=$(grep -aoE "$CRASH_RE" "$ERRLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -1)
  if [ -n "$CL" ]; then
    echo "CRASH/ABORT at tick $i ($((i*15))s): $CL"; report
    echo "--- last 20 ---"; tail -20 "$ERRLOG"; break
  fi
  DONE=$(grep -ac 'completed with result' "$ERRLOG" 2>/dev/null); DONE=${DONE:-0}
  if [ "$DONE" -ge "$GAMES" ]; then
    echo "=== ${RUN} FINAL: $GAMES games (heart/act3 split) ==="; report
    echo "shadow ok=$(grep -ac '\[shadow ok\]' "$ERRLOG")  ERR=$(grep -ac '\[shadow ERR\]' "$ERRLOG")  nudges=$(grep -ac "sending 'state'" "$ERRLOG")"
    echo "--- per-game ---"; grep -aE 'completed with result' "$ERRLOG" | tail -"$GAMES"
    break
  fi
  SZ=$(stat -c %s "$CAP" 2>/dev/null); SZ=${SZ:-0}
  if [ "$SZ" != "$LASTSZ" ]; then LASTSZ=$SZ; LASTSZ_CHG=$i; fi
  if [ $((i-LASTSZ_CHG)) -ge 60 ]; then
    echo "HUNG: capture flat 15min at tick $i"; report
    echo "nudges: $(grep -ac "sending 'state'" "$ERRLOG")"; tail -15 "$ERRLOG"; break
  fi
done
echo "${RUN} monitor exit; completions=$(grep -ac 'completed with result' "$ERRLOG" 2>/dev/null)/$GAMES"
