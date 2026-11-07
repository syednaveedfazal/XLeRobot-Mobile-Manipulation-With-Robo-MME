#!/usr/bin/env bash
# Submit the teleop job, wait until it is up, start a relay on this login node, print the URL.
#   bash scripts/start_teleop.sh                      # 1 h on mlgpu_devel
#   PARTITION=mlgpu_short TIME=04:00:00 bash scripts/start_teleop.sh
#   bash scripts/start_teleop.sh --shader rt-fast     # extra args go to teleop_web.py
#   SBATCH_SCRIPT=slurm/teleop_isaac.sbatch bash scripts/start_teleop.sh   # Isaac Lab version
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
PARTITION="${PARTITION:-mlgpu_devel}"
TIME="${TIME:-01:00:00}"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/teleop.sbatch}"  # Isaac Lab: slurm/teleop_isaac.sbatch
JOB=$(sbatch --parsable --partition="$PARTITION" --time="$TIME" "$SBATCH_SCRIPT" "$@")
LOG="logs/teleop-$JOB.out"
echo "submitted job $JOB on $PARTITION ($TIME). log: $LOG"

until [ -f "$LOG" ] && grep -q "env ready" "$LOG"; do
    state=$(squeue -j "$JOB" -h -o "%T %R" 2>/dev/null || true)
    if [ -z "$state" ]; then echo; echo "job ended early:"; cat "$LOG" 2>/dev/null; exit 1; fi
    if [ "${state%% *}" = "RUNNING" ]; then state="RUNNING - loading scene"; fi
    printf "\r  %-60s" "$state"
    sleep 3
done
echo

# "Teleop server on mlgpu017:22874" and "Open: http://localhost:22874/?token=XYZ"
NODE_PORT=$(grep -oP "Teleop server on \K\S+" "$LOG")
NODE=${NODE_PORT%:*}; PORT=${NODE_PORT#*:}
TOKEN=$(grep -oP "token=\K\S+" "$LOG" | head -1)

# relay: localhost:<LPORT> on this login node -> NODE:PORT (relay picks the first free port >= PORT)
RLOG="logs/relay-$JOB.out"
setsid nohup bash -c "
    python3 -u scripts/relay.py $NODE $PORT &
    RP=\$!
    while squeue -j $JOB -h 2>/dev/null | grep -q .; do sleep 10; done
    kill \$RP" > "$RLOG" 2>&1 < /dev/null &
for _ in $(seq 50); do grep -q "\[relay\]" "$RLOG" 2>/dev/null && break; sleep 0.2; done
LPORT=$(grep -oP "\[relay\] localhost:\K[0-9]+" "$RLOG" || true)
if [ -z "$LPORT" ]; then echo "relay failed to start:"; cat "$RLOG"; exit 1; fi

cat <<EOF

 Teleop is running on $NODE (job $JOB, login relay on $(hostname -s):$LPORT)

 Open in your laptop's browser:   http://localhost:$LPORT/?token=$TOKEN

 VS Code forwards localhost:$LPORT automatically (check the Ports tab; add $LPORT there if missing).
 Without VS Code, run on your laptop (must reach this login node, $(hostname -s)):
   ssh -N -L $LPORT:localhost:$LPORT $USER@$(hostname -I | tr ' ' '\n' | grep -m1 '^131\.220')

 Stop:  scancel $JOB      (the relay stops by itself when the job ends)
EOF
