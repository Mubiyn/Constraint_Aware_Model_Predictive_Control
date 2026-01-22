#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"
PLOT_DIR="$RESULTS_DIR/plots"
VIDEO_DIR="$RESULTS_DIR/videos"
ATP_DIR="$RESULTS_DIR/atp/latest"
PYTHON="python"

log() {
  printf "[%s] %s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1"
}

cleanup_dirs() {
  log "Cleaning previous artifacts"
  rm -rf "$RESULTS_DIR"
  mkdir -p "$LOG_DIR" "$PLOT_DIR" "$VIDEO_DIR" "$ATP_DIR"
}

run_tests() {
  log "Running pytest"
  (cd "$ROOT_DIR" && "$PYTHON" -m pytest)
}

run_evals() {
  log "Baseline evaluation"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 1000 --eval > "$LOG_DIR/baseline_eval.txt")

  log "Gait policy evaluation"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 1000 --eval \
      --policy-mode gait --policy rl/gait_cem_v2.npz > "$LOG_DIR/gait_eval.txt")
}

generate_long_rollouts() {
  log "Long baseline rollout (log)"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 5000 > "$LOG_DIR/baseline_long.log")

  log "Long RL rollout (log)"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 5000 \
      --policy-mode gait --policy rl/gait_cem_v2.npz > "$LOG_DIR/gait_long.log")
}

run_atp() {
  log "Running ATP test suite"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --run-atp --atp-out "$ATP_DIR")
}

plot_atp_results() {
  log "Plotting ATP logs"
  (cd "$ROOT_DIR" && "$PYTHON" plot_atp.py --in "$ATP_DIR")
  cp -r "$ATP_DIR/plots" "$PLOT_DIR"
}

record_videos() {
  log "Recording baseline walk"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 2000 --record "$VIDEO_DIR/baseline_walk.mp4")

  log "Recording RL walk"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 2000 \
      --policy-mode gait --policy rl/gait_cem_v2.npz --record "$VIDEO_DIR/rl_walk.mp4")

  log "Recording disturbance recovery"
  (cd "$ROOT_DIR" && "$PYTHON" main_mpc.py --no-gui --max-steps 1500 --disturbance 100 \
      --record "$VIDEO_DIR/disturbance.mp4")
}

log "Starting finalization script"
cleanup_dirs
run_tests
run_evals
generate_long_rollouts
run_atp
plot_atp_results
record_videos
log "Finalization complete. Results located under $RESULTS_DIR"
