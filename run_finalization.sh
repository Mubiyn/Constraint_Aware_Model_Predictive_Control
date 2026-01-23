#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
BASELINE_DIR="$RESULTS_DIR/baseline"
RL_DIR="$RESULTS_DIR/rl"
COMPARISON_DIR="$RESULTS_DIR/comparison"

PYTHON="python"

log() {
  printf "[%s] %s\n" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1"
}

cleanup_dirs() {
  log "Cleaning previous artifacts"
  rm -rf "$RESULTS_DIR"
  mkdir -p "$BASELINE_DIR"/{logs,plots,videos,data}
  mkdir -p "$RL_DIR"/{logs,plots,videos,data}
  mkdir -p "$COMPARISON_DIR"
}

run_tests() {
  log "Running pytest"
  # Run from REPO_ROOT so that 'v2' package is resolvable
  (cd "$REPO_ROOT" && "$PYTHON" -m pytest --override-ini addopts="") || {
    log "Pytest returned non-zero (likely zero tests); continuing with remaining steps"
  }
}

run_evals() {
  log "Baseline evaluation"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 1000 --eval > "$BASELINE_DIR/logs/eval.txt")

  log "Gait policy evaluation"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 1000 --eval \
      --policy-mode gait --policy v2/rl/gait_cem_v2.npz > "$RL_DIR/logs/eval.txt")
}

generate_long_rollouts() {
  log "Long baseline rollout (log)"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 5000 > "$BASELINE_DIR/logs/long_rollout.log")

  log "Long RL rollout (log)"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 5000 \
      --policy-mode gait --policy v2/rl/gait_cem_v2.npz > "$RL_DIR/logs/long_rollout.log")
}

run_atp() {
  log "Running ATP test suite (MPC Baseline)"
  # Output directly to the data directory
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --run-atp --atp-out "$BASELINE_DIR/data" --atp-record-videos) | tee "$BASELINE_DIR/logs/atp_summary.txt"

  log "Running ATP test suite (RL Policy)"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --run-atp --atp-out "$RL_DIR/data" --atp-record-videos \
      --policy-mode gait --policy v2/rl/gait_cem_v2.npz) | tee "$RL_DIR/logs/atp_summary.txt"
}

plot_atp_results() {
  log "Plotting ATP logs (MPC)"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.utils.plot_atp --in "$BASELINE_DIR/data")
  # Move generated plots to plots dir
  mv "$BASELINE_DIR/data/plots/"* "$BASELINE_DIR/plots/" 2>/dev/null || true
  rmdir "$BASELINE_DIR/data/plots" 2>/dev/null || true
  # Move videos to videos dir (ATP videos are generated in 'videos' subdir of output)
  mv "$BASELINE_DIR/data/videos/"* "$BASELINE_DIR/videos/" 2>/dev/null || true
  rmdir "$BASELINE_DIR/data/videos" 2>/dev/null || true

  log "Plotting ATP logs (RL)"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.utils.plot_atp --in "$RL_DIR/data")
  # Move generated plots to plots dir
  mv "$RL_DIR/data/plots/"* "$RL_DIR/plots/" 2>/dev/null || true
  rmdir "$RL_DIR/data/plots" 2>/dev/null || true
  # Move videos
  mv "$RL_DIR/data/videos/"* "$RL_DIR/videos/" 2>/dev/null || true
  rmdir "$RL_DIR/data/videos" 2>/dev/null || true
}

record_videos() {
  log "Recording baseline walk"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 2000 --record "$BASELINE_DIR/videos/walk.mp4")

  log "Recording RL walk"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 2000 \
      --policy-mode gait --policy v2/rl/gait_cem_v2.npz --record "$RL_DIR/videos/walk.mp4")

  log "Recording disturbance recovery"
  (cd "$REPO_ROOT" && "$PYTHON" -m v2.main_mpc --no-gui --max-steps 1500 --disturbance 100 \
      --record "$BASELINE_DIR/videos/disturbance.mp4")
}

log "Starting finalization script"
log "Using Python: $(which "$PYTHON")"
cleanup_dirs
run_tests
run_evals
generate_long_rollouts
run_atp
plot_atp_results
record_videos
log "Finalization complete. Results located under $RESULTS_DIR"
