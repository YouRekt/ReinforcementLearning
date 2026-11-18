#!/usr/bin/env bash
# Self-play comparison: MAPPO vs IPPO and network sizes, 3 seeds x 20M steps each.
# Two runs at a time (7 env workers each); finished runs (with final.pt) are skipped.
#   caffeinate -i bash scripts/sweep.sh        # macOS: keep the machine awake
# Override the defaults with env vars, e.g.:
#   CONFIGS="mappo_h128 ippo_h128" SEEDS="0 1" caffeinate -i bash scripts/sweep.sh
set -u
cd "$(dirname "$0")/.."
args_of() {  # bash 3 (macOS) has no associative arrays
  case "$1" in
    mappo_h128)   echo "--hidden 128 --layers 2 --critic central" ;;
    ippo_h128)    echo "--hidden 128 --layers 2 --critic local" ;;
    mappo_h64)    echo "--hidden 64 --layers 2 --critic central" ;;
    mappo_h256)   echo "--hidden 256 --layers 2 --critic central" ;;
    mappo_h256x3) echo "--hidden 256 --layers 3 --critic central" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}
jobs=()
CONFIGS="${CONFIGS:-mappo_h128 ippo_h128 mappo_h64 mappo_h256 mappo_h256x3}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-2e7}"
for seed in $SEEDS; do
  for name in $CONFIGS; do
    run="${name}_s${seed}"
    [ -f "runs/$run/final.pt" ] && continue
    jobs+=("$run|$(args_of $name) --seed $seed --steps $STEPS")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -I{} bash -c '
  run="${1%%|*}"; args="${1#*|}"
  mkdir -p runs/$run
  echo "$(date +%H:%M) start $run"
  uv run python scripts/train_mappo.py --run $run --train adv good --workers 7 --eval-every 50 $args \
    > runs/$run/train.log 2>&1
  rc=$?
  echo "$(date +%H:%M) done  $run (exit $rc)"
' _ {}
