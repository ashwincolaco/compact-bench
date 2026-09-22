#!/usr/bin/env bash
# COMPACT-Bench scale ladder: the replication subset at six model/precision points.
#
# Each rung re-runs only the cells the paper's three findings rest on, not the full
# reference sweep, and writes one JSON per (task, rung) under runs/scale/.
#
#   ./scripts/scale_sweep.sh            # every rung, in cheapest-first order
#   ./scripts/scale_sweep.sh qwen7b     # one rung by tag
#   TRIALS=3 ./scripts/scale_sweep.sh   # thinner cells for a quick smoke run
#
# Trials default to 8, giving 48 generations per frontier cell. Three is enough to
# check the pipeline runs but too thin to locate a collapse point: neighbouring
# cells then differ by one or two correct answers.
#
# Rungs are resumable: a rung whose four JSONs already exist is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=runs/scale
mkdir -p "$OUT"

# tag|model|quant
RUNGS=(
  "qwen0.5b|Qwen/Qwen2.5-0.5B-Instruct|fp16"
  "qwen1.5b|Qwen/Qwen2.5-1.5B-Instruct|fp16"
  "qwen1.5b-nf4|Qwen/Qwen2.5-1.5B-Instruct|nf4"
  "qwen3b|Qwen/Qwen2.5-3B-Instruct|nf4"
  "phi3.5mini|microsoft/Phi-3.5-mini-instruct|nf4"
  "qwen7b|Qwen/Qwen2.5-7B-Instruct|nf4"
)

run_rung () {
  local tag=$1 model=$2 quant=$3
  echo "=== rung $tag ($model, $quant) started $(date +%F\ %T)"
  local common=(--model "$model" --quant "$quant")

  # F1 frontier collapse: does it sit at a fixed budget fraction or a fixed byte count?
  [ -f "$OUT/frontier-$tag.json" ] || compactbench frontier "${common[@]}" \
      --lengths 2000 4000 --positions 0.1 0.5 0.9 \
      --ratios 0.1 0.25 0.5 0.75 0.9 --trials ${TRIALS:-8} \
      --out "$OUT/frontier-$tag.json" 2>&1 | tee -a "$OUT/$tag.log" | tail -2

  # F2 reversibility crossover: cheap, kept whole.
  [ -f "$OUT/reversibility-$tag.json" ] || compactbench reversibility "${common[@]}" \
      --budgets 0.1 0.25 0.5 0.75 1.0 --trials ${TRIALS:-8} \
      --out "$OUT/reversibility-$tag.json" 2>&1 | tee -a "$OUT/$tag.log" | tail -2

  # F3 attribution is structural: positional auditable, content-scored overclaiming.
  [ -f "$OUT/attribution-$tag.json" ] || compactbench attribution "${common[@]}" \
      --methods SnapKV StreamingLLM Random --lengths 2000 4000 \
      --positions 0.2 0.5 0.8 --ratios 0.5 0.9 --trials ${TRIALS:-8} \
      --out "$OUT/attribution-$tag.json" 2>&1 | tee -a "$OUT/$tag.log" | tail -2

  # F4 calibration blindness: ECE rises while stated confidence stays flat.
  [ -f "$OUT/confidence-$tag.json" ] || compactbench confidence "${common[@]}" \
      --methods SnapKV StreamingLLM Random --lengths 2000 4000 \
      --positions 0.2 0.5 0.8 --ratios 0.0 0.5 0.9 --trials ${TRIALS:-8} \
      --out "$OUT/confidence-$tag.json" 2>&1 | tee -a "$OUT/$tag.log" | tail -2

  echo "=== rung $tag finished $(date +%F\ %T)"
}

want=${1:-all}
for spec in "${RUNGS[@]}"; do
  IFS='|' read -r tag model quant <<< "$spec"
  [ "$want" = all ] || [ "$want" = "$tag" ] || continue
  run_rung "$tag" "$model" "$quant"
done
echo "sweep done; analyse with: compactbench scale --runs $OUT"
