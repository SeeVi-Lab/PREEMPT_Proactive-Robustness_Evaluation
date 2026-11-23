#!/usr/bin/env bash
# PREEMPT: Proactive Robustness Evaluation under Editing for Media Protection Techniques
#
# Usage: bash run_preempt.sh <step> [extra args passed to underlying script]
#
# Steps (in order; see README):
#   prompts                      (optional) generate editing prompts with GPT-4.1 mini
#   metadata                     1. index clean and protected images
#   one-shot-transformation      2. one-shot image transformations
#   one-shot-semantic            3. one-shot semantic edits
#   sequential-simple            4. simple chains C1-C4
#   sequential-complex-generate  5a. sample mixed chains CX1-CX4
#   sequential-complex           5b. run mixed chains CX1-CX4
#   metrics                      6. metrics of steps 2-4
#   depth0                       7. S(0) of protected images
#   sequential-complex-metrics   8. metrics of mixed chains
#   psr                          9. PSR, survival curves, PAUC
#   all                          steps 1-9
#   watermark                    score edited images with your watermark decoder

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${CONDA_ENV:-preempt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
EDITOR_NAME="${EDITOR_NAME:-ip2p}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PREEMPT_DATA_ROOT="${PREEMPT_DATA_ROOT:-${PROJECT_ROOT}/data}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV}"
  fi
  mkdir -p "${HF_HOME}" "${TMPDIR}" "${PROJECT_ROOT}/records/metrics"
}

STEP="${1:-}"
shift || true

activate_env
cd "${PROJECT_ROOT}"

case "${STEP}" in
  prompts)
    python scripts/generate_editing_prompts.py "$@"
    ;;
  metadata)
    python scripts/prepare_metadata.py "$@"
    ;;
  one-shot-transformation)
    python scripts/run_one_shot_transformation.py "$@"
    ;;
  one-shot-semantic)
    python scripts/run_one_shot_semantic.py --editor "${EDITOR_NAME}" --device "${DEVICE}" --resume "$@"
    ;;
  sequential-simple)
    python scripts/run_sequential_simple.py --editor "${EDITOR_NAME}" --device "${DEVICE}" --resume "$@"
    ;;
  sequential-complex-generate)
    python scripts/generate_sequential_complex.py --editor "${EDITOR_NAME}" "$@"
    ;;
  sequential-complex)
    python scripts/run_sequential_complex.py --editor "${EDITOR_NAME}" --device "${DEVICE}" --resume "$@"
    ;;
  sequential-complex-metrics)
    python scripts/eval_complex_chains.py --device "${DEVICE}" --editor "${EDITOR_NAME}" \
      --records records/sequential_complex_records.csv \
      --out records/metrics/sequential_complex_metrics.csv "$@"
    ;;
  psr)
    python scripts/compute_survival.py "$@"
    ;;
  watermark)
    python scripts/score_watermark.py "$@"
    ;;
  metrics)
    python scripts/compute_metrics.py --device "${DEVICE}" \
      --level1_records records/one_shot_transformation_records.csv \
      --level2_records records/one_shot_semantic_records.csv \
      --level3_records records/sequential_simple_records.csv "$@"
    ;;
  depth0)
    python scripts/compute_metrics.py --device "${DEVICE}" \
      --skip_level1 --skip_level2 --skip_level3 \
      --level3_records records/sequential_simple_records.csv \
      --compute-depth0 "$@"
    ;;
  all)
    for s in metadata one-shot-transformation one-shot-semantic sequential-simple \
             sequential-complex-generate sequential-complex metrics depth0 \
             sequential-complex-metrics psr; do
      echo "=== ${s} ==="
      bash "${PROJECT_ROOT}/run_preempt.sh" "${s}"
    done
    ;;
  *)
    echo "Unknown step: ${STEP}"
    echo "Valid: prompts metadata one-shot-transformation one-shot-semantic sequential-simple"
    echo "       sequential-complex-generate sequential-complex metrics depth0"
    echo "       sequential-complex-metrics psr all watermark"
    exit 1
    ;;
esac
