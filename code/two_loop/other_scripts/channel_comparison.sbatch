#!/bin/bash
#SBATCH --job-name=c1-channel-table
#SBATCH --partition=serial_requeue
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail

: "${C1_COMPARISON_ROOT:?Set C1_COMPARISON_ROOT to a new run directory}"
: "${C1_PYTHON:?Set C1_PYTHON to a Python with the numerical dependencies}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script using sbatch}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

row=$((SLURM_ARRAY_TASK_ID / 2 + 1))
channel=glasses
if ((SLURM_ARRAY_TASK_ID % 2 == 1)); then channel=theta; fi
cd "${C1_COMPARISON_ROOT}/source"
exec "${C1_PYTHON}" -u -B run.py recompute_channel_comparison evaluate \
    --row "${row}" --channel "${channel}" \
    --output "${C1_COMPARISON_ROOT}/receipts" \
    --block-order 12 --nodes 12 --dps 50 --period-tolerance 2e-10
