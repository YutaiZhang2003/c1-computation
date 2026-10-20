#!/bin/bash
#SBATCH --job-name=c1-channel-convergence
#SBATCH --partition=serial_requeue
#SBATCH --array=0-20%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
set -euo pipefail

: "${C1_CONVERGENCE_ROOT:?Set C1_CONVERGENCE_ROOT to a new run directory}"
: "${C1_PYTHON:?Set C1_PYTHON to a Python with the numerical dependencies}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script using sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# First ten: change only momentum resolution. Next ten: change only block
# order. Last task: rerun the row-one glasses baseline as a driver control.
index=$((SLURM_ARRAY_TASK_ID % 10))
row=$((index / 2 + 1))
channel=glasses
if ((index % 2 == 1)); then channel=theta; fi
block_order=12
nodes=16
if ((SLURM_ARRAY_TASK_ID >= 10)); then block_order=18; nodes=12; fi
if ((SLURM_ARRAY_TASK_ID == 20)); then block_order=12; nodes=12; fi
cd "${C1_CONVERGENCE_ROOT}/source"
exec "${C1_PYTHON}" -u -B run.py check_channel_convergence \
    --row "${row}" --channel "${channel}" \
    --block-order "${block_order}" --nodes "${nodes}" --dps 50 \
    --output "${C1_CONVERGENCE_ROOT}/receipts"
