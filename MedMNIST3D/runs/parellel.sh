#!/bin/bash
#SBATCH --job-name=qbc
#SBATCH --output=logs/qbc_%A_%a.out
#SBATCH --error=logs/qbc_%A_%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SEED=$SLURM_ARRAY_TASK_ID

python run_one_seed.py --seed $SEED --eval_every 1
