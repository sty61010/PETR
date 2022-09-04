#!/usr/bin/env bash
#SBATCH --account=MST111023
#SBATCH --partition=gp2d
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -x

CONFIG=$1
CHECKPOINT=$2
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python3 -u tools/test.py ${CONFIG} ${CHECKPOINT}--launcher="slurm" ${PY_ARGS}
