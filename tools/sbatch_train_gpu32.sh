#!/usr/bin/env bash
#SBATCH --account=MST111023
#SBATCH --partition=gp2d
#SBATCH --cpus-per-task=4
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -x

CONFIG=$1
PY_ARGS=${@:2}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python3 -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
