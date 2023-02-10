#!/bin/sh
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/clip_em/train_em.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd $SUMMER_CLIP_PATH || exit
python -u $PY_PATH
