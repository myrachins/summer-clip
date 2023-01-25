#!/bin/sh
#SBATCH -A proj_1329
#SBATCH --time=1-00:00:00 --cpus-per-task=8
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/clip_prompt/tokenize_dataset.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd $SUMMER_CLIP_PATH || exit
python -u $PY_PATH
