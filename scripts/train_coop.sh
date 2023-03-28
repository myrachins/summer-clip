#!/bin/sh
#SBATCH --gres=gpu:1 --time=1-00:00:00
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
nvidia-smi
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/clip_prompt/train_coop.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd $SUMMER_CLIP_PATH || exit
python -u $PY_PATH
# kernprof -l $PY_PATH
