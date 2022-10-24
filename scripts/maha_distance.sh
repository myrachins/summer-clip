#!/bin/sh
#SBATCH -A proj_1329
#SBATCH --gres=gpu:1
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
nvidia-smi
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/clip_searcher/maha_distance.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_API_KEY="4f00c0670d096e6fa7b5959867f34e3b14015f0b"
export WANDB_MODE="offline"
export WANDB_CONSOLE="off"

cd $SUMMER_CLIP_PATH || exit
python -u $PY_PATH
