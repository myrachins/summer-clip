#!/bin/sh
#SBATCH -A proj_1329
#SBATCH --gres=gpu:1
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
nvidia-smi
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/clip_searcher/save_image_labels.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE="offline"
export WANDB_CONSOLE="off"

cd $SUMMER_CLIP_PATH || exit
python -u $PY_PATH
