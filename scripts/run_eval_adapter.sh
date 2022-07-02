#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/results_reproduce/eval_adapter.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd $SUMMER_CLIP_PATH || exit
python -u $PY_PATH
