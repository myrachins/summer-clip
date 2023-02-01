#!/bin/sh
#SBATCH --gres=gpu:1 --time=1-00:00:00 --cpus-per-task=8
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
nvidia-smi
date

PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/clip_prompt/train_gpt.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"
ACCELERATE_CFG_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/conf/accelerate/cfg_v1.yaml"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd $SUMMER_CLIP_PATH || exit
accelerate launch --config_file $ACCELERATE_CFG_PATH $PY_PATH
# python -u $PY_PATH
# kernprof -l $PY_PATH
