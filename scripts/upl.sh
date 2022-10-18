#!/bin/sh
#SBATCH -A proj_1329
#SBATCH --gres=gpu:1
#SBATCH --output=/home/myurachinskiy/CLIP/summer-clip/scripts/outs/slurm-%j.out
nvidia-smi
date

# PY_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/tip_adapter/tip_adapter.py"
# SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"
UPL_PATH="/home/myurachinskiy/CLIP/summer-clip/summer_clip/upl/UPL/scripts"

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd $UPL_PATH || exit
# CUDA_VISIBLE_DEVICES=0 bash get_info.sh ssstanford_cars anay_rn50 end 16 -1 False
# CUDA_VISIBLE_DEVICES=0 bash upl_train.sh ssstanford_cars rn50_ep50 end 16 16 False True rn50_random_init
bash upl_test_existing_logits.sh ssstanford_cars rn50_ep50 end 16 16 False True
