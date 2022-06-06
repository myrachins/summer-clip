#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

EVAL_PY_PATH="/home/myurachinskiy/CLIP/summer-clip/results_reproduce/zero_shot.py"
# SUMMER_WSI_PATH="/home/myurachinskiy/WSI/summer-wsi"

# export PYTHONPATH="${PYTHONPATH}:${SUMMER_WSI_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# cd $SUMMER_WSI_PATH || exit
python -u $EVAL_PY_PATH
# python -u -m cProfile -o GLM/scripts/program.prof $EVAL_PY_PATH --config-name eval_v1 model_vec=glm
# python -u $REPR_PY_PATH --config-name eval_v1 +run_dir=/home/myurachinskiy/WSI/summer-wsi/outputs/2022-02-16_21-33-24
