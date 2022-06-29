#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

EVAL_PY_PATH="/home/myurachinskiy/CLIP/summer-clip/results_reproduce/adapter_eval.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

DATASET_NAME="MNIST"
CHECKPOINT_DIR="/home/myurachinskiy/CLIP/summer-clip/data/checkpoints/mnist-test-4/epoch_2"
#BATCH_SIZE=512

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd $SUMMER_CLIP_PATH || exit
python -u $EVAL_PY_PATH --dataset-name $DATASET_NAME --checkpoint-dir $CHECKPOINT_DIR
