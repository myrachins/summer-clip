#!/bin/sh
#SBATCH --gres=gpu:1
nvidia-smi
date

EVAL_PY_PATH="/home/myurachinskiy/CLIP/summer-clip/results_reproduce/clip_adapter.py"
SUMMER_CLIP_PATH="/home/myurachinskiy/CLIP/summer-clip"

MODEL_NAME="ViT-B/16"
DATASET_NAME="MNIST"
CHECKPOINTS_DIR="/home/myurachinskiy/CLIP/summer-clip/data/checkpoints/mnist-test-4"
EPOCHS_NUM=2
#BATCH_SIZE=512
#LEARNING_RATE=0.00001

export PYTHONPATH="${PYTHONPATH}:${SUMMER_CLIP_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd $SUMMER_CLIP_PATH || exit
# python -u $EVAL_PY_PATH --model-name $MODEL_NAME --dataset-name $DATASET_NAME --checkpoints-dir $CHECKPOINTS_DIR --epochs-num $EPOCHS_NUM --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE
python -u $EVAL_PY_PATH --model-name $MODEL_NAME --dataset-name $DATASET_NAME --checkpoints-dir $CHECKPOINTS_DIR --epochs-num $EPOCHS_NUM
