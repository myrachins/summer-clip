# CLIP-like Models Adaptation for the Image Classification Task

This repository contains the experiments conducted during my Master's thesis at HSE.

## CLIP-search

CLIP-search is a novel method for image classification that does not require model retraining or labeled data. By incorporating image attention mechanisms and pseudo-labeling strategies, CLIP-search overcomes limitations of existing methods such as UPL and Tip-Adapter, and offers an efficient solution for online classification scenarios where gathering new labeled data or retraining is expensive or impractical.

To reproduce the results of the CLIP-search method, please follow the steps below. Although the instructions are provided for the SUN397 dataset, they are generally applicable to any other dataset unless stated otherwise.

### Step 1: Set up the environment

Create the same conda environment as described in the [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) repository.

### Step 2: Prepare the datasets

Refer to the [DATASET.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) file to install the ImageNet dataset and other 10 datasets mentioned in the CoOp repository.

### Step 3: Generate image features

1. Open the ```summer_clip/conf/save_features.yaml``` configuration file and set ```dataset_name: sun397``` and ```defaults.prompting: tip_sun397```. For the ImageNet dataset, also set ```dataset@train_dataset: imagenet_train``` and ```dataset@test_dataset: imagenet_val```.
1. Run the ```scripts/save_features.sh``` script. This script is compatible with the [slurm](https://slurm.schedmd.com/sbatch.html) manager.
1. Open the ```summer_clip/conf/saved_paths/clip_paths.yaml``` file and set the paths for ```image_features.SUN397_tip_train-RN50```, ```image_features.SUN397_tip_test-RN50```, and ```logits.SUN397_tip_train-RN50-tip```, which were obtained from the previous run. The exact key names can be found in ```summer_clip/conf/img_attn_dataset/sun397.yaml```.

### Step 4: Run the image attention

1. Open the ```summer_clip/conf/image_attention.yaml``` file and set ```defaults.img_attn_dataset@dataset_cfg: sun397```. For the ImageNet dataset, also set ```dataset@train_dataset: imagenet_train_no_image``` and ```dataset@test_dataset: imagenet_val_no_image```.
1. Run the ```scripts/image_attention.sh``` script. This script is also compatible with the [slurm](https://slurm.schedmd.com/sbatch.html) manager.
1. Examine the output file ```outputs/.../.../image_attention.log```. It contains the results of the CLIP-search method with the specified parameters in JSON format.

## Main Packages Used:
- [PyTorch](https://pytorch.org/)
- [Hydra](https://hydra.cc/docs/intro/)
- [CLIP](https://github.com/openai/CLIP)
- [seaborn](https://seaborn.pydata.org/)
- [transformers](https://github.com/huggingface/transformers)

## Main Repositories Used:
- https://github.com/openai/CLIP
- https://github.com/KaiyangZhou/CoOp
- https://github.com/gaopengcuhk/Tip-Adapter
- https://github.com/tonyhuang2022/UPL
- https://github.com/huggingface/transformers
- https://github.com/ucinlp/autoprompt

## Contributors:
- Maxim Rachinskiy
- Aibek Alanov
- Dmitry Vetrov
