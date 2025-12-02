#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH -J unet_train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=40G

# 激活环境
source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv

# WandB上线
wandb online

# 进入项目文件夹
cd /home/kongw0a/inverseproblem

# 运行训练脚本，传参
python Train.py \
    --train_root "/ibex/project/c2266/wkkong/data/DIV2K/Train" \
    --valid_root "/ibex/project/c2266/wkkong/data/DIV2K/Valid" \
    --batch_size 4 \
    --lr 1e-4 \
    --num_epochs 200 \
    --img_size 256 \
    --lambda_sarm 0.05

