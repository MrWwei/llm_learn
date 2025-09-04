#!/bin/bash

# 设置环境变量解决tokenizers并行处理问题
export TOKENIZERS_PARALLELISM=false

# 设置其他有用的环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# 运行训练
echo "开始训练..."
echo "使用GPU: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"

python train.py --config configs/training_config_rtx3060.yaml