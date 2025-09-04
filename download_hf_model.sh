#!/bin/bash
# 用法：./download_hf_model.sh meta-llama/Llama-2-7b-chat-hf [可选保存目录]
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
export HF_HOME=/home/ubuntu/wtwei/transformer_learn/llm_sft_lora/hf_models          # 可选：统一缓存目录

model=$1
dir=${2:-./$model}
pip install -U huggingface_hub[cli]        # 自动装依赖
huggingface-cli download "$model" --local-dir "$dir" --resume-download