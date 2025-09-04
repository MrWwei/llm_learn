#!/usr/bin/env python3
"""
简化版大模型SFT微调训练脚本
适合RTX 3060等中端显卡
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    """检查GPU状态"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        logger.info(f"GPU: {device_name}")
        logger.info(f"显存: {total_memory:.1f} GB")
        return True
    else:
        logger.warning("未检测到CUDA GPU，将使用CPU训练（速度会很慢）")
        return False

def create_simple_dataset():
    """创建简单的示例数据集"""
    # 确保目录存在
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./data/raw", exist_ok=True)
    
    # 创建小规模高质量训练数据
    train_data = [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
        },
        {
            "instruction": "Explain what is Python",
            "input": "",
            "output": "Python is a high-level, interpreted programming language known for its simple syntax and readability. It's widely used for web development, data science, automation, and artificial intelligence."
        },
        {
            "instruction": "How to sort a list in Python?",
            "input": "",
            "output": "You can sort a list in Python using the sort() method for in-place sorting: my_list.sort(), or using the sorted() function to return a new sorted list: sorted_list = sorted(my_list)."
        },
        {
            "instruction": "What is the difference between AI and ML?",
            "input": "",
            "output": "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a smart way. ML (Machine Learning) is a subset of AI that focuses on machines learning from data to make predictions or decisions."
        },
        {
            "instruction": "Translate to Chinese",
            "input": "Hello world",
            "output": "你好，世界"
        },
        {
            "instruction": "Write a simple function",
            "input": "Calculate factorial",
            "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "instruction": "Explain deep learning",
            "input": "",
            "output": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for tasks like image recognition and natural language processing."
        },
        {
            "instruction": "What is LoRA?",
            "input": "",
            "output": "LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large language models by adding small trainable matrices to existing model weights, significantly reducing the number of trainable parameters."
        }
    ]
    
    # 验证数据（简单分割）
    val_data = [
        {
            "instruction": "What is GPU?",
            "input": "",
            "output": "GPU (Graphics Processing Unit) is a specialized processor designed to handle graphics rendering and parallel computations, making it ideal for machine learning and AI tasks."
        },
        {
            "instruction": "How to install Python packages?",
            "input": "",
            "output": "You can install Python packages using pip: 'pip install package_name' in the command line, or using conda: 'conda install package_name' if you're using Anaconda."
        }
    ]
    
    # 保存数据集
    with open("./data/processed/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("./data/processed/val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练数据: {len(train_data)} 条")
    logger.info(f"验证数据: {len(val_data)} 条")
    
    return train_data, val_data

def simple_training_demo():
    """简单的训练演示（不需要复杂依赖）"""
    logger.info("=" * 60)
    logger.info("大模型SFT微调项目 - RTX 3060版本")
    logger.info("=" * 60)
    
    # 检查GPU
    has_gpu = check_gpu()
    
    # 创建数据集
    logger.info("创建示例数据集...")
    train_data, val_data = create_simple_dataset()
    
    # 显示配置信息
    logger.info("\n推荐配置（适合RTX 3060 12GB）:")
    logger.info("- 模型: microsoft/DialoGPT-medium (350M参数)")
    logger.info("- 备选: distilgpt2 (82M), gpt2 (124M), TinyLlama-1.1B")
    logger.info("- 量化: 4bit量化")
    logger.info("- LoRA rank: 16")
    logger.info("- 批大小: 4")
    logger.info("- 序列长度: 512")
    logger.info("- 精度: FP16")
    
    # 显示内存使用建议
    if has_gpu:
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"\n当前GPU内存使用:")
        logger.info(f"- 已分配: {allocated:.2f} GB")
        logger.info(f"- 已保留: {reserved:.2f} GB")
    
    logger.info("\n要开始训练，请按照以下步骤:")
    logger.info("1. 安装依赖: pip install -r requirements.txt")
    logger.info("2. 运行训练: python train.py --config configs/training_config.yaml")
    logger.info("3. 推理测试: python inference.py --model_path ./output/final_model")
    
    logger.info("\n注意事项:")
    logger.info("- 确保有足够的磁盘空间（至少5GB）")
    logger.info("- 训练过程中监控GPU温度")
    logger.info("- 可以根据显存使用情况调整batch_size")
    
    return True

def create_requirements_txt():
    """创建适合RTX 3060的requirements.txt"""
    requirements = """# 核心依赖（RTX 3060优化版本）
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0
bitsandbytes>=0.41.0

# 基础依赖
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0

# 可选依赖（用于评估）
# sacrebleu>=2.3.0
# rouge-score>=0.1.2
# nltk>=3.8

# 如果需要wandb日志
# wandb>=0.15.0
"""
    
    with open("requirements_rtx3060.txt", 'w') as f:
        f.write(requirements)
    
    logger.info("已创建 requirements_rtx3060.txt")

if __name__ == "__main__":
    # 创建适合RTX 3060的依赖文件
    create_requirements_txt()
    
    # 运行简单演示
    simple_training_demo()
    
    print("\n" + "="*60)
    print("项目创建完成！")
    print("="*60)
    print("\n下一步:")
    print("1. pip install -r requirements_rtx3060.txt")
    print("2. 运行数据准备: python data/prepare_dataset.py")
    print("3. 开始训练: python train.py --config configs/training_config.yaml")
    print("\n注意: 如果遇到显存不足，可以:")
    print("- 减少batch_size到2或1")
    print("- 使用更小的模型如distilgpt2")
    print("- 减少max_seq_length到256")
