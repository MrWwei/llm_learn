# 大模型SFT微调项目 (结合LoRA)

这是一个完整的大语言模型监督微调(SFT)项目，集成了LoRA(Low-Rank Adaptation)技术，实现高效的模型微调。

## 项目结构

```
llm_sft_lora/
├── data/                   # 数据集目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理后的数据
│   └── prepare_dataset.py # 数据准备脚本
├── models/                # 模型相关
│   ├── __init__.py
│   ├── lora_model.py     # LoRA模型实现
│   └── trainer.py        # 训练器
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── data_utils.py     # 数据处理工具
│   ├── train_utils.py    # 训练工具
│   └── eval_utils.py     # 评估工具
├── configs/               # 配置文件
│   ├── training_config.yaml
│   └── model_config.yaml
├── train.py              # 主训练脚本
├── inference.py          # 推理脚本
├── evaluate.py           # 评估脚本
└── requirements.txt      # 依赖包
```

## 功能特性

- 支持多种预训练模型 (LLaMA, ChatGLM, Baichuan等)
- 集成LoRA微调技术，显著减少训练参数
- 支持多种数据格式 (JSON, CSV, JSONL)
- 完整的数据预处理流水线
- 支持分布式训练和梯度累积
- 集成WandB实验跟踪
- 支持模型量化 (4bit/8bit)
- 完整的评估和推理流水线

## 快速开始

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 准备数据集
```bash
python data/prepare_dataset.py
```

3. 开始训练
```bash
python train.py --config configs/training_config.yaml
```

4. 模型推理
```bash
python inference.py --model_path ./output/checkpoint-xxx
```

## 数据格式

支持的数据格式示例：

### 指令微调格式
```json
{
    "instruction": "请解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
}
```

### 对话格式
```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
    ]
}
```

## 配置说明

详细的配置选项请参考 `configs/` 目录下的配置文件。

## 注意事项

- 确保有足够的GPU内存进行训练
- 建议使用V100或A100等高性能GPU
- 根据显存大小调整batch_size和序列长度
