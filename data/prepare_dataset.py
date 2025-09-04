"""
数据集准备脚本
支持多种格式的数据集预处理和转换
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

class DatasetPreparer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs("./data/raw", exist_ok=True)
        os.makedirs("./data/processed", exist_ok=True)
        
    def create_sample_dataset(self):
        """创建示例数据集"""
        # 指令微调样本数据
        alpaca_samples = [
            {
                "instruction": "解释什么是机器学习",
                "input": "",
                "output": "机器学习是人工智能的一个分支，它使用算法和统计模型来让计算机系统能够在没有明确编程的情况下从数据中学习和改进性能。机器学习的核心思想是通过大量数据训练模型，使其能够识别模式并做出预测或决策。"
            },
            {
                "instruction": "用Python写一个计算斐波那契数列的函数",
                "input": "",
                "output": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# 更高效的版本\ndef fibonacci_optimized(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```"
            },
            {
                "instruction": "翻译以下英文句子",
                "input": "The weather is beautiful today.",
                "output": "今天天气很美好。"
            },
            {
                "instruction": "解释深度学习中的反向传播算法",
                "input": "",
                "output": "反向传播算法是深度学习中用于训练神经网络的核心算法。它通过计算损失函数相对于网络参数的梯度，然后使用这些梯度来更新参数以最小化损失。算法分为前向传播和反向传播两个阶段：1) 前向传播：输入数据通过网络层层计算得到输出；2) 反向传播：从输出层开始，利用链式法则逐层计算梯度并传播到输入层。"
            },
            {
                "instruction": "给出一些学习编程的建议",
                "input": "",
                "output": "学习编程的建议：1) 选择一门语言深入学习，建议从Python开始；2) 多写代码，通过实际项目练习；3) 阅读他人的优秀代码，学习最佳实践；4) 参与开源项目，与社区互动；5) 坚持不懈，编程需要大量练习；6) 学会调试和解决问题的能力；7) 关注算法和数据结构的学习。"
            }
        ]
        
        # 对话格式样本数据
        conversation_samples = [
            {
                "conversations": [
                    {"role": "user", "content": "你好！"},
                    {"role": "assistant", "content": "你好！我是一个AI助手，很高兴为你服务。有什么可以帮助你的吗？"},
                    {"role": "user", "content": "请介绍一下人工智能的发展历史"},
                    {"role": "assistant", "content": "人工智能的发展历史可以分为几个重要阶段：\n\n1. 起源期(1940s-1950s)：图灵提出图灵测试，标志着AI概念的诞生\n2. 早期发展(1950s-1960s)：专家系统和符号主义AI的兴起\n3. 第一次AI寒冬(1970s-1980s)：由于技术限制和过高期望导致的发展停滞\n4. 复兴期(1980s-1990s)：机器学习和神经网络的重新兴起\n5. 现代AI时代(2000s至今)：深度学习革命，大数据和计算能力的突破"}
                ]
            },
            {
                "conversations": [
                    {"role": "user", "content": "如何提高工作效率？"},
                    {"role": "assistant", "content": "提高工作效率的方法有很多，以下是一些建议：\n\n1. 时间管理：使用番茄工作法、时间块等技巧\n2. 任务优先级：重要紧急四象限法\n3. 减少干扰：关闭不必要的通知，创造专注环境\n4. 工具利用：使用自动化工具和效率软件\n5. 健康作息：保证充足睡眠和适度运动\n6. 学习新技能：持续提升自己的能力"}
                ]
            }
        ]
        
        return alpaca_samples, conversation_samples
        
    def convert_to_alpaca_format(self, data: List[Dict]) -> List[Dict]:
        """转换为Alpaca格式"""
        alpaca_data = []
        for item in data:
            if "conversations" in item:
                # 将对话格式转换为指令格式
                conversations = item["conversations"]
                if len(conversations) >= 2:
                    user_msg = conversations[-2]["content"]
                    assistant_msg = conversations[-1]["content"]
                    alpaca_data.append({
                        "instruction": user_msg,
                        "input": "",
                        "output": assistant_msg
                    })
            else:
                alpaca_data.append(item)
        return alpaca_data
        
    def create_training_data(self, samples: List[Dict], train_ratio: float = 0.8):
        """创建训练和验证数据"""
        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]
        
        return train_data, val_data
        
    def save_dataset(self, data: List[Dict], filepath: str):
        """保存数据集"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据集已保存到: {filepath}")
        
    def load_custom_dataset(self, filepath: str) -> List[Dict]:
        """加载自定义数据集"""
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return []
            
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filepath.endswith('.jsonl'):
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        elif filepath.endswith('.csv'):
            # 简单的CSV读取，不使用pandas
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            return data
        else:
            print(f"不支持的文件格式: {filepath}")
            return []
            
    def prepare_dataset(self, custom_data_path: Optional[str] = None):
        """准备完整的数据集"""
        print("开始准备数据集...")
        
        # 创建示例数据
        alpaca_samples, conversation_samples = self.create_sample_dataset()
        
        # 如果提供了自定义数据，加载并合并
        all_samples = alpaca_samples.copy()
        if custom_data_path and os.path.exists(custom_data_path):
            print(f"加载自定义数据: {custom_data_path}")
            custom_data = self.load_custom_dataset(custom_data_path)
            custom_alpaca = self.convert_to_alpaca_format(custom_data)
            all_samples.extend(custom_alpaca)
            
        # 添加对话样本（转换为alpaca格式）
        conversation_alpaca = self.convert_to_alpaca_format(conversation_samples)
        all_samples.extend(conversation_alpaca)
        
        print(f"总共准备了 {len(all_samples)} 个训练样本")
        
        # 分割训练和验证集
        train_data, val_data = self.create_training_data(all_samples)
        
        # 保存数据集
        self.save_dataset(train_data, "./data/processed/train.json")
        self.save_dataset(val_data, "./data/processed/val.json")
        
        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        
        # 保存原始示例数据
        self.save_dataset(alpaca_samples, "./data/raw/alpaca_samples.json")
        self.save_dataset(conversation_samples, "./data/raw/conversation_samples.json")
        
        return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description="数据集准备工具")
    parser.add_argument("--custom_data", type=str, help="自定义数据集路径")
    parser.add_argument("--format", type=str, default="alpaca", 
                       choices=["alpaca", "conversation"], help="数据格式")
    
    args = parser.parse_args()
    
    config = {}
    preparer = DatasetPreparer(config)
    
    # 准备数据集
    train_data, val_data = preparer.prepare_dataset(args.custom_data)
    
    print("\n数据集准备完成！")
    print("可以使用以下命令开始训练：")
    print("python train.py --config configs/training_config.yaml")

if __name__ == "__main__":
    main()
