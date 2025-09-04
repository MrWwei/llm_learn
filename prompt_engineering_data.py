#!/usr/bin/env python3
"""
提示词工程增强的数据准备模块
结合多种提示词技术提升训练效果
"""

import json
import os
from typing import List, Dict, Any

class PromptEngineeringDataset:
    """提示词工程数据集创建器"""
    
    def __init__(self):
        self.prompt_templates = {
            "role_playing": {
                "ai_expert": "你是一名资深的人工智能专家，拥有丰富的理论知识和实践经验。",
                "programming_tutor": "你是一名编程导师，善于用通俗易懂的语言解释复杂的技术概念。",
                "technical_writer": "你是一名技术写作专家，能够清晰准确地表达技术内容。"
            },
            "task_instructions": {
                "explain": "请详细解释以下概念，包括定义、特点和应用场景：",
                "compare": "请比较以下概念的异同点：",
                "step_by_step": "请提供分步骤的详细说明：",
                "example": "请提供具体的例子来说明："
            },
            "output_formats": {
                "structured": "请按以下格式回答：\n1. 定义\n2. 特点\n3. 应用场景\n4. 实例",
                "simple": "请用简单易懂的语言回答。",
                "detailed": "请提供详细全面的回答。"
            }
        }
    
    def create_base_samples(self) -> List[Dict[str, str]]:
        """创建基础样本数据"""
        return [
            {
                "topic": "机器学习",
                "question": "什么是机器学习？",
                "answer": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。主要包括监督学习、无监督学习和强化学习三种类型。监督学习使用标记数据进行训练，无监督学习发现数据中的隐藏模式，强化学习通过奖励机制优化决策。"
            },
            {
                "topic": "深度学习",
                "question": "深度学习是什么？",
                "answer": "深度学习是机器学习的子集，使用多层神经网络来模拟人脑的工作方式。它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。深度学习需要大量数据和计算资源，但能够自动学习特征表示。"
            },
            {
                "topic": "Python编程",
                "question": "为什么Python适合机器学习？",
                "answer": "Python适合机器学习的原因包括：语法简洁易学、拥有丰富的科学计算库（如NumPy、Pandas、Scikit-learn）、强大的深度学习框架（如TensorFlow、PyTorch）、活跃的社区支持，以及良好的可视化工具。"
            },
            {
                "topic": "神经网络",
                "question": "神经网络如何工作？",
                "answer": "神经网络由输入层、隐藏层和输出层组成。每个神经元接收输入，应用权重和偏置，然后通过激活函数产生输出。反向传播算法用于训练网络，通过调整权重来最小化损失函数，从而使网络能够学习复杂的模式。"
            },
            {
                "topic": "自然语言处理",
                "question": "什么是NLP？",
                "answer": "自然语言处理(NLP)是人工智能的一个分支，专注于计算机与人类语言的交互。它包括文本分析、情感分析、机器翻译、问答系统等任务。现代NLP主要基于Transformer架构，如BERT、GPT等模型。"
            }
        ]
    
    def create_role_playing_samples(self, base_samples: List[Dict]) -> List[Dict[str, str]]:
        """创建角色扮演提示样本"""
        samples = []
        
        for sample in base_samples:
            for role, role_desc in self.prompt_templates["role_playing"].items():
                formatted_sample = {
                    "instruction": f"{role_desc}请回答以下问题：",
                    "input": sample["question"],
                    "output": f"作为一名专家，我来回答您的问题。{sample['answer']}"
                }
                samples.append(formatted_sample)
        
        return samples
    
    def create_chain_of_thought_samples(self, base_samples: List[Dict]) -> List[Dict[str, str]]:
        """创建思维链提示样本"""
        samples = []
        
        cot_templates = [
            {
                "instruction": "请按照以下步骤思考并回答问题：\n1. 首先理解问题的核心\n2. 分析相关概念\n3. 给出详细解释\n4. 提供实际例子",
                "format": "让我按步骤来分析这个问题：\n\n1. 核心概念：{core}\n2. 详细分析：{analysis}\n3. 实际应用：{application}"
            }
        ]
        
        for sample in base_samples:
            for template in cot_templates:
                formatted_sample = {
                    "instruction": template["instruction"],
                    "input": sample["question"],
                    "output": template["format"].format(
                        core=f"这个问题询问的是{sample['topic']}",
                        analysis=sample["answer"],
                        application="这在实际应用中非常重要，是AI领域的基础概念。"
                    )
                }
                samples.append(formatted_sample)
        
        return samples
    
    def create_few_shot_samples(self, base_samples: List[Dict]) -> List[Dict[str, str]]:
        """创建Few-Shot提示样本"""
        samples = []
        
        # 创建示例格式
        examples = base_samples[:2]  # 使用前两个作为示例
        target_samples = base_samples[2:]  # 剩余的作为目标
        
        example_text = "以下是一些问答示例：\n\n"
        for i, example in enumerate(examples, 1):
            example_text += f"示例{i}：\n问：{example['question']}\n答：{example['answer']}\n\n"
        
        for sample in target_samples:
            formatted_sample = {
                "instruction": example_text + "现在请回答以下问题：",
                "input": sample["question"],
                "output": sample["answer"]
            }
            samples.append(formatted_sample)
        
        return samples
    
    def create_structured_output_samples(self, base_samples: List[Dict]) -> List[Dict[str, str]]:
        """创建结构化输出样本"""
        samples = []
        
        for sample in base_samples:
            formatted_sample = {
                "instruction": "请按照以下结构回答问题：\n【定义】\n【特点】\n【应用】\n【总结】",
                "input": sample["question"],
                "output": f"【定义】\n{sample['answer'].split('。')[0]}。\n\n【特点】\n{sample['topic']}具有重要的技术特征。\n\n【应用】\n广泛应用于AI和机器学习领域。\n\n【总结】\n{sample['topic']}是现代人工智能的重要组成部分。"
            }
            samples.append(formatted_sample)
        
        return samples
    
    def generate_enhanced_dataset(self) -> List[Dict[str, str]]:
        """生成增强的训练数据集"""
        base_samples = self.create_base_samples()
        
        all_samples = []
        
        # 基础样本
        for sample in base_samples:
            all_samples.append({
                "instruction": "请回答以下问题：",
                "input": sample["question"],
                "output": sample["answer"]
            })
        
        # 角色扮演样本
        all_samples.extend(self.create_role_playing_samples(base_samples))
        
        # 思维链样本
        all_samples.extend(self.create_chain_of_thought_samples(base_samples))
        
        # Few-Shot样本
        all_samples.extend(self.create_few_shot_samples(base_samples))
        
        # 结构化输出样本
        all_samples.extend(self.create_structured_output_samples(base_samples))
        
        return all_samples

def create_prompt_engineering_dataset():
    """创建提示词工程增强的数据集"""
    print("🎯 创建提示词工程增强数据集...")
    
    # 创建目录
    os.makedirs("./data/processed", exist_ok=True)
    os.makedirs("./data/prompt_engineering", exist_ok=True)
    
    # 生成数据集
    dataset_creator = PromptEngineeringDataset()
    enhanced_samples = dataset_creator.generate_enhanced_dataset()
    
    # 分割训练和验证数据
    split_index = int(len(enhanced_samples) * 0.8)
    train_data = enhanced_samples[:split_index]
    val_data = enhanced_samples[split_index:]
    
    # 保存数据
    train_file = "./data/processed/train_prompt_enhanced.json"
    val_file = "./data/processed/val_prompt_enhanced.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    # 保存提示模板
    template_file = "./data/prompt_engineering/prompt_templates.json"
    with open(template_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_creator.prompt_templates, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 提示词工程数据集创建完成：")
    print(f"   训练样本: {len(train_data)} 条")
    print(f"   验证样本: {len(val_data)} 条")
    print(f"   提示模板: {template_file}")
    
    return train_data, val_data

if __name__ == "__main__":
    create_prompt_engineering_dataset()
