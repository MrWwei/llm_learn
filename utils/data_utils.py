"""
数据处理工具函数
"""

import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_sharegpt_to_alpaca(conversations: List[Dict]) -> Dict[str, str]:
    """将ShareGPT格式转换为Alpaca格式"""
    if len(conversations) < 2:
        return None
        
    # 找到最后一轮对话
    user_message = None
    assistant_message = None
    
    for i in range(len(conversations) - 1, -1, -1):
        conv = conversations[i]
        if conv.get("from") == "gpt" or conv.get("role") == "assistant":
            assistant_message = conv.get("value") or conv.get("content")
        elif conv.get("from") == "human" or conv.get("role") == "user":
            user_message = conv.get("value") or conv.get("content")
            break
            
    if user_message and assistant_message:
        return {
            "instruction": user_message,
            "input": "",
            "output": assistant_message
        }
    return None

def format_prompt_alpaca(instruction: str, input_text: str = "") -> str:
    """格式化Alpaca风格的prompt"""
    if input_text:
        return f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:"
    else:
        return f"### 指令:\n{instruction}\n\n### 回答:"

def format_prompt_chatglm(instruction: str, input_text: str = "") -> str:
    """格式化ChatGLM风格的prompt"""
    if input_text:
        prompt = f"问：{instruction}\n补充信息：{input_text}\n答："
    else:
        prompt = f"问：{instruction}\n答："
    return prompt

def format_prompt_baichuan(instruction: str, input_text: str = "", system_prompt: str = "") -> str:
    """格式化Baichuan风格的prompt"""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    user_content = instruction
    if input_text:
        user_content += f"\n\n补充信息：{input_text}"
        
    messages.append({"role": "user", "content": user_content})
    
    # 构建对话格式
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<reserved_102>{msg['content']}<reserved_103>"
        elif msg["role"] == "user":
            prompt += f"<reserved_104>{msg['content']}<reserved_105>"
        elif msg["role"] == "assistant":
            prompt += f"<reserved_106>{msg['content']}<reserved_107>"
    
    # 添加助手开始标记
    prompt += "<reserved_106>"
    
    return prompt

def clean_text(text: str) -> str:
    """清理文本"""
    if not text:
        return ""
        
    # 移除多余的空白字符
    text = ' '.join(text.split())
    
    # 移除特殊字符
    text = text.replace('\r', ' ').replace('\n', ' ')
    
    return text.strip()

def split_dataset(data: List[Dict], train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1,
                 seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """分割数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    random.seed(seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]
    
    return train_data, val_data, test_data

def filter_by_length(data: List[Dict], min_length: int = 10, 
                    max_length: int = 2048, field: str = "output") -> List[Dict]:
    """根据长度过滤数据"""
    filtered_data = []
    for item in data:
        text = item.get(field, "")
        if min_length <= len(text) <= max_length:
            filtered_data.append(item)
    
    logger.info(f"长度过滤: {len(data)} -> {len(filtered_data)}")
    return filtered_data

def deduplicate_data(data: List[Dict], key: str = "instruction") -> List[Dict]:
    """数据去重"""
    seen = set()
    unique_data = []
    
    for item in data:
        key_value = item.get(key, "")
        if key_value not in seen:
            seen.add(key_value)
            unique_data.append(item)
    
    logger.info(f"去重处理: {len(data)} -> {len(unique_data)}")
    return unique_data

def validate_data_format(data: List[Dict], required_fields: List[str] = None) -> List[Dict]:
    """验证数据格式"""
    if required_fields is None:
        required_fields = ["instruction", "output"]
    
    valid_data = []
    for item in data:
        if all(field in item and item[field] for field in required_fields):
            valid_data.append(item)
    
    logger.info(f"格式验证: {len(data)} -> {len(valid_data)}")
    return valid_data

def create_conversation_data(instruction: str, response: str, 
                           system_prompt: str = None) -> Dict:
    """创建对话格式数据"""
    conversations = []
    
    if system_prompt:
        conversations.append({
            "role": "system",
            "content": system_prompt
        })
    
    conversations.extend([
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ])
    
    return {"conversations": conversations}

def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """合并多个数据集"""
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    
    logger.info(f"合并数据集，总计 {len(merged)} 条")
    return merged

def sample_data(data: List[Dict], n_samples: int, seed: int = 42) -> List[Dict]:
    """采样数据"""
    if n_samples >= len(data):
        return data
    
    random.seed(seed)
    return random.sample(data, n_samples)

def balance_dataset_by_category(data: List[Dict], category_field: str, 
                              max_per_category: int = None) -> List[Dict]:
    """按类别平衡数据集"""
    from collections import defaultdict
    
    category_data = defaultdict(list)
    for item in data:
        category = item.get(category_field, "unknown")
        category_data[category].append(item)
    
    balanced_data = []
    for category, items in category_data.items():
        if max_per_category and len(items) > max_per_category:
            items = random.sample(items, max_per_category)
        balanced_data.extend(items)
    
    logger.info(f"数据平衡: {len(data)} -> {len(balanced_data)}")
    return balanced_data

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def process_raw_data(self, input_file: str, output_file: str, 
                        data_format: str = "alpaca"):
        """处理原始数据"""
        # 加载数据
        if input_file.endswith('.jsonl'):
            data = load_jsonl(input_file)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        processed_data = []
        
        for item in data:
            if data_format == "sharegpt":
                # 处理ShareGPT格式
                conversations = item.get("conversations", [])
                alpaca_item = convert_sharegpt_to_alpaca(conversations)
                if alpaca_item:
                    processed_data.append(alpaca_item)
            else:
                # 假设已经是Alpaca格式
                processed_data.append(item)
        
        # 数据清理和验证
        processed_data = validate_data_format(processed_data)
        processed_data = deduplicate_data(processed_data)
        processed_data = filter_by_length(processed_data)
        
        # 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据处理完成，保存到: {output_file}")
        return processed_data
