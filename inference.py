#!/usr/bin/env python3
"""
模型推理脚本
支持LoRA微调后的模型推理
"""

import os
import sys

# 设置环境变量解决tokenizers并行处理问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
from pathlib import Path
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.lora_model import ModelForInference
from utils.train_utils import load_config, setup_logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="微调后的模型路径"
    )
    parser.add_argument(
        "--base_model_path", 
        type=str,
        help="基础模型路径（LoRA微调时需要）"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="配置文件路径"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="交互式对话模式"
    )
    parser.add_argument(
        "--input_file", 
        type=str,
        help="输入文件路径（批量推理）"
    )
    parser.add_argument(
        "--output_file", 
        type=str,
        help="输出文件路径（批量推理）"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="生成温度"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="nucleus sampling参数"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="top-k sampling参数"
    )
    parser.add_argument(
        "--repetition_penalty", 
        type=float, 
        default=1.1,
        help="重复惩罚"
    )
    
    return parser.parse_args()

def format_prompt(instruction: str, input_text: str = "") -> str:
    """格式化输入prompt"""
    if input_text:
        return f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:"
    else:
        return f"### 指令:\n{instruction}\n\n### 回答:"

def interactive_chat(model_inference: ModelForInference, generation_kwargs: dict):
    """交互式对话"""
    logger.info("进入交互式对话模式，输入'quit'或'exit'退出")
    
    while True:
        try:
            # 获取用户输入
            instruction = input("\n请输入指令: ").strip()
            
            if instruction.lower() in ['quit', 'exit', '退出']:
                logger.info("退出对话")
                break
            
            if not instruction:
                continue
            
            # 构建prompt
            prompt = format_prompt(instruction)
            
            # 生成回复
            logger.info("生成中...")
            response = model_inference.generate_response(prompt, **generation_kwargs)
            
            # 显示结果
            print(f"\n助手: {response}")
            
        except KeyboardInterrupt:
            logger.info("\n检测到中断信号，退出对话")
            break
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")

def batch_inference(model_inference: ModelForInference, input_file: str, 
                   output_file: str, generation_kwargs: dict):
    """批量推理"""
    import json
    
    logger.info(f"从 {input_file} 加载数据进行批量推理")
    
    # 加载输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    results = []
    
    for i, item in enumerate(data):
        logger.info(f"处理第 {i+1}/{len(data)} 个样本")
        
        # 提取指令和输入
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        
        if not instruction:
            logger.warning(f"样本 {i+1} 缺少指令，跳过")
            continue
        
        # 构建prompt
        prompt = format_prompt(instruction, input_text)
        
        try:
            # 生成回复
            response = model_inference.generate_response(prompt, **generation_kwargs)
            
            # 保存结果
            result = {
                'instruction': instruction,
                'input': input_text,
                'output': response,
                'expected_output': item.get('output', '')  # 如果有预期输出
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"处理样本 {i+1} 时出错: {str(e)}")
            results.append({
                'instruction': instruction,
                'input': input_text,
                'output': f"ERROR: {str(e)}",
                'expected_output': item.get('output', '')
            })
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"批量推理完成，结果保存到: {output_file}")

def single_inference(model_inference: ModelForInference, instruction: str, 
                    input_text: str = "", generation_kwargs: dict = None):
    """单次推理"""
    if generation_kwargs is None:
        generation_kwargs = {}
    
    prompt = format_prompt(instruction, input_text)
    response = model_inference.generate_response(prompt, **generation_kwargs)
    
    print(f"指令: {instruction}")
    if input_text:
        print(f"输入: {input_text}")
    print(f"回答: {response}")
    
    return response

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(log_level="INFO")
    
    # 加载配置（如果提供）
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": True
    }
    
    # 从配置文件更新生成参数
    model_config = config.get('model', {})
    generation_config = model_config.get('generation', {})
    generation_kwargs.update(generation_config)
    
    try:
        # 初始化推理模型
        logger.info("加载模型...")
        model_inference = ModelForInference(
            model_path=args.model_path,
            base_model_path=args.base_model_path
        )
        model_inference.load_model()
        logger.info("模型加载完成")
        
        # 根据模式进行推理
        if args.interactive:
            # 交互式模式
            interactive_chat(model_inference, generation_kwargs)
            
        elif args.input_file and args.output_file:
            # 批量推理模式
            batch_inference(model_inference, args.input_file, args.output_file, generation_kwargs)
            
        else:
            # 演示模式
            logger.info("演示模式 - 使用示例指令进行推理")
            
            demo_examples = [
                {
                    "instruction": "解释什么是机器学习",
                    "input": ""
                },
                {
                    "instruction": "用Python写一个冒泡排序算法",
                    "input": ""
                },
                {
                    "instruction": "翻译以下英文句子",
                    "input": "The weather is beautiful today."
                }
            ]
            
            for example in demo_examples:
                print("\n" + "="*50)
                single_inference(
                    model_inference, 
                    example["instruction"], 
                    example["input"], 
                    generation_kwargs
                )
    
    except Exception as e:
        logger.error(f"推理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
