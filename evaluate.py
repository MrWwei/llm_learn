#!/usr/bin/env python3
"""
模型评估脚本
对微调后的模型进行全面评估
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.lora_model import ModelForInference
from utils.eval_utils import ModelEvaluator, evaluate_generation_quality, calculate_diversity_metrics
from utils.train_utils import load_config, setup_logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型评估")
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
        "--test_file", 
        type=str, 
        required=True,
        help="测试数据文件路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="配置文件路径"
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
        "--batch_size", 
        type=int, 
        default=1,
        help="批处理大小"
    )
    parser.add_argument(
        "--num_samples", 
        type=int,
        help="评估样本数量（如果不指定则使用全部）"
    )
    
    return parser.parse_args()

def load_test_data(test_file: str, num_samples: int = None):
    """加载测试数据"""
    logger.info(f"加载测试数据: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        if test_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    if num_samples and num_samples < len(data):
        import random
        random.seed(42)
        data = random.sample(data, num_samples)
        logger.info(f"随机采样 {num_samples} 个样本进行评估")
    
    logger.info(f"加载了 {len(data)} 个测试样本")
    return data

def evaluate_model_predictions(model_inference: ModelForInference, test_data: list, 
                             generation_kwargs: dict, output_dir: str):
    """评估模型预测结果"""
    logger.info("开始生成预测结果...")
    
    predictions = []
    references = []
    detailed_results = []
    
    for i, example in enumerate(test_data):
        logger.info(f"处理样本 {i+1}/{len(test_data)}")
        
        # 提取数据
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        reference = example.get('output', '')
        
        if not instruction:
            logger.warning(f"样本 {i+1} 缺少指令，跳过")
            continue
        
        # 构建prompt
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 回答:"
        
        try:
            # 生成预测
            prediction = model_inference.generate_response(prompt, **generation_kwargs)
            
            predictions.append(prediction)
            references.append(reference)
            
            # 保存详细结果
            detailed_results.append({
                'index': i,
                'instruction': instruction,
                'input': input_text,
                'reference': reference,
                'prediction': prediction
            })
            
        except Exception as e:
            logger.error(f"处理样本 {i+1} 时出错: {str(e)}")
            predictions.append(f"ERROR: {str(e)}")
            references.append(reference)
            
            detailed_results.append({
                'index': i,
                'instruction': instruction,
                'input': input_text,
                'reference': reference,
                'prediction': f"ERROR: {str(e)}"
            })
    
    # 保存详细预测结果
    predictions_file = os.path.join(output_dir, "predictions.json")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"详细预测结果已保存到: {predictions_file}")
    
    return predictions, references, detailed_results

def calculate_metrics(predictions: list, references: list, output_dir: str):
    """计算评估指标"""
    logger.info("计算评估指标...")
    
    # 过滤掉错误的预测
    valid_predictions = []
    valid_references = []
    
    for pred, ref in zip(predictions, references):
        if not pred.startswith("ERROR:"):
            valid_predictions.append(pred)
            valid_references.append(ref)
    
    logger.info(f"有效预测数量: {len(valid_predictions)}/{len(predictions)}")
    
    if not valid_predictions:
        logger.error("没有有效的预测结果")
        return {}
    
    # 计算生成质量指标
    quality_metrics = evaluate_generation_quality(valid_predictions, valid_references)
    
    # 计算多样性指标
    diversity_metrics = calculate_diversity_metrics(valid_predictions)
    
    # 合并所有指标
    all_metrics = {
        'quality_metrics': quality_metrics,
        'diversity_metrics': diversity_metrics,
        'statistics': {
            'total_samples': len(predictions),
            'valid_samples': len(valid_predictions),
            'error_samples': len(predictions) - len(valid_predictions),
            'success_rate': len(valid_predictions) / len(predictions) * 100 if predictions else 0
        }
    }
    
    # 保存指标
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估指标已保存到: {metrics_file}")
    
    # 打印主要指标
    logger.info("评估结果:")
    logger.info(f"  成功率: {all_metrics['statistics']['success_rate']:.2f}%")
    
    for key, value in quality_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.2f}")
    
    for key, value in diversity_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
    
    return all_metrics

def generate_evaluation_report(metrics: dict, detailed_results: list, output_dir: str):
    """生成评估报告"""
    logger.info("生成评估报告...")
    
    report_lines = []
    report_lines.append("# 模型评估报告\n")
    
    # 基本统计
    stats = metrics.get('statistics', {})
    report_lines.append("## 基本统计")
    report_lines.append(f"- 总样本数: {stats.get('total_samples', 0)}")
    report_lines.append(f"- 有效样本数: {stats.get('valid_samples', 0)}")
    report_lines.append(f"- 错误样本数: {stats.get('error_samples', 0)}")
    report_lines.append(f"- 成功率: {stats.get('success_rate', 0):.2f}%\n")
    
    # 质量指标
    quality_metrics = metrics.get('quality_metrics', {})
    if quality_metrics:
        report_lines.append("## 生成质量指标")
        for key, value in quality_metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"- {key}: {value:.2f}")
        report_lines.append("")
    
    # 多样性指标
    diversity_metrics = metrics.get('diversity_metrics', {})
    if diversity_metrics:
        report_lines.append("## 多样性指标")
        for key, value in diversity_metrics.items():
            if isinstance(value, (int, float)):
                report_lines.append(f"- {key}: {value:.4f}")
        report_lines.append("")
    
    # 样本分析
    report_lines.append("## 样本分析")
    
    # 找出最好和最差的几个样本（基于长度作为简单指标）
    valid_samples = [r for r in detailed_results if not r['prediction'].startswith('ERROR:')]
    
    if valid_samples:
        # 按预测长度排序
        valid_samples.sort(key=lambda x: len(x['prediction']), reverse=True)
        
        report_lines.append("### 最长回答样本:")
        if valid_samples:
            sample = valid_samples[0]
            report_lines.append(f"**指令**: {sample['instruction']}")
            report_lines.append(f"**预测**: {sample['prediction'][:200]}...")
            report_lines.append("")
        
        report_lines.append("### 最短回答样本:")
        if valid_samples:
            sample = valid_samples[-1]
            report_lines.append(f"**指令**: {sample['instruction']}")
            report_lines.append(f"**预测**: {sample['prediction']}")
            report_lines.append("")
    
    # 保存报告
    report_file = os.path.join(output_dir, "evaluation_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"评估报告已保存到: {report_file}")

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(
        log_level="INFO",
        log_file=os.path.join(args.output_dir, "evaluation.log")
    )
    
    # 加载配置（如果提供）
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True
    }
    
    # 从配置文件更新生成参数
    model_config = config.get('model', {})
    generation_config = model_config.get('generation', {})
    generation_kwargs.update(generation_config)
    
    try:
        # 加载测试数据
        test_data = load_test_data(args.test_file, args.num_samples)
        
        # 初始化推理模型
        logger.info("加载模型...")
        model_inference = ModelForInference(
            model_path=args.model_path,
            base_model_path=args.base_model_path
        )
        model_inference.load_model()
        logger.info("模型加载完成")
        
        # 生成预测结果
        predictions, references, detailed_results = evaluate_model_predictions(
            model_inference, test_data, generation_kwargs, args.output_dir
        )
        
        # 计算评估指标
        metrics = calculate_metrics(predictions, references, args.output_dir)
        
        # 生成评估报告
        generate_evaluation_report(metrics, detailed_results, args.output_dir)
        
        logger.info(f"评估完成！结果保存在: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"评估过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
