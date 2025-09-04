#!/usr/bin/env python3
"""
修复版训练脚本 - 解决梯度计算问题
专为RTX 3060优化
"""

import os
import sys

# 设置环境变量解决tokenizers并行处理问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_DISABLED"] = "true"

import torch
import json
import logging
import yaml
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(config_path):
    """设置模型和tokenizer"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get("training", {})
    model_name = training_config.get("model_name_or_path", "distilgpt2")
    
    logger.info(f"加载模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 获取LoRA配置
    lora_config_dict = training_config.get("lora", {})
    lora_config = LoraConfig(
        r=lora_config_dict.get("r", 16),
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.1),
        target_modules=lora_config_dict.get("target_modules", ["c_attn", "c_proj"]),
        bias=lora_config_dict.get("bias", "none"),
        task_type=getattr(TaskType, lora_config_dict.get("task_type", "CAUSAL_LM")),
    )
    
    logger.info(f"LoRA配置: {lora_config}")
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 确保模型处于训练模式
    model.train()
    
    # 显式设置LoRA参数需要梯度
    trainable_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_count += 1
            logger.info(f"设置LoRA参数 {name} 需要梯度")
        else:
            param.requires_grad = False
    
    if trainable_count == 0:
        logger.error("错误：没有找到LoRA参数！")
        # 打印所有参数名称进行调试
        for name, param in model.named_parameters():
            logger.info(f"参数: {name}, requires_grad: {param.requires_grad}")
        raise ValueError("没有可训练的LoRA参数")
    
    # 验证梯度设置
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, tokenizer, config

def load_data(data_path):
    """加载训练数据"""
    logger.info(f"加载数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为简单的文本格式
    texts = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        if input_text:
            text = f"Human: {instruction}\nInput: {input_text}\nAssistant: {output_text}"
        else:
            text = f"Human: {instruction}\nAssistant: {output_text}"
        
        texts.append({"text": text})
    
    return Dataset.from_list(texts)

def tokenize_data(dataset, tokenizer, max_length=256):
    """tokenize数据"""
    def tokenize_function(examples):
        # 简单tokenization，自动处理labels
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",  # 启用填充
            max_length=max_length,
            return_tensors=None
        )
        # 对于因果语言模型，labels与input_ids相同
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        batched=True
    )
    
    return tokenized_dataset

def main():
    """主函数"""
    logger.info("RTX 3060 修复版训练启动")
    logger.info("="*50)
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # 配置文件路径
        config_path = "configs/training_config_rtx3060.yaml"
        
        # 设置模型
        model, tokenizer, config = setup_model_and_tokenizer(config_path)
        
        # 加载数据
        train_dataset = load_data("./data/processed/train.json")
        tokenized_dataset = tokenize_data(train_dataset, tokenizer)
        
        # 验证数据
        eval_dataset = None
        if os.path.exists("./data/processed/val.json"):
            eval_data = load_data("./data/processed/val.json")
            eval_dataset = tokenize_data(eval_data, tokenizer)
        
        # 训练参数
        training_config = config.get("training", {})
        training_args = TrainingArguments(
            output_dir=training_config.get("output_dir", "./output"),
            num_train_epochs=training_config.get("num_train_epochs", 300),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 0.0002),
            weight_decay=training_config.get("weight_decay", 0.01),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            warmup_steps=training_config.get("warmup_steps", 10),
            logging_steps=training_config.get("logging_steps", 1),
            save_steps=training_config.get("save_steps", 50),
            save_total_limit=training_config.get("save_total_limit", 2),
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=training_config.get("eval_steps", 50),
            fp16=training_config.get("fp16", True),
            dataloader_num_workers=0,  # 避免多进程问题
            remove_unused_columns=False,
            gradient_checkpointing=False,  # 暂时禁用以简化调试
            report_to="none",
            load_best_model_at_end=False,  # 暂时禁用
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言模型
            pad_to_multiple_of=8,  # 优化GPU性能
        )
        
        # 自定义训练器，确保梯度计算正确
        class FixedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """自定义损失计算"""
                labels = inputs.get("labels")
                outputs = model(**inputs)
                
                if labels is not None:
                    # 确保labels的形状正确
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # 计算损失
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss = outputs.loss
                
                return (loss, outputs) if return_outputs else loss
        
        # 训练器
        trainer = FixedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # 验证模型状态
        logger.info("验证模型状态...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"可训练参数: {name}, shape: {param.shape}")
        
        # 开始训练
        logger.info("开始训练...")
        train_result = trainer.train()
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model("./output/final_model")
        
        logger.info("训练完成！")
        logger.info(f"训练结果: {train_result.metrics}")
        
        # 简单测试
        logger.info("进行简单测试...")
        # test_text = "Human: What is machine learning?\nAssistant:"
        test_text = "请介绍一下人工智能的发展历史\n:"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"测试输出: {response}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
