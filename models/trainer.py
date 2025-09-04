"""
训练器实现
基于Transformers Trainer的自定义训练器
"""

import torch
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTDataCollator(DataCollatorForSeq2Seq):
    """SFT专用数据收集器"""
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 处理输入数据
        batch = super().__call__(features)
        
        # 设置labels，忽略pad token的损失
        if "labels" in batch:
            batch["labels"] = torch.where(
                batch["labels"] == self.tokenizer.pad_token_id,
                -100,
                batch["labels"]
            )
            
        return batch

class SFTTrainer(Trainer):
    """监督微调训练器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算损失函数"""
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
        
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """保存模型"""
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # 保存LoRA权重
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)
            
        # 保存tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        logger.info(f"模型已保存到: {output_dir}")

def load_dataset_from_json(file_path: str, tokenizer, max_seq_length: int = 2048) -> Dataset:
    """从JSON文件加载数据集"""
    logger.info(f"加载数据集: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    def format_example(example):
        """格式化单个样本"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        
        # 构建prompt
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 回答:"
            
        # 完整的文本
        full_text = prompt + output_text + tokenizer.eos_token
        
        return {"text": full_text, "prompt": prompt, "response": output_text}
    
    def tokenize_function(examples):
        """tokenize函数"""
        # tokenize完整文本
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # tokenize prompt用于计算标签
        prompt_inputs = tokenizer(
            examples["prompt"],
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # 设置labels，只计算回答部分的损失
        labels = model_inputs["input_ids"].copy()
        prompt_length = len(prompt_inputs["input_ids"])
        
        # 将prompt部分的标签设为-100，不计算损失
        for i in range(prompt_length):
            labels[i] = -100
            
        model_inputs["labels"] = labels
        
        return model_inputs
    
    # 格式化数据
    formatted_data = [format_example(item) for item in data]
    
    # 创建Dataset对象
    dataset = Dataset.from_list(formatted_data)
    
    # tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    logger.info(f"数据集加载完成，共 {len(tokenized_dataset)} 个样本")
    return tokenized_dataset

def create_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """创建训练参数"""
    training_config = config.get("training", {})
    
    args = TrainingArguments(
        output_dir=training_config.get("output_dir", "./output"),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 2e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_steps=training_config.get("warmup_steps", 100),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_strategy=training_config.get("eval_strategy", training_config.get("evaluation_strategy", "steps")),  # 兼容新旧参数名
        eval_steps=training_config.get("eval_steps", 500),
        logging_steps=training_config.get("logging_steps", 50),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        report_to=training_config.get("report_to", "none"),
        run_name=training_config.get("run_name", "llm_sft_lora"),
        ddp_find_unused_parameters=training_config.get("ddp_find_unused_parameters", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        deepspeed=training_config.get("deepspeed", None),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    return args

def setup_trainer(model, tokenizer, training_args, train_dataset, eval_dataset=None):
    """设置训练器"""
    
    # 数据收集器
    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer
