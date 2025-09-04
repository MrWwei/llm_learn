# """
# LoRA模型实现
# 结合PEFT库实现高效的LoRA微调
# """

# import torch
# import torch.nn as nn
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TrainingArguments
# )
# from peft import (
#     LoraConfig, 
#     get_peft_model, 
#     TaskType,
#     prepare_model_for_kbit_training
# )
# import yaml
# from typing import Dict, Any, Optional
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class LoRAModelManager:
#     """LoRA模型管理器"""
    
#     def __init__(self, config: Dict[str, Any]):
#         self.config = config
#         self.model = None
#         self.tokenizer = None
#         self.peft_config = None
        
#     def load_base_model(self, model_name_or_path: str, 
#                        quantization_config: Optional[Dict] = None,
#                        device_map: str = "auto"):
#         """加载基础模型"""
#         logger.info(f"加载基础模型: {model_name_or_path}")
        
#         # 配置量化
#         bnb_config = None
#         if quantization_config and quantization_config.get("load_in_4bit", False):
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=getattr(torch, quantization_config.get("bnb_4bit_compute_dtype", "float16")),
#                 bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
#                 bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
#             )
            
#         # 加载模型
#         model_kwargs = {
#             "torch_dtype": torch.float16,
#             "device_map": device_map,
#             "trust_remote_code": self.config.get("trust_remote_code", True)
#         }
        
#         if bnb_config:
#             model_kwargs["quantization_config"] = bnb_config
            
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path,
#             **model_kwargs
#         )
        
#         # 加载tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name_or_path,
#             trust_remote_code=self.config.get("trust_remote_code", True),
#             padding_side="left"
#         )
        
#         # 设置pad_token
#         if self.tokenizer.pad_token is None:
#             if self.tokenizer.eos_token is not None:
#                 self.tokenizer.pad_token = self.tokenizer.eos_token
#             else:
#                 # 对于GPT2/DistilGPT2，使用eos_token作为pad_token
#                 self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
            
#         logger.info("基础模型加载完成")
#         return self.model, self.tokenizer
        
#     def setup_lora_config(self, lora_config: Dict[str, Any]):
#         """设置LoRA配置"""
#         self.peft_config = LoraConfig(
#             r=lora_config.get("r", 8),
#             lora_alpha=lora_config.get("lora_alpha", 32),
#             lora_dropout=lora_config.get("lora_dropout", 0.1),
#             target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
#             bias=lora_config.get("bias", "none"),
#             task_type=getattr(TaskType, lora_config.get("task_type", "CAUSAL_LM")),
#         )
#         logger.info(f"LoRA配置: {self.peft_config}")
        
#     def prepare_model_for_training(self):
#         """准备模型进行训练"""
#         if self.model is None:
#             raise ValueError("请先加载基础模型")
            
#         # 准备量化模型
#         if hasattr(self.model, "config") and getattr(self.model.config, "quantization_config", None):
#             self.model = prepare_model_for_kbit_training(self.model)
            
#         # 应用LoRA
#         if self.peft_config:
#             self.model = get_peft_model(self.model, self.peft_config)
            
#         # 确保模型处于训练模式
#         self.model.train()
        
#         # 确保LoRA参数需要梯度
#         for name, param in self.model.named_parameters():
#             if "lora_" in name or "adapter" in name:
#                 param.requires_grad = True
#                 logger.info(f"设置参数 {name} 需要梯度")
#             else:
#                 param.requires_grad = False
        
#         # 验证是否有参数需要梯度
#         trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
#         if not trainable_params:
#             logger.warning("警告：没有参数需要梯度！")
#             # 如果没有LoRA参数，可能需要重新检查配置
#             for name, param in self.model.named_parameters():
#                 if any(target in name for target in self.peft_config.target_modules):
#                     param.requires_grad = True
#                     logger.info(f"强制设置参数 {name} 需要梯度")
        
#         # 打印可训练参数
#         self.print_trainable_parameters()
        
#         return self.model
        
#     def print_trainable_parameters(self):
#         """打印可训练参数统计"""
#         trainable_params = 0
#         all_param = 0
        
#         for _, param in self.model.named_parameters():
#             all_param += param.numel()
#             if param.requires_grad:
#                 trainable_params += param.numel()
                
#         logger.info(
#             f"可训练参数: {trainable_params:,} || "
#             f"总参数: {all_param:,} || "
#             f"可训练参数比例: {100 * trainable_params / all_param:.2f}%"
#         )
        
#     def save_model(self, output_dir: str):
#         """保存LoRA模型"""
#         if self.model is None:
#             raise ValueError("模型未初始化")
            
#         self.model.save_pretrained(output_dir)
#         if self.tokenizer:
#             self.tokenizer.save_pretrained(output_dir)
#         logger.info(f"模型已保存到: {output_dir}")
        
#     def load_lora_model(self, model_path: str, base_model_name_or_path: str):
#         """加载LoRA微调后的模型"""
#         from peft import PeftModel
        
#         # 加载基础模型
#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_name_or_path,
#             torch_dtype=torch.float16,
#             device_map="auto",
#             trust_remote_code=True
#         )
        
#         # 加载LoRA权重
#         self.model = PeftModel.from_pretrained(base_model, model_path)
        
#         # 加载tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             trust_remote_code=True
#         )
        
#         logger.info(f"LoRA模型加载完成: {model_path}")
#         return self.model, self.tokenizer

# def create_model_from_config(config_path: str) -> LoRAModelManager:
#     """从配置文件创建模型管理器"""
#     with open(config_path, 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)
    
#     model_manager = LoRAModelManager(config)
    
#     # 加载基础模型
#     training_config = config.get("training", {})
#     model_manager.load_base_model(
#         training_config.get("model_name_or_path"),
#         training_config.get("quantization")
#     )
    
#     # 设置LoRA配置
#     lora_config = training_config.get("lora", {})
#     model_manager.setup_lora_config(lora_config)
    
#     # 准备训练
#     model_manager.prepare_model_for_training()
    
#     return model_manager

# class ModelForInference:
#     """推理专用模型类"""
    
#     def __init__(self, model_path: str, base_model_path: str = None):
#         self.model_path = model_path
#         self.base_model_path = base_model_path
#         self.model = None
#         self.tokenizer = None
        
#     def load_model(self):
#         """加载模型用于推理"""
#         if self.base_model_path:
#             # 加载LoRA微调模型
#             manager = LoRAModelManager({})
#             self.model, self.tokenizer = manager.load_lora_model(
#                 self.model_path, self.base_model_path
#             )
#         else:
#             # 直接加载完整模型
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_path,
#                 trust_remote_code=True
#             )
            
#         self.model.eval()
        
#     def generate_response(self, prompt: str, **generation_kwargs) -> str:
#         """生成回复"""
#         if self.model is None or self.tokenizer is None:
#             self.load_model()
            
#         inputs = self.tokenizer(prompt, return_tensors="pt")
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
#         # 默认生成参数
#         default_kwargs = {
#             "max_new_tokens": 512,
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "do_sample": True,
#             "repetition_penalty": 1.1,
#             "pad_token_id": self.tokenizer.eos_token_id
#         }
#         default_kwargs.update(generation_kwargs)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 **default_kwargs
#             )
            
#         response = self.tokenizer.decode(
#             outputs[0][inputs["input_ids"].shape[1]:], 
#             skip_special_tokens=True
#         )
        
#         return response.strip()
