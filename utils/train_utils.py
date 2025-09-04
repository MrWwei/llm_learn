"""
训练工具函数
"""

import torch
import numpy as np
import random
import os
import logging
from typing import Dict, Any, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保可重现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_device() -> torch.device:
    """获取设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    return device

def print_gpu_memory():
    """打印GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU内存 - 已分配: {allocated:.2f} GB, 已缓存: {cached:.2f} GB")

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """保存配置文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def create_output_dir(output_dir: str):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

def count_parameters(model) -> Dict[str, int]:
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params * 100
    }

def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def calculate_model_size(model) -> Dict[str, float]:
    """计算模型大小"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    return {
        "model_size_mb": model_size,
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024
    }

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志"""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": []
        }
        
    def log_metrics(self, metrics: Dict[str, float], step: int, epoch: int):
        """记录指标"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        self.metrics["step"].append(step)
        self.metrics["epoch"].append(epoch)
        
    def get_best_metric(self, metric_name: str, mode: str = "min") -> float:
        """获取最佳指标"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
            
        values = self.metrics[metric_name]
        if mode == "min":
            return min(values)
        else:
            return max(values)
    
    def save_metrics(self, filepath: str):
        """保存指标"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

def check_memory_usage():
    """检查内存使用情况"""
    import psutil
    
    # CPU内存
    memory = psutil.virtual_memory()
    logger.info(f"CPU内存使用: {memory.percent}% ({memory.used / 1e9:.1f} GB / {memory.total / 1e9:.1f} GB)")
    
    # GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i} 内存: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total")

def optimize_memory():
    """优化内存使用"""
    import gc
    
    # 清理Python垃圾
    gc.collect()
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("内存优化完成")

def save_checkpoint(model, tokenizer, optimizer, scheduler, step: int, 
                   output_dir: str, config: Dict[str, Any]):
    """保存检查点"""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存模型
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(checkpoint_dir)
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
    
    # 保存tokenizer
    if tokenizer:
        tokenizer.save_pretrained(checkpoint_dir)
    
    # 保存优化器和调度器状态
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'config': config
    }, os.path.join(checkpoint_dir, "training_state.pt"))
    
    logger.info(f"检查点已保存到: {checkpoint_dir}")

def load_checkpoint(checkpoint_dir: str, model, optimizer=None, scheduler=None):
    """加载检查点"""
    # 加载模型
    if hasattr(model, 'load_state_dict'):
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
    
    # 加载训练状态
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        
        if optimizer and 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in state and state['scheduler_state_dict']:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        
        step = state.get('step', 0)
        config = state.get('config', {})
        
        logger.info(f"检查点已加载，步数: {step}")
        return step, config
    
    return 0, {}

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件"""
    required_keys = [
        "training.model_name_or_path",
        "training.output_dir",
        "training.data_path"
    ]
    
    def get_nested_value(d, key_path):
        keys = key_path.split('.')
        value = d
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    for key in required_keys:
        if get_nested_value(config, key) is None:
            logger.error(f"配置文件缺少必需的键: {key}")
            return False
    
    return True
