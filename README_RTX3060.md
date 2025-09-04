# RTX 3060 大模型微调指南

这是一个专门针对 NVIDIA GeForce RTX 3060 (12GB) 优化的大语言模型SFT微调项目。

## 硬件要求

- **GPU**: NVIDIA GeForce RTX 3060 (12GB显存)
- **内存**: 建议16GB以上
- **存储**: 至少10GB可用空间
- **CUDA**: 11.8或更高版本

## 快速开始

### 1. 环境测试
```bash
# 测试环境是否准备就绪
python test_rtx3060.py
```

### 2. 安装依赖
```bash
# 使用RTX 3060优化版本的依赖
pip install -r requirements_rtx3060.txt
```

### 3. 数据准备
```bash
# 运行数据准备脚本
python data/prepare_dataset.py
```

### 4. 开始训练
```bash
# 使用RTX 3060优化配置
python train.py --config configs/training_config_rtx3060.yaml
```

## 模型选择

### 推荐模型（按显存占用排序）

1. **distilgpt2** (82M参数) - 最推荐
   - 显存占用: ~2GB
   - 训练速度: 快
   - 适合: 初学者，快速实验

2. **gpt2** (124M参数)
   - 显存占用: ~3GB
   - 训练速度: 中等
   - 适合: 平衡性能和速度

3. **microsoft/DialoGPT-medium** (350M参数)
   - 显存占用: ~6GB
   - 训练速度: 较慢
   - 适合: 更好的对话效果

4. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B参数)
   - 显存占用: ~8-10GB（4bit量化）
   - 训练速度: 慢
   - 适合: 追求更好性能

## 配置调优

### 显存不足时的解决方案

1. **减少批大小**
```yaml
per_device_train_batch_size: 2  # 从8减到2
per_device_eval_batch_size: 2
```

2. **减少序列长度**
```yaml
max_seq_length: 256  # 从512减到256
```

3. **使用更小的模型**
```yaml
model_name_or_path: "distilgpt2"  # 使用最小模型
```

4. **启用梯度检查点**
```yaml
gradient_checkpointing: true  # 已默认启用
```

### 性能优化

1. **增加批大小**（如果显存充足）
```yaml
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
```

2. **调整LoRA参数**
```yaml
lora:
  r: 64          # 增加rank提高容量
  lora_alpha: 128
```

## 训练监控

### 显存监控
```python
import torch
print(f"已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"已保留: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

### 温度监控
```bash
# 使用nvidia-smi监控GPU温度
watch -n 1 nvidia-smi
```

## 常见问题

### Q: 显存不足怎么办？
A: 按顺序尝试：
1. 减少batch_size到1
2. 减少max_seq_length到128
3. 使用distilgpt2模型
4. 启用4bit量化

### Q: 训练速度太慢？
A: 优化建议：
1. 使用FP16精度（已默认启用）
2. 适当增加batch_size
3. 减少数据预处理workers
4. 使用SSD存储

### Q: 内存不足？
A: 解决方案：
1. 减少dataloader_num_workers到1
2. 关闭其他程序释放内存
3. 使用更小的数据集

## 性能基准

基于RTX 3060的典型性能：

| 模型 | 参数量 | 批大小 | 序列长度 | 显存占用 | 训练速度 |
|------|--------|--------|----------|----------|----------|
| distilgpt2 | 82M | 8 | 512 | ~4GB | ~2 步/秒 |
| gpt2 | 124M | 6 | 512 | ~6GB | ~1.5 步/秒 |
| DialoGPT-medium | 350M | 4 | 512 | ~8GB | ~1 步/秒 |
| TinyLlama-1.1B | 1.1B | 2 | 256 | ~10GB | ~0.5 步/秒 |

## 高级功能

### 混合精度训练
```yaml
fp16: true  # 已默认启用
```

### 梯度累积
```yaml
gradient_accumulation_steps: 4  # 模拟更大batch_size
```

### 学习率调度
```yaml
lr_scheduler_type: "cosine"
warmup_steps: 30
```

## 故障排除

### 1. CUDA内存错误
```bash
# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 2. 模型下载失败
```bash
# 设置代理或使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 训练中断恢复
```bash
# 从检查点恢复
python train.py --config configs/training_config_rtx3060.yaml --resume_from_checkpoint ./output/checkpoint-100
```

## 联系支持

如果遇到问题：
1. 检查GPU驱动版本
2. 验证CUDA安装
3. 运行测试脚本确认环境
4. 查看训练日志

Happy Training! 🚀
