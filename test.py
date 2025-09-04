#!/usr/bin/env python3
"""
模型测试脚本 - 测试训练完成的LoRA模型
支持交互式对话和批量测试
"""

import os
import sys
import torch
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    """模型测试类"""
    
    def __init__(self, model_path="./output/final_model", base_model="distilgpt2"):
        """
        初始化测试器
        
        Args:
            model_path: 训练好的LoRA模型路径
            base_model: 基础模型名称
        """
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型和tokenizer"""
        logger.info("正在加载模型...")
        
        try:
            # 检查模型路径是否存在
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 加载tokenizer
            logger.info(f"加载tokenizer: {self.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            logger.info(f"加载基础模型: {self.base_model}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # 加载LoRA适配器
            logger.info(f"加载LoRA适配器: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info("✅ 模型加载成功！")
            
            # 显示模型信息
            self._show_model_info()
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def _show_model_info(self):
        """显示模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("=" * 50)
        logger.info("模型信息:")
        logger.info(f"  基础模型: {self.base_model}")
        logger.info(f"  LoRA路径: {self.model_path}")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  设备: {next(self.model.parameters()).device}")
        logger.info("=" * 50)
    
    def generate_response(self, prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True):
        """
        生成回答
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            do_sample: 是否采样
            
        Returns:
            生成的回答
        """
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 只返回新生成的部分
            new_response = response[len(prompt):].strip()
            
            return new_response
            
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            return f"错误: {e}"
    
    def test_single_prompt(self, prompt):
        """测试单个提示"""
        logger.info(f"📝 输入: {prompt}")
        response = self.generate_response(prompt)
        logger.info(f"🤖 输出: {response}")
        print(f"\n{'='*60}")
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        print(f"{'='*60}\n")
        return response
    
    def run_batch_tests(self):
        """运行批量测试"""
        logger.info("🧪 开始批量测试...")
        
        test_prompts = [
            "Human: What is machine learning?\nAssistant:",
            "Human: 请介绍一下人工智能\nAssistant:",
            "Human: How does a neural network work?\nAssistant:",
            "Human: 什么是深度学习？\nAssistant:",
            "Human: Explain the concept of supervised learning\nAssistant:",
            "Human: 请解释什么是自然语言处理\nAssistant:",
            "Human: What are the benefits of using AI?\nAssistant:",
            "Human: 机器学习有哪些应用？\nAssistant:"
        ]
        
        results = []
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"测试 {i}/{len(test_prompts)}")
            response = self.test_single_prompt(prompt)
            results.append({"prompt": prompt, "response": response})
        
        return results
    
    def interactive_chat(self):
        """交互式聊天"""
        logger.info("🎯 进入交互式聊天模式")
        print("\n" + "="*60)
        print("🤖 AI助手已就绪！输入 'quit' 或 'exit' 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 格式化输入
                prompt = f"Human: {user_input}\nAssistant:"
                
                # 生成回答
                response = self.generate_response(prompt, max_new_tokens=150)
                print(f"🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 出错了: {e}")
    
    def save_test_results(self, results, filename="test_results.json"):
        """保存测试结果"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 测试结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")

def main():
    """主函数"""
    print("🚀 LoRA模型测试器启动")
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("⚠️  使用CPU模式")
    
    # 创建测试器
    tester = ModelTester()
    
    try:
        # 加载模型
        tester.load_model()
        
        # 询问测试模式
        print("\n请选择测试模式:")
        print("1. 批量测试 (运行预设的测试用例)")
        print("2. 交互式聊天 (与AI进行对话)")
        print("3. 单次测试 (输入一个问题)")
        
        while True:
            choice = input("\n请输入选择 (1/2/3): ").strip()
            
            if choice == "1":
                # 批量测试
                results = tester.run_batch_tests()
                tester.save_test_results(results)
                break
                
            elif choice == "2":
                # 交互式聊天
                tester.interactive_chat()
                break
                
            elif choice == "3":
                # 单次测试
                prompt = input("请输入您的问题: ").strip()
                if prompt:
                    formatted_prompt = f"Human: {prompt}:\nAssistant:"
                    tester.test_single_prompt(formatted_prompt)
                break
                
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
    
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
