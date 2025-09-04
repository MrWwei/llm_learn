#!/usr/bin/env python3
"""
自动批量测试脚本 - 无需交互输入
"""

import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def auto_test():
    """自动测试函数"""
    print("🚀 开始自动测试训练好的模型...")
    print("="*60)
    
    model_path = "./output/final_model"
    base_model = "distilgpt2"
    
    # 检查模型
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    try:
        print("📦 加载模型和tokenizer...")
        
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
        
        print("✅ 模型加载成功！")
        
        # 显示模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型信息:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   设备: {next(model.parameters()).device}")
        
        # 测试用例
        test_cases = [
            {
                "category": "英文AI问题",
                "prompts": [
                    "What is machine learning?",
                    "How does artificial intelligence work?",
                    "Explain neural networks",
                    "What are the benefits of AI?"
                ]
            },
            {
                "category": "中文AI问题", 
                "prompts": [
                    "什么是人工智能？",
                    "机器学习有什么用？",
                    "深度学习是什么？",
                    "神经网络如何工作？"
                ]
            },
            {
                "category": "技术问题",
                "prompts": [
                    "How does a computer work?",
                    "What is programming?",
                    "Explain algorithms",
                    "什么是数据结构？"
                ]
            }
        ]
        
        results = []
        test_count = 0
        
        print("\n🧪 开始批量测试...")
        print("="*60)
        
        for category_data in test_cases:
            category = category_data["category"]
            prompts = category_data["prompts"]
            
            print(f"\n📂 测试类别: {category}")
            print("-" * 40)
            
            for i, question in enumerate(prompts, 1):
                test_count += 1
                prompt = f"Human: {question}\nAssistant:"
                
                print(f"\n测试 {test_count}: {question}")
                
                # 生成回答
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response[len(prompt):].strip()
                
                print(f"回答: {answer}")
                
                # 保存结果
                results.append({
                    "category": category,
                    "question": question,
                    "prompt": prompt,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
        
        # 保存测试结果
        output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("✅ 测试完成！")
        print(f"📊 总共测试: {test_count} 个问题")
        print(f"💾 结果保存至: {output_file}")
        print("="*60)
        
        # 显示部分结果摘要
        print("\n📋 测试结果摘要:")
        for category_data in test_cases:
            category = category_data["category"]
            category_results = [r for r in results if r["category"] == category]
            print(f"  {category}: {len(category_results)} 个测试")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    auto_test()
