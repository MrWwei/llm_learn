#!/usr/bin/env python3
"""
快速演示脚本 - 展示提示词工程和RAG技术的效果
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag_system import RAGSystem

def quick_demo():
    """快速演示三种模式的效果差异"""
    print("🚀 大模型SFT + 提示词工程 + RAG 技术演示")
    print("="*70)
    
    # 检查模型是否存在
    model_path = "./output/final_model"
    base_model = "distilgpt2"
    
    print("📦 加载模型...")
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if os.path.exists(model_path):
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
        model = PeftModel.from_pretrained(base, model_path)
        print("✅ 使用微调后的模型")
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
        print("⚠️ 使用基础模型进行演示")
    
    model.eval()
    
    # 初始化RAG系统
    print("🔍 初始化RAG系统...")
    try:
        from rag_system import KnowledgeBase
        kb = KnowledgeBase()
        kb.build_index()
        print("✅ RAG系统准备完成")
    except Exception as e:
        print(f"⚠️ RAG系统初始化失败: {e}")
        kb = None
    
    # 演示问题
    demo_question = "什么是机器学习？"
    
    print(f"\n🎯 演示问题: {demo_question}")
    print("="*70)
    
    # 1. 基础模式
    print("\n🔸 模式1: 基础回答")
    print("-" * 30)
    basic_prompt = f"Human: {demo_question}\nAssistant:"
    inputs = tokenizer(basic_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    basic_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    basic_answer = basic_response[len(basic_prompt):].strip()
    print(f"回答: {basic_answer}")
    
    # 2. 提示词工程模式
    print("\n🔸 模式2: 提示词工程增强")
    print("-" * 30)
    pe_prompt = f"""你是一名资深的人工智能专家。请按照以下结构详细回答问题：

【定义】
【主要类型】
【应用场景】

问题: {demo_question}

请提供专业、准确的回答:"""
    
    inputs = tokenizer(pe_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    pe_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pe_answer = pe_response[len(pe_prompt):].strip()
    print(f"回答: {pe_answer}")
    
    # 3. RAG增强模式
    print("\n🔸 模式3: RAG检索增强")
    print("-" * 30)
    
    if kb:
        # 检索相关文档
        relevant_docs = kb.search(demo_question, top_k=2)
        
        if relevant_docs:
            print(f"📚 检索到 {len(relevant_docs)} 个相关文档:")
            for doc in relevant_docs:
                print(f"   - {doc['metadata']['title']} (相似度: {doc['similarity']:.3f})")
            
            # 构建RAG提示
            context = "\n".join([f"参考资料: {doc['content'][:150]}..." for doc in relevant_docs])
            rag_prompt = f"""{context}

基于上述参考资料，请回答: {demo_question}

回答:"""
            
            inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            
            rag_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            rag_answer = rag_response[len(rag_prompt):].strip()
            print(f"回答: {rag_answer}")
        else:
            print("❌ 未检索到相关文档")
    else:
        print("❌ RAG系统不可用")
    
    # 总结
    print("\n" + "="*70)
    print("📊 三种模式对比总结:")
    print("="*70)
    print("🔸 基础模式:")
    print("   - 优点: 速度快，资源占用少")
    print("   - 缺点: 回答可能不够准确或详细")
    print()
    print("🔸 提示词工程:")
    print("   - 优点: 回答更结构化，质量提升")
    print("   - 缺点: 仍依赖模型训练时的知识")
    print()
    print("🔸 RAG增强:")
    print("   - 优点: 基于实际知识库，回答准确可靠")
    print("   - 缺点: 速度相对较慢，需要维护知识库")
    print()
    print("🎯 推荐使用场景:")
    print("   - 快速问答: 基础模式")
    print("   - 结构化输出: 提示词工程")
    print("   - 专业领域: RAG增强模式")
    print("="*70)

if __name__ == "__main__":
    quick_demo()
